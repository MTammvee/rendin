#!/usr/bin/env python3
"""
Stage 1: Back-project depth frames into a world-space RGB point cloud.

Pipeline per frame:
  1. Load depth (uint16 mm), confidence (uint8), RGB (HEIC), and JSON metadata.
  2. Skip frames where ARKit tracking is not "normal" (bad extrinsics).
  3. Scale intrinsics from calibration resolution (1920×1440) to depth resolution (192×144).
  4. Back-project valid depth pixels into camera space, then world space via worldFromCamera.
  5. Color each point from the corresponding RGB pixel.

Bonus applied:
  - Voxel downsampling (2 cm grid).
  - Statistical outlier removal (k-NN distance thresholding).
"""

import json
import sys
import numpy as np
from pathlib import Path

from pillow_heif import register_heif_opener
from PIL import Image

register_heif_opener()

DATASETS_ROOT = Path("/Users/martin/Documents/martin/rendin/datasets")
OUTPUT_ROOT   = Path("/Users/martin/Documents/martin/rendin/output")
OUTPUT_ROOT.mkdir(exist_ok=True)

# Accept dataset name as CLI arg, default to continuous_frames
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "continuous_frames"
DATA_DIR   = DATASETS_ROOT / dataset_name
OUTPUT_PLY = OUTPUT_ROOT / f"stage1_{dataset_name}.ply"

MAX_DEPTH_M    = 5.0   # discard points farther than 5 m
MIN_CONFIDENCE = 1     # 0=low → skip, 1=medium, 2=high
VOXEL_SIZE     = 0.02  # 2 cm voxel grid for downsampling
SOR_K          = 10    # statistical outlier removal: k neighbours
SOR_STD_RATIO  = 2.0   # reject if mean-neighbour-dist > mean + ratio*std


def parse_col_major(values, shape):
    """Reshape a flat column-major list into a numpy matrix."""
    return np.array(values, dtype=np.float64).reshape(shape, order='F')


def load_frame(idx: int):
    stem = DATA_DIR / f"frame_{idx:06d}"

    with open(f"{stem}.json") as fh:
        meta = json.load(fh)

    if meta.get("trackingState") != "normal":
        return None  # pose unreliable during initialisation

    depth_w = meta["depthResolutionStored"]["width"]   # 192
    depth_h = meta["depthResolutionStored"]["height"]  # 144
    rgb_w   = meta["rgbResolutionStored"]["width"]     # 960
    rgb_h   = meta["rgbResolutionStored"]["height"]    # 720
    ref_w   = meta["intrinsicsReferenceResolution"]["width"]   # 1920
    ref_h   = meta["intrinsicsReferenceResolution"]["height"]  # 1440

    # --- depth (mm uint16 little-endian → float32 metres) ---
    depth = np.fromfile(f"{stem}.depth_u16.bin", dtype='<u2').reshape(depth_h, depth_w)
    depth = depth.astype(np.float32) * 0.001
    depth[depth == 0.0] = np.nan  # sentinel 0 → invalid

    # --- confidence (uint8: 0=low, 1=medium, 2=high) ---
    conf = np.fromfile(f"{stem}.conf_u8.bin", dtype='u1').reshape(depth_h, depth_w)

    # --- RGB (HEIC) ---
    # pillow_heif applies the EXIF orientation tag (90° CW for iOS portrait captures),
    # producing a portrait array (960, 720, 3).  We rotate 90° CCW to restore the
    # sensor-native landscape layout (720, 960, 3) that the depth map is aligned to.
    rgb_img = Image.open(f"{stem}.rgb.heic")
    rgb = np.rot90(np.array(rgb_img), k=1)  # portrait → landscape: (720, 960, 3)

    # --- intrinsics: scale from calibration res → depth res ---
    K = parse_col_major(meta["intrinsics"]["values"], (3, 3))
    sx = depth_w / ref_w
    sy = depth_h / ref_h
    fx = K[0, 0] * sx
    fy = K[1, 1] * sy
    cx = K[0, 2] * sx
    cy = K[1, 2] * sy

    # --- extrinsics: column-major 4x4 camera-to-world ---
    wfc = parse_col_major(meta["worldFromCamera"]["values"], (4, 4))

    # --- validity mask ---
    vs, us = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
    mask = (~np.isnan(depth)) & (conf >= MIN_CONFIDENCE) & (depth <= MAX_DEPTH_M)

    d = depth[mask].astype(np.float64)
    u = us[mask].astype(np.float64)
    v = vs[mask].astype(np.float64)

    # --- back-project to camera space ---
    # ARKit camera frame: +X right, +Y up, -Z forward (OpenGL convention).
    # Depth is the perpendicular distance along the optical axis (always positive).
    # Scene points are in front → z_cam = -depth (negative, since -Z is forward).
    # Image v increases downward; camera +Y is up → y_cam = (cy - v) * depth / fy.
    x_cam =  (u - cx) * d / fx
    y_cam =  (cy - v) * d / fy   # flip: camera +Y up vs image v down
    z_cam = -d                    # flip: -Z forward

    # --- camera → world ---
    ones  = np.ones_like(d)
    p_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=1)  # (N, 4)
    xyz   = (wfc @ p_cam.T).T[:, :3]                       # (N, 3)

    # --- colour: map depth pixel (u,v) → nearest RGB pixel ---
    u_rgb = np.clip(np.round(u * (rgb_w / depth_w)).astype(int), 0, rgb_w - 1)
    v_rgb = np.clip(np.round(v * (rgb_h / depth_h)).astype(int), 0, rgb_h - 1)
    colors = rgb[v_rgb, u_rgb]  # (N, 3)

    return xyz, colors


# ── collect frames ────────────────────────────────────────────────────────────

all_xyz, all_rgb = [], []
indices = sorted(int(p.stem.split('_')[1]) for p in DATA_DIR.glob("frame_*.json"))
print(f"Found {len(indices)} frames in {DATA_DIR.name}")

skipped = 0
for idx in indices:
    result = load_frame(idx)
    if result is None:
        skipped += 1
        continue
    xyz, colors = result
    all_xyz.append(xyz)
    all_rgb.append(colors)

print(f"Skipped {skipped} frames (tracking not normal)")

xyz = np.concatenate(all_xyz, axis=0)
rgb = np.concatenate(all_rgb, axis=0)
print(f"Total raw points: {len(xyz):,}")


# ── voxel downsampling ────────────────────────────────────────────────────────

def voxel_downsample(xyz, rgb, voxel_size):
    coords = np.floor(xyz / voxel_size).astype(np.int64)
    # Cantor-style hash unlikely to collide at room scale
    C0, C1 = np.int64(1_000_000_007), np.int64(1_000_003)
    keys = coords[:, 0] * C0 * C1 + coords[:, 1] * C1 + coords[:, 2]
    _, first = np.unique(keys, return_index=True)
    return xyz[first], rgb[first]

xyz, rgb = voxel_downsample(xyz, rgb, VOXEL_SIZE)
print(f"After voxel downsample ({VOXEL_SIZE*100:.0f} cm): {len(xyz):,} points")


# ── statistical outlier removal ───────────────────────────────────────────────

from scipy.spatial import cKDTree  # available via pyntcloud install

print("Running statistical outlier removal (k-NN)…")
tree = cKDTree(xyz)
dists, _ = tree.query(xyz, k=SOR_K + 1)  # includes self at dist 0
mean_nn  = dists[:, 1:].mean(axis=1)
threshold = mean_nn.mean() + SOR_STD_RATIO * mean_nn.std()
keep = mean_nn < threshold
xyz = xyz[keep]
rgb = rgb[keep]
print(f"After SOR (k={SOR_K}, thresh={threshold:.4f} m): {len(xyz):,} points")


# ── save PLY (binary little-endian) ──────────────────────────────────────────

def save_ply_binary(path, xyz, rgb):
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()

    xyz32 = xyz.astype(np.float32)
    rgb8  = rgb.astype(np.uint8)

    # interleave: 3×float32 + 3×uint8 = 15 bytes per vertex
    data = np.empty(n, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'),  ('g', 'u1'),  ('b', 'u1'),
    ])
    data['x'] = xyz32[:, 0]; data['y'] = xyz32[:, 1]; data['z'] = xyz32[:, 2]
    data['r'] = rgb8[:, 0];  data['g'] = rgb8[:, 1];  data['b'] = rgb8[:, 2]

    with open(path, 'wb') as fh:
        fh.write(header)
        fh.write(data.tobytes())

    print(f"Saved {n:,} points → {path}")

save_ply_binary(OUTPUT_PLY, xyz, rgb)
