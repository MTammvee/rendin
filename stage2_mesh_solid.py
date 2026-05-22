#!/usr/bin/env python3
"""
Stage 2 (solid surfaces): Truncated SDF + Marching Cubes.

Why not Poisson or BPA:
  - Screened Poisson: crashes in PyMeshLab 2025.7 on ARM Mac at every depth.
  - BPA: reuses input points as-is → inherits all point-cloud holes and noise,
    large flat surfaces (floor/walls) stay noisy and holey.

This approach:
  1. Estimate surface normals from the point cloud (PCA on k-NN, oriented
     toward scene centroid so they consistently point into the room interior).
  2. Build a signed distance field (SDF) on a 3cm voxel grid:
       SDF(q) = dist(q, nearest_surface_point) * sign(dot(q - p, n))
     Inside the room → positive; behind the wall → negative.
  3. Run Marching Cubes (Lorensen 1987) at isovalue=0.
     The zero level set is a smooth, gap-free surface.
  4. Transfer colors from point cloud (nearest-neighbour KDTree).

Tradeoffs vs Poisson:
  - Poisson optimises over all points globally → smoother.
  - Our SDF is local (1-NN) → slightly rougher, but robust and fast.
  - Both give solid, watertight surfaces; Poisson just doesn't run here.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "continuous_frames"
INPUT_PLY  = Path(f"/Users/martin/Documents/martin/rendin/output/stage1_{dataset_name}_hq.ply")
OUTPUT_PLY = Path(f"/Users/martin/Documents/martin/rendin/output/stage2_{dataset_name}_solid.ply")

VOXEL_SIZE = 0.03   # 3 cm grid — balances detail vs speed/memory
TRUNC      = 0.12   # truncation: only keep SDF within ±12 cm of surface
NORMAL_K   = 20     # neighbours for normal PCA


# ── read binary PLY ───────────────────────────────────────────────────────────

def read_ply(path):
    with open(path, 'rb') as f:
        n_verts = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            if line == 'end_header':
                break
        dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                          ('r','u1'),('g','u1'),('b','u1')])
        data = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)
    xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float64)
    rgb = np.column_stack([data['r'], data['g'], data['b']])
    return xyz, rgb

print(f"Loading {INPUT_PLY.name}…")
xyz, rgb = read_ply(INPUT_PLY)
print(f"  {len(xyz):,} points")


# ── normal estimation (batched PCA to keep memory under control) ──────────────

print("Building KDTree…")
tree = cKDTree(xyz)

print(f"Estimating normals (k={NORMAL_K}, batched PCA)…")
_, nn_idx = tree.query(xyz, k=NORMAL_K, workers=-1)

BATCH   = 8_000
normals = np.empty_like(xyz)
for start in range(0, len(xyz), BATCH):
    end   = min(start + BATCH, len(xyz))
    nbrs  = xyz[nn_idx[start:end]]          # (B, k, 3)
    c     = nbrs.mean(axis=1, keepdims=True)
    nbrs -= c
    cov   = np.einsum('nij,nik->njk', nbrs, nbrs)   # (B, 3, 3)
    _, vecs = np.linalg.eigh(cov)                    # smallest eigval → col 0
    normals[start:end] = vecs[:, :, 0]

# Orient normals to point INTO the room (toward scene centroid)
centroid   = xyz.mean(axis=0)
toward_cen = centroid - xyz                          # (N, 3)
flip       = (normals * toward_cen).sum(axis=1) < 0
normals[flip] *= -1
print("  Normals done")


# ── build SDF grid ────────────────────────────────────────────────────────────

pad      = TRUNC + VOXEL_SIZE
bbox_min = xyz.min(axis=0) - pad
bbox_max = xyz.max(axis=0) + pad
shape    = np.ceil((bbox_max - bbox_min) / VOXEL_SIZE).astype(int)
print(f"SDF grid: {shape} ({shape.prod():,} voxels) at {VOXEL_SIZE*100:.0f} cm resolution")

# Grid in world coordinates
gx = np.arange(shape[0]) * VOXEL_SIZE + bbox_min[0]
gy = np.arange(shape[1]) * VOXEL_SIZE + bbox_min[1]
gz = np.arange(shape[2]) * VOXEL_SIZE + bbox_min[2]
XI, YI, ZI = np.meshgrid(gx, gy, gz, indexing='ij')
grid_pts = np.column_stack([XI.ravel(), YI.ravel(), ZI.ravel()])

print("Querying SDF (nearest-neighbour signed distance)…")
dists, idxs = tree.query(grid_pts, k=1, workers=-1)

# Sign: positive inside room, negative behind surface
diff = grid_pts - xyz[idxs]
sign = np.sign((diff * normals[idxs]).sum(axis=1))
sign[sign == 0] = 1.0

sdf = (dists * sign).clip(-TRUNC, TRUNC).reshape(shape)
print("  SDF done")


# ── marching cubes at zero level set ─────────────────────────────────────────

print("Running Marching Cubes…")
verts, faces, _, _ = marching_cubes(sdf, level=0,
                                     spacing=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE))
verts += bbox_min   # shift from grid-local to world coords
print(f"  {len(verts):,} vertices, {len(faces):,} faces")


# ── color transfer ────────────────────────────────────────────────────────────

print("Transferring colors…")
_, cidx  = tree.query(verts, k=1, workers=-1)
colors   = rgb[cidx].astype(np.uint8)


# ── save binary PLY ───────────────────────────────────────────────────────────

header = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {len(verts)}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    f"element face {len(faces)}\n"
    "property list uchar int vertex_indices\n"
    "end_header\n"
).encode()

vdtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                   ('r','u1'),('g','u1'),('b','u1')])
vdata          = np.empty(len(verts), dtype=vdtype)
vdata['x']     = verts[:,0]; vdata['y'] = verts[:,1]; vdata['z'] = verts[:,2]
vdata['r']     = colors[:,0]; vdata['g'] = colors[:,1]; vdata['b'] = colors[:,2]

fdtype         = np.dtype([('n','u1'),('v0','<i4'),('v1','<i4'),('v2','<i4')])
fdata          = np.empty(len(faces), dtype=fdtype)
fdata['n']     = 3
fdata['v0']    = faces[:,0]; fdata['v1'] = faces[:,1]; fdata['v2'] = faces[:,2]

with open(OUTPUT_PLY, 'wb') as fh:
    fh.write(header)
    fh.write(vdata.tobytes())
    fh.write(fdata.tobytes())

print(f"\nSaved → {OUTPUT_PLY}")
print(f"Vertices: {len(verts):,}   Faces: {len(faces):,}")
