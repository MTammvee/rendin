#!/usr/bin/env python3
"""
Stage 3: Structural plane extraction (floor / ceiling / walls) via RANSAC.

Approach:
  Iterative RANSAC plane fitting on the HQ point cloud.
  Each iteration: randomly sample 3 points → fit plane → count inliers →
  keep best → refit on inliers → remove inliers → repeat.

  Planes are classified by their normal relative to the ARKit world frame
  (Y is up, gravity-aligned):
    |normal.Y| > 0.85  →  horizontal  (floor = lowest, ceiling = highest)
    |normal.Y| < 0.4   →  vertical    (wall)
    otherwise          →  oblique     (ignored)

Outputs:
  - Coloured point cloud PLY: floor=blue, ceiling=green, walls=distinct colours,
    unclassified=grey.
  - Top-down PNG floorplan: wall inliers projected onto XZ plane.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "continuous_frames"
INPUT_PLY    = Path(f"/Users/martin/Documents/martin/rendin/output/stage1_{dataset_name}_hq.ply")
OUT_PLY      = Path(f"/Users/martin/Documents/martin/rendin/output/stage3_{dataset_name}_planes.ply")
OUT_PNG      = Path(f"/Users/martin/Documents/martin/rendin/output/stage3_{dataset_name}_floorplan.png")

# ── RANSAC parameters ──────────────────────────────────────────────────────────
N_PLANES        = 8      # max planes to extract
RANSAC_ITER     = 300    # iterations per plane (on subsample)
RANSAC_SUBSAMP  = 30_000 # subsample size for RANSAC search (full cloud for inlier count)
INLIER_THRESH   = 0.04   # 4 cm distance threshold
MIN_INLIERS     = 3_000  # skip planes smaller than this
HORIZ_THRESH    = 0.85   # |n.y| > this → horizontal plane
VERT_THRESH     = 0.40   # |n.y| < this → vertical (wall)

# ── colours ────────────────────────────────────────────────────────────────────
WALL_COLOUR    = [255,  80,  80]   # red   — all walls same colour
FLOOR_COLOUR   = [ 30, 120, 255]   # blue
CEILING_COLOUR = [ 50, 200,  80]   # green
TABLE_COLOUR   = [255, 165,   0]   # orange — intermediate horizontal surfaces
UNCLASS_COLOUR = [160, 160, 160]   # grey


# ── helpers ────────────────────────────────────────────────────────────────────

def read_ply(path):
    with open(path, 'rb') as f:
        n = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            if line == 'end_header':
                break
        dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                       ('r','u1'),('g','u1'),('b','u1')])
        d = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    xyz = np.column_stack([d['x'], d['y'], d['z']]).astype(np.float64)
    rgb = np.column_stack([d['r'], d['g'], d['b']])
    return xyz, rgb


def fit_plane_ransac(pts_full, pts_sub, n_iter, threshold):
    """RANSAC on pts_sub (fast), then count inliers on pts_full (accurate)."""
    N = len(pts_sub)
    best_normal = None
    best_d      = None
    best_count  = 0

    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = pts_sub[idx]
        v1 = p1 - p0;  v2 = p2 - p0
        n  = np.cross(v1, v2)
        nm = np.linalg.norm(n)
        if nm < 1e-9:
            continue
        n /= nm
        d  = -n @ p0
        # Count inliers on subsample only (fast)
        cnt = (np.abs(pts_sub @ n + d) < threshold).sum()
        if cnt > best_count:
            best_count  = cnt
            best_normal = n
            best_d      = d

    if best_normal is None:
        return None, None, None

    # Refit via SVD on full inliers for accuracy
    mask_full  = np.abs(pts_full @ best_normal + best_d) < threshold
    inlier_pts = pts_full[mask_full]
    if len(inlier_pts) < 3:
        return best_normal, best_d, mask_full
    cen = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - cen, full_matrices=False)
    normal = Vt[-1]
    d      = -normal @ cen
    mask_full = np.abs(pts_full @ normal + d) < threshold
    return normal, d, mask_full


def classify_plane(normal):
    ny = abs(normal[1])
    if ny > HORIZ_THRESH:
        return 'horizontal'
    if ny < VERT_THRESH:
        return 'wall'
    return 'oblique'


def plane_area_m2(pts, normal):
    """
    Minimum bounding rectangle area of pts projected onto the fitted plane (m²).

    Convex hull only covers scanned pixels — if a room corner is missing the
    hull shrinks away from it and the area is underestimated.  The minimum
    bounding rectangle (rotating-calipers over hull edges) fits the tightest
    rectangle at the best angle, giving the correct area even when corners are
    partially unscanned.
    """
    from scipy.spatial import ConvexHull
    # Project to 2-D on the plane
    ref  = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u    = np.cross(normal, ref);  u /= np.linalg.norm(u)
    v    = np.cross(normal, u);    v /= np.linalg.norm(v)
    pts2 = (pts - pts.mean(axis=0)) @ np.column_stack([u, v])

    try:
        hull     = ConvexHull(pts2)
        hull_pts = pts2[hull.vertices]
    except Exception:
        return 0.0

    # Rotating calipers: try every hull edge as a candidate rectangle orientation
    n        = len(hull_pts)
    min_area = np.inf
    for i in range(n):
        edge  = hull_pts[(i + 1) % n] - hull_pts[i]
        angle = np.arctan2(edge[1], edge[0])
        c, s  = np.cos(-angle), np.sin(-angle)
        rot   = np.array([[c, -s], [s, c]])
        rotated = hull_pts @ rot.T
        w     = rotated[:, 0].max() - rotated[:, 0].min()
        h     = rotated[:, 1].max() - rotated[:, 1].min()
        min_area = min(min_area, w * h)

    return min_area


def save_ply(path, xyz, rgb):
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    d = np.empty(n, dtype=dt)
    d['x']=xyz[:,0]; d['y']=xyz[:,1]; d['z']=xyz[:,2]
    d['r']=rgb[:,0]; d['g']=rgb[:,1]; d['b']=rgb[:,2]
    with open(path, 'wb') as f:
        f.write(header); f.write(d.tobytes())


# ── main ───────────────────────────────────────────────────────────────────────

OUT_CLEAN = Path(f"/Users/martin/Documents/martin/rendin/output/stage3_{dataset_name}_clean.ply")

print(f"Loading {INPUT_PLY.name}…", flush=True)
xyz, rgb_orig = read_ply(INPUT_PLY)
print(f"  {len(xyz):,} points", flush=True)

colors     = np.full((len(xyz), 3), UNCLASS_COLOUR, dtype=np.uint8)
remaining  = np.arange(len(xyz))   # indices of unassigned points

planes       = []   # (normal, d, label, global_mask)
horiz_planes = []   # for floor/ceiling distinction

wall_idx = 0

for i in range(N_PLANES):
    if len(remaining) < MIN_INLIERS:
        break

    pts_full = xyz[remaining]
    # Subsample for fast RANSAC search
    sub_idx  = np.random.choice(len(pts_full),
                                min(RANSAC_SUBSAMP, len(pts_full)),
                                replace=False)
    pts_sub  = pts_full[sub_idx]
    print(f"\nPlane {i+1}: RANSAC on {len(pts_sub):,} pts (full={len(pts_full):,})…", flush=True)

    normal, d, local_mask = fit_plane_ransac(pts_full, pts_sub, RANSAC_ITER, INLIER_THRESH)
    if normal is None or local_mask.sum() < MIN_INLIERS:
        print("  → too few inliers, stopping", flush=True)
        break

    global_idx = remaining[local_mask]
    kind = classify_plane(normal)
    n_in = len(global_idx)
    print(f"  normal={np.round(normal,3)}  inliers={n_in:,}  → {kind}", flush=True)

    if kind == 'oblique':
        remaining = remaining[~local_mask]
        continue

    if kind == 'horizontal':
        mean_y = xyz[global_idx, 1].mean()
        horiz_planes.append((mean_y, n_in, normal, d, global_idx))
    else:  # wall
        colors[global_idx] = WALL_COLOUR
        planes.append(('wall', normal, d, global_idx, WALL_COLOUR))
        wall_idx += 1

    remaining = remaining[~local_mask]

# Assign floor / ceiling / table among horizontal planes.
# Floor = lowest-Y plane with enough points to be a real surface.
# Ceiling = highest-Y plane with enough points.
# Small spurious planes (< MIN_HORIZ_PTS) are skipped entirely.
MIN_HORIZ_PTS = 20_000   # below this → too small to be floor or ceiling

if horiz_planes:
    horiz_planes.sort(key=lambda x: x[0])   # sort by mean Y ascending
    # Filter out tiny planes that are likely scan artefacts
    large  = [(my, n_in, normal, d, gidx) for my, n_in, normal, d, gidx in horiz_planes
               if n_in >= MIN_HORIZ_PTS]
    small  = [(my, n_in, normal, d, gidx) for my, n_in, normal, d, gidx in horiz_planes
               if n_in <  MIN_HORIZ_PTS]

    for my, n_in, normal, d, gidx in small:
        print(f"  Horizontal plane → unclassified (too small, {n_in:,} pts)  (mean Y={my:.2f} m)", flush=True)

    for j, (my, n_in, normal, d, gidx) in enumerate(large):
        if j == 0:
            label = 'floor';   col = FLOOR_COLOUR
        elif j == len(large) - 1:
            label = 'ceiling'; col = CEILING_COLOUR
        else:
            print(f"  Horizontal plane → unclassified (table/shelf, {n_in:,} pts)  (mean Y={my:.2f} m)", flush=True)
            continue
        colors[gidx] = col
        planes.append((label, normal, d, gidx, col))
        print(f"  Horizontal plane → {label}  ({n_in:,} pts, mean Y={my:.2f} m)", flush=True)

# ── save coloured point cloud ──────────────────────────────────────────────────
print(f"\nSaving coloured point cloud → {OUT_PLY.name}")
save_ply(OUT_PLY, xyz, colors)

# ── top-down floorplan ─────────────────────────────────────────────────────────
print(f"Generating floorplan → {OUT_PNG.name}")

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')

legend_entries = []
wall_counter = 0

for label, normal, d, gidx, col in planes:
    c = [v/255 for v in col]
    pts = xyz[gidx]

    if label in ('floor', 'ceiling', 'floor_mid'):
        # Project onto XZ for floor/ceiling: show as convex hull outline
        from scipy.spatial import ConvexHull
        xz = pts[:, [0, 2]]
        try:
            hull = ConvexHull(xz)
            hull_pts = xz[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(hull_pts[:,0], hull_pts[:,1], color=c, linewidth=2, alpha=0.8)
            ax.fill(hull_pts[:-1,0], hull_pts[:-1,1], color=c, alpha=0.08)
        except Exception:
            pass
        legend_entries.append(mpatches.Patch(color=c, label=label))
    else:
        # Walls: scatter XZ positions (subsample for speed)
        step = max(1, len(pts) // 3000)
        wall_counter += 1
        ax.scatter(pts[::step, 0], pts[::step, 2], s=1, color=c, alpha=0.6)
        legend_entries.append(mpatches.Patch(color=c, label=f'wall {wall_counter}'))

ax.set_xlabel('X (m)', color='white')
ax.set_ylabel('Z (m)', color='white')
ax.set_title(f'Top-down floorplan — {dataset_name}', color='white', fontsize=14)
ax.tick_params(colors='white')
ax.set_aspect('equal')
ax.legend(handles=legend_entries, facecolor='#2a2a4e', labelcolor='white',
          loc='upper right')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor=fig.get_facecolor())
plt.close()

# ── summary ────────────────────────────────────────────────────────────────────
print("\n── Plane summary ──")
classified = sum(len(p[3]) for p in planes)
print(f"Total points classified : {classified:,} / {len(xyz):,} "
      f"({100*classified/len(xyz):.1f}%)\n")
print(f"  {'label':12s}  {'points':>8}  {'area (m²)':>10}  normal")
print(f"  {'-'*12}  {'-'*8}  {'-'*10}  ------")
for label, normal, d, gidx, col in planes:
    area = plane_area_m2(xyz[gidx], normal)
    print(f"  {label:12s}  {len(gidx):>8,}  {area:>10.2f}  {np.round(normal,2)}")
print(f"\nOutputs:\n  {OUT_PLY}\n  {OUT_PNG}")

# ── plane-based room crop ──────────────────────────────────────────────────────
# Use the fitted floor/ceiling/wall planes as a convex room boundary.
# Any point more than PLANE_MARGIN outside a bounding plane is discarded.
PLANE_MARGIN = 0.20   # 20 cm buffer outside each plane (catches furniture edges)

if planes:
    classified_pts = np.concatenate([xyz[p[3]] for p in planes], axis=0)
    centroid = classified_pts.mean(axis=0)

    inside = np.ones(len(xyz), dtype=bool)
    for label, normal, d, gidx, col in planes:
        cen_side = np.sign(normal @ centroid + d)
        signed_dist = cen_side * (xyz @ normal + d)
        inside &= signed_dist > -PLANE_MARGIN

    xyz_clean    = xyz[inside]
    rgb_clean    = rgb_orig[inside]
    colors_clean = colors[inside]
    print(f"\nPlane-crop: {inside.sum():,} / {len(xyz):,} points kept "
          f"({100*inside.mean():.1f}%)")
    # Overwrite both outputs with the cropped versions
    save_ply(OUT_PLY,   xyz_clean, colors_clean)
    save_ply(OUT_CLEAN, xyz_clean, rgb_clean)
    print(f"Saved cropped coloured cloud → {OUT_PLY}")
    print(f"Saved cropped original-RGB cloud → {OUT_CLEAN}")
