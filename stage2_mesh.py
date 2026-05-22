#!/usr/bin/env python3
"""
Stage 2: Surface mesh via Ball-Pivoting Algorithm (BPA).

Preferred algorithm: Screened Poisson (Kazhdan & Hoppe 2013).
  - Implicit surface, handles noise/gaps, produces a watertight mesh.
  - Requires normals; slightly over-smoothed at sharp corners.
  - LIMITATION: PyMeshLab 2025.7 on ARM Mac segfaults in the Poisson solver
    (bad-average-root / failed-to-close-loop), so Poisson is not used here.

Fallback used: Ball-Pivoting Algorithm (BPA, Bernardini 1999).
  - Rolls virtual balls of a given radius over the point cloud and stitches
    triangles wherever three points are co-tangent.
  - Faster than Poisson, no normals required, but leaves holes where the
    ball cannot fit (gaps in coverage) and is more sensitive to point density.
  - Key advantage here: BPA reuses the input vertices directly, so vertex
    colors from the point cloud carry through to the mesh for free.

Post-processing:
  - Remove small disconnected components (floating artefacts).
  - Save as binary PLY with vertex RGB.
"""

import sys
import numpy as np
from pathlib import Path
import pymeshlab

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "continuous_frames"
INPUT_PLY  = Path(f"/Users/martin/Documents/martin/rendin/output/stage1_{dataset_name}_hq.ply")
OUTPUT_PLY = Path(f"/Users/martin/Documents/martin/rendin/output/stage2_{dataset_name}_mesh.ply")

print(f"Dataset : {dataset_name}")
print(f"Input   : {INPUT_PLY}")
print(f"Output  : {OUTPUT_PLY}")

# ── load point cloud ─────────────────────────────────────────────────────────
ms = pymeshlab.MeshSet()
ms.load_new_mesh(str(INPUT_PLY))
print(f"Loaded {ms.current_mesh().vertex_number():,} points")

# ── estimate normals (needed for BPA orientation) ─────────────────────────────
ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
print("Normals estimated (k=20 PCA)")

# ── Ball-Pivoting reconstruction ──────────────────────────────────────────────
# Default ball radius = auto (PyMeshLab picks based on point spacing).
# BPA reuses the input vertices, so colors are already on the mesh.
ms.generate_surface_reconstruction_ball_pivoting()
mesh = ms.current_mesh()
print(f"BPA mesh: {mesh.vertex_number():,} vertices, {mesh.face_number():,} faces")

# ── remove small disconnected components ─────────────────────────────────────
before_v = mesh.vertex_number()
ms.meshing_remove_connected_component_by_face_number(mincomponentsize=500)
mesh = ms.current_mesh()
print(f"After cleanup: {mesh.vertex_number():,} vertices "
      f"(removed {before_v - mesh.vertex_number():,})")

# ── save as binary PLY with vertex colors ────────────────────────────────────
mesh_xyz = mesh.vertex_matrix()                         # (M, 3) float64
faces    = mesh.face_matrix()                           # (F, 3) int32
rgba_f   = mesh.vertex_color_matrix()                   # (M, 4) RGBA float [0,1]
rgba     = (np.clip(rgba_f, 0, 1) * 255).astype(np.uint8)

header = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {len(mesh_xyz)}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    f"element face {len(faces)}\n"
    "property list uchar int vertex_indices\n"
    "end_header\n"
).encode()

vertex_dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                         ('r','u1'),('g','u1'),('b','u1')])
vdata = np.empty(len(mesh_xyz), dtype=vertex_dtype)
vdata['x'] = mesh_xyz[:,0]; vdata['y'] = mesh_xyz[:,1]; vdata['z'] = mesh_xyz[:,2]
vdata['r'] = rgba[:,0];     vdata['g'] = rgba[:,1];     vdata['b'] = rgba[:,2]

face_dtype = np.dtype([('n','u1'), ('v0','<i4'), ('v1','<i4'), ('v2','<i4')])
fdata = np.empty(len(faces), dtype=face_dtype)
fdata['n'] = 3
fdata['v0'] = faces[:,0]; fdata['v1'] = faces[:,1]; fdata['v2'] = faces[:,2]

with open(OUTPUT_PLY, 'wb') as fh:
    fh.write(header)
    fh.write(vdata.tobytes())
    fh.write(fdata.tobytes())

print(f"\nSaved → {OUTPUT_PLY}")
print(f"Vertices: {len(mesh_xyz):,}   Faces: {len(faces):,}")
