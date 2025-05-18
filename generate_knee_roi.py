import argparse
import os
import pathlib
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh

# ---------- utility wrappers you already have -----------------
from anato_utils import (
    GetFilteredMeshPaths,  # (root, group_str, file_str, ext) -> [(dir, file)]
    GetMeshFromParquet,  # parquet -> trimesh.Trimesh (with .curvatures)
)


def get_trochlea_roi(mesh, height_ratio, anterior_crop):
    """
    Get the region of interest (ROI) for the trochlea from a 3D mesh.
    """
    V = mesh.vertices - mesh.centroid

    xy = V[:, :2]
    C = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    long_xy = eigvecs[:, eigvals.argmax()]
    ap_xy = eigvecs[:, eigvals.argmin()]
    angle = -np.arctan2(long_xy[1], long_xy[0])  # align to +x

    # two candidate rotations: to +x (phi) or to –x (phi + π)
    candidates = [angle, angle + np.pi]

    def normalize(a):
        """wrap angle to (‑π, π] so we can compare magnitudes cleanly"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    chosen = None
    for ang in sorted(candidates, key=lambda a: abs(normalize(a))):
        Rz = R.from_euler("z", ang).as_matrix()
        ap_new = Rz @ np.append(ap_xy, 0.0)  # rotate A–P vector
        if ap_new[1] > 0:  # still pointing anterior?
            chosen = normalize(ang)  # keep the smaller |angle|
            break

    if chosen is None:  # extremely rare degeneracy (both posterior)
        # fall back to the smaller candidate anyway
        chosen = normalize(candidates[0])

    Rz = R.from_euler("z", chosen).as_matrix()
    V_rot = (Rz @ V.T).T

    # V_rot = V_rot + mesh.centroid
    mesh_oriented = mesh.copy()
    mesh_oriented.vertices = V_rot + mesh.centroid

    x_mid = np.median(V_rot[:, 0])

    # Boolean masks for each half
    left_mask = V_rot[:, 0] <= x_mid  # ← typically lateral condyle
    right_mask = ~left_mask  # ← typically medial condyle

    # helper: grab distal 5 % from whichever mask we pass in
    def distal_five_percent(mask):
        z_cut = np.percentile(V_rot[mask, 2], 5)  # 5 % lowest z in that half
        return V_rot[mask & (V_rot[:, 2] <= z_cut)]

    condyle_L = distal_five_percent(left_mask)
    condyle_R = distal_five_percent(right_mask)

    # Single “cluster” point for each condyle (use mean or min‑z if you prefer)
    center_L = condyle_L.mean(axis=0)
    center_R = condyle_R.mean(axis=0)

    # distal tip (min‑z vertex) of each condyle
    tip_L = condyle_L[np.argmin(condyle_L[:, 2])]
    tip_R = condyle_R[np.argmin(condyle_R[:, 2])]

    # distal tip of each condyle
    z_L_tip = condyle_L[:, 2].min()  # most distal on left half
    z_R_tip = condyle_R[:, 2].min()  # most distal on right half

    height_mm = height_ratio * np.linalg.norm(tip_L - tip_R)

    # "higher" condyle is the one with the larger (less negative) z
    z_bottom = min(z_L_tip, z_R_tip)
    z_top = max(z_L_tip, z_R_tip) + height_mm

    y_post = 0
    if anterior_crop:
        y_post = np.median(V_rot[:, 1])
    else:
        y_post = np.min(V_rot[:, 1])

    # use those two x‑positions as the medial–lateral walls of the ROI
    roi_min = np.array([min(tip_L[0], tip_R[0]), y_post, z_bottom])

    roi_max = np.array([max(tip_L[0], tip_R[0]), np.max(V_rot[:, 1]), z_top])

    inside = np.all((V_rot >= roi_min) & (V_rot <= roi_max), axis=1)
    face_mask = inside[mesh.faces].all(axis=1)
    faces_kept = mesh.faces[face_mask]
    preserved_vidx = np.unique(faces_kept)

    roi_mesh = mesh_oriented.submesh([face_mask], append=True)
    roi_mesh.curvatures = mesh.curvatures[preserved_vidx]

    return roi_mesh


def process_one_mesh(
    input_dir, filename, input_root, output_root, height_ratio, anterior_crop
):
    input_path = os.path.join(input_dir, filename)
    mesh = GetMeshFromParquet(input_path)

    if "femur" in filename.lower():  # crop only femora
        mesh = get_trochlea_roi(mesh, height_ratio, anterior_crop)

    rel_dir = os.path.relpath(input_dir, start=input_root)
    dest_dir = os.path.join(output_root, rel_dir)
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    dest_path = os.path.join(dest_dir, filename)

    vertices = pd.DataFrame(mesh.vertices, columns=["X", "Y", "Z"])
    triangles = pd.DataFrame(mesh.faces, columns=["T1", "T2", "T3"])
    curvatures = pd.DataFrame(mesh.curvatures, columns=["K1", "K2"])
    concatenated_data = pd.concat([vertices, triangles, curvatures], axis=1)
    concatenated_data.to_parquet(dest_path, compression="gzip", index=False)

    print(f"saved {dest_path}")


input_root = "Knee_7"
output_root = "Knee_7_ROI"
height_ratio = 1
anterior_crop = False

paths = GetFilteredMeshPaths(input_root, ["CI", "TI"], ["M5"], ext=".parquet")

if not paths:
    raise RuntimeError("No meshes matched the supplied filters.")

for input_dir, filename in paths:
    process_one_mesh(
        input_dir, filename, input_root, output_root, height_ratio, anterior_crop
    )
