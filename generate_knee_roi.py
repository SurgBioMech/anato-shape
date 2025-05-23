from __future__ import annotations

import trimesh
from scipy.spatial.transform import Rotation as R
from trimesh.repair import fix_normals

import os
import pathlib
import pandas as pd
import numpy as np


# ---------- utility wrappers you already have -----------------
from anato_utils import (
    GetFilteredMeshPaths,  # (root, group_str, file_str, ext) -> [(dir, file)]
    GetMeshFromParquet,  # parquet -> trimesh.Trimesh (with .curvatures)
)


# NOTE: No Guarantee of correction of A/P orientation
# NOTE: MUST Verify empirically on the dataset


def planar_pca(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Principal‑component analysis in the x‑y plane.

    Returns
    -------
    ML_xy : np.ndarray, shape (2,)
        Unit vector of the medial–lateral (long) axis in the x‑y plane.
    phi   : float
        Angle (rad) that rotates ML onto +x.
    """
    xy = vertices[:, :2]
    C = np.cov(xy.T)
    eigval, eigvec = np.linalg.eigh(C)
    ML_xy = eigvec[:, eigval.argmax()]
    phi = -np.arctan2(ML_xy[1], ML_xy[0])
    return ML_xy, phi


def submesh_between(
    mesh: trimesh.Trimesh,
    V_rot: np.ndarray,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
) -> trimesh.Trimesh:
    """
    Return a sub‑mesh whose vertices lie inside [roi_min, roi_max]
    in the *rotated* frame given by `V_rot`.
    """
    inside = np.all((V_rot >= roi_min) & (V_rot <= roi_max), axis=1)
    face_mask = inside[mesh.faces].all(axis=1)
    m_roi = mesh.submesh([face_mask], append=True)

    # keep a correct curvature array if present
    if hasattr(mesh, "curvatures"):
        kept = np.unique(mesh.faces[face_mask])
        m_roi.curvatures = mesh.curvatures[kept]

    return m_roi


# --------------------------------------------------------------------------
#  Distal femur – trochlear ROI
# --------------------------------------------------------------------------


def trochlear_roi(
    femur: trimesh.Trimesh,
    height_ratio: float = 1.5,
    anterior_crop: bool = True,
) -> trimesh.Trimesh:
    """
    Extract the trochlear (patellar) surface of an arbitrarily‑posed femur.

    Parameters
    ----------
    femur          : trimesh.Trimesh
    height_ratio   : float
        Proximal extent expressed as a multiple of the inter‑condylar distance.
    anterior_crop  : bool
        If True keep only the anterior half in y; if False keep full depth.

    Returns
    -------
    trimesh.Trimesh (sub‑mesh)
    """
    V0 = femur.vertices - femur.centroid
    _, phi = planar_pca(V0)

    Rz = R.from_euler("z", phi).as_matrix()  # best_angle
    V = femur.vertices - femur.centroid
    V_rot = (Rz @ V.T).T

    femur_oriented = femur.copy()
    femur_oriented.vertices = V_rot + femur.centroid  # back to world coords
    femur_oriented.curvatures = femur.curvatures  # keep curvatures if present

    # --- find condyles -------------------------------------------------
    x_mid = np.median(V_rot[:, 0])
    L_mask = V_rot[:, 0] <= x_mid
    R_mask = ~L_mask

    def distal_percent(mask, pct=5):
        z_cut = np.percentile(V_rot[mask, 2], pct)
        return V_rot[mask & (V_rot[:, 2] <= z_cut)]

    cond_L, cond_R = distal_percent(L_mask), distal_percent(R_mask)
    tip_L, tip_R = cond_L[np.argmin(cond_L[:, 2])], cond_R[np.argmin(cond_R[:, 2])]
    z_L_tip, z_R_tip = tip_L[2], tip_R[2]

    intercond = np.linalg.norm(tip_L - tip_R)
    z_bottom = min(z_L_tip, z_R_tip)
    z_top = max(z_L_tip, z_R_tip) + height_ratio * intercond
    y_post_cut = np.median(V_rot[:, 1]) if anterior_crop else np.min(V_rot[:, 1])

    roi_min = np.array([min(tip_L[0], tip_R[0]), y_post_cut, z_bottom])
    roi_max = np.array([max(tip_L[0], tip_R[0]), np.max(V_rot[:, 1]), z_top])

    return submesh_between(femur_oriented, V_rot, roi_min, roi_max)


# --------------------------------------------------------------------------
#  Proximal tibia ROI (simple anterior cropping)
# --------------------------------------------------------------------------


def tibial_roi(
    tibia: trimesh.Trimesh,
    anterior_crop: bool = True,
) -> trimesh.Trimesh:
    """
    Extract the full proximal tibia plateau, optionally keeping only
    the anterior half in y.

    Parameters
    ----------
    tibia          : trimesh.Trimesh
    anterior_crop  : bool
        If True keep only the anterior half in y.

    Returns
    -------
    trimesh.Trimesh (sub‑mesh)
    """
    V0 = tibia.vertices - tibia.centroid
    _, phi = planar_pca(V0)

    Rz = R.from_euler("z", phi).as_matrix()  # best_angle
    V = tibia.vertices - tibia.centroid
    V_rot = (Rz @ V.T).T

    tibia_oriented = tibia.copy()
    tibia_oriented.vertices = V_rot + tibia.centroid  # back to world coords
    tibia_oriented.curvatures = tibia.curvatures  # keep curvatures if present

    y_post_cut = np.median(V_rot[:, 1]) if anterior_crop else np.min(V_rot[:, 1])
    roi_min = np.array([V_rot[:, 0].min(), y_post_cut, V_rot[:, 2].min()])
    roi_max = np.array([V_rot[:, 0].max(), np.max(V_rot[:, 1]), V_rot[:, 2].max()])

    return submesh_between(tibia_oriented, V_rot, roi_min, roi_max)


def process_one_mesh(
    input_dir, filename, input_root, output_root, height_ratio, anterior_crop
):
    input_path = os.path.join(input_dir, filename)
    mesh = GetMeshFromParquet(input_path)

    if "femur" in filename.lower():  # crop only femora
        mesh = trochlear_roi(mesh, height_ratio, anterior_crop)
    elif "tibia" in filename.lower():  # crop only tibiae
        mesh = tibial_roi(mesh, anterior_crop)

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
output_root = "Knee_7_ROI_AnteriorCrop"
height_ratio = 1
anterior_crop = True

paths = GetFilteredMeshPaths(input_root, ["CI", "TI"], ["M5"], ext=".parquet")

if not paths:
    raise RuntimeError("No meshes matched the supplied filters.")

for input_dir, filename in paths:
    process_one_mesh(
        input_dir, filename, input_root, output_root, height_ratio, anterior_crop
    )
