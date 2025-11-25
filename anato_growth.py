import copy
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
import scipy.io as sio
import time as _time
import trimesh
from IPython.display import display
from pycpd import DeformableRegistration
from sklearn.cluster import KMeans

from unravel import unravel_elems

import anato_curv as ac
import anato_mesh as am
def plot_registration(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


def calculate_translation_matrix(pcd_source, pcd_target):
    """
    Calculates the translation transformation matrix to move the center of mass
    of pcd_source to the center of mass of pcd_target.

    Args:
        pcd_source (o3d.geometry.PointCloud): The source point cloud.
        pcd_target (o3d.geometry.PointCloud): The target point cloud.

    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix.
    """
    # Get the center of coordinates for both point clouds
    center_source = pcd_source.get_center()
    center_target = pcd_target.get_center()

    # Calculate the translation vector
    translation_vector = center_target - center_source

    # Create 4x4 homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def visualize_segment_registration(iteration, error, X, Y, ax=None, division=0):
    """
    Plotly-based interactive callback for registration.

    This creates a single FigureWidget per division and updates its traces
    in-place on each callback invocation so the widget remains movable and
    interactive in Jupyter.
    """
    if iteration > 2 and iteration % 2 != 0:
        return

    # ensure arrays are numpy arrays
    X = _np.asarray(X)
    Y = _np.asarray(Y)

    # attach storage to function so figs persist across calls
    if not hasattr(visualize_segment_registration, "_figs"):
        visualize_segment_registration._figs = {}

    # create a new FigureWidget for this division the first time
    if division not in visualize_segment_registration._figs:
        figw = go.FigureWidget()
        figw.add_trace(
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.7),
                name="Target (final)",
            )
        )
        figw.add_trace(
            go.Scatter3d(
                x=Y[:, 0],
                y=Y[:, 1],
                z=Y[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.7),
                name="Source (initial)",
            )
        )
        figw.update_layout(
            title=f"Division {division+1}/{mdiv} — Iter {iteration}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=800,
            height=600,
        )
        visualize_segment_registration._figs[division] = figw
        display(figw)
    else:
        # update existing FigureWidget in-place for smooth interactivity
        figw = visualize_segment_registration._figs[division]
        with figw.batch_update():
            figw.data[0].x = X[:, 0]
            figw.data[0].y = X[:, 1]
            figw.data[0].z = X[:, 2]
            figw.data[1].x = Y[:, 0]
            figw.data[1].y = Y[:, 1]
            figw.data[1].z = Y[:, 2]
            figw.layout.title = f"Division {division+1}/{mdiv} — Iter {iteration}"

    return figw


def align_segment(args):
    """
    Align a single segment for deformable registration.
    Returns the index and transformed points.
    """
    (
        i,
        source_centers,
        target_centers,
        source_idx,
        target_idx,
        alpha,
        beta,
        max_iterations,
        plot_figures,
    ) = args

    print(f"Aligning segment {i+1}")

    Y = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(source_centers[source_idx, :])
    )
    X = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target_centers[target_idx, :])
    )

    Y.transform(calculate_translation_matrix(Y, X))

    deform_reg = DeformableRegistration(
        X=np.asarray(X.points),
        Y=np.asarray(Y.points),
        alpha=alpha,
        beta=beta,
        max_iterations=max_iterations,
    )

    callback = (
        partial(visualize_segment_registration, ax=ax, division=i)
        if plot_figures
        else None
    )

    TY, _ = deform_reg.register(callback)

    if TY.shape[0] != source_idx.shape[0]:
        raise ValueError(
            f"Segment {i} mismatch: source {source_idx.shape[0]}, transformed {TY.shape[0]}"
        )

    print(f"Segment {i+1} aligned.")
    return i, TY


def segment_registration(
    source_mesh,
    target_mesh,
    source_grps,
    target_grps,
    n_segments,
    alpha,
    beta,
    max_iterations=30,
    plot_figures=False,
    parallel=False,
):
    """
    Perform segment-wise deformable registration between source and target meshes.

    Parameters:
    - source_mesh: The source mesh object.
    - target_mesh: The target mesh object.
    - source_grps: List of groups (segments) in the source mesh.
    - target_grps: List of groups (segments) in the target mesh.
    - n_segments: Number of segments to register.
    - alpha: Regularization parameter alpha for deformable registration.
    - beta: Regularization parameter beta for deformable registration.
    - max_iterations: Maximum number of iterations for registration.
    - plot_figures: Boolean flag to enable/disable plotting of registration progress.
    - parallel: Boolean flag to enable/disable parallel processing.

    Returns:
    - trans_source_triangles_center: Transformed source mesh triangle centers after registration.
    """

    trans_source_triangles_center = source_mesh.triangles_center.copy()

    if parallel and not plot_figures:
        args_list = []
        for i in range(n_segments):
            args_list.append(
                (
                    i,
                    source_mesh.triangles_center,
                    target_mesh.triangles_center,
                    source_grps[i][0],
                    target_grps[i][0],
                    alpha,
                    beta,
                    max_iterations,
                    plot_figures,
                )
            )

        n_jobs = min(n_segments, os.cpu_count())
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(align_segment, args_list)

        for i, TY in results:
            trans_source_triangles_center[source_grps[i][0], :] = TY
    else:
        for i in range(n_segments):
            _, TY = align_segment(
                (
                    i,
                    source_mesh.triangles_center,
                    target_mesh.triangles_center,
                    source_grps[i][0],
                    target_grps[i][0],
                    alpha,
                    beta,
                    max_iterations,
                    plot_figures,
                )
            )
            trans_source_triangles_center[source_grps[i][0], :] = TY

    return trans_source_triangles_center


def growth_mapping(
    initial_mesh: trimesh.Trimesh,
    final_mesh: trimesh.Trimesh,
    ncluster: int,
    manifold_curvatures: pd.DataFrame,
    curv_m: int = 1,
    plot_figures: bool = False,
    parallel: bool = False,
    unravel: bool = False,
    mdiv: int = 1,
    alpha: float = 1.0,
    beta: float = 10.0,
    cline_initial_pos: Optional[np.ndarray] = None,
    cline_initial_div: Optional[np.ndarray] = None,
    cline_final_pos: Optional[np.ndarray] = None,
    cline_final_div: Optional[np.ndarray] = None,
):
    """
    Perform growth mapping between initial and final meshes.

    Function performs a growth mapping between the initial mesh and the
    final mesh, meaning that it finds a per-element growth rate that
    characterizes the geometric change between the two geometries.

    Steps:
        1. Rotate/transform initial geometry onto final geometry
        2. (optional) Unravel initial/final geometries and perform secondary
           non-rigid transformation piecewise along centerline
        3. Perform k-means on corresponding clusters between the initial
           and final geometries
        4. Calculate area change per cluster.
        5. Calculate growth rate per cluster.

    Parameters:
        initial_mesh (trimesh.Trimesh): The initial mesh.
        final_mesh (trimesh.Trimesh): The final mesh.
        ncluster (int): Number of clusters for k-means.
        manifold_curvatures (pd.DataFrame): DataFrame from GetAnatoMeshResults containing manifold curvature data for both meshes.
        curv_m (int): m value to use for manifold_curvatures
        plot_figures (bool): Whether to plot figures during processing.
        parallel (bool): Whether to use parallel processing for segment registration.

        For unraveling:
            unravel (bool): Whether to perform unraveling and segment-wise registration.
            mdiv (int): Number of divisions for unraveling.
            alpha (float): Alpha parameter for deformable registration.
            beta (float): Beta parameter for deformable registration.
            cline_initial_pos (np.ndarray): Centerline positions for initial mesh.
            cline_initial_div (np.ndarray): Centerline derivatives for initial mesh.
            cline_final_pos (np.ndarray): Centerline positions for final mesh.
            cline_final_div (np.ndarray): Centerline derivatives for final mesh.
    Returns:
        area_changes (np.ndarray): Array of area changes per cluster.
        intgaussian_changes (np.ndarray): Array of integrated Gaussian curvature changes per cluster.
        initial_kms (KMeans): KMeans object for initial mesh clustering.
        final_kms (KMeans): KMeans object for final mesh clustering.
    """

    # Rigid registration
    initial_pcd = o3d.geometry.PointCloud()
    final_pcd = o3d.geometry.PointCloud()
    initial_pcd.points = o3d.utility.Vector3dVector(initial_mesh.vertices)
    final_pcd.points = o3d.utility.Vector3dVector(final_mesh.vertices)
    initial_pcd.normals = o3d.utility.Vector3dVector(initial_mesh.vertex_normals)
    final_pcd.normals = o3d.utility.Vector3dVector(final_mesh.vertex_normals)

    translation_matrix = calculate_translation_matrix(initial_pcd, final_pcd)
    icp_result = o3d.pipelines.registration.registration_icp(
        source=initial_pcd,
        target=final_pcd,
        max_correspondence_distance=1,
        init=translation_matrix,
    )

    initial_translated_mesh = copy.deepcopy(initial_mesh)
    initial_translated_mesh.apply_transform(icp_result.transformation)

    if plot_figures:
        plot_registration(initial_pcd, final_pcd, icp_result.transformation)

    if unravel:
        # Perform unraveling
        initial_unraveled_grps = unravel_elems(
            initial_mesh,
            cline_initial_pos,
            cline_initial_div,
            m=mdiv,
            n=1,
            plot_figures=plot_figures,
        )
        final_unraveled_grps = unravel_elems(
            final_mesh,
            cline_final_pos,
            cline_final_div,
            m=mdiv,
            n=1,
            plot_figures=plot_figures,
        )
        # Perform segment-wise non-rigid registration
        aligned_source_triangles_center = segment_registration(
            initial_translated_mesh,
            final_mesh,
            initial_unraveled_grps,
            final_unraveled_grps,
            n_segments=mdiv,
            alpha=1,
            beta=10,
            max_iterations=30,
            parallel=False,
        )
    else:
        aligned_source_triangles_center = initial_translated_mesh.triangles_center

    initial_kms = KMeans(ncluster).fit(aligned_source_triangles_center)
    final_kms = KMeans(ncluster, init=initial_kms.cluster_centers_, max_iter=5).fit(
        final_mesh.triangles_center
    )

    # Extract integrated Gaussian curvature data
    initial_manifold_data = {}
    final_manifold_data = {}

    for m, name in manifold_curvatures.keys():
        if name == initial_name:
            initial_manifold_data[m] = {
                "patch_data": manifold_curvatures[(m, name)][0],
                "patch_labels": manifold_curvatures[(m, name)][1],
            }
        elif name == final_name:
            final_manifold_data[m] = {
                "patch_data": manifold_curvatures[(m, name)][0],
                "patch_labels": manifold_curvatures[(m, name)][1],
            }
        else:
            raise ValueError(f"Unknown manifold name: {name}")

    initial_mesh.intgaussian_faces = (
        initial_manifold_data[curv_m]["patch_data"]
        .loc[initial_manifold_data[curv_m]["patch_labels"], "IntGaussian"]
        .values
    )
    final_mesh.intgaussian_faces = (
        final_manifold_data[curv_m]["patch_data"]
        .loc[final_manifold_data[curv_m]["patch_labels"], "IntGaussian"]
        .values
    )

    # Ensure labels cover all clusters
    assert set(initial_kms.labels_) == set(
        range(ncluster)
    ), "Initial clustering has empty clusters!"
    assert set(final_kms.labels_) == set(
        range(ncluster)
    ), "Final clustering has empty clusters!"

    # Compute sums per cluster
    initial_Ks = np.bincount(
        initial_kms.labels_, weights=initial_mesh.intgaussian_faces, minlength=ncluster
    )
    final_Ks = np.bincount(
        final_kms.labels_, weights=final_mesh.intgaussian_faces, minlength=ncluster
    )

    initial_As = np.bincount(
        initial_kms.labels_, weights=initial_mesh.area_faces, minlength=ncluster
    )
    final_As = np.bincount(
        final_kms.labels_, weights=final_mesh.area_faces, minlength=ncluster
    )

    # Compute area changes per cluster, with only nonnegative growth allowed
    area_changes = (np.sqrt(final_As) - np.sqrt(initial_As)) / np.sqrt(initial_As)
    area_changes = np.maximum(0, area_changes)

    intgaussian_changes = final_Ks - initial_Ks

    return area_changes, intgaussian_changes, initial_kms, final_kms
