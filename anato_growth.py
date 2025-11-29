import copy
import os
from functools import partial
from typing import Optional

import multiprocessing as mp
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
import trimesh
from IPython.display import display
from pycpd import DeformableRegistration
from sklearn.cluster import KMeans

from anato_unravel import unravel_elems


def plot_registration(source_transformed, target, save_path=None, show=False):
    """
    Create a Plotly 3D figure showing source (transformed) and target point clouds,
    optionally save it to file, and return the figure object.

    Args:
        source_transformed: array-like, transformed source point cloud.
        target: array-like, target point cloud.
        save_path: optional path to save the figure.
        show: whether to call fig.show()

    Returns:
        fig: plotly.graph_objects.Figure
    """
    # build Plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            mode="markers",
            marker=dict(size=2, color="#00A6EE", opacity=0.7),
            name="Target (final)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=source_transformed[:, 0],
            y=source_transformed[:, 1],
            z=source_transformed[:, 2],
            mode="markers",
            marker=dict(size=2, color="#FFB400", opacity=0.7),
            name="Source (transformed)",
        )
    )
    fig.update_layout(
        title="Registration result",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800,
        height=600,
    )

    # show in notebook or interactive session
    if show:
        fig.show()

    # save file if requested
    if save_path:
        fig.write_html(save_path)

    return fig


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
    transformation_matrix = np.eye(4, dtype=np.float64)
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def save_segment_registration(
    iteration, error, X, Y, dir_path=None, max_iterations=None, division=None
):
    """
    Saves a snapshot of the registration process to disk.
    """
    if iteration % 5 != 0 and iteration != max_iterations - 1:
        return

    # Ensure arrays are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # 2. Create a standard Figure (not FigureWidget)
    fig = go.Figure()

    # Add Target (X)
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(size=2, color="red", opacity=0.7),
            name="Target (final)",
        )
    )

    # Add Source (Y)
    fig.add_trace(
        go.Scatter3d(
            x=Y[:, 0],
            y=Y[:, 1],
            z=Y[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.7),
            name="Source (initial)",
        )
    )

    # Update Layout
    fig.update_layout(
        title=f"Division {division+1} — Iter {iteration} (Error: {error:.4f})",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    os.makedirs(os.path.join(dir_path, "segment_alignment_figs"), exist_ok=True)
    filename = os.path.join(
        dir_path, "segment_alignment_figs", f"div{division}_iter_{iteration:04d}.html"
    )
    fig.write_html(filename)


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
        dir_path,
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
        partial(
            save_segment_registration,
            dir_path=dir_path,
            max_iterations=max_iterations,
            division=i,
        )
        if plot_figures
        else None
    )

    TY, _ = deform_reg.register(callback)

    if TY.shape[0] != source_idx.shape[0]:
        raise ValueError(
            f"Segment {i} mismatch: source {source_idx.shape[0]}, transformed {TY.shape[0]}"
        )

    return i, TY


def segment_registration(
    source_mesh,
    target_mesh,
    source_grps,
    target_grps,
    n_segments,
    alpha,
    beta,
    max_iterations,
    plot_figures,
    dir_path,
    parallel,
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
    - dir_path: Directory path to save figures.
    - parallel: Boolean flag to enable/disable parallel processing.

    Returns:
    - trans_source_triangles_center: Transformed source mesh triangle centers after registration.
    """

    trans_source_triangles_center = source_mesh.triangles_center.copy()

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
                dir_path,
            )
        )

    if parallel and not plot_figures:
        n_jobs = min(n_segments, os.cpu_count())
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(align_segment, args_list)

        for i, TY in results:
            trans_source_triangles_center[source_grps[i][0], :] = TY
    else:
        for i in range(n_segments):
            _, TY = align_segment(args_list[i])
            trans_source_triangles_center[source_grps[i][0], :] = TY

    return trans_source_triangles_center


def growth_mapping(
    initial_mesh: trimesh.Trimesh,
    final_mesh: trimesh.Trimesh,
    initial_manifold_data: dict,
    final_manifold_data: dict,
    ncluster: int,
    plot_figures: bool = False,
    dir_path: Optional[str] = None,
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
        initial_manifold_data (dict): Manifold data for the initial mesh.
        final_manifold_data (dict): Manifold data for the final mesh.
        ncluster (int): Number of clusters for k-means.
        dir_path (str, optional): Directory path to save intermediate results. If None, no files are saved.
        plot_figures (bool): Whether to plot figures during processing. If True, set dir_path to true
        dir_path (str, optional): Directory path to save results and figures (if plotting).
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
    print("Performing rigid registration")
    initial_pcd = o3d.geometry.PointCloud()
    final_pcd = o3d.geometry.PointCloud()

    iv_cpy = initial_mesh.vertices.copy().astype(np.float64)
    fv_cpy = final_mesh.vertices.copy().astype(np.float64)
    initial_pcd.points = o3d.utility.Vector3dVector(iv_cpy)
    final_pcd.points = o3d.utility.Vector3dVector(fv_cpy)

    ivn_cpy = initial_mesh.vertex_normals.copy().astype(np.float64)
    fvn_cpy = final_mesh.vertex_normals.copy().astype(np.float64)

    initial_pcd.normals = o3d.utility.Vector3dVector(ivn_cpy)
    final_pcd.normals = o3d.utility.Vector3dVector(fvn_cpy)

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
        assert dir_path is not None, "dir_path must be provided to save figures"
        fig = plot_registration(
            np.asarray(initial_translated_mesh.triangles_center),
            np.asarray(final_mesh.triangles_center),
            save_path=(os.path.join(dir_path, "rigid_registration.html")),
        )

    if unravel:
        # Perform unraveling
        initial_unraveled_grps = unravel_elems(
            initial_manifold_data["name"],
            initial_mesh,
            cline_initial_pos,
            cline_initial_div,
            m=mdiv,
            n=1,
            plot_figures=plot_figures,
            dir_path=dir_path,
        )
        final_unraveled_grps = unravel_elems(
            final_manifold_data["name"],
            final_mesh,
            cline_final_pos,
            cline_final_div,
            m=mdiv,
            n=1,
            plot_figures=plot_figures,
            dir_path=dir_path,
        )

        print("Performing segment-wise deformable registration")
        aligned_source_triangles_center = segment_registration(
            initial_translated_mesh,
            final_mesh,
            initial_unraveled_grps,
            final_unraveled_grps,
            n_segments=mdiv,
            alpha=1,
            beta=10,
            max_iterations=30,
            plot_figures=plot_figures,
            dir_path=dir_path,
            parallel=parallel,
        )
        if plot_figures:
            fig = plot_registration(
                aligned_source_triangles_center,
                np.asarray(final_mesh.triangles_center),
                save_path=(os.path.join(dir_path, "deformable_registration.html")),
            )
    else:
        aligned_source_triangles_center = initial_translated_mesh.triangles_center

    initial_kms = KMeans(ncluster).fit(aligned_source_triangles_center)
    final_kms = KMeans(ncluster, init=initial_kms.cluster_centers_, max_iter=5).fit(
        final_mesh.triangles_center
    )

    initial_mesh.intgaussian_faces = (
        initial_manifold_data["patch_data"]
        .loc[initial_manifold_data["patch_labels"], "IntGaussian"]
        .values
    )
    final_mesh.intgaussian_faces = (
        final_manifold_data["patch_data"]
        .loc[final_manifold_data["patch_labels"], "IntGaussian"]
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

    if dir_path:
        # Save results to CSV
        results_df = pd.DataFrame(
            {
                "Cluster": np.arange(ncluster),
                "AreaChange": area_changes,
                "InitialIntGaussian": initial_Ks,
                "FinalIntGaussian": final_Ks,
                "IntGaussianChange": intgaussian_changes,
                "InitialArea": initial_As,
                "FinalArea": final_As,
            }
        )
        results_csv_path = os.path.join(dir_path, "growth_mapping_results.csv")
        results_df.to_csv(results_csv_path, index=False)

        # Save the k-means cluster info
        np.savez(
            os.path.join(dir_path, "kmeans_clusters.npz"),
            initial_centers=initial_kms.cluster_centers_,
            final_centers=final_kms.cluster_centers_,
            initial_labels=initial_kms.labels_,
            final_labels=final_kms.labels_,
        )

    return area_changes, intgaussian_changes, initial_kms, final_kms
