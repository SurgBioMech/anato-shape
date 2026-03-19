import copy
import os
import shutil
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
from scipy.interpolate import RBFInterpolator
from sklearn.cluster import KMeans
import time
import json
from anato_unravel import unravel_elems


PLOT_UNRAVEL_SLICE_FIGS = False  # Used for debugging unraveling process


def transform_points_via_correspondence(
    points: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    smoothing: float = 0.0,
    max_control_points: Optional[int] = None,
) -> np.ndarray:
    """
    Transform points from source space to target space using RBF interpolation.

    Given a correspondence between source_coords and target_coords (same number of points),
    this function learns a smooth mapping and applies it to transform arbitrary points.

    Performance: For large meshes, automatically subsamples control points
    to improve speed. RBF interpolation is O(N³) in the number of control points.

    Args:
        points: (N, 3) array of points to transform.
        source_coords: (M, 3) array of original/source coordinates.
        target_coords: (M, 3) array of corresponding transformed/target coordinates.
        smoothing: RBF smoothing parameter (0 = exact interpolation).
        max_control_points: Maximum control points for RBF. If None, automatically
                           scales based on the number of points to transform:
                           max(2000, 3 * n_points) to ensure good coverage.
                           Set explicitly to control speed/accuracy tradeoff.

    Returns:
        (N, 3) array of transformed points.
    """
    # Determine max_control_points if not specified
    n_points = points.shape[0]
    if max_control_points is None:
        # Scale control points with query points for accuracy
        # Use at least 2000, or 3x the number of points to transform
        max_control_points = max(2000, 3 * n_points)

    # Subsample if we have too many control points
    n_source = source_coords.shape[0]
    if n_source > max_control_points:
        # Random subsample for speed
        indices = np.random.choice(n_source, max_control_points, replace=False)
        source_subset = source_coords[indices]
        target_subset = target_coords[indices]
        print(
            f"RBF: Subsampling {n_source} -> {max_control_points} control points for speed"
        )
    else:
        source_subset = source_coords
        target_subset = target_coords

    interpolator = RBFInterpolator(
        source_subset,
        target_subset,
        smoothing=smoothing,
        kernel="thin_plate_spline",  # Often faster than default 'multiquadric'
    )
    return interpolator(points)


def fit_corresponding_kmeans(
    n_clusters: int,
    initial_coords: np.ndarray,
    aligned_coords: np.ndarray,
    final_coords: np.ndarray,
    init_centers: Optional[np.ndarray] = None,
    initial_max_iter: int = 300,
    final_max_iter: int = 5,
    max_control_points: Optional[int] = None,
):
    """
    Fit K-means on initial mesh and a corresponding K-means on final mesh.

    The initial K-means cluster centers are transformed via RBF interpolation
    (using the correspondence between initial_coords and aligned_coords) to
    initialize the final K-means. This ensures cluster correspondence between
    the two meshes.

    Args:
        n_clusters: Number of clusters.
        initial_coords: (N, 3) coordinates of the original initial mesh.
        aligned_coords: (N, 3) coordinates after registration (same point order).
        final_coords: (M, 3) coordinates of the final mesh.
        init_centers: Optional (n_clusters, 3) array to seed initial K-means.
                      If None, uses random k-means++ initialization.
        initial_max_iter: Max iterations for initial K-means (use 1 to preserve seeding).
        final_max_iter: Max iterations for final K-means.
        max_control_points: Maximum control points for RBF interpolation when
                           transforming cluster centers. If None, automatically
                           scales based on n_clusters (default: max(2000, 3*n_clusters)).

    Returns:
        (initial_kms, final_kms): Fitted KMeans objects for initial and final meshes.
    """
    # Fit K-means on ORIGINAL initial mesh coordinates
    if init_centers is not None:
        initial_kms = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,  # Required when using custom init
            max_iter=initial_max_iter,
        ).fit(initial_coords)
    else:
        initial_kms = KMeans(n_clusters=n_clusters).fit(initial_coords)

    # Transform cluster centers from initial space to aligned/final space
    transformed_centers = transform_points_via_correspondence(
        initial_kms.cluster_centers_,
        initial_coords,
        aligned_coords,
        max_control_points=max_control_points,
    )

    # Fit final K-means seeded with transformed centers
    final_kms = KMeans(
        n_clusters=n_clusters,
        init=transformed_centers,
        n_init=1,  # Required when using custom init
        max_iter=final_max_iter,
    ).fit(final_coords)

    return initial_kms, final_kms


def generate_random_colors(n, seed=42):
    """Generates N distinct random RGB color strings."""
    np.random.seed(seed)
    colors = []
    for _ in range(n):
        r, g, b = np.random.randint(0, 255, 3)
        colors.append(f"rgb({r},{g},{b})")
    return np.array(colors)


def reassign_per_face_intgaussian(mesh, manifold_data):
    """
    Calculate per-face integrated Gaussian curvature based on manifold data.
    Per-face values are computed as:
        intgaussian_face = area_face * (patch IntGaussian) / (patch area)
    """
    return (
        mesh.area_faces
        * (
            manifold_data["patch_data"]
            .loc[manifold_data["patch_labels"], "IntGaussian"]
            .values
        )
    ) / (
        manifold_data["patch_data"]
        .loc[manifold_data["patch_labels"], "Patch_Area"]
        .values
    )


def plot_mesh_patch_values(
    mesh: trimesh.Trimesh,
    patch_labels: np.ndarray,
    patch_values: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    discrete: bool = False,
    colormap: Optional[np.ndarray] = None,
):
    """
    Plot a mesh colored by patch data.

    Args:
        mesh: trimesh.Trimesh object.
        patch_labels: Array of shape (n_faces,) mapping each Face -> Patch Index/ID.
        patch_values: Optional array mapping Patch Index -> Value.
                      - If None and discrete=True, 'patch_labels' are used as the values (coloring by ID).
                      - If provided, we map patch_values[patch_labels] to get per-face data.
        title: Plot title.
        save_path: Optional path to save HTML.
        discrete: If True, uses flat coloring (no interpolation).
                  If False, calculates vertex averages for a smooth heatmap (requires patch_values).
        colormap: Array of color strings. If None, random colors are generated.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # 1. Standardize Inputs
    patch_labels = np.asarray(patch_labels).ravel()
    if patch_labels.size != faces.shape[0]:
        raise ValueError(
            f"patch_labels length ({patch_labels.size}) must match face count ({faces.shape[0]})."
        )

    # 2. Determine Data Per Face
    if patch_values is not None:
        # Map: Face -> Patch Index -> Value
        pv = np.asarray(patch_values).ravel()
        face_data = pv[patch_labels]
    else:
        # Fallback: Face -> Patch Index (Use the label itself as the value)
        if not discrete:
            raise ValueError(
                "patch_values is required for continuous (smooth) heatmaps."
            )
        face_data = patch_labels

    # 3. Create Trace based on Mode
    if discrete:
        # --- Discrete Mode (Flat Coloring by ID) ---
        face_ids = face_data.astype(int)

        # Handle Colormap
        if colormap is None:
            # We need enough colors for the maximum ID found
            n_colors = int(face_ids.max()) + 1
            # Simple random RGB generator if helper not available
            colormap = np.array(
                [
                    f"rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})"
                    for _ in range(n_colors)
                ]
            )

        # Safety check for colormap size
        if len(colormap) <= face_ids.max():
            # Extend colormap randomly if it's too short
            extra_needed = int(face_ids.max()) - len(colormap) + 1
            extra_colors = [
                f"rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})"
                for _ in range(extra_needed)
            ]
            colormap = np.concatenate([colormap, extra_colors])

        face_colors = colormap[face_ids]

        mesh3d = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            facecolor=face_colors,
            flatshading=True,
            name=title,
        )

    else:
        # --- Continuous Mode (Smooth Heatmap) ---
        # Interpolate face values to vertices for smooth gradients
        n_verts = verts.shape[0]
        vertex_accum = np.zeros(n_verts, dtype=np.float64)
        vertex_counts = np.zeros(n_verts, dtype=np.int32)

        # Vectorized accumulation is hard with numpy alone, using simple loop for clarity/safety
        # (Or use scipy.sparse for speed if meshes are huge, but loop is fine for plotting)
        for f_idx, (i, j, k) in enumerate(faces):
            val = face_data[f_idx]
            vertex_accum[[i, j, k]] += val
            vertex_counts[[i, j, k]] += 1

        # Avoid division by zero
        vertex_counts[vertex_counts == 0] = 1
        vertex_values = vertex_accum / vertex_counts

        mesh3d = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vertex_values,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=title or "Value"),
            flatshading=False,
            opacity=1.0,
            name=title,
        )

    # 4. Finalize Layout
    fig = go.Figure(data=[mesh3d])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


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
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
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


def perform_rigid_registration(
    initial_mesh, final_mesh, mesh_m, plot_figures=False, dir_path=None
):
    """
    Perform rigid registration of initial_mesh onto final_mesh using ICP.
    Args:
        initial_mesh: trimesh.Trimesh
        final_mesh: trimesh.Trimesh
        mesh_m: mesh parameter (for figure naming)
        plot_figures: whether to plot registration result
        dir_path: directory path to save figures (if plot_figures is True)
    Returns the transformed initial_mesh.
    """
    print(f"mesh_m = {mesh_m}, performing rigid registration")
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
            save_path=(os.path.join(dir_path, f"rigid_registration_{mesh_m}.html")),
        )
    return initial_translated_mesh


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
    iteration,
    error,
    X,
    Y,
    dir_path=None,
    max_iterations=None,
    division=None,
    plot_figures=False,
):
    """
    Saves a snapshot of the registration process to disk.

    Args:
        dir_path: Direct directory path where figures will be saved.
    """
    if iteration % 5 != 0 and iteration != max_iterations - 1:
        return

    filename = os.path.join(dir_path, f"div{division}_iter_{iteration:04d}.html")
    if os.path.exists(filename):
        return  # already saved

    if not plot_figures:
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
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    os.makedirs(dir_path, exist_ok=True)
    fig.write_html(filename)


def align_segment(args):
    """
    Align a single segment for deformable registration.
    Returns the index and transformed points.
    """
    (
        i,
        source_segment_centers,
        target_segment_centers,
        alpha,
        beta,
        max_iterations,
        plot_figures,
        dir_path,
    ) = args

    print(f"Aligning segment {i+1}")

    Y = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_segment_centers))
    X = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_segment_centers))

    Y.transform(calculate_translation_matrix(Y, X))

    deform_reg = DeformableRegistration(
        X=np.asarray(X.points),
        Y=np.asarray(Y.points),
        alpha=alpha,
        beta=beta,
        max_iterations=max_iterations,
    )

    callback = partial(
        save_segment_registration,
        dir_path=dir_path,
        max_iterations=max_iterations,
        division=i,
        plot_figures=plot_figures,
    )

    TY, _ = deform_reg.register(callback)

    if TY.shape[0] != source_segment_centers.shape[0]:
        raise ValueError(
            f"Segment {i} mismatch: source {source_segment_centers.shape[0]}, transformed {TY.shape[0]}"
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
        src = np.ascontiguousarray(
            source_mesh.triangles_center[source_grps[i][0], :], dtype=np.float64
        )
        tgt = np.ascontiguousarray(
            target_mesh.triangles_center[target_grps[i][0], :], dtype=np.float64
        )
        args_list.append(
            (
                i,
                src,
                tgt,
                float(alpha),
                float(beta),
                int(max_iterations),
                bool(plot_figures),
                dir_path,
            )
        )

    if parallel:
        # reserve one core for responsiveness
        n_jobs = min(n_segments, max(1, os.cpu_count()))
        print(f"Using {n_jobs} parallel jobs for segment registration")

        ctx = mp.get_context("spawn")
        pool = None
        results_buffer = [None] * n_segments
        try:
            pool = ctx.Pool(processes=n_jobs)
            # use imap_unordered to get results as they arrive (better throughput)
            for done_count, res in enumerate(
                pool.imap_unordered(align_segment, args_list, chunksize=1), start=1
            ):
                idx, TY = res
                results_buffer[idx] = TY
                print(f"Completed {done_count}/{n_segments} segments", end="\r")
        except Exception:
            if pool is not None:
                pool.terminate()
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            print("")  # finish progress line

        # check results and write back
        for i, TY in enumerate(results_buffer):
            if TY is None:
                raise RuntimeError(f"Segment {i} failed during parallel registration")
            trans_source_triangles_center[source_grps[i][0], :] = TY
    else:
        for arg in args_list:
            idx, TY = align_segment(arg)
            trans_source_triangles_center[source_grps[idx][0], :] = TY

    return trans_source_triangles_center


def growth_mapping(
    initial_meshes: dict,
    final_meshes: dict,
    initial_manifold_datas: dict,
    final_manifold_datas: dict,
    ncluster_input: int,
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

    - If `ncluster_input` is None, clustering is "pegged" to the intrinsic patch count
      defined in `initial_manifold_datas` for each curvature scale (Dynamic Mode).

    - Results are saved to a CSV file, and a `growth_metadata.json` registry is
      updated. This registry contains an `n_clusters` field (int or "dynamic") to
      allow reliable lookup of the correct results file for any configuration.

    Steps:
        1. Rotate/transform initial geometry onto final geometry
        2. (optional) Unravel initial/final geometries and perform secondary
           non-rigid transformation piecewise along centerline
        3. Perform k-means on corresponding clusters between the initial
           and final geometries
        4. Calculate area change per cluster.
        5. Calculate growth rate per cluster.

    Parameters:
        initial_meshes (dict[trimesh.Trimesh]): A dictionary mapping mesh parameters to the initial meshes.
        final_meshes (dict[trimesh.Trimesh]): A dictionary mapping mesh parameters to the final meshes.
        initial_manifold_datas (dict): Manifold data for the initial meshes.
        final_manifold_datas (dict): Manifold data for the final meshes.
        ncluster (int): Number of clusters for k-means.
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
    """
    assert set(initial_meshes.keys()) == set(
        final_meshes.keys()
    ), "Initial and final mesh keys must match."

    assert set(initial_manifold_datas.keys()) == set(
        final_manifold_datas.keys()
    ), "Initial and final manifold data keys must match."

    mesh_ms = list(initial_manifold_datas.keys())
    curv_ms = list(set(initial_manifold_datas[mesh_ms[0]].keys()).difference({"name"}))

    growth_data = {}
    for mesh_m in initial_meshes.keys():
        growth_data[mesh_m] = {}
        initial_mesh = initial_meshes[mesh_m]
        final_mesh = final_meshes[mesh_m]
        initial_manifold_data = initial_manifold_datas[mesh_m]
        final_manifold_data = final_manifold_datas[mesh_m]

        initial_translated_mesh = perform_rigid_registration(
            initial_mesh, final_mesh, mesh_m, plot_figures, dir_path
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
                plot_unravel_figs=PLOT_UNRAVEL_SLICE_FIGS,
                plot_unravel_groups_figs=plot_figures,
                dir_path=dir_path,
            )
            final_unraveled_grps = unravel_elems(
                final_manifold_data["name"],
                final_mesh,
                cline_final_pos,
                cline_final_div,
                m=mdiv,
                n=1,
                plot_unravel_figs=PLOT_UNRAVEL_SLICE_FIGS,
                plot_unravel_groups_figs=plot_figures,
                dir_path=dir_path,
            )

            print(
                f"mesh_m = {mesh_m}, performing deformable registration of {mdiv} segments"
            )
            segment_fig_dir = os.path.join(
                dir_path, "segment_alignment_figs", f"mesh_{mesh_m}"
            )
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
                dir_path=segment_fig_dir,
                parallel=parallel,
            )
            if plot_figures:
                fig = plot_registration(
                    aligned_source_triangles_center,
                    np.asarray(final_mesh.triangles_center),
                    save_path=(
                        os.path.join(dir_path, f"deformable_registration_{mesh_m}.html")
                    ),
                )
        else:
            aligned_source_triangles_center = initial_translated_mesh.triangles_center

        initial_As = None
        final_As = None
        initial_kms = None
        final_kms = None

        # ─────────────────────────────────────────────────────────────────────
        # Static mode: K-means with random initialization, run once per mesh_m.
        # Clusters are shared across all curv_m values since n_clusters is fixed.
        # ─────────────────────────────────────────────────────────────────────
        if ncluster_input:
            ncluster_label = str(ncluster_input)
            ncluster = ncluster_input
            print(f"Calculating mapping with static ncluster = {ncluster}")

            initial_kms, final_kms = fit_corresponding_kmeans(
                n_clusters=ncluster,
                initial_coords=initial_mesh.triangles_center,
                aligned_coords=aligned_source_triangles_center,
                final_coords=final_mesh.triangles_center,
                init_centers=None,  # Random k-means++ initialization
                max_control_points=None,  # Default scaling based on n_clusters
            )

            # Ensure labels cover all clusters
            assert set(initial_kms.labels_) == set(
                range(ncluster)
            ), "Initial clustering has empty clusters!"
            assert set(final_kms.labels_) == set(
                range(ncluster)
            ), "Final clustering has empty clusters!"

            # Compute area sums per cluster
            initial_As = np.bincount(
                initial_kms.labels_, weights=initial_mesh.area_faces, minlength=ncluster
            )
            final_As = np.bincount(
                final_kms.labels_, weights=final_mesh.area_faces, minlength=ncluster
            )
            if dir_path:
                print(f"Saving k-means cluster info for mesh_m={mesh_m} to {dir_path}")
                np.savez(
                    os.path.join(
                        dir_path,
                        f"kmeans_clusters_{mesh_m}_{ncluster}.npz",
                    ),
                    initial_centers=initial_kms.cluster_centers_,
                    final_centers=final_kms.cluster_centers_,
                    initial_labels=initial_kms.labels_,
                    final_labels=final_kms.labels_,
                )
        else:
            ncluster_label = "dynamic"

        for curv_m in curv_ms:
            print(f"Processing mesh_m={mesh_m}, curv_m={curv_m}...")

            initial_mesh.intgaussian_faces = reassign_per_face_intgaussian(
                initial_mesh, initial_manifold_data[curv_m]
            )
            final_mesh.intgaussian_faces = reassign_per_face_intgaussian(
                final_mesh, final_manifold_data[curv_m]
            )

            # ─────────────────────────────────────────────────────────────────
            # Dynamic mode: K-means seeded from curvature patch centroids.
            # n_clusters varies per curv_m based on patch count, so K-means
            # must run inside the curv_m loop. Uses max_iter=1 to preserve
            # the patch structure from curvature calculation.
            # ─────────────────────────────────────────────────────────────────
            if ncluster_input is None:
                patch_labels = initial_manifold_data[curv_m]["patch_labels"]
                current_n_clusters = np.unique(patch_labels).shape[0]
                print(
                    f"Calculating mapping with dynamic n_clusters = {current_n_clusters} "
                    "based on patches from curvature calculation"
                )

                # Calculate patch centroids in ORIGINAL initial mesh space
                df_orig = pd.DataFrame(
                    initial_mesh.triangles_center, columns=["x", "y", "z"]
                )
                df_orig["label"] = patch_labels
                init_centers = df_orig.groupby("label").mean().sort_index().values

                initial_kms, final_kms = fit_corresponding_kmeans(
                    n_clusters=current_n_clusters,
                    initial_coords=initial_mesh.triangles_center,
                    aligned_coords=aligned_source_triangles_center,
                    final_coords=final_mesh.triangles_center,
                    init_centers=init_centers,  # Seed from patch centroids
                    initial_max_iter=1,  # Preserve patch structure
                    max_control_points=None,  # Default scaling based on n_clusters
                )

                # Ensure labels cover all clusters
                assert set(initial_kms.labels_) == set(
                    range(current_n_clusters)
                ), "Initial clustering has empty clusters!"
                assert set(final_kms.labels_) == set(
                    range(current_n_clusters)
                ), "Final clustering has empty clusters!"

                # Compute areas
                initial_As = np.bincount(
                    initial_kms.labels_,
                    weights=initial_mesh.area_faces,
                    minlength=current_n_clusters,
                )
                final_As = np.bincount(
                    final_kms.labels_,
                    weights=final_mesh.area_faces,
                    minlength=current_n_clusters,
                )
                if dir_path:
                    print(
                        f"Saving k-means cluster info for mesh_m={mesh_m} curv_m={curv_m} to {dir_path}"
                    )
                    np.savez(
                        os.path.join(
                            dir_path,
                            f"kmeans_clusters_{mesh_m}_{curv_m}_{current_n_clusters}.npz",
                        ),
                        initial_centers=initial_kms.cluster_centers_,
                        final_centers=final_kms.cluster_centers_,
                        initial_labels=initial_kms.labels_,
                        final_labels=final_kms.labels_,
                    )
            else:
                current_n_clusters = ncluster

            # Compute curvature sums per cluster
            initial_Ks = np.bincount(
                initial_kms.labels_,
                weights=initial_mesh.intgaussian_faces,
                minlength=current_n_clusters,
            )
            final_Ks = np.bincount(
                final_kms.labels_,
                weights=final_mesh.intgaussian_faces,
                minlength=current_n_clusters,
            )

            # Ensure curvatures sum correctly
            assert np.isclose(
                initial_Ks.sum(),
                initial_manifold_data[curv_m]["patch_data"]["IntGaussian"].sum(),
            ), "Initial integrated Gaussian curvature mismatch"
            assert np.isclose(
                final_Ks.sum(),
                final_manifold_data[curv_m]["patch_data"]["IntGaussian"].sum(),
            ), "Final integrated Gaussian curvature mismatch"

            growth_data[mesh_m][curv_m] = {}
            growth_data[mesh_m][curv_m]["ncluster"] = current_n_clusters
            growth_data[mesh_m][curv_m]["initial_As"] = initial_As
            growth_data[mesh_m][curv_m]["final_As"] = final_As
            growth_data[mesh_m][curv_m]["initial_Ks"] = initial_Ks
            growth_data[mesh_m][curv_m]["final_Ks"] = final_Ks

            if plot_figures:
                # Organize figures by mesh_m/curv_m subdirectories
                fig_dir = os.path.join(
                    dir_path,
                    "heatmap_figs",
                    f"mesh_{mesh_m}",
                    f"curv_{curv_m}",
                    f"{ncluster_label}_clusters",
                )
                os.makedirs(fig_dir, exist_ok=True)

                # Plot initial/final cluster IDs
                colormap = generate_random_colors(current_n_clusters)
                plot_mesh_patch_values(
                    mesh=initial_mesh,
                    patch_labels=initial_kms.labels_,
                    title=f"Initial KMeans Clusters [M={mesh_m}, C={curv_m}]",
                    save_path=os.path.join(fig_dir, "initial_clusters.html"),
                    discrete=True,
                    colormap=colormap,
                )
                plot_mesh_patch_values(
                    mesh=final_mesh,
                    patch_labels=final_kms.labels_,
                    title=f"Final KMeans Clusters [M={mesh_m}, C={curv_m}]",
                    save_path=os.path.join(fig_dir, "final_clusters.html"),
                    discrete=True,
                    colormap=colormap,
                )

                # Compute changes per cluster
                area_changes = (np.sqrt(final_As) - np.sqrt(initial_As)) / np.sqrt(
                    initial_As
                )
                intgaussian_changes = final_Ks - initial_Ks

                # Define plot specs: (value_key, short_name)
                plot_specs = [
                    ("Patch_Area", "area"),
                    ("IntGaussian", "intgaussian"),
                ]
                for value_key, short_name in plot_specs:
                    # Curvature-based patches
                    plot_mesh_patch_values(
                        mesh=initial_mesh,
                        patch_labels=initial_manifold_data[curv_m]["patch_labels"],
                        patch_values=initial_manifold_data[curv_m]["patch_data"][
                            value_key
                        ].values,
                        title=f"Initial {value_key} (Curv Patches)",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_initial_curv_patches.html"
                        ),
                    )
                    plot_mesh_patch_values(
                        mesh=final_mesh,
                        patch_labels=final_manifold_data[curv_m]["patch_labels"],
                        patch_values=final_manifold_data[curv_m]["patch_data"][
                            value_key
                        ].values,
                        title=f"Final {value_key} (Curv Patches)",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_final_curv_patches.html"
                        ),
                    )

                    # Growth mapping patches
                    growth_initial = (
                        initial_As if value_key == "Patch_Area" else initial_Ks
                    )
                    growth_final = final_As if value_key == "Patch_Area" else final_Ks
                    growth_change = (
                        area_changes
                        if value_key == "Patch_Area"
                        else intgaussian_changes
                    )

                    plot_mesh_patch_values(
                        mesh=initial_mesh,
                        patch_labels=initial_kms.labels_,
                        patch_values=growth_initial,
                        title=f"Initial {value_key} (Growth Patches)",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_initial_growth_patches.html"
                        ),
                    )
                    plot_mesh_patch_values(
                        mesh=final_mesh,
                        patch_labels=final_kms.labels_,
                        patch_values=growth_final,
                        title=f"Final {value_key} (Growth Patches)",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_final_growth_patches.html"
                        ),
                    )
                    plot_mesh_patch_values(
                        mesh=final_mesh,
                        patch_labels=final_kms.labels_,
                        patch_values=growth_change,
                        title=f"Change in {value_key}",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_change_on_final_growth_patches.html"
                        ),
                    )

    if dir_path:
        print(f"Saving growth mapping results to {dir_path}")
        results_df = pd.DataFrame()
        for mesh_m in initial_meshes.keys():
            for curv_m in curv_ms:
                data = growth_data[mesh_m][curv_m]
                current_n = data["ncluster"]

                row = pd.DataFrame(
                    {
                        "Mesh_M": mesh_m,
                        "Curv_M": curv_m,
                        "N_Segments": mdiv if unravel else 0,
                        "Cluster": np.arange(current_n),
                        "InitialIntGaussian": data["initial_Ks"],
                        "FinalIntGaussian": data["final_Ks"],
                        "InitialArea": data["initial_As"],
                        "FinalArea": data["final_As"],
                    }
                )
                results_df = pd.concat([results_df, row])

        csv_filename = f"growth_mapping_{ncluster_label}_results.csv"

        results_csv_path = os.path.join(dir_path, csv_filename)
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved CSV: {csv_filename}")

        # Update JSON Registry
        meta_path = os.path.join(dir_path, "growth_metadata.json")
        registry = {}

        # Load existing registry if available
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    registry = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Metadata file corrupted, creating new registry.")

        # Update specific key
        registry[ncluster_label] = {
            "csv_file": csv_filename,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "dynamic" if ncluster_input is None else "static",
            "n_clusters": ncluster_label,
        }

        with open(meta_path, "w") as f:
            json.dump(registry, f, indent=4)
        print(f"Updated metadata registry: {ncluster_label} -> {csv_filename}")


def parse_inp_growth_rates(inp_path):
    """Parse growth rates assigned to each solid element from an Abaqus INP file.

    Reads element sets, solid section assignments, and User Material definitions
    to map each element to its growth rate (7th constant in ``*User Material``).

    Parameters
    ----------
    inp_path : str or Path
        Path to the ``.inp`` file with growth material assignments.

    Returns
    -------
    node_coords : ndarray, shape (N, 3)
    node_ids : ndarray, shape (N,)
    elem_ids : ndarray, shape (E,)
    elem_connectivity : ndarray, shape (E, 4)
        Node IDs (1-based) for each C3D4 tetrahedron.
    element_growth_rates : ndarray, shape (E,)
        Growth rate for each element.
    """
    import re
    from pathlib import Path as _Path
    inp_path = _Path(inp_path)
    with open(inp_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # --- Parse nodes ---
    node_start = None
    elem_start = None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if node_start is None and (low == "*node" or low.startswith("*node,")):
            node_start = i + 1
        elif elem_start is None and low.startswith("*element,"):
            elem_start = i + 1

    node_rows = []
    for i in range(node_start, len(lines)):
        if lines[i].strip().startswith("*"):
            break
        parts = lines[i].split(",")
        if len(parts) >= 4:
            node_rows.append([float(p.strip()) for p in parts[:4]])
    node_arr = np.array(node_rows)
    node_ids = node_arr[:, 0].astype(int)
    node_coords = node_arr[:, 1:4]

    # --- Parse elements ---
    elem_rows = []
    for i in range(elem_start, len(lines)):
        if lines[i].strip().startswith("*"):
            break
        parts = lines[i].split(",")
        vals = [int(p.strip()) for p in parts if p.strip()]
        elem_rows.append(vals)
    elem_arr = np.array(elem_rows)
    elem_ids = elem_arr[:, 0]
    elem_connectivity = elem_arr[:, 1:5]

    # --- Parse element sets (Set-N -> element IDs) ---
    elsets = {}
    i = 0
    while i < len(lines):
        low = lines[i].strip().lower()
        if low.startswith("*elset") and "elset=" in low:
            m = re.search(r"elset\s*=\s*(\S+)", lines[i], re.IGNORECASE)
            if m:
                name = m.group(1).rstrip(",")
                ids = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("*"):
                    for p in lines[i].split(","):
                        p = p.strip()
                        if p:
                            ids.append(int(p))
                    i += 1
                elsets[name] = ids
                continue
        i += 1

    # --- Parse section assignments (elset -> material name) ---
    elset_to_material = {}
    for line in lines:
        if line.strip().lower().startswith("*solid section"):
            es = re.search(r"elset\s*=\s*(\S+)", line, re.IGNORECASE)
            mt = re.search(r"material\s*=\s*(\S+)", line, re.IGNORECASE)
            if es and mt:
                elset_to_material[es.group(1).rstrip(",")] = mt.group(1).rstrip(",")

    # --- Parse growth rates from User Material (7th constant) ---
    material_growth = {}
    i = 0
    while i < len(lines):
        low = lines[i].strip().lower()
        if low.startswith("*material"):
            m = re.search(r"name\s*=\s*(\S+)", lines[i], re.IGNORECASE)
            if m:
                mat_name = m.group(1).rstrip(",")
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip().lower().startswith("*user material"):
                        constants = []
                        for k in range(j + 1, min(j + 5, len(lines))):
                            if lines[k].strip().startswith("*"):
                                break
                            for p in lines[k].split(","):
                                p = p.strip()
                                if p:
                                    constants.append(float(p))
                        if len(constants) >= 7:
                            material_growth[mat_name] = constants[6]
                        break
        i += 1

    # --- Map element ID -> growth rate ---
    elem_id_to_idx = {int(eid): idx for idx, eid in enumerate(elem_ids)}
    element_growth_rates = np.zeros(len(elem_ids))
    for elset_name, material_name in elset_to_material.items():
        if elset_name not in elsets or material_name not in material_growth:
            continue
        gr = material_growth[material_name]
        for eid in elsets[elset_name]:
            if eid in elem_id_to_idx:
                element_growth_rates[elem_id_to_idx[eid]] = gr

    return node_coords, node_ids, elem_ids, elem_connectivity, element_growth_rates


def visualize_inp_growth(inp_path, output_path=None):
    """Visualize growth rates from a finalized growth INP file.

    Produces a two-panel interactive HTML figure:
      - Left: outer surface mesh colored by growth rate
      - Right: solid mesh element centroids colored by growth rate

    Parameters
    ----------
    inp_path : str or Path
        Path to the growth ``.inp`` file.
    output_path : str or Path or None
        Where to save the HTML file. Defaults to ``<inp_stem>_growth_viz.html``
        next to the INP file.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from pathlib import Path as _Path
    from plotly.subplots import make_subplots
    from scipy.spatial import cKDTree
    from anato_odb import parse_inp_surface

    inp_path = _Path(inp_path)
    if output_path is None:
        output_path = inp_path.parent / f"{inp_path.stem}_growth_viz.html"
    output_path = _Path(output_path)

    node_coords, node_ids, elem_ids, elem_conn, elem_growth = \
        parse_inp_growth_rates(inp_path)

    # Node ID -> 0-based index
    id_to_idx = np.empty(node_ids.max() + 1, dtype=int)
    id_to_idx[:] = -1
    id_to_idx[node_ids] = np.arange(len(node_ids))

    # Element centroids
    corner_idx = id_to_idx[elem_conn]
    elem_coms = node_coords[corner_idx].mean(axis=1)

    # Extract outer surface and map growth rates to surface faces
    surface_verts, surface_tris, _, _, _ = parse_inp_surface(inp_path)
    surface_face_coms = surface_verts[surface_tris].mean(axis=1)
    tree = cKDTree(elem_coms)
    _, nearest_elem = tree.query(surface_face_coms)
    surface_growth = elem_growth[nearest_elem]

    # Shared color range (exclude zeros = non-growth elements)
    gr_nonzero = elem_growth[elem_growth != 0]
    if len(gr_nonzero) > 0:
        cmin, cmax = gr_nonzero.min(), gr_nonzero.max()
    else:
        cmin, cmax = elem_growth.min(), elem_growth.max()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Surface Growth Rate", "Solid Mesh Growth Rate"],
    )

    # Left: surface mesh colored by growth rate
    fig.add_trace(
        go.Mesh3d(
            x=surface_verts[:, 0], y=surface_verts[:, 1], z=surface_verts[:, 2],
            i=surface_tris[:, 0], j=surface_tris[:, 1], k=surface_tris[:, 2],
            intensity=surface_growth,
            intensitymode="cell",
            colorscale="Jet",
            cmin=cmin, cmax=cmax,
            colorbar=dict(title="Growth Rate", x=0.45, len=0.8),
            lighting=dict(ambient=0.6, diffuse=0.5),
        ),
        row=1, col=1,
    )

    # Right: solid mesh element COMs as scatter
    fig.add_trace(
        go.Scatter3d(
            x=elem_coms[:, 0], y=elem_coms[:, 1], z=elem_coms[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=elem_growth,
                colorscale="Jet",
                cmin=cmin, cmax=cmax,
                colorbar=dict(title="Growth Rate", x=1.0, len=0.8),
            ),
        ),
        row=1, col=2,
    )

    for scene in ["scene", "scene2"]:
        fig.update_layout(**{scene: dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )})

    fig.update_layout(
        title=inp_path.stem,
        width=1400, height=600,
        showlegend=False,
    )

    fig.write_html(str(output_path))
    print(f"Saved growth visualization to {output_path}")
    return fig
