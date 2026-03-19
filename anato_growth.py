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
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import time
import json
from anato_unravel import unravel_elems
from anato_patching import optimized_patching


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


def transfer_patches_nearest_neighbor(
    source_face_patch_id: np.ndarray,
    source_aligned_centers: np.ndarray,
    target_centers: np.ndarray,
    target_mesh: trimesh.Trimesh,
    min_faces: int = 6,
) -> np.ndarray:
    """
    Transfer patch assignments from source (registered) mesh to target mesh
    using nearest-neighbor face matching + connectivity cleanup.

    1. For each target face, find nearest source face (in aligned space)
       and assign its patch ID.
    2. For each patch on target mesh, keep only the largest connected component.
    3. Iteratively assign orphaned faces to neighboring patches.

    Args:
        source_face_patch_id: (N,) face-to-patch mapping for source mesh.
        source_aligned_centers: (N, 3) source face centroids in aligned/target space.
        target_centers: (M, 3) target mesh face centroids.
        target_mesh: Target trimesh for adjacency computation.
        min_faces: Minimum faces per patch.

    Returns:
        target_face_patch_id: (M,) face-to-patch mapping for target mesh.
    """
    # Step 1: Nearest-neighbor transfer
    tree = cKDTree(source_aligned_centers)
    _, nn_indices = tree.query(target_centers)
    target_pid = source_face_patch_id[nn_indices].copy()

    # Step 2: Connectivity cleanup - keep largest component per patch
    F = int(target_mesh.faces.shape[0])
    nbrs = [[] for _ in range(F)]
    fa = np.asarray(target_mesh.face_adjacency, dtype=int)
    if fa.size:
        for a, b in fa:
            a = int(a)
            b = int(b)
            nbrs[a].append(b)
            nbrs[b].append(a)

    unique_pids = np.unique(target_pid[target_pid >= 0])
    for p in unique_pids:
        p = int(p)
        fidx = np.where(target_pid == p)[0]
        if fidx.size < min_faces:
            target_pid[fidx] = -1
            continue

        # Find connected components within this patch
        face_set = set(int(f) for f in fidx)
        visited = set()
        components = []
        for start in fidx:
            s = int(start)
            if s in visited:
                continue
            stack = [s]
            visited.add(s)
            comp = []
            while stack:
                f = stack.pop()
                comp.append(f)
                for nb in nbrs[f]:
                    if nb in face_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            components.append(np.asarray(comp, dtype=np.int64))

        # Keep only the largest component, mark rest as unassigned
        if len(components) > 1:
            components.sort(key=lambda c: c.size, reverse=True)
            for comp in components[1:]:
                target_pid[comp] = -1

    # Step 3: Iteratively assign unassigned faces to best neighbor
    for _ in range(5):
        unassigned = np.where(target_pid == -1)[0]
        if unassigned.size == 0:
            break

        changed = 0
        for f in unassigned:
            f = int(f)
            neighbor_pids = [int(target_pid[int(nb)]) for nb in nbrs[f] if int(target_pid[int(nb)]) >= 0]
            if neighbor_pids:
                # Majority vote among neighbors
                counts = {}
                for np_id in neighbor_pids:
                    counts[np_id] = counts.get(np_id, 0) + 1
                best = max(counts.items(), key=lambda x: x[1])[0]
                target_pid[f] = best
                changed += 1

        if changed == 0:
            break

    # Final fallback: assign any remaining -1 to nearest assigned face
    remaining = np.where(target_pid == -1)[0]
    if remaining.size > 0:
        assigned_mask = target_pid >= 0
        if np.any(assigned_mask):
            assigned_centers = target_centers[assigned_mask]
            assigned_pids = target_pid[assigned_mask]
            tree2 = cKDTree(assigned_centers)
            _, nn_idx = tree2.query(target_centers[remaining])
            target_pid[remaining] = assigned_pids[nn_idx]

    return target_pid


def fit_corresponding_optimized_patches(
    initial_mesh: trimesh.Trimesh,
    aligned_source_centers: np.ndarray,
    final_mesh: trimesh.Trimesh,
    patching_params: dict,
) -> tuple:
    """
    Run optimized patching on initial mesh, then transfer to final mesh
    via nearest-neighbor face correspondence.

    Args:
        initial_mesh: Initial time point mesh.
        aligned_source_centers: (N, 3) initial mesh face centroids after registration
                                (in final mesh coordinate space).
        final_mesh: Final time point mesh.
        patching_params: Parameters for optimized_patching().

    Returns:
        (initial_face_patch_id, final_face_patch_id, patch_stats, n_patches)
    """
    # Run optimized patching on initial mesh
    result = optimized_patching(initial_mesh, **patching_params)
    initial_pid = result["face_patch_id"]
    patch_stats = result["patch_stats"]
    n_patches = result["n_patches"]

    # Transfer to final mesh via nearest-neighbor correspondence
    final_pid = transfer_patches_nearest_neighbor(
        source_face_patch_id=initial_pid,
        source_aligned_centers=aligned_source_centers,
        target_centers=final_mesh.triangles_center,
        target_mesh=final_mesh,
        min_faces=patching_params.get("min_faces", 6),
    )

    return initial_pid, final_pid, patch_stats, n_patches


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

    # 4. Compute patch boundary edges
    #    An edge is a boundary if its two adjacent faces have different patch labels.
    edge_to_faces = {}
    for f_idx, (v0, v1, v2) in enumerate(faces):
        for edge in [(min(v0, v1), max(v0, v1)),
                     (min(v1, v2), max(v1, v2)),
                     (min(v0, v2), max(v0, v2))]:
            edge_to_faces.setdefault(edge, []).append(f_idx)

    bx, by, bz = [], [], []
    for (va, vb), adj_faces in edge_to_faces.items():
        if len(adj_faces) == 2:
            if patch_labels[adj_faces[0]] != patch_labels[adj_faces[1]]:
                bx.extend([verts[va, 0], verts[vb, 0], None])
                by.extend([verts[va, 1], verts[vb, 1], None])
                bz.extend([verts[va, 2], verts[vb, 2], None])

    boundary_trace = go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="skip",
        showlegend=False,
    )

    # 5. Finalize Layout
    fig = go.Figure(data=[mesh3d, boundary_trace])
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
    patching_method: str = "kmeans",
    patching_params: Optional[dict] = None,
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
        initial_labels = None
        final_labels = None

        # ─────────────────────────────────────────────────────────────────────
        # Optimized patching mode: geometry-aware patches via quadratic fits.
        # Runs once per mesh_m (independent of curv_m).
        # ─────────────────────────────────────────────────────────────────────
        if patching_method == "optimized":
            assert patching_params is not None, "patching_params required for optimized mode"
            target_patches = patching_params.get("target_patches", 1000)
            ncluster_label = f"optimized_{target_patches}"
            print(f"Calculating mapping with optimized patching (target={target_patches})")

            initial_labels, final_labels, patch_stats, n_patches = (
                fit_corresponding_optimized_patches(
                    initial_mesh=initial_mesh,
                    aligned_source_centers=aligned_source_triangles_center,
                    final_mesh=final_mesh,
                    patching_params=patching_params,
                )
            )

            current_n_clusters = n_patches

            # Compute area sums per patch
            initial_As = np.bincount(
                initial_labels, weights=initial_mesh.area_faces, minlength=n_patches
            )
            final_As = np.bincount(
                final_labels, weights=final_mesh.area_faces, minlength=n_patches
            )

            if dir_path:
                print(f"Saving optimized patch info for mesh_m={mesh_m} to {dir_path}")
                np.savez(
                    os.path.join(
                        dir_path,
                        f"optimized_patches_{mesh_m}_{target_patches}.npz",
                    ),
                    initial_labels=initial_labels,
                    final_labels=final_labels,
                )

        # ─────────────────────────────────────────────────────────────────────
        # Static mode: K-means with random initialization, run once per mesh_m.
        # Clusters are shared across all curv_m values since n_clusters is fixed.
        # ─────────────────────────────────────────────────────────────────────
        elif ncluster_input:
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
            if patching_method == "optimized":
                # Optimized patching already computed above; reuse labels
                pass
            elif ncluster_input is None:
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

            # Resolve labels to generic arrays for curvature computation
            if patching_method == "optimized":
                _init_labels = initial_labels
                _final_labels = final_labels
            elif initial_kms is not None:
                _init_labels = initial_kms.labels_
                _final_labels = final_kms.labels_
            else:
                raise RuntimeError("No labels available for curvature computation")

            # Compute curvature sums per cluster
            initial_Ks = np.bincount(
                _init_labels,
                weights=initial_mesh.intgaussian_faces,
                minlength=current_n_clusters,
            )
            final_Ks = np.bincount(
                _final_labels,
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
                method_label = "Optimized" if patching_method == "optimized" else "KMeans"
                plot_mesh_patch_values(
                    mesh=initial_mesh,
                    patch_labels=_init_labels,
                    title=f"Initial {method_label} Clusters [M={mesh_m}, C={curv_m}]",
                    save_path=os.path.join(fig_dir, "initial_clusters.html"),
                    discrete=True,
                    colormap=colormap,
                )
                plot_mesh_patch_values(
                    mesh=final_mesh,
                    patch_labels=_final_labels,
                    title=f"Final {method_label} Clusters [M={mesh_m}, C={curv_m}]",
                    save_path=os.path.join(fig_dir, "final_clusters.html"),
                    discrete=True,
                    colormap=colormap,
                )

                # Compute changes per cluster (direct proportional area change)
                area_changes = (final_As - initial_As) / initial_As
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
                        patch_labels=_init_labels,
                        patch_values=growth_initial,
                        title=f"Initial {value_key} (Growth Patches)",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_initial_growth_patches.html"
                        ),
                    )
                    plot_mesh_patch_values(
                        mesh=final_mesh,
                        patch_labels=_final_labels,
                        patch_values=growth_final,
                        title=f"Final {value_key} (Growth Patches)",
                        save_path=os.path.join(
                            fig_dir, f"{short_name}_final_growth_patches.html"
                        ),
                    )
                    plot_mesh_patch_values(
                        mesh=final_mesh,
                        patch_labels=_final_labels,
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
    """Visualize solid growth rates from a finalized growth INP file.

    Produces an interactive HTML figure showing solid mesh element centroids
    colored by growth rate.

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

    # Color range (exclude zeros = non-growth elements)
    gr_nonzero = elem_growth[elem_growth != 0]
    if len(gr_nonzero) > 0:
        cmin, cmax = gr_nonzero.min(), gr_nonzero.max()
    else:
        cmin, cmax = elem_growth.min(), elem_growth.max()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=elem_coms[:, 0], y=elem_coms[:, 1], z=elem_coms[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=elem_growth,
                colorscale="Jet",
                cmin=cmin, cmax=cmax,
                colorbar=dict(title="Growth Rate"),
            ),
        ),
    )

    fig.update_layout(
        title=inp_path.stem,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        width=900, height=700,
        showlegend=False,
    )

    fig.write_html(str(output_path))
    print(f"Saved growth visualization to {output_path}")
    return fig


def parse_abaqus_inp(inp_path: str) -> tuple:
    """
    Parse an Abaqus INP file to extract nodes, elements, and raw lines.

    Supports linear tetrahedral (C3D4) and quadratic tetrahedral (C3D10)
    element types.

    Args:
        inp_path: Path to the .inp file.

    Returns:
        (nodes, elements, lines) where:
            nodes: (N, 3) array of node coordinates.
            elements: (M, K) array of element connectivity (0-indexed node indices).
            lines: List of raw lines from the INP file.
    """
    with open(inp_path, "r") as f:
        lines = f.readlines()

    lines = [line.rstrip("\n") for line in lines]

    # Locate the *Node and *Element data sections.
    # Match "*node" exactly or "*node," (with options) but not keywords like
    # "*node output", "*node print", "*node file", "*nset", etc.
    node_start = None
    elem_start = None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if node_start is None and (low == "*node" or low.startswith("*node,")):
            node_start = i + 1
        elif elem_start is None and low.startswith("*element,"):
            elem_start = i + 1

    if node_start is None:
        raise ValueError("Could not find *Node section in INP file")
    if elem_start is None:
        raise ValueError("Could not find *Element section in INP file")

    # Read node data rows (id, x, y, z) until the next keyword
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

    # Read element connectivity rows (id, n1, n2, ...) until the next keyword
    elem_rows = []
    for i in range(elem_start, len(lines)):
        if lines[i].strip().startswith("*"):
            break
        parts = lines[i].split(",")
        vals = [int(p.strip()) for p in parts if p.strip()]
        elem_rows.append(vals)
    elem_arr = np.array(elem_rows)
    elem_connectivity = elem_arr[:, 1:]  # drop element ID column

    # Remap original node IDs to 0-based indices.
    # Fast path for contiguous 1-based IDs (the common case).
    if np.array_equal(node_ids, np.arange(1, len(node_ids) + 1)):
        elements = elem_connectivity - 1
    else:
        remap = np.empty(node_ids.max() + 1, dtype=int)
        remap[node_ids] = np.arange(len(node_ids))
        elements = remap[elem_connectivity]

    return node_coords, elements, lines


def calc_element_coms(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Calculate center of mass of each element.

    Args:
        nodes: (N, 3) array of node coordinates.
        elements: (M, K) array of element connectivity (0-indexed node indices).

    Returns:
        (M, 3) array of element centers of mass.
    """
    return nodes[elements].mean(axis=1)


def _round_to_sig_figs(arr: np.ndarray, n: int) -> np.ndarray:
    """Round array values to *n* significant figures. Zeros stay zero."""
    result = np.zeros_like(arr)
    nonzero = arr != 0
    if np.any(nonzero):
        x = arr[nonzero]
        magnitude = np.floor(np.log10(np.abs(x)))
        scale = 10.0 ** (n - 1 - magnitude)
        result[nonzero] = np.round(x * scale) / scale
    return result


def map_surface_growth_to_solid(
    surface_face_centroids: np.ndarray,
    surface_face_growth_rates: np.ndarray,
    element_centroids: np.ndarray,
    smoothing_radius: float = 50.0,
) -> np.ndarray:
    """
    Map surface face growth rates to solid mesh elements.

    Assumes a thin-walled geometry so that each solid element can be
    assigned the growth rate of its nearest surface face.

    Steps:
        1. For each solid element, find nearest surface face by centroid
           distance.
        2. Smooth growth rates on the surface via KNN averaging, excluding
           surface faces that were not matched to any solid element.
        3. Round smoothed rates to 3 significant figures to reduce the
           number of unique material definitions in the output INP.
        4. Map the smoothed surface rates back to solid elements.

    Args:
        surface_face_centroids: (F, 3) surface face centroids.
        surface_face_growth_rates: (F,) growth rate per surface face.
        element_centroids: (E, 3) solid element centroids.
        smoothing_radius: Controls KNN smoothing.  n_neighbors =
            round(n_faces / smoothing_radius), so a *higher* value gives
            fewer neighbors and *less* smoothing.

    Returns:
        (E,) array of growth rates for each solid element.
    """
    n_faces = len(surface_face_centroids)

    # Find closest surface face for each solid element
    surface_tree = cKDTree(surface_face_centroids)
    _, closest_faces = surface_tree.query(element_centroids)

    # Mark which surface faces were actually matched to at least one element
    is_mapped = np.zeros(n_faces, dtype=bool)
    is_mapped[np.unique(closest_faces)] = True

    # KNN smoothing on the surface mesh
    n_neighbors = max(1, round(n_faces / smoothing_radius))
    _, all_neighbors = surface_tree.query(surface_face_centroids, k=n_neighbors)
    # all_neighbors shape: (F, n_neighbors)

    # Build masked rates: zero out contributions from unmapped neighbors
    neighbor_is_mapped = is_mapped[all_neighbors]          # (F, K) bool
    neighbor_rates = surface_face_growth_rates[all_neighbors]  # (F, K)
    masked_rates = np.where(neighbor_is_mapped, neighbor_rates, 0.0)
    valid_counts = neighbor_is_mapped.sum(axis=1)            # (F,)

    # Faces with zero mapped neighbors fall back to their nearest neighbor
    no_valid = valid_counts == 0
    valid_counts[no_valid] = 1
    masked_rates[no_valid, 0] = surface_face_growth_rates[
        all_neighbors[no_valid, 0]
    ]

    raw_means = masked_rates.sum(axis=1) / valid_counts

    # Round to 3 significant figures to limit the number of unique materials
    growth_rates_smoothed = _round_to_sig_figs(raw_means, 3)

    # Map smoothed surface rates to solid elements
    return growth_rates_smoothed[closest_faces]


def _write_element_set(f, set_index: int, element_indices: np.ndarray):
    """Write an Abaqus *Elset definition. Element IDs are written 1-based."""
    f.write(f"*Elset, elset=Set-{set_index}\n")
    for c, el in enumerate(element_indices, start=1):
        if c % 16 == 0:
            f.write(f"      {el + 1}\n")
        else:
            f.write(f"      {el + 1},")
    f.write("\n")


def _write_section(f, set_index: int):
    """Write an Abaqus solid section assignment."""
    f.write(f"** Section: Section-{set_index}\n")
    f.write(f"*Solid Section, elset=Set-{set_index}, material=Material-{set_index}\n,\n")


def _write_nh_growth_material(
    f, name: int, density: float, c10: float, growth_rate: float
):
    """
    Write a neo-Hookean growth UMAT material definition.

    The material uses 2 state-dependent variables and 9 user material
    constants:
        c10, D, 0, 0.039, 0, 0, growth_rate, 0.001, 0.001
    where D = 1 / (20 * c10) enforces near-incompressibility.
    """
    D = 1.0 / (20.0 * c10)
    f.write(f"*Material, name=Material-{name}\n")
    f.write("*Density\n")
    f.write(f" {density},\n")
    f.write("*Depvar\n")
    f.write("      2,\n")
    f.write("*User Material, constants=9\n")
    f.write(f" {c10},   {D},    0., 0.039,    0.,    0.,  {growth_rate},    0.001\n")
    f.write("0.001,\n")


def write_growth_inp(
    inp_path: str,
    output_path: str,
    surface_face_centroids: np.ndarray,
    surface_face_growth_rates: np.ndarray,
    density: float,
    c10: float,
    smoothing_radius: float = 50.0,
):
    """
    Generate an Abaqus INP file with spatially-varying growth materials.

    Takes growth rates defined on a surface triangular mesh and produces a
    new solid-mesh INP where every unique growth rate gets its own element
    set, solid section, and neo-Hookean growth UMAT material definition.

    Pipeline:
        1. Parse the template INP (solid tetrahedral mesh).
        2. Map surface growth rates to solid elements via nearest-centroid
           matching (thin-wall assumption).
        3. Smooth rates on the surface with KNN averaging and round to 3
           significant figures.
        4. Clamp negative rates to 0 (only positive expansion).
        5. Insert element sets and sections before *End Part, and material
           definitions after *End Assembly.

    Args:
        inp_path: Path to the template Abaqus .inp file.  Must contain a
            solid tetrahedral mesh with *Part / *End Part and
            *Assembly / *End Assembly blocks.
        output_path: Path for the generated .inp file.
        surface_face_centroids: (F, 3) centroids of the surface triangular
            mesh (e.g., trimesh.Trimesh.triangles_center).
        surface_face_growth_rates: (F,) growth rate per surface face.
            Negative values are clamped to 0.
        density: Material density (e.g., 0.00112 for mm-MPa-ms units).
        c10: Neo-Hookean c10 parameter (e.g., 0.05).
        smoothing_radius: KNN smoothing control.  n_neighbors =
            round(n_faces / smoothing_radius), so a *higher* value gives
            fewer neighbors and *less* smoothing.  Default 50.

    Returns:
        element_growth_rates: (E,) per-element growth rates written to the
            output INP.
    """
    # 1. Parse the template INP
    nodes, elements, lines = parse_abaqus_inp(inp_path)
    element_centroids = calc_element_coms(nodes, elements)

    print(f"Parsed INP: {len(nodes)} nodes, {len(elements)} elements")
    print(f"Surface mesh: {len(surface_face_centroids)} faces")

    # 2. Map surface rates to solid elements, then smooth and round
    clamped_rates = np.maximum(surface_face_growth_rates, 0.0)

    element_growth_rates = map_surface_growth_to_solid(
        surface_face_centroids, clamped_rates, element_centroids, smoothing_radius
    )

    print(
        f"Growth rates: min={element_growth_rates.min():.4f}, "
        f"max={element_growth_rates.max():.4f}, "
        f"mean={element_growth_rates.mean():.4f}"
    )

    # 3. Identify unique growth rates; each gets its own material
    unique_rates, inverse_indices = np.unique(
        element_growth_rates, return_inverse=True
    )
    print(f"Unique growth rates: {len(unique_rates)}")

    # 4. Locate insertion points in the original INP
    end_part_line = None
    end_assembly_line = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if end_part_line is None and stripped == "*end part":
            end_part_line = i
        if end_assembly_line is None and stripped == "*end assembly":
            end_assembly_line = i

    if end_part_line is None:
        raise ValueError("Could not find '*End Part' in INP file")
    if end_assembly_line is None:
        raise ValueError("Could not find '*End Assembly' in INP file")

    # Find the first ** comment line after *End Assembly (materials go here)
    star_comment_after_assembly = None
    for i in range(end_assembly_line + 1, len(lines)):
        if lines[i].strip().startswith("**"):
            star_comment_after_assembly = i
            break

    # 5. Write the new INP
    with open(output_path, "w") as f:
        # Original content up to (not including) *End Part
        for i in range(end_part_line):
            f.write(lines[i] + "\n")

        # Element sets — one per unique growth rate
        for gi in range(len(unique_rates)):
            elem_indices = np.where(inverse_indices == gi)[0]
            _write_element_set(f, gi + 1, elem_indices)

        # Solid section assignments — one per unique growth rate
        for gi in range(len(unique_rates)):
            _write_section(f, gi + 1)

        # Original content from *End Part through the ** after *End Assembly
        end_block = (
            star_comment_after_assembly
            if star_comment_after_assembly is not None
            else end_assembly_line + 1
        )
        for i in range(end_part_line, end_block + 1):
            f.write(lines[i] + "\n")

        # Material definitions
        f.write("** MATERIALS\n**\n")
        for gi in range(len(unique_rates)):
            _write_nh_growth_material(
                f, gi + 1, density, c10, unique_rates[gi]
            )
        f.write("**\n")

        # Remainder of the original file (steps, outputs, etc.)
        for i in range(end_block + 1, len(lines)):
            f.write(lines[i] + "\n")

    print(f"Wrote growth INP: {output_path}")
    return element_growth_rates
