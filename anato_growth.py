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
    patch_values: np.ndarray,
    patch_labels: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Plot a mesh colored by per-patch scalar values
    Args:
        mesh: trimesh.Trimesh
        patch_values: numpy array of length n_patches
        patch_labels: patch_labels maps each face -> patch index (len = n_faces).
        title: plot title (if patch_labels was passed positionally as string, that will be used as title)
        save_path: optional path to save HTML
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    n_faces = faces.shape[0]

    pv = np.asarray(patch_values).ravel()

    patch_labels = np.asarray(patch_labels).ravel()
    if patch_labels.size != n_faces:
        raise ValueError(
            f"patch_labels length ({patch_labels.size}) must equal number of faces ({n_faces})."
        )
    # pv is per-patch: map to faces
    face_values = pv[patch_labels]

    # Build per-vertex values by averaging adjacent face values
    n_verts = verts.shape[0]
    vertex_accum = np.zeros(n_verts, dtype=np.float64)
    vertex_counts = np.zeros(n_verts, dtype=np.int32)

    for f_idx, (i, j, k) in enumerate(faces):
        v = float(face_values[f_idx])
        vertex_accum[i] += v
        vertex_accum[j] += v
        vertex_accum[k] += v
        vertex_counts[i] += 1
        vertex_counts[j] += 1
        vertex_counts[k] += 1

    # avoid division by zero
    vertex_counts = np.where(vertex_counts == 0, 1, vertex_counts)
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
    )

    fig = go.Figure(data=[mesh3d])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
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

        print(f"Performing deformable registration of {mdiv} segments")
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

    initial_mesh.intgaussian_faces = reassign_per_face_intgaussian(
        initial_mesh, initial_manifold_data
    )
    final_mesh.intgaussian_faces = reassign_per_face_intgaussian(
        final_mesh, final_manifold_data
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

    # Ensure curvatures sum correctly
    assert np.isclose(
        initial_Ks.sum(),
        initial_manifold_data["patch_data"]["IntGaussian"].sum(),
    ), "Initial integrated Gaussian curvature mismatch"
    assert np.isclose(
        final_Ks.sum(),
        final_manifold_data["patch_data"]["IntGaussian"].sum(),
    ), "Final integrated Gaussian curvature mismatch"

    if plot_figures:
        # Plot initial/final integrated Gaussian per cluster
        # using (curvature calculation vs. growth mapping patches)
        os.makedirs(os.path.join(dir_path, "heatmap_figs"), exist_ok=True)

        # Compute area changes per cluster
        area_changes = (np.sqrt(final_As) - np.sqrt(initial_As)) / np.sqrt(initial_As)
        intgaussian_changes = final_Ks - initial_Ks

        growth_values = {}
        growth_values["initial"] = {"Patch_Area": initial_As, "IntGaussian": initial_Ks}
        growth_values["final"] = {"Patch_Area": final_As, "IntGaussian": final_Ks}
        growth_changes = {
            "Patch_Area": area_changes,
            "IntGaussian": intgaussian_changes,
        }
        for value in ("Patch_Area", "IntGaussian"):
            plot_mesh_patch_values(
                initial_mesh,
                initial_manifold_data["patch_data"][value].values,
                initial_manifold_data["patch_labels"],
                f"Initial mesh — {value}",
                save_path=os.path.join(
                    dir_path,
                    "heatmap_figs",
                    f"initial_{value.lower()}_curv_patches.html",
                ),
            )

            plot_mesh_patch_values(
                final_mesh,
                final_manifold_data["patch_data"][value].values,
                final_manifold_data["patch_labels"],
                f"Final mesh — {value}",
                save_path=os.path.join(
                    dir_path, "heatmap_figs", f"final_{value.lower()}_curv_patches.html"
                ),
            )

            plot_mesh_patch_values(
                initial_mesh,
                growth_values["initial"][value],
                initial_kms.labels_,
                f"Initial mesh — Integrated {value}",
                save_path=os.path.join(
                    dir_path,
                    "heatmap_figs",
                    f"initial_{value.lower()}_growth_patches.html",
                ),
            )

            plot_mesh_patch_values(
                final_mesh,
                growth_values["final"][value],
                final_kms.labels_,
                f"Final mesh — Integrated {value}",
                save_path=os.path.join(
                    dir_path,
                    "heatmap_figs",
                    f"final_{value.lower()}_growth_patches.html",
                ),
            )

            plot_mesh_patch_values(
                final_mesh,
                growth_changes[value],
                final_kms.labels_,
                f"Change in Integrated {value}",
                save_path=os.path.join(
                    dir_path,
                    "heatmap_figs",
                    f"change_in_{value.lower()}_growth_patches.html",
                ),
            )

    if dir_path:
        # Save results to CSV
        results_df = pd.DataFrame(
            {
                "Cluster": np.arange(ncluster),
                "InitialIntGaussian": initial_Ks,
                "FinalIntGaussian": final_Ks,
                "InitialArea": initial_As,
                "FinalArea": final_As,
            }
        )
        results_csv_path = os.path.join(
            dir_path, f"growth_mapping_{ncluster}_results.csv"
        )
        results_df.to_csv(results_csv_path, index=False)

        # Save the k-means cluster info
        np.savez(
            os.path.join(dir_path, f"kmeans_clusters_{ncluster}.npz"),
            initial_centers=initial_kms.cluster_centers_,
            final_centers=final_kms.cluster_centers_,
            initial_labels=initial_kms.labels_,
            final_labels=final_kms.labels_,
        )

    return area_changes, intgaussian_changes, initial_kms, final_kms
