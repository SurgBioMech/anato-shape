import colorsys
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


def _safe_normalize(v, axis=-1, eps=1e-12):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, np.maximum(norm, eps))


def _rotation_from_a_to_b(a, b):
    """
    Rodrigues' rotation formula: compute rotation matrix that maps unit vector a to unit vector b.
    Returns 3x3 matrix. If vectors are (nearly) parallel, returns identity or 180-degree rotation.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-12:
        # a and b are parallel or anti-parallel
        if c > 0.0:
            return np.eye(3)
        else:
            # 180-degree rotation: choose any perpendicular axis
            # find axis orthogonal to a
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - np.dot(axis, a) * a
            axis = axis / np.linalg.norm(axis)
            # Rodrigues for 180 deg: R = I + 2 * (K^2) where K is skew of axis
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + (K @ K) * ((1 - c) / (s**2))
    return R


def straighten_vertices(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    cline: np.ndarray,
    rotation_matrices: np.ndarray,
    straightened_line: np.ndarray,
):
    """
    Straighten the mesh vertices according to the centerline and its first derivative.
    The straightening aligns the centerline along the z-axis and rotates the mesh accordingly.

    Inputs:
    - vertices: (n,3) array of vertex coordinates
    - vertex_normals: (n,3) array of normals per vertex
    - cline: (k,3) centerline points
    - rotation_matrices: (k,3,3) rotation matrices along the centerline
    - straightened_line: (k,3) straightened centerline points
    Outputs:
        - straightened_vertices: (n,3) vertices after straightening
    """
    n = vertices.shape[0]
    k = cline.shape[0]

    # Straighten each vertex by rotating local tangent to [0,0,1] then translating to straightened_line[idx]
    straightened_vertices = np.full((n, 3), np.nan, dtype=float)
    closest_cline_indices = np.empty(n, dtype=int)
    unit_vertex_normals = _safe_normalize(vertex_normals)

    # For each vertex, find closest cline point based on angle with normal and distance
    for i in range(n):
        u = unit_vertex_normals[i]
        # compute angle between u and vector from vertex to each cline point
        vecs = cline - vertices[i]  # shape (k,3)
        dists = np.linalg.norm(vecs, axis=1)
        # avoid divide by zero

        # Smart search for closest point (using normals)
        with np.errstate(invalid="ignore"):
            uv = vecs / np.maximum(dists[:, None], 1e-12)
            cos_theta = np.clip((uv @ u), -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))

        idc1 = int(np.argmin(dists))
        search_region = max(1, int(round(k / 12)))
        start_pt = max(0, idc1 - search_region)
        end_pt = min(k, idc1 + search_region + 1)
        centerline_look = np.arange(start_pt, end_pt)

        if centerline_look.size == 0:
            centerline_look = np.arange(k)

        narrowed_angles = theta_deg[centerline_look]
        max_angle = np.max(narrowed_angles)
        mask = narrowed_angles > 0.85 * max_angle
        candidate_indices = centerline_look[mask]

        if len(candidate_indices) == 0:
            # Fallback if filtering removes all points
            chosen_local = idc1
        else:
            cand_dists = np.linalg.norm(vertices[i] - cline[candidate_indices], axis=1)
            chosen_local = candidate_indices[int(np.argmin(cand_dists))]

        closest_cline_indices[i] = chosen_local

        R = rotation_matrices[chosen_local]

        vec = vertices[i] - cline[chosen_local]
        rotated_vec = R.dot(vec)
        straightened_vertices[i] = straightened_line[chosen_local] + rotated_vec
        # ------------------------------
    return straightened_vertices


def _compute_transport_rotations(tangents, initial_rotation=None):
    """
    Computes rotation matrices along the centerline using Parallel Transport.

    Args:
        tangents: (k, 3) tangent vectors
        initial_rotation: (3, 3) optional starting rotation matrix.
                          If None, defaults to simple Z-alignment.
    """
    k = len(tangents)
    rotations = np.zeros((k, 3, 3))

    # 1. Initialize first frame
    t0 = tangents[0]
    t0 = t0 / np.linalg.norm(t0)

    if initial_rotation is not None:
        rotations[0] = initial_rotation
    else:
        z_axis = np.array([0.0, 0.0, 1.0])
        rotations[0] = _rotation_from_a_to_b(t0, z_axis)

    # 2. Propagate frames minimizing twist (Parallel Transport)
    for i in range(1, k):
        t_prev = tangents[i - 1] / np.linalg.norm(tangents[i - 1])
        t_curr = tangents[i] / np.linalg.norm(tangents[i])

        # Calculate relative rotation from prev tangent to curr tangent
        R_rel = _rotation_from_a_to_b(t_prev, t_curr)

        # Apply the rotation update
        # R_curr = R_prev @ R_rel.T
        rotations[i] = rotations[i - 1] @ R_rel.T

    return rotations


def compute_initial_rotation(vertices, start_tangent):
    """
    Computes R0 aligning local X to the first principal component.
    """
    pca = PCA(n_components=1)
    # Fit on XY projection as requested
    pca.fit(vertices[:, :2])
    pc1_2d = pca.components_[0]
    # Reconstruct 3D vector (flat in Z)
    pc1 = np.array([pc1_2d[0], pc1_2d[1], 0.0])

    # Check orientation: Point PC1 towards centroid
    to_centroid = np.mean(vertices, axis=0) - vertices[0]
    if np.dot(pc1, to_centroid) < 0:
        pc1 = -pc1

    # Build Basis
    z_axis = start_tangent / np.linalg.norm(start_tangent)

    # Y is perp to Z and PC1
    y_axis = np.cross(z_axis, pc1)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.cross(z_axis, np.array([1, 0, 0]))
    y_axis = y_axis / np.linalg.norm(y_axis)

    # X is perp to Y and Z.
    # Since PC1 pointed 'in', and Y is perp to PC1, X will point 'in' (Lesser Curvature)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    return np.stack([x_axis, y_axis, z_axis])


def unravel(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    cline: np.ndarray,
    cline_deriv1: np.ndarray,
):
    # Validate shapes
    n = vertices.shape[0]
    k = cline.shape[0]
    assert vertices.shape[1] == 3

    # Build straightened centerline
    straightened_line = np.zeros((k, 3), dtype=float)
    min_z_idx = int(np.argmin(cline[:, 2]))
    flipz = min_z_idx == k - 1

    for i in range(1, k):
        dist = np.linalg.norm(cline[i] - cline[i - 1])
        straightened_line[i] = straightened_line[i - 1] + np.array([0.0, 0.0, dist])

    # Compute rotations
    R0 = compute_initial_rotation(vertices, cline_deriv1[0])
    rotation_matrices = _compute_transport_rotations(cline_deriv1, R0)

    # Straighten vertices
    straightened_vertices = straighten_vertices(
        vertices, vertex_normals, cline, rotation_matrices, straightened_line
    )

    # Prepare for Unraveling
    twod_vertices = np.full((n, 2), np.nan, dtype=float)
    translated = straightened_vertices.copy()

    zmin = np.nanmin(translated[:, 2]) - 1e-6
    zmax = np.nanmax(translated[:, 2]) + 1e-6
    zs = np.linspace(zmin, zmax, 100)

    for zi in range(len(zs) - 1):
        mask_slice = (translated[:, 2] > zs[zi]) & (translated[:, 2] < zs[zi + 1])
        slice_idx = np.nonzero(mask_slice)[0]
        if slice_idx.size == 0:
            continue

        vs = translated[slice_idx][:, [0, 1, 2]]

        if vs.shape[0] == 1:
            x_sign = -1 if vs[0, 0] < 0 else 1
            twod_vertices[slice_idx[0]] = np.array([0.0 * x_sign, vs[0, 2]])
            continue

        # --- Graph Construction (Standard) ---
        XY = vs[:, :2]
        pairwise = squareform(pdist(XY))
        ascend_idx = np.argsort(pairwise, axis=1)
        last_connect = min(5, pairwise.shape[0])
        d2 = np.zeros_like(pairwise)

        for j in range(pairwise.shape[0]):
            neighbors = ascend_idx[j, 1 : last_connect + 1]
            if neighbors.size > 0:
                d2[j, neighbors] = pairwise[j, neighbors]
                d2[neighbors, j] = pairwise[neighbors, j]

        try:
            hull = ConvexHull(XY)
            hull_indices = hull.vertices
            for m in range(len(hull_indices)):
                a_idx = hull_indices[m]
                b_idx = hull_indices[(m + 1) % len(hull_indices)]
                d2[a_idx, b_idx] = pairwise[a_idx, b_idx]
                d2[b_idx, a_idx] = pairwise[b_idx, a_idx]
        except Exception:
            for m in range(pairwise.shape[0] - 1):
                d2[m, m + 1] = pairwise[m, m + 1]
                d2[m + 1, m] = pairwise[m + 1, m]

        G = nx.Graph()
        for a in range(pairwise.shape[0]):
            for b in range(a + 1, pairwise.shape[0]):
                if d2[a, b] > 0:
                    G.add_edge(a, b, weight=float(d2[a, b]))

        if G.number_of_nodes() == 0:
            continue
        for node in range(pairwise.shape[0]):
            if node not in G:
                G.add_node(node)

        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            comps_sorted = sorted(comps, key=len, reverse=True)
            main_nodes = set(comps_sorted[0])
            for comp in comps_sorted[1:]:
                for lone in comp:
                    found = None
                    for candidate in ascend_idx[lone]:
                        if candidate in main_nodes:
                            found = candidate
                            break
                    if found is None:
                        found = int(np.argmin(pairwise[lone]))
                    G.add_edge(lone, found, weight=float(pairwise[lone, found]))

        sourcenode = int(np.argmin(vs[:, 0]))

        # Calculate geodesic distances
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, sourcenode, weight="weight"
            )
        except nx.NetworkXNoPath:
            # Fallback if graph is somehow broken, though component fix above handles most
            lengths = {node: 0.0 for node in G.nodes()}

        for local_j, global_idx in enumerate(slice_idx):
            z_val = translated[global_idx, 2]
            length = lengths.get(local_j, 0.0)

            y_val = straightened_vertices[global_idx, 1]

            if y_val < 0:
                twod_vertices[global_idx, :] = np.array([-length, z_val])
            else:
                twod_vertices[global_idx, :] = np.array([length, z_val])

    # Post-processing flip
    if flipz:
        mz = np.nanmean(twod_vertices[:, 1])
        twod_vertices[:, 1] = (twod_vertices[:, 1] - mz) * -1.0 + mz
        mvalue = np.nanmin(twod_vertices[:, 1])
        twod_vertices[:, 1] = twod_vertices[:, 1] - mvalue + 1.0

        mz_line = np.nanmean(straightened_line[:, 2])
        straightened_line[:, 2] = (straightened_line[:, 2] - mz_line) * -1.0 + mz_line
        straightened_line[:, 2] = straightened_line[:, 2] - mvalue + 1.0

        mz_vs = np.nanmean(straightened_vertices[:, 2])
        straightened_vertices[:, 2] = (
            straightened_vertices[:, 2] - mz_vs
        ) * -1.0 + mz_vs
        straightened_vertices[:, 2] = straightened_vertices[:, 2] - mvalue + 1.0

    return twod_vertices, straightened_line, straightened_vertices


def plot_unravel_groups(twodvertices, points, cline, grps, mdiv, ndiv, marker_size=3):
    """
    Combined interactive 2D + 3D Plotly visualization of unravel groups.
    - twodvertices : (N,2) array-like
    - points       : (N,3) array-like
    - cline        : (K,3) array-like
    - grps         : list-of-lists of index arrays [mdiv][ndiv]
    - mdiv, ndiv   : Integers specifying the grid dimensions of the groups
    - marker_size  : Size of the scatter points
    """

    pts2d = np.asarray(twodvertices)
    pts3d = np.asarray(points)

    # Flatten all groups while keeping order
    groups = []
    for i in range(mdiv):
        for j in range(ndiv):
            idx = grps[i][j]
            if idx is None:
                continue
            idx = np.asarray(idx)
            if idx.size == 0:
                continue
            groups.append((i, j, idx))

    n_groups = max(len(groups), 1)

    fig = go.Figure()

    # Golden angle approximation (approx 137.5 degrees in radians or 0.3819 fraction)
    # Using this ensures that consecutive groups have widely different hues.
    golden_ratio_conjugate = 0.618033988749895

    # --- 2D traces (scene = scene2 / xy plane) ---
    for k, (i, j, idx) in enumerate(groups):
        # Calculate hue using golden angle to maximize distinctness
        hue = (k * golden_ratio_conjugate) % 1.0

        # High saturation and value for visibility
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color = "#%02x%02x%02x" % (int(255 * r), int(255 * g), int(255 * b))

        fig.add_trace(
            go.Scatter(
                x=pts2d[idx, 0],
                y=pts2d[idx, 1],
                mode="markers",
                marker=dict(
                    size=marker_size, color=color, opacity=1.0
                ),  # Explicit opacity
                name=f"2D div{i+1}_grp{j+1}",
                legendgroup=f"group_{k}",
                showlegend=False,  # avoid double legends
            )
        )

    # --- 3D traces (scene = scene) ---
    for k, (i, j, idx) in enumerate(groups):
        # Same color calculation for matching groups
        hue = (k * golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color = "#%02x%02x%02x" % (int(255 * r), int(255 * g), int(255 * b))

        fig.add_trace(
            go.Scatter3d(
                x=pts3d[idx, 0],
                y=pts3d[idx, 1],
                z=pts3d[idx, 2],
                mode="markers",
                # Opacity set to 1.0 for opaque points
                marker=dict(size=marker_size, color=color, opacity=1.0),
                name=f"div{i+1}_grp{j+1}",
                legendgroup=f"group_{k}",
            )
        )

    # --- centerline for 3D ---
    if cline is not None and len(cline) > 0:
        cline = np.asarray(cline)
        fig.add_trace(
            go.Scatter3d(
                x=cline[:, 0],
                y=cline[:, 1],
                z=cline[:, 2],
                mode="lines",
                line=dict(color="black", width=4),
                name="Centerline",
                legendgroup="centerline",
            )
        )

    fig.update_layout(
        width=1100,
        height=600,
        title="2D and 3D Unraveled Groups (Interactive)",
        scene=dict(  # 3D scene
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            domain=dict(x=[0.55, 1.0], y=[0, 1]),
        ),
        xaxis=dict(  # 2D xaxis domain
            domain=[0, 0.45], scaleanchor="y", scaleratio=1  # <-- enforce equal aspect
        ),
        yaxis=dict(domain=[0, 1]),  # 2D yaxis domain
    )

    return fig


def unravel_elems(mesh, cline, cline1stderiv, m, n, plot_figures=False):
    """
    This function projects 3D mesh face centers onto a 2D plane using an unraveling
    function, then divides them into a grid of groups based on their positions relative
    to a centerline and orthogonal direction.

    Args:
    mesh : trimesh.Trimesh
        The input 3D mesh object.
    cline : numpy.ndarray
        Matrix with centerline point coordinates.
    cline1stderiv : numpy.ndarray
        Matrix with corresponding first derivatives of centerline points.
    m : int or array-like
        EITHER the number of segments along the centerline direction (e.g., n=1)
        OR an array of divisions specifying the segments by proportion of total
        centerline length (e.g., n=[0, 1]).
    n : int or array-like
        EITHER the number of divisions orthogonal to the centerline direction
        (e.g., m=3) OR an array of divisions specifying the segments by proportion
        of total centerline length (e.g., m=[0, 0.1, 0.3, 1.0]).
    plot_figures : bool
        Whether to plot the unraveling results.
    Returns
    grps : list of list of numpy.ndarray
        A 2D list (m_eff x n_eff) where each element contains indices of faces
        in that group. grps[i][j] is an array of face indices for the (i,j) bin.

    """

    # ---- Unravel the COMs into 2D ----
    twodvertices, _, _ = unravel(
        mesh.triangles_center, mesh.face_normals, cline, cline1stderiv
    )

    # ---- Determine global bounds for divisions ----
    x_min, x_max = np.min(twodvertices[:, 0]), np.max(twodvertices[:, 0])
    y_min, y_max = np.min(twodvertices[:, 1]), np.max(twodvertices[:, 1])

    # Expand bounds by 1% in the appropriate direction
    x_margin = 0.01 * np.abs(x_max - x_min)
    y_margin = 0.01 * np.abs(y_max - y_min)

    xst, xed = x_min - x_margin, x_max + x_margin
    yst, yed = y_min - y_margin, y_max + y_margin

    # ---- Determine y-divisions (m direction) ----
    if np.isscalar(m):
        ydivs = np.linspace(yst, yed, int(m) + 1)
        m_eff = int(m)
    else:
        m = np.asarray(m)
        ydivs = m * (yed - yst) + yst
        m_eff = len(m) - 1

    # ---- Determine x-divisions (n direction) ----
    if np.isscalar(n):
        # equal segments
        xdivs = np.linspace(xst, xed, int(n) + 1)
        n_eff = int(n)
    else:
        # proportional divisions
        n = np.asarray(n)
        xdivs = n * (xed - xst) + xst
        n_eff = len(n) - 1

    # ---- Allocate groups ----
    grps = [[[] for _ in range(n_eff)] for _ in range(m_eff)]

    total_count = 0

    # ---- Assign faces to bins ----
    for i in range(m_eff):
        for j in range(n_eff):
            mask = (
                (twodvertices[:, 1] > ydivs[i])
                & (twodvertices[:, 1] < ydivs[i + 1])
                & (twodvertices[:, 0] > xdivs[j])
                & (twodvertices[:, 0] < xdivs[j + 1])
            )
            idx = np.where(mask)[0]
            grps[i][j] = idx
            total_count += len(idx)

    # ---- Ensure all faces are assigned ----
    if total_count != mesh.faces.shape[0]:
        raise RuntimeError("Error dividing into groups: counts do not match.")

    if plot_figures:
        fig = plot_unravel_groups(
            twodvertices, mesh.triangles_center, cline, grps, m_eff, n_eff
        )
        fig.show()

    return grps
