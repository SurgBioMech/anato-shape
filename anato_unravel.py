
import colorsys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

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


def unravel(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    cline: np.ndarray,
    cline_deriv1: np.ndarray,
):
    """
    Unravel a 3D mesh surface around a centerline into a 2D representation.
    The unraveling is done by straightening the centerline along the z-axis and then
    flattening the surface in z-slices based on geodesic distances around the centerline

    Inputs:
      - vertices: (n,3) array of vertex coordinates
      - vertex_normals: (n,3) array of normals per vertex
      - cline: (k,3) centerline points
      - cline_deriv1: (k,3) first derivatives (tangent vectors) of cline
    Outputs:
      - twod_vertices: (n,2) unraveled coordinates [x_along_cut_or_signed_length, z_straightened]
      - straightened_line: (k,3) straightened centerline (z increases with centerline arc length)
      - straightened_vertices: (n,3) vertices after straightening (in same 3D coords as straightened_line)
    """
    # Validate shapes
    n = vertices.shape[0]
    k = cline.shape[0]
    assert vertices.shape[1] == 3 and cline.shape[1] == 3 and cline_deriv1.shape[1] == 3
    assert vertex_normals.shape[0] == n and vertex_normals.shape[1] == 3

    # Build straightened centerline: accumulate distances along original cline as z coordinate.
    straightened_line = np.zeros((k, 3), dtype=float)
    min_z_idx = int(np.argmin(cline[:, 2]))
    if min_z_idx == 0:
        flipy = False
    elif min_z_idx == k - 1:
        flipy = True
    else:
        raise ValueError(
            "Unexpected location of min z on centerline; expected end points."
        )

    for i in range(1, k):
        dist = np.linalg.norm(cline[i] - cline[i - 1])
        straightened_line[i] = straightened_line[i - 1] + np.array([0.0, 0.0, dist])

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
        with np.errstate(invalid="ignore"):
            uv = vecs / np.maximum(dists[:, None], 1e-12)
        cos_theta = np.clip((uv @ u), -1.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos_theta))

        # choose a search neighborhood around closest cline point
        idc1 = int(np.argmin(dists))
        search_region = max(1, int(round(k / 12)))
        start_pt = max(0, idc1 - search_region)
        end_pt = min(k, idc1 + search_region + 1)
        centerline_look = np.arange(start_pt, end_pt)
        if centerline_look.size == 0:
            centerline_look = np.arange(k)

        narrowed_angles = theta_deg[centerline_look]
        max_angle = np.max(narrowed_angles)

        # take points within top 15% of the max angle
        mask = narrowed_angles > 0.85 * max_angle
        candidate_indices = centerline_look[mask]

        # among candidates pick cline point closest in Euclidean distance to vertex
        cand_dists = np.linalg.norm(vertices[i] - cline[candidate_indices], axis=1)
        chosen_local = candidate_indices[int(np.argmin(cand_dists))]
        closest_cline_indices[i] = chosen_local

        # compute rotation to align tangent at chosen point to [0,0,1]
        vec_from = cline_deriv1[chosen_local]
        if np.linalg.norm(vec_from) < 1e-12:
            vec_from = np.array([0.0, 0.0, 1.0])
        else:
            vec_from = vec_from / np.linalg.norm(vec_from)
        vec_to = np.array([0.0, 0.0, 1.0])
        R = _rotation_from_a_to_b(vec_from, vec_to)

        vec = vertices[i] - cline[chosen_local]
        rotated_vec = R.dot(vec)
        straightened_vertices[i] = straightened_line[chosen_local] + rotated_vec

    # Now "unravel" per z-slices.
    twod_vertices = np.full((n, 2), np.nan, dtype=float)
    # Translate so minimum y is at zero
    translated = straightened_vertices.copy()
    translated[:, 1] -= np.nanmin(straightened_vertices[:, 1])

    zmin = np.nanmin(translated[:, 2]) - 1e-6
    zmax = np.nanmax(translated[:, 2]) + 1e-6
    zs = np.linspace(zmin, zmax, 100)

    for zi in range(len(zs) - 1):
        mask_slice = (translated[:, 2] > zs[zi]) & (translated[:, 2] < zs[zi + 1])
        slice_idx = np.nonzero(mask_slice)[0]
        if slice_idx.size == 0:
            continue
        vs = translated[slice_idx][:, [0, 1, 2]]  # X,Y,Z for nodes in slice

        if vs.shape[0] == 1:
            # single point: length 0, x sign determines sign
            x_sign = -1 if vs[0, 0] < 0 else 1
            twod_vertices[slice_idx[0]] = np.array([0.0 * x_sign, vs[0, 2]])
            continue

        # pairwise Euclidean distances of the slice points (2D spatial distance XY)
        # Use XY distances for neighborhood and boundary calculations
        XY = vs[:, :2]
        pairwise = squareform(pdist(XY))

        # adjacency: connect each node to its few nearest neighbours
        ascend_idx = np.argsort(pairwise, axis=1)  # ascending indices per row
        last_connect = min(5, pairwise.shape[0])  # number of neighbors to connect
        d2 = np.zeros_like(pairwise)
        for j in range(pairwise.shape[0]):
            # Always connect at least one neighbor (the nearest, excluding self)
            neighbors = ascend_idx[j, 1 : last_connect + 1]  # skip self at index 0

            if neighbors.size > 0:
                d2[j, neighbors] = pairwise[j, neighbors]
                d2[neighbors, j] = pairwise[neighbors, j]

        # ensure a closed curve by connecting hull edges
        try:
            hull = ConvexHull(XY)
            hull_indices = hull.vertices
            # connect hull edges (note: hull.vertices cycles through hull order)
            for m in range(len(hull_indices)):
                a_idx = hull_indices[m]
                b_idx = hull_indices[(m + 1) % len(hull_indices)]
                # map hull indices (which are into XY) back to indices in slice_idx-order
                d2[a_idx, b_idx] = pairwise[a_idx, b_idx]
                d2[b_idx, a_idx] = pairwise[b_idx, a_idx]
        except Exception:
            # ConvexHull can fail if points are collinear; in that case, connect sequentially
            for m in range(pairwise.shape[0] - 1):
                d2[m, m + 1] = pairwise[m, m + 1]
                d2[m + 1, m] = pairwise[m + 1, m]

        # Fix isolated components: build graph and connect smaller components to main
        G = nx.Graph()
        for a in range(pairwise.shape[0]):
            for b in range(a + 1, pairwise.shape[0]):
                if d2[a, b] > 0:
                    G.add_edge(a, b, weight=float(d2[a, b]))
        if G.number_of_nodes() == 0:
            continue
        # ensure all nodes present
        for node in range(pairwise.shape[0]):
            if node not in G:
                G.add_node(node)

        # Ensure that every slice’s point cloud forms a single connected graph by attaching lone nodes to the nearest
        # “real” boundary component
        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            # find largest component
            comps_sorted = sorted(comps, key=lambda c: len(c), reverse=True)
            main_comp = comps_sorted[0]
            main_nodes = set(main_comp)
            for comp in comps_sorted[1:]:
                for lone in comp:
                    # find nearest main node (in ascend_idx ordering)
                    # ascend_idx indexes into nodes of this slice
                    found = None
                    for candidate in ascend_idx[lone]:
                        if candidate in main_nodes:
                            found = candidate
                            break
                    if found is None:
                        # fallback to absolute nearest by pairwise
                        found = int(np.argmin(pairwise[lone]))
                    G.add_edge(lone, found, weight=float(pairwise[lone, found]))

        # Ensure graph is connected now
        if not nx.is_connected(G):
            raise RuntimeError("Graph is still not connected after fixing components.")

        # For each node in slice, compute shortest path distance from the source (point nearest to x=0 line at same z)
        for local_j, global_idx in enumerate(slice_idx):
            # find the source node index (closest to x=0 at same z)
            z_val = translated[global_idx, 2]

            z_target = translated[global_idx, 2]
            slice_pts = translated[slice_idx]
            ref_point = np.array([0.0, 0.0, z_target])
            dists = np.linalg.norm(slice_pts - ref_point, axis=1)
            sourcenode = int(np.argmin(dists))
            try:
                length = nx.shortest_path_length(
                    G, source=sourcenode, target=local_j, weight="weight"
                )
            except nx.NetworkXNoPath:
                raise RuntimeError(
                    "No path found in slice graph for shortest path calculation."
                )
            if translated[global_idx, 0] < 0:
                twod_vertices[global_idx, :] = np.array([-length, z_val])
            else:
                twod_vertices[global_idx, :] = np.array([length, z_val])

    # Post-processing flip if required (mirror about mean z then shift so minimum y becomes 1)
    if flipy:
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
    - twodvertices : (N,2)
    - points       : (N,3)
    - cline        : (K,3)
    - grps         : list-of-lists of index arrays [mdiv][ndiv]
    """

    pts2d = np.asarray(twodvertices)
    pts3d = np.asarray(points)

    # flatten all groups while keeping order
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

    # make consistent number of colors
    n_groups = max(len(groups), 1)

    fig = go.Figure()

    # --- 2D traces (scene = scene2) ---
    for k, (i, j, idx) in enumerate(groups):
        hue = (k / n_groups) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
        color = "#%02x%02x%02x" % (int(255 * r), int(255 * g), int(255 * b))

        fig.add_trace(
            go.Scatter(
                x=pts2d[idx, 0],
                y=pts2d[idx, 1],
                mode="markers",
                marker=dict(size=marker_size, color=color),
                name=f"2D div{i+1}_grp{j+1}",
                legendgroup=f"group_{k}",
                showlegend=False,  # avoid double legends
            )
        )

    # --- 3D traces (scene = scene) ---
    for k, (i, j, idx) in enumerate(groups):
        hue = (k / n_groups) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
        color = "#%02x%02x%02x" % (int(255 * r), int(255 * g), int(255 * b))

        fig.add_trace(
            go.Scatter3d(
                x=pts3d[idx, 0],
                y=pts3d[idx, 1],
                z=pts3d[idx, 2],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.8),
                name=f"div{i+1}_grp{j+1}",
                legendgroup=f"group_{k}",
            )
        )

    # --- centerline for 3D ---
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
        plot_unravel_groups(
            twodvertices, mesh.triangles_center, cline, grps, m_eff, n_eff
        )

    return grps
