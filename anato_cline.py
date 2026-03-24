# -*- coding: utf-8 -*-
"""
Automated centerline extraction for tubular (aortic) surface meshes.

Computes a centerline from a closed triangular mesh by:
1. Detecting flat end-caps via per-vertex principal curvatures (K1, K2)
2. Computing a geodesic distance field along the tube wall
3. Extracting level-set centroids at regular distance intervals
4. Fitting a smooth parametric spline and resampling uniformly

Output is directly compatible with parse_cline() / unravel() / growth_mapping().

@author: kkhabaz
"""

from collections import deque
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import splprep, splev
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import trimesh

from anato_utils import GetMeshFromParquet


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_normalize(v, axis=-1, eps=1e-12):
    """Normalize vectors along *axis*, guarding against zero-length."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


# ---------------------------------------------------------------------------
# MHD volume helpers
# ---------------------------------------------------------------------------


def _voxel_to_world(ijk, vox_spacing, offset):
    """Convert voxel (k,j,i) coords to world (x,y,z)."""
    world = np.zeros((len(ijk), 3), dtype=float)
    world[:, 0] = ijk[:, 2] * vox_spacing[0] + offset[0]
    world[:, 1] = ijk[:, 1] * vox_spacing[1] + offset[1]
    world[:, 2] = ijk[:, 0] * vox_spacing[2] + offset[2]
    return world


def _skeleton_to_graph(skel_coords):
    """Build adjacency graph from skeleton voxel coordinates using 26-connectivity."""
    tree = cKDTree(skel_coords)
    pairs = tree.query_pairs(r=1.75)
    n = len(skel_coords)
    adj = [[] for _ in range(n)]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _bfs_path(adj, start, end):
    """BFS from *start* to *end*, return path as list of node indices."""
    n = len(adj)
    dist = np.full(n, -1, dtype=int)
    parent = np.full(n, -1, dtype=int)
    dist[start] = 0
    parent[start] = start
    q = deque([start])
    while q:
        u = q.popleft()
        if u == end:
            break
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    path = []
    node = end
    while parent[node] != node:
        path.append(node)
        node = parent[node]
    path.append(node)
    path.reverse()
    return path


def _bfs_dist(adj, start):
    """BFS from a single node, return distance array (-1 = unreached)."""
    n = len(adj)
    dist = np.full(n, -1, dtype=int)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _trace_to_edge(ep_idx, skel_ijk, skel_world, adj, mask, vox_spacing, offset, ds):
    """Trace from a skeleton endpoint outward to the actual mask edge,
    returning the cross-section centroid at that edge.

    Walks along the **dominant axis** of the local tube direction (whichever
    of k/j/i has the largest component), tracking the connected-component
    centroid at each slice.  This works for any tube orientation and keeps
    the centroid at the center of the cross-section throughout.
    """
    # Determine outward direction from the skeleton endpoint
    path = [ep_idx]
    cur = ep_idx
    for _ in range(10):
        nbrs = [n for n in adj[cur] if n not in path]
        if not nbrs:
            break
        cur = nbrs[0]
        path.append(cur)

    if len(path) >= 2:
        inward = skel_world[path[-1]] - skel_world[path[0]]
    else:
        inward = np.array([0.0, 0.0, 1.0])

    # Convert inward direction to voxel space (k, j, i)
    inward_vox = np.array(
        [
            inward[2] / (abs(vox_spacing[2]) + 1e-12),
            inward[1] / (abs(vox_spacing[1]) + 1e-12),
            inward[0] / (abs(vox_spacing[0]) + 1e-12),
        ]
    )
    # Outward = opposite of inward
    outward_vox = -inward_vox

    # Dominant axis = the voxel axis most aligned with the tube direction
    dominant = int(np.argmax(np.abs(outward_vox)))  # 0=k, 1=j, 2=i
    step_sign = 1 if outward_vox[dominant] > 0 else -1

    # Starting position in original mask coords
    sk, sj, si = skel_ijk[ep_idx]
    pos = [int(sk * ds), int(sj * ds), int(si * ds)]  # [k, j, i]
    # Clamp to mask bounds
    pos[0] = min(pos[0], mask.shape[0] - 1)
    pos[1] = min(pos[1], mask.shape[1] - 1)
    pos[2] = min(pos[2], mask.shape[2] - 1)

    # Track the last valid cross-section centroid (as float [k, j, i])
    last_center = [float(pos[0]), float(pos[1]), float(pos[2])]

    # Walk along dominant axis, tracking the connected-component centroid
    while True:
        pos[dominant] += step_sign
        if pos[dominant] < 0 or pos[dominant] >= mask.shape[dominant]:
            break

        # Extract 2D slice perpendicular to dominant axis
        if dominant == 0:
            slc = mask[pos[0], :, :]
            ref = (int(round(last_center[1])), int(round(last_center[2])))
        elif dominant == 1:
            slc = mask[:, pos[1], :]
            ref = (int(round(last_center[0])), int(round(last_center[2])))
        else:
            slc = mask[:, :, pos[2]]
            ref = (int(round(last_center[0])), int(round(last_center[1])))

        labeled, _ = ndimage.label(slc)
        ra = min(max(ref[0], 0), slc.shape[0] - 1)
        rb = min(max(ref[1], 0), slc.shape[1] - 1)
        comp = labeled[ra, rb]

        if comp > 0:
            comp_vox = np.argwhere(labeled == comp)
            ca, cb = comp_vox[:, 0].mean(), comp_vox[:, 1].mean()
            if dominant == 0:
                last_center = [float(pos[0]), ca, cb]
            elif dominant == 1:
                last_center = [ca, float(pos[1]), cb]
            else:
                last_center = [ca, cb, float(pos[2])]
        else:
            break

    # Convert final centroid [k, j, i] -> world [x, y, z]
    centroid = np.array(
        [
            last_center[2] * vox_spacing[0] + offset[0],
            last_center[1] * vox_spacing[1] + offset[1],
            last_center[0] * vox_spacing[2] + offset[2],
        ]
    )
    return centroid


def load_mhd(mhd_path):
    """Load an MHD + raw file pair.

    Parameters
    ----------
    mhd_path : str or Path
        Path to the ``.mhd`` header file.

    Returns
    -------
    volume : np.ndarray, shape (Z, Y, X)
    spacing : np.ndarray, shape (3,)  — voxel spacing in mm (x, y, z)
    offset : np.ndarray, shape (3,)   — world-coordinate origin (x, y, z)
    """
    mhd_path = Path(mhd_path)
    header = {}
    with open(mhd_path) as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                header[key.strip()] = val.strip()

    dims = [int(x) for x in header["DimSize"].split()]
    spacing = [float(x) for x in header["ElementSpacing"].split()]
    offset = [float(x) for x in header["Offset"].split()]

    dtype_map = {
        "MET_UCHAR": np.uint8,
        "MET_USHORT": np.uint16,
        "MET_SHORT": np.int16,
        "MET_FLOAT": np.float32,
    }
    dtype = dtype_map.get(header["ElementType"], np.uint8)

    raw_path = mhd_path.parent / header["ElementDataFile"]
    vol = np.fromfile(raw_path, dtype=dtype).reshape(dims[2], dims[1], dims[0])

    return vol, np.array(spacing), np.array(offset)


def compute_centerline_from_mhd(
    mhd_path, n_points=1200, spacing=None, smoothing_factor=None, downsample=3
):
    """Compute centerline from an MHD segmentation volume.

    Extracts the medial axis via 3D skeletonization, identifies the two main
    tube endpoints by cross-sectional area (distinguishing flat cut faces from
    branch vessel tips), traces to the true mask edges, and fits a smooth
    spline through the skeleton path.

    Parameters
    ----------
    mhd_path : str or Path
        Path to the ``.mhd`` file.
    n_points : int
        Number of uniformly spaced output points (default 1200).
        Ignored if *spacing* is provided.
    spacing : float, optional
        Uniform spacing between output points (mm). Overrides *n_points*.
    smoothing_factor : float, optional
        Spline smoothing parameter. Auto-calibrated if None.
    downsample : int
        Downsample factor before skeletonization (default 3).

    Returns
    -------
    cline_pos : np.ndarray, shape (N, 3)
        Centerline positions in world coordinates (ascending root -> descending end).
    cline_div : np.ndarray, shape (N, 3)
        Unit tangent vectors at each position.
    """
    vol, vox_spacing, offset = load_mhd(mhd_path)
    mask = vol > 0

    # Downsample for cleaner skeleton
    ds = downsample
    mask_ds = mask[::ds, ::ds, ::ds]
    spacing_ds = vox_spacing * ds

    # Morphological closing to fill small holes
    struct = ndimage.generate_binary_structure(3, 1)
    mask_ds = ndimage.binary_closing(mask_ds, structure=struct, iterations=2)
    mask_ds = ndimage.binary_fill_holes(mask_ds)

    # Skeletonize
    skel = skeletonize(mask_ds).astype(np.uint8)
    skel_ijk = np.argwhere(skel > 0)

    if len(skel_ijk) < 10:
        raise ValueError(f"Skeleton too small ({len(skel_ijk)} voxels).")

    skel_world = _voxel_to_world(skel_ijk, spacing_ds, offset)

    # Build graph and find degree-1 endpoints
    adj = _skeleton_to_graph(skel_ijk)
    degrees = np.array([len(a) for a in adj])
    endpoints = np.where(degrees == 1)[0]

    if len(endpoints) < 2:
        z_vals = skel_world[:, 2]
        endpoints = np.array([np.argmax(z_vals), np.argmin(z_vals)])

    # --- Identify the two tube endpoints ---
    # Descending end: always the endpoint with lowest Z (bottom of candy cane).
    # Ascending root: the endpoint farthest from descending by graph distance
    # (this finds the other end of the main tube, not a branch tip).
    ep_z = skel_world[endpoints, 2]
    desc_node = endpoints[np.argmin(ep_z)]

    desc_dist = _bfs_dist(adj, desc_node)
    remaining = [ep for ep in endpoints if ep != desc_node]
    asc_node = max(remaining, key=lambda ep: desc_dist[ep])

    # Trace from each skeleton endpoint outward to the actual mask edge
    asc_edge = _trace_to_edge(
        asc_node, skel_ijk, skel_world, adj, mask, vox_spacing, offset, ds
    )
    desc_edge = _trace_to_edge(
        desc_node, skel_ijk, skel_world, adj, mask, vox_spacing, offset, ds
    )

    # BFS shortest path between the two main endpoints
    path_indices = _bfs_path(adj, asc_node, desc_node)
    raw_path = skel_world[path_indices]

    # Prepend/append edge centroids (center of cross-section at each cut face)
    raw_path = np.vstack([asc_edge, raw_path, desc_edge])

    # Spline smoothing
    if smoothing_factor is None:
        smoothing_factor = len(raw_path) * (spacing_ds[0] * 1.5) ** 2

    tck, u = splprep(
        [raw_path[:, 0], raw_path[:, 1], raw_path[:, 2]], s=smoothing_factor, k=3
    )

    u_dense = np.linspace(0, 1, 5000)
    pts_dense = np.array(splev(u_dense, tck)).T
    total_length = np.sum(np.linalg.norm(np.diff(pts_dense, axis=0), axis=1))

    if spacing is not None:
        n_points = max(int(total_length / spacing) + 1, 4)

    u_new = np.linspace(0, 1, n_points)
    cline_pos = np.array(splev(u_new, tck)).T
    cline_div = np.array(splev(u_new, tck, der=1)).T
    cline_div = _safe_normalize(cline_div)

    # Trim points that overshoot outside the mask
    def _inside(pt):
        i = int(round((pt[0] - offset[0]) / vox_spacing[0]))
        j = int(round((pt[1] - offset[1]) / vox_spacing[1]))
        k = int(round((pt[2] - offset[2]) / vox_spacing[2]))
        if 0 <= k < mask.shape[0] and 0 <= j < mask.shape[1] and 0 <= i < mask.shape[2]:
            return mask[k, j, i]
        return False

    lo = 0
    while lo < len(cline_pos) - 1 and not _inside(cline_pos[lo]):
        lo += 1
    hi = len(cline_pos) - 1
    while hi > lo and not _inside(cline_pos[hi]):
        hi -= 1
    cline_pos = cline_pos[lo : hi + 1]
    cline_div = cline_div[lo : hi + 1]

    # Orient: ascending root first (higher Z of the two endpoints)
    if cline_pos[0, 2] < cline_pos[-1, 2]:
        cline_pos = cline_pos[::-1]
        cline_div = -cline_div[::-1]

    # Extend each end along the spline tangent: walk until outside the
    # mask, then continue 3% of centerline length further.
    seg_lens = np.linalg.norm(np.diff(cline_pos, axis=0), axis=1)
    cline_len = seg_lens.sum()
    ext_bonus = 0.03 * cline_len
    pt_sp = spacing if spacing else cline_len / len(cline_pos)

    def _extend(start_pt, direction):
        """Walk from start_pt along direction until outside mask,
        then 3 % more.  Returns list of evenly spaced extension points."""
        step = pt_sp
        pts = []
        d = step
        outside = False
        while d < cline_len:  # safety cap
            pt = start_pt + direction * d
            pts.append(pt)
            if not outside and not _inside(pt):
                outside = True
                remaining = ext_bonus
            if outside:
                remaining -= step
                if remaining <= 0:
                    break
            d += step
        return np.array(pts) if pts else np.empty((0, 3))

    ext_start = _extend(cline_pos[0], -cline_div[0])[::-1]  # outward from asc root
    ext_end = _extend(cline_pos[-1], cline_div[-1])  # outward from desc end

    cline_pos = np.vstack([ext_start, cline_pos, ext_end])

    # Recompute tangents for the full extended centerline
    cline_div = np.zeros_like(cline_pos)
    cline_div[1:-1] = cline_pos[2:] - cline_pos[:-2]
    cline_div[0] = cline_pos[1] - cline_pos[0]
    cline_div[-1] = cline_pos[-1] - cline_pos[-2]
    cline_div = _safe_normalize(cline_div)

    return cline_pos, cline_div
