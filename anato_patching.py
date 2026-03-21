"""
Optimized patching module for geometry-aware surface decomposition.

Provides a multi-stage patching pipeline that creates curvature-consistent,
connectivity-preserving patches on triangular meshes. The pipeline:

1. K-means initialization + polynomial refinement (splitting above RMSE tolerance)
2. Curvature-based trimming (relative deviation from patch centroid curvature)
3. Iterative cleanup loop (promote unassigned, refine new patches, outlier removal)
4. Border reassignment (assign remaining unassigned faces)
5. Omega-preserving upsampling (split to target count, preserving integrated curvature)

Extracted from CleanOptimizedPatches.ipynb.
"""

import time
import heapq
from collections import deque
from typing import Optional

import numpy as np
import trimesh
from sklearn.cluster import KMeans


# =============================================================================
# Low-level geometry helpers
# =============================================================================


def _per_face_area(mesh):
    """Compute per-face area from vertex positions."""
    v = mesh.vertices
    f = mesh.faces
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    nvec = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(nvec, axis=1)


def _fit_quadratic_frame(centroids, face_ids):
    """
    Fit a quadratic surface Z = c0 + c1*x + c2*y + c3*x^2 + c4*xy + c5*y^2
    to face centroids in a local PCA frame.

    Returns (coefs, rmse, C0, u, v, n) where:
        coefs: 6 quadratic coefficients
        rmse: root mean square error of the fit
        C0: centroid of the patch
        u, v, n: local coordinate frame axes
    """
    ids = np.asarray(face_ids, dtype=int)
    P = centroids[ids]
    C0 = P.mean(axis=0)
    Q = P - C0

    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1, :]
    n /= np.linalg.norm(n) + 1e-15

    u = Vt[0, :]
    u -= n * np.dot(u, n)
    u /= np.linalg.norm(u) + 1e-15

    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-15

    xr = Q @ u
    yr = Q @ v
    zr = Q @ n

    X = np.column_stack([np.ones_like(xr), xr, yr, xr**2, xr * yr, yr**2])
    coefs, *_ = np.linalg.lstsq(X, zr, rcond=None)

    resid = zr - (X @ coefs)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return coefs, rmse, C0, u, v, n


def _K_quadratic_vec(coefs, x, y):
    """Compute Gaussian curvature K at (x, y) from quadratic surface coefficients."""
    c0, c1, c2, c3, c4, c5 = coefs
    fx = c1 + 2 * c3 * x + c4 * y
    fy = c2 + c4 * x + 2 * c5 * y
    fxx = 2 * c3
    fxy = c4
    fyy = 2 * c5
    num = fxx * fyy - fxy**2
    den = (1.0 + fx * fx + fy * fy) ** 2
    return num / (den + 1e-30)


# =============================================================================
# Face adjacency and connectivity helpers
# =============================================================================


def _build_face_neighbors(mesh):
    """Build face adjacency list from mesh face_adjacency."""
    F = int(mesh.faces.shape[0])
    nbrs = [[] for _ in range(F)]
    fa = np.asarray(mesh.face_adjacency, dtype=int)
    if fa.size:
        for a, b in fa:
            a = int(a)
            b = int(b)
            nbrs[a].append(b)
            nbrs[b].append(a)
    return [np.array(v, dtype=int) if len(v) else np.array([], dtype=int) for v in nbrs]


def _masked_bfs_dist(neighbors, mask, seed):
    """BFS distance from seed within masked faces."""
    INF = 10**9
    dist = np.full(mask.size, INF, int)
    seed = int(seed)
    if not mask[seed]:
        return dist
    q = deque([seed])
    dist[seed] = 0
    while q:
        v = q.popleft()
        for u in neighbors[v]:
            u = int(u)
            if (not mask[u]) or dist[u] != INF:
                continue
            dist[u] = dist[v] + 1
            q.append(u)
    return dist


def _farthest_pair_in_mask(neighbors, mask):
    """Find approximate farthest pair of faces within masked region."""
    I = np.where(mask)[0]
    if I.size == 0:
        return None, None
    s0 = int(I[0])
    d0 = _masked_bfs_dist(neighbors, mask, s0)
    s1 = int(I[np.argmax(d0[I])])
    d1 = _masked_bfs_dist(neighbors, mask, s1)
    s2 = int(I[np.argmax(d1[I])])
    return s1, s2


def _connected_components(face_indices, neighbors):
    """Find connected components among a set of face indices."""
    face_indices = np.asarray(face_indices, dtype=np.int64)
    face_set = set(int(f) for f in face_indices)
    visited = set()
    comps = []
    for start in face_indices:
        s = int(start)
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in neighbors[f]:
                if nb in face_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(np.asarray(comp, dtype=np.int64))
    comps.sort(key=lambda c: c.size, reverse=True)
    return comps


def _largest_component(face_indices, neighbors):
    """Return largest connected component of a set of face indices."""
    if len(face_indices) == 0:
        return np.array([], dtype=np.int64)

    face_set = set(int(f) for f in face_indices)
    visited = set()
    largest = []

    for start in face_indices:
        s = int(start)
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in neighbors[f]:
                if nb in face_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(comp) > len(largest):
            largest = comp

    return np.asarray(largest, dtype=np.int64)


# =============================================================================
# Patch splitting functions
# =============================================================================


def _split_patch_by_farthest_pair(neighbors, ids, min_faces=6):
    """Split a patch into two halves using BFS from farthest pair."""
    idxs = np.asarray(ids, int)
    if idxs.size < 2 * int(min_faces):
        return None
    mask = np.zeros(len(neighbors), bool)
    mask[idxs] = True
    a, b = _farthest_pair_in_mask(neighbors, mask)
    if a is None or a == b:
        return None
    da = _masked_bfs_dist(neighbors, mask, int(a))
    db = _masked_bfs_dist(neighbors, mask, int(b))
    assignA = da[idxs] <= db[idxs]
    A = idxs[assignA]
    B = idxs[~assignA]
    if A.size < min_faces or B.size < min_faces:
        return None
    return A, B


def _fit_quadratic_rmse_on_faces(face_centroids, ids):
    """Compute RMSE of quadratic surface fit on a set of faces."""
    pts = face_centroids[np.asarray(ids, dtype=int)]
    C0 = pts.mean(axis=0)
    Q = pts - C0

    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1, :]
    n /= np.linalg.norm(n) + 1e-15
    u = Vt[0, :]
    u -= n * np.dot(u, n)
    u /= np.linalg.norm(u) + 1e-15
    v = np.cross(n, u)

    xr = Q @ u
    yr = Q @ v
    zr = Q @ n

    X = np.column_stack([np.ones_like(xr), xr, yr, xr**2, xr * yr, yr**2])
    coefs, *_ = np.linalg.lstsq(X, zr, rcond=None)
    resid = zr - X @ coefs
    rmse = float(np.sqrt(np.mean(resid**2)))
    return rmse


def _enforce_min_faces(patch_id, neighbors, min_faces=6, max_passes=10):
    """Merge patches smaller than min_faces into their best neighbor."""
    pid = np.asarray(patch_id, int).copy()
    for _ in range(int(max_passes)):
        changed = False
        uniq, cnt = np.unique(pid[pid >= 0], return_counts=True)
        sizes = dict(zip(uniq.tolist(), cnt.tolist()))
        small = [p for p, c in sizes.items() if c < min_faces]
        if not small:
            break

        for p in small:
            idx = np.where(pid == p)[0]
            votes = {}
            for f in idx:
                for g in neighbors[int(f)]:
                    q = int(pid[int(g)])
                    if q >= 0 and q != p:
                        votes[q] = votes.get(q, 0) + 1
            if votes:
                q_best = max(votes.items(), key=lambda kv: kv[1])[0]
            else:
                others = [q for q in uniq if q != p]
                if not others:
                    continue
                q_best = max(others, key=lambda q: sizes.get(int(q), 0))
            pid[idx] = int(q_best)
            changed = True

        if not changed:
            break
    return pid


def split_patch_longest_axis(mesh, face_ids, min_faces=6):
    """
    Split a set of faces into two halves along the patch's longest axis (PCA in 3D).
    Returns (A_faces, B_faces) or None if cannot split with min_faces constraint.
    """
    fidx = np.asarray(face_ids, dtype=np.int64)
    n = int(fidx.size)
    if n < 2 * int(min_faces):
        return None

    C = mesh.triangles_center[fidx]
    C0 = C.mean(axis=0)
    Q = C - C0

    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    d = Vt[0, :]
    d /= np.linalg.norm(d) + 1e-15

    t = Q @ d
    order = np.argsort(t)

    split = n // 2
    split = max(split, int(min_faces))
    split = min(split, n - int(min_faces))

    A = fidx[order[:split]]
    B = fidx[order[split:]]
    if A.size < int(min_faces) or B.size < int(min_faces):
        return None
    return A, B


def _approx_farthest_pair(face_ids, centers):
    """Fast farthest-pair approx: pick farthest from mean, then farthest from that."""
    C = centers[face_ids]
    m = C.mean(axis=0)
    a = face_ids[int(np.argmax(np.sum((C - m) ** 2, axis=1)))]
    Ca = centers[a]
    b = face_ids[int(np.argmax(np.sum((C - Ca) ** 2, axis=1)))]
    return int(a), int(b)


def _bfs_distances(seed, face_set, neighbors):
    """BFS graph distances in induced subgraph (faces restricted to face_set)."""
    INF = 10**15
    dist = {int(f): INF for f in face_set}
    s = int(seed)
    if s not in dist:
        return dist
    dist[s] = 0
    q = [s]
    qi = 0
    while qi < len(q):
        f = q[qi]
        qi += 1
        df = dist[f]
        for nb in neighbors[f]:
            nb = int(nb)
            if nb in dist and dist[nb] == INF:
                dist[nb] = df + 1
                q.append(nb)
    return dist


def split_patch_connectivity_bfs(neighbors, face_ids, centers, min_faces=6):
    """
    Connectivity-preserving split:
      - choose 2 farthest seeds (approx)
      - BFS distances from both seeds in induced adjacency
      - assign each face to closer seed
    """
    fidx = np.asarray(face_ids, dtype=np.int64)
    n = int(fidx.size)
    if n < 2 * int(min_faces):
        return None

    face_set = set(int(f) for f in fidx)
    a, b = _approx_farthest_pair(fidx, centers)

    da = _bfs_distances(a, face_set, neighbors)
    db = _bfs_distances(b, face_set, neighbors)

    A = []
    B = []
    for f in fidx:
        f = int(f)
        d1 = da.get(f, 10**15)
        d2 = db.get(f, 10**15)
        if d1 >= 10**14 and d2 >= 10**14:
            pa = np.sum((centers[f] - centers[a]) ** 2)
            pb = np.sum((centers[f] - centers[b]) ** 2)
            (A if pa <= pb else B).append(f)
        else:
            (A if d1 <= d2 else B).append(f)

    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)

    if A.size < int(min_faces) or B.size < int(min_faces):
        return None
    return A, B


# =============================================================================
# Core patching stages
# =============================================================================


def run_polyfit_split_on_mesh(
    mesh,
    seed_patches=300,
    poly_rmse_tol=0.05,
    min_faces=6,
    max_passes=20,
    verbose=True,
):
    """
    Stage 1: K-means initialization + iterative polynomial splitting.

    Runs K-means clustering on face centroids, then iteratively splits patches
    whose quadratic surface fit RMSE exceeds poly_rmse_tol.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces.")

    if mesh.face_adjacency is None or len(mesh.face_adjacency) == 0:
        mesh = mesh.copy()
        mesh.face_adjacency  # triggers computation

    C = mesh.triangles_center
    F = int(mesh.faces.shape[0])
    neighbors = _build_face_neighbors(mesh)

    K_max = max(1, F // max(1, int(min_faces)))
    K_used = int(min(int(seed_patches), int(K_max)))

    t0 = time.time()
    if verbose:
        print(
            f"faces={F} | seed_patches={seed_patches} -> K_used={K_used} | tol(RMSE)={poly_rmse_tol}"
        )

    patch_id = (
        KMeans(n_clusters=K_used, n_init=10, max_iter=2000, random_state=0)
        .fit_predict(C)
        .astype(int)
    )
    patch_id = _enforce_min_faces(
        patch_id, neighbors, min_faces=min_faces, max_passes=10
    )

    # Refinement loop
    did_split = True
    passes = 0
    while did_split and passes < int(max_passes):
        did_split = False
        passes += 1

        pids = [int(p) for p in np.unique(patch_id[patch_id >= 0])]
        next_pid = (max(pids) + 1) if pids else 0

        splits = 0
        viol = 0
        worst = 0.0

        rmse_map = {}
        for pid in pids:
            idx = np.where(patch_id == pid)[0]
            if idx.size < int(min_faces):
                continue
            rmse = _fit_quadratic_rmse_on_faces(C, idx)
            rmse_map[pid] = (rmse, int(idx.size))
            worst = max(worst, rmse)

        for pid in pids:
            if pid not in rmse_map:
                continue
            rmse, nfaces = rmse_map[pid]
            if not (np.isfinite(rmse) and rmse > float(poly_rmse_tol)):
                continue
            viol += 1
            if nfaces < 2 * int(min_faces):
                continue

            idx = np.where(patch_id == pid)[0]
            split = _split_patch_by_farthest_pair(neighbors, idx, min_faces=min_faces)
            if split is None:
                continue

            A, B = split
            patch_id[A] = pid
            patch_id[B] = next_pid
            next_pid += 1
            did_split = True
            splits += 1

        if verbose:
            print(
                f"  [pass {passes:02d}] patches={len(pids)} | viol={viol} | "
                f"splits={splits} | worst_RMSE={worst:.6g}"
            )

        if splits == 0:
            break

    final_pids = [int(p) for p in np.unique(patch_id[patch_id >= 0])]
    final_pids.sort()

    patch_rmse = np.array(
        [
            _fit_quadratic_rmse_on_faces(C, np.where(patch_id == pid)[0])
            for pid in final_pids
        ],
        float,
    )
    n_total = int(patch_rmse.size)
    n_above = int(np.sum(np.isfinite(patch_rmse) & (patch_rmse > float(poly_rmse_tol))))
    dt = time.time() - t0

    out = {
        "K_used": int(K_used),
        "K_poly": int(len(final_pids)),
        "poly_rmse_tol": float(poly_rmse_tol),
        "seed_patches_requested": int(seed_patches),
        "patch_ids": np.array(final_pids, int),
        "patch_rmse": patch_rmse,
        "n_above_tol": int(n_above),
        "frac_above_tol": float(n_above / max(1, n_total)),
        "rmse_median": float(np.nanmedian(patch_rmse)),
        "rmse_max": float(np.nanmax(patch_rmse)),
        "runtime_s": float(dt),
        "face_patch_id": patch_id,
    }
    if verbose:
        print(
            f"FINAL: K_poly={out['K_poly']} | above_tol={out['n_above_tol']} "
            f"({100*out['frac_above_tol']:.1f}%) | rmse_med={out['rmse_median']:.6g} "
            f"rmse_max={out['rmse_max']:.6g} | {dt:.2f}s"
        )
    return out


def trim_patches_by_relK(
    mesh,
    face_patch_id,
    rel_tol=0.075,
    min_faces=6,
    k0_eps=1e-10,
    keep_largest_component=True,
    verbose=True,
):
    """
    Stage 2: Curvature-based trimming.

    For each patch, fit a quadratic surface and compute K0 at the centroid.
    Unassign faces where |K(x,y) - K0| / |K0| > rel_tol.
    Optionally keep only the largest connected component of remaining faces.
    """
    pid = np.asarray(face_patch_id, dtype=np.int64).copy()
    C = mesh.triangles_center
    neighbors = _build_face_neighbors(mesh) if keep_largest_component else None

    pids = np.unique(pid[pid >= 0]).astype(int)

    n_unassigned_total = 0
    n_patches_trimmed = 0
    n_patches_skipped_k0 = 0
    n_patches_reverted_small = 0

    for p in pids:
        fidx = np.where(pid == p)[0]
        if fidx.size < int(min_faces):
            continue

        coefs, rmse, C0, u, v, n = _fit_quadratic_frame(C, fidx)
        K0 = float(_K_quadratic_vec(coefs, np.array([0.0]), np.array([0.0]))[0])

        if abs(K0) < float(k0_eps):
            n_patches_skipped_k0 += 1
            continue

        Q = C[fidx] - C0
        x = Q @ u
        y = Q @ v
        Kvals = _K_quadratic_vec(coefs, x, y)
        rel = np.abs(Kvals - K0) / (abs(K0) + 1e-30)

        keep = fidx[rel <= rel_tol]

        if keep_largest_component:
            keep = _largest_component(keep, neighbors)

        if keep.size < int(min_faces):
            n_patches_reverted_small += 1
            continue

        to_unassign = np.setdiff1d(fidx, keep, assume_unique=False)
        pid[to_unassign] = -1

        n_removed = int(to_unassign.size)
        if n_removed > 0:
            n_unassigned_total += n_removed
            n_patches_trimmed += 1

    info = {
        "n_unassigned_total": int(n_unassigned_total),
        "n_patches_trimmed": int(n_patches_trimmed),
        "n_patches_skipped_k0_small": int(n_patches_skipped_k0),
        "n_patches_reverted_small": int(n_patches_reverted_small),
    }

    if verbose:
        print(
            f"[trim] rel_tol={rel_tol} | unassigned={n_unassigned_total} faces | "
            f"trimmed={n_patches_trimmed} | skipped_k0={n_patches_skipped_k0} | "
            f"reverted_small={n_patches_reverted_small}"
        )

    return pid, info


def promote_unassigned_components_to_patches(mesh, face_patch_id, min_faces=6):
    """
    Promote connected components of unassigned (pid==-1) faces with
    size >= min_faces to new patch IDs.
    """
    pid = np.asarray(face_patch_id, dtype=np.int64).copy()

    unassigned = np.where(pid == -1)[0]
    if unassigned.size == 0:
        return pid, set(), []

    neighbors = _build_face_neighbors(mesh)
    comps = _connected_components(unassigned, neighbors)

    max_pid = int(pid[pid >= 0].max()) if np.any(pid >= 0) else -1
    next_pid = max_pid + 1

    new_patch_ids = set()
    small_components = []

    for comp in comps:
        if comp.size >= int(min_faces):
            pid[comp] = next_pid
            new_patch_ids.add(next_pid)
            next_pid += 1
        else:
            small_components.append(comp)

    return pid, new_patch_ids, small_components


def refine_new_patches_split_if_above_tol(
    mesh,
    face_patch_id,
    new_patch_ids,
    poly_rmse_tol,
    min_faces=6,
    max_total_splits=5000,
    verbose=True,
):
    """
    Refine only newly created patches by splitting those with RMSE > poly_rmse_tol.
    """
    pid = np.asarray(face_patch_id, dtype=np.int64).copy()
    refine_set = set(int(x) for x in new_patch_ids)

    stack = list(refine_set)
    max_pid = int(pid[pid >= 0].max()) if np.any(pid >= 0) else -1

    splits_done = 0
    rmse_log = {}
    centroids = mesh.triangles_center

    while stack:
        p = int(stack.pop())
        fidx = np.where(pid == p)[0]
        if fidx.size < 2 * int(min_faces):
            continue

        _, rmse, *_ = _fit_quadratic_frame(centroids, fidx)
        rmse = float(rmse)
        rmse_log[p] = rmse

        if not (np.isfinite(rmse) and rmse > float(poly_rmse_tol)):
            continue

        split = split_patch_longest_axis(mesh, fidx, min_faces=min_faces)
        if split is None:
            continue

        A, B = split
        max_pid += 1
        p_new = int(max_pid)

        pid[A] = p
        pid[B] = p_new

        refine_set.add(p_new)
        stack.append(p)
        stack.append(p_new)

        splits_done += 1
        if splits_done >= int(max_total_splits):
            if verbose:
                print(
                    f"[refine] hit max_total_splits={max_total_splits}, stopping early."
                )
            break

    if verbose:
        print(
            f"[refine] poly_rmse_tol={poly_rmse_tol} | splits_done={splits_done} | "
            f"refined_set={len(refine_set)}"
        )

    return pid, refine_set, rmse_log, splits_done


def build_fixed_patching_for_mesh(
    mesh,
    pid0,
    poly_rmse_tol,
    rel_tol,
    min_faces=6,
    max_total_splits=5000,
):
    """
    Build a fixed patching from an initial polyfit result:
    1. Trim by relative curvature deviation
    2. Promote unassigned components to new patches
    3. Reassign remaining unassigned back to original patches
    4. Refine only new patches by splitting above tolerance
    """
    pid_trim, _ = trim_patches_by_relK(
        mesh,
        pid0,
        rel_tol=rel_tol,
        min_faces=min_faces,
        keep_largest_component=True,
        k0_eps=1e-10,
        verbose=False,
    )

    pid_promoted, new_patch_ids, _ = promote_unassigned_components_to_patches(
        mesh,
        pid_trim,
        min_faces=min_faces,
    )

    pid_after_new = np.asarray(pid_promoted, dtype=np.int64).copy()
    still_unassigned = np.where(pid_after_new == -1)[0]
    pid_after_new[still_unassigned] = pid0[still_unassigned]

    pid_refined, refined_patch_ids, rmse_log, splits_done = (
        refine_new_patches_split_if_above_tol(
            mesh,
            pid_after_new,
            new_patch_ids=new_patch_ids,
            poly_rmse_tol=float(poly_rmse_tol),
            min_faces=min_faces,
            max_total_splits=max_total_splits,
            verbose=False,
        )
    )

    return pid_refined, refined_patch_ids, rmse_log, splits_done


def assign_unassigned_faces_to_best_border_patch(
    mesh,
    face_patch_id,
    min_faces=6,
    k0_eps=1e-10,
    passes=2,
    verbose=False,
):
    """
    Assign remaining unassigned faces (pid==-1) to the bordering patch that
    minimizes relative curvature deviation |K(x,y) - K0| / |K0|.
    """
    pid = np.asarray(face_patch_id, dtype=np.int64).copy()
    C_all = np.asarray(mesh.triangles_center, dtype=np.float64)
    neighbors = _build_face_neighbors(mesh)

    reassigned_total = 0

    for p in range(int(passes)):
        unassigned = np.where(pid == -1)[0]
        if unassigned.size == 0:
            break

        cand = set()
        for f in unassigned:
            for nb in neighbors[int(f)]:
                pp = int(pid[int(nb)])
                if pp >= 0:
                    cand.add(pp)

        if not cand:
            cand = set(np.unique(pid[pid >= 0]).astype(int).tolist())

        fit_cache = {}
        for pp in cand:
            fidx = np.where(pid == int(pp))[0]
            if fidx.size < int(min_faces):
                continue
            coefs, rmse, C0, u, v, n = _fit_quadratic_frame(C_all, fidx)
            K0 = float(_K_quadratic_vec(coefs, np.array([0.0]), np.array([0.0]))[0])
            fit_cache[int(pp)] = (coefs, C0, u, v, K0)

        if not fit_cache:
            break

        patch_ids = np.array(list(fit_cache.keys()), dtype=int)
        C0s = np.stack([fit_cache[int(pp)][1] for pp in patch_ids], axis=0)

        changed_this_pass = 0

        for f in unassigned:
            f = int(f)
            border = sorted(
                {int(pid[int(nb)]) for nb in neighbors[f] if int(pid[int(nb)]) >= 0}
            )

            if border:
                best_pp = None
                best_eps = np.inf

                cf = C_all[f]
                for pp in border:
                    if pp not in fit_cache:
                        continue
                    coefs, C0, u, v, K0 = fit_cache[int(pp)]
                    Q = cf - C0
                    x = float(Q @ u)
                    y = float(Q @ v)

                    Kxy = float(
                        _K_quadratic_vec(coefs, np.array([x]), np.array([y]))[0]
                    )
                    denom = max(abs(K0), float(k0_eps))
                    eps = abs(Kxy - K0) / denom

                    if np.isfinite(eps) and eps < best_eps:
                        best_eps = eps
                        best_pp = int(pp)

                if best_pp is None:
                    best_pp = int(np.bincount(np.array(border, dtype=int)).argmax())

                pid[f] = best_pp
                changed_this_pass += 1
            else:
                cf = C_all[f]
                d = np.linalg.norm(C0s - cf[None, :], axis=1)
                best_pp = int(patch_ids[int(np.argmin(d))])
                pid[f] = best_pp
                changed_this_pass += 1

        reassigned_total += changed_this_pass
        if verbose:
            print(
                f"[assign pass {p+1}] reassigned {changed_this_pass} faces | "
                f"remaining -1 = {int(np.sum(pid==-1))}"
            )

        if changed_this_pass == 0:
            break

    return pid, int(reassigned_total)


# =============================================================================
# Omega computation and outlier handling
# =============================================================================


def compute_patchOmega_for_result_raw3d(mesh, face_patch_id, min_faces=6):
    """
    Compute per-patch integrated Gaussian curvature (Omega) and area using
    raw 3D face areas and quadratic surface fit curvature.

    Returns:
        omega_by_patch: dict {pid: Omega}
        area_by_patch: dict {pid: Area}
    """
    face_patch_id = np.asarray(face_patch_id, dtype=int)
    C = mesh.triangles_center
    Af = _per_face_area(mesh)

    omega_by_patch = {}
    area_by_patch = {}

    pids = np.unique(face_patch_id[face_patch_id >= 0]).astype(int)
    for pid in pids:
        fidx = np.where(face_patch_id == pid)[0]
        if fidx.size < int(min_faces):
            continue

        coefs, rmse, C0, u, v, n = _fit_quadratic_frame(C, fidx)
        Q = C[fidx] - C0
        x = Q @ u
        y = Q @ v

        Kvals = _K_quadratic_vec(coefs, x, y)
        Apatch = Af[fidx]

        omega_by_patch[int(pid)] = float(np.nansum(Kvals * Apatch))
        area_by_patch[int(pid)] = float(np.sum(Apatch))

    return omega_by_patch, area_by_patch


def _dissolve_small_patches_to_unassigned(face_patch_id, min_faces=6):
    """Dissolve patches with fewer than min_faces faces to unassigned (-1)."""
    pid = np.asarray(face_patch_id, dtype=np.int64).copy()
    assigned = pid[pid >= 0]
    if assigned.size == 0:
        return pid, 0, []

    max_pid = int(assigned.max())
    counts = np.bincount(assigned, minlength=max_pid + 1)

    small = np.where(counts < int(min_faces))[0]
    small = [int(p) for p in small if np.any(pid == int(p))]
    if not small:
        return pid, 0, []

    dissolved_faces = 0
    for p in small:
        idx = np.where(pid == p)[0]
        dissolved_faces += int(idx.size)
        pid[idx] = -1

    return pid, dissolved_faces, small


def dissolve_outlier_patches_by_omega(mesh, face_patch_id, z=3.0, min_faces=6):
    """
    Dissolve patches whose Omega is a z-score outlier (|Omega - mean| > z * std).
    """
    pid = np.asarray(face_patch_id, dtype=np.int64).copy()

    omega_dict, _ = compute_patchOmega_for_result_raw3d(
        mesh, pid, min_faces=int(min_faces)
    )
    if not omega_dict:
        return pid, [], 0, {"mean": np.nan, "std": np.nan, "n": 0}

    w = np.asarray(list(omega_dict.values()), dtype=float)
    w = w[np.isfinite(w)]
    if w.size < 2:
        mu = float(np.mean(w)) if w.size else np.nan
        return pid, [], 0, {"mean": mu, "std": 0.0, "n": int(w.size)}

    mu = float(np.mean(w))
    sd = float(np.std(w, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-30:
        return pid, [], 0, {"mean": mu, "std": sd, "n": int(w.size)}

    outlier_pids = []
    for p, om in omega_dict.items():
        if not np.isfinite(om):
            continue
        if abs(float(om) - mu) > float(z) * sd:
            outlier_pids.append(int(p))

    if not outlier_pids:
        return pid, [], 0, {"mean": mu, "std": sd, "n": int(w.size)}

    n_faces_dissolved = 0
    for p in outlier_pids:
        idx = np.where(pid == int(p))[0]
        n_faces_dissolved += int(idx.size)
        pid[idx] = -1

    return (
        pid,
        outlier_pids,
        n_faces_dissolved,
        {"mean": mu, "std": sd, "n": int(w.size)},
    )


# =============================================================================
# Omega-preserving upsampling
# =============================================================================


def upsample_to_target_patches_omega_preserving_raw3d(
    mesh,
    patch_id,
    omega_parent_dict,
    target_patches=1000,
    min_faces_per_child=6,
    verbose=True,
):
    """
    Split largest-area patches to reach target_patches count.
    Children inherit Area/2 and Omega/2 exactly, preserving K0 = Omega/Area.
    """
    UPSAMPLE_EPS_CHECK = 1e-9

    pid = np.asarray(patch_id, dtype=np.int64).copy()
    neighbors = _build_face_neighbors(mesh)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    Af = np.asarray(mesh.area_faces, dtype=np.float64)

    def _safe_k0(Om, A):
        Om = float(Om) if np.isfinite(Om) else np.nan
        A = float(A) if np.isfinite(A) else np.nan
        if np.isfinite(Om) and np.isfinite(A) and abs(A) > 1e-30:
            return float(Om / A)
        return np.nan

    pids = np.unique(pid[pid >= 0]).astype(int)
    stats = {}
    heap = []

    for p in pids:
        idx = np.where(pid == int(p))[0]
        if idx.size < int(min_faces_per_child):
            continue

        A = float(np.sum(Af[idx]))
        Om = float(omega_parent_dict.get(int(p), np.nan))
        K0 = _safe_k0(Om, A)

        stats[int(p)] = {"Area": A, "Omega": Om, "K0": K0}
        heapq.heappush(heap, (-A, int(p)))

    omega_vals0 = np.asarray([v["Omega"] for v in stats.values()], dtype=float)
    omega_vals0 = omega_vals0[np.isfinite(omega_vals0)]
    maxabs0 = float(np.max(np.abs(omega_vals0))) if omega_vals0.size else 0.0

    splits_done = 0
    unsplittable = set()
    stopped_early = False

    while len(stats) < int(target_patches) and heap:
        negA, p = heapq.heappop(heap)
        if p in unsplittable or p not in stats:
            continue

        idx = np.where(pid == int(p))[0]
        if idx.size < 2 * int(min_faces_per_child):
            unsplittable.add(int(p))
            continue

        split = split_patch_connectivity_bfs(
            neighbors, idx, centers, min_faces=min_faces_per_child
        )
        if split is None:
            unsplittable.add(int(p))
            continue

        A_idx, B_idx = split
        if A_idx.size < int(min_faces_per_child) or B_idx.size < int(
            min_faces_per_child
        ):
            unsplittable.add(int(p))
            continue

        new_pid = int(pid.max()) + 1
        pid[B_idx] = new_pid

        parent = stats[int(p)]
        parent_A = float(parent.get("Area", np.nan))
        parent_Om = float(parent.get("Omega", np.nan))
        parent_K0 = float(parent.get("K0", np.nan))

        childA_A = 0.5 * parent_A if np.isfinite(parent_A) else np.nan
        childA_Om = 0.5 * parent_Om if np.isfinite(parent_Om) else np.nan
        childB_A = childA_A
        childB_Om = childA_Om

        child_K0 = (
            parent_K0 if np.isfinite(parent_K0) else _safe_k0(childA_Om, childA_A)
        )

        childA = {"Area": childA_A, "Omega": childA_Om, "K0": child_K0}
        childB = {"Area": childB_A, "Omega": childB_Om, "K0": child_K0}

        stats[int(p)] = childA
        stats[int(new_pid)] = childB

        heapq.heappush(
            heap,
            (-float(childA["Area"]) if np.isfinite(childA["Area"]) else 0.0, int(p)),
        )
        heapq.heappush(
            heap,
            (
                -float(childB["Area"]) if np.isfinite(childB["Area"]) else 0.0,
                int(new_pid),
            ),
        )

        splits_done += 1
        if splits_done > 10_000_000:
            stopped_early = True
            break

    if len(stats) < int(target_patches):
        stopped_early = True

    # Compact IDs to 0..K-1
    uniq = np.unique(pid[pid >= 0]).astype(int)
    remap = {int(old): i for i, old in enumerate(uniq)}
    if any(remap[int(u)] != int(u) for u in uniq):
        for old, new in remap.items():
            pid[pid == int(old)] = int(new)
        stats = {int(remap[k]): v for k, v in stats.items() if int(k) in remap}

    omega_vals1 = np.asarray([v["Omega"] for v in stats.values()], dtype=float)
    omega_vals1 = omega_vals1[np.isfinite(omega_vals1)]
    maxabs1 = float(np.max(np.abs(omega_vals1))) if omega_vals1.size else 0.0

    report = {
        "splits_done": int(splits_done),
        "stopped_early": bool(stopped_early),
        "K_before": int(len(pids)),
        "K_after": int(len(stats)),
        "maxabsOmega_before": float(maxabs0),
        "maxabsOmega_after": float(maxabs1),
    }

    if maxabs1 > maxabs0 + UPSAMPLE_EPS_CHECK:
        raise RuntimeError(
            f"[UPSAMPLE INVARIANT VIOLATION] max|Omega| increased: "
            f"before={maxabs0} after={maxabs1}"
        )

    if verbose:
        print(f"[upsample] {report}")

    return pid, stats, report


# =============================================================================
# Top-level orchestrator
# =============================================================================


def optimized_patching(
    mesh: trimesh.Trimesh,
    seed_patches: int = 300,
    poly_rmse_tol: float = 0.05,
    rel_tol: float = 0.075,
    min_faces: int = 6,
    target_patches: int = 1000,
    max_total_splits: int = 5000,
    max_loop_iters: int = 30,
    outlier_z: float = 3.0,
    outlier_drop_passes: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Run the full optimized patching pipeline on a single mesh.

    Pipeline stages:
        1. K-means init + polynomial RMSE splitting
        2. Build fixed patching (trim by curvature, promote, refine)
        3. Iterative loop: trim -> dissolve small -> promote -> refine
        4. Assign remaining unassigned faces
        5. Outlier removal by Omega z-score
        6. Omega-preserving upsampling to target_patches

    Args:
        mesh: Input triangular mesh.
        seed_patches: Initial K-means cluster count.
        poly_rmse_tol: Quadratic fit RMSE threshold for splitting.
        rel_tol: Relative curvature deviation threshold for trimming.
        min_faces: Minimum faces per patch (default 6 needed to fit polynomial).
        target_patches: Target number of patches after upsampling.
        max_total_splits: Safety cap on refinement splits.
        max_loop_iters: Safety cap on cleanup loop iterations.
        outlier_z: Z-score threshold for outlier patch removal.
        outlier_drop_passes: Number of outlier removal passes.
        verbose: Print progress information.

    Returns:
        dict with:
            'face_patch_id': np.ndarray (n_faces,) - face-to-patch mapping (0-indexed)
            'patch_stats': dict {pid: {'Area': float, 'Omega': float, 'K0': float}}
            'n_patches': int - number of patches
            'report': dict with pipeline metadata
    """
    t0 = time.time()

    if verbose:
        print(
            f"[optimized_patching] mesh faces={len(mesh.faces)} | "
            f"seed={seed_patches} | tol={poly_rmse_tol} | rel_tol={rel_tol} | "
            f"target={target_patches}"
        )

    # Stage 1: K-means + polynomial splitting
    polyfit_result = run_polyfit_split_on_mesh(
        mesh,
        seed_patches=seed_patches,
        poly_rmse_tol=poly_rmse_tol,
        min_faces=min_faces,
        max_passes=20,
        verbose=verbose,
    )
    pid0 = np.asarray(polyfit_result["face_patch_id"], dtype=np.int64)

    # Stage 2: Build fixed patching (trim + promote + refine)
    pid_curr, _, _, _ = build_fixed_patching_for_mesh(
        mesh,
        pid0,
        poly_rmse_tol=poly_rmse_tol,
        rel_tol=rel_tol,
        min_faces=min_faces,
        max_total_splits=max_total_splits,
    )
    pid_curr = np.asarray(pid_curr, dtype=np.int64)

    # Stage 3: Iterative cleanup loop
    for it in range(max_loop_iters):
        pid_kicked, _ = trim_patches_by_relK(
            mesh,
            pid_curr,
            rel_tol=rel_tol,
            min_faces=min_faces,
            keep_largest_component=True,
            k0_eps=1e-10,
            verbose=False,
        )

        pid_clean, dissolved_faces, dissolved_pids = (
            _dissolve_small_patches_to_unassigned(
                pid_kicked,
                min_faces=min_faces,
            )
        )

        pid_promoted, new_patch_ids, _ = promote_unassigned_components_to_patches(
            mesh,
            pid_clean,
            min_faces=min_faces,
        )
        n_new = int(len(new_patch_ids))

        if n_new > 0:
            pid_refined, _, _, _ = refine_new_patches_split_if_above_tol(
                mesh,
                pid_promoted,
                new_patch_ids=new_patch_ids,
                poly_rmse_tol=poly_rmse_tol,
                min_faces=min_faces,
                max_total_splits=max_total_splits,
                verbose=False,
            )
        else:
            pid_refined = pid_promoted

        pid_curr = np.asarray(pid_refined, dtype=np.int64)

        if verbose:
            K = int(np.unique(pid_curr[pid_curr >= 0]).size)
            unassigned = int(np.sum(pid_curr == -1))
            print(
                f"  [loop {it+1}] new_patches={n_new} | K={K} | unassigned={unassigned}"
            )

        if n_new == 0:
            break

    # Stage 4: Assign remaining unassigned faces
    pid_assigned, n_reassigned = assign_unassigned_faces_to_best_border_patch(
        mesh,
        pid_curr,
        min_faces=min_faces,
        k0_eps=1e-10,
        passes=2,
        verbose=verbose,
    )
    pid_assigned = np.asarray(pid_assigned, dtype=np.int64)

    # Stage 5: Outlier removal
    pid_drop = pid_assigned.copy()
    for dp in range(int(outlier_drop_passes)):
        pid_drop, outlier_pids, outlier_faces, omega_stats = (
            dissolve_outlier_patches_by_omega(
                mesh,
                pid_drop,
                z=outlier_z,
                min_faces=min_faces,
            )
        )
        pid_drop, _ = assign_unassigned_faces_to_best_border_patch(
            mesh,
            pid_drop,
            min_faces=min_faces,
            k0_eps=1e-10,
            passes=2,
            verbose=False,
        )
        pid_drop = np.asarray(pid_drop, dtype=np.int64)
        if len(outlier_pids) == 0:
            break

    # Compute Omega for the looped state (this is what upsampling will preserve)
    omega_looped_dict, _ = compute_patchOmega_for_result_raw3d(
        mesh,
        pid_drop,
        min_faces=min_faces,
    )

    # Stage 6: Omega-preserving upsampling
    pid_final, stats_final, upsample_report = (
        upsample_to_target_patches_omega_preserving_raw3d(
            mesh,
            pid_drop,
            omega_parent_dict=omega_looped_dict,
            target_patches=target_patches,
            min_faces_per_child=min_faces,
            verbose=verbose,
        )
    )

    dt = time.time() - t0

    report = {
        "polyfit_K": polyfit_result["K_poly"],
        "K_pre_upsample": int(np.unique(pid_drop[pid_drop >= 0]).size),
        "K_final": int(len(stats_final)),
        "upsample_splits": upsample_report["splits_done"],
        "upsample_stopped_early": upsample_report["stopped_early"],
        "total_runtime_s": dt,
    }

    if verbose:
        print(f"[optimized_patching] done in {dt:.1f}s | K_final={report['K_final']}")

    return {
        "face_patch_id": pid_final,
        "patch_stats": stats_final,
        "n_patches": report["K_final"],
        "report": report,
    }
