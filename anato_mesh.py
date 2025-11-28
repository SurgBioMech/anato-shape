# -*- coding: utf-8 -*-
"""
The main functions for calculating the partition-level curvatures using python to reproduce 
the algorithm originally published by K. Khabaz here: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011815

@authors: joepugar & kameelkhabaz
"""

from anato_utils import *
import pandas as pd
import numpy as np
import math
import trimesh
import time
import concurrent.futures
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
import queue

class Curvatures:
    """Class to store all curvature metric functions."""
    def __init__(self, pcs, ta):
        self.k1, self.k2, self.ta = pcs[:, 0], pcs[:, 1], ta
    def Gaussian(self):
        return np.mean(self.k1 * self.k2)
    def Mean(self):
        return np.mean(0.5 * (self.k1 + self.k2))
    def IntGaussian(self):
        return self.Gaussian() * self.ta
    def IntMeanSquared(self):
        return self.Mean()**2 * self.ta
    def Willmore(self):
        return 4 * self.Mean()**2 - 2 * self.Gaussian()
    def IntWillmore(self):
        return self.Willmore() * self.ta
    def Casorati(self):
        return np.mean(np.sqrt(0.5 * (self.k1**2 + self.k2**2)))
    def ShapeIndex(self):
        return np.mean((2 / np.pi) * np.arctan((self.k2 + self.k1) / (self.k2 - self.k1)))

def GetStatFeatures(partition_df, quantities):
    """Reducing each distribution of curvature quantities down to statistical scalar features."""
    stats = {}
    areas = partition_df['Patch_Area'].values
    total_area = np.sum(areas)
    for q in quantities:
        quant = partition_df[q].values
        mean_q, var_q = np.mean(quant), np.var(quant)
        weighted_sum = np.sum(quant * areas)
        weighted_sum_sq = np.sum((quant**2) * areas)
        stats.update({
            f"{q}_Mean": mean_q,
            f"{q}_Var": var_q,
            f"{q}_Fluct": (weighted_sum_sq / total_area) - (weighted_sum / total_area)**2,
            f"{q}_Sum": np.sum(quant)})
    return pd.DataFrame([stats])

def get_areas(v, f):
    """Calculate the area of each triangle."""
    P1, P2, P3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(P2 - P1, P3 - P1), axis=1)
    
def calcCOMs(mesh):
    """Returns the triangles' center of mass."""
    v, f = mesh.vertices, mesh.faces
    return v[f].mean(axis=1)

def triangle_indices_faces(cluster_id, v, f):
    """Per cluster extraction of assoicated vertices and triangles."""
    triangles = f[cluster_id]
    unique_triangle_nodes = np.unique(triangles)
    return triangles, unique_triangle_nodes

def triangle_indices_vertices(kept_vertices, v, f):
    """For a subset of vertices, returns the associated triangles."""
    sts_idx = np.all(np.isin(f, kept_vertices), axis=1)
    triangles = f[sts_idx]
    vsidx2, indices = np.unique(triangles, return_inverse=True)
    vertices = v[vsidx2]
    if not np.array_equal(kept_vertices, np.arange(len(v))):
        triangles = indices.reshape(triangles.shape)
    return triangles, vertices, vsidx2

def calc_num_patches(mesh, m):
    """Calculate the number of partitions per manifold using scaling law."""
    v, f, pcs = mesh.vertices, mesh.faces, mesh.curvatures
    median_R2_squared = np.median(1 / pcs[:, 0])**2
    total_area = np.sum(get_areas(v, f))
    k = int(m * round(total_area / median_R2_squared))
    return min(k, len(f))

def calculate_quantities(mesh, quantities, k, cluster_ids):
    """Calculates per-patch curvature quantities."""
    v, f, pcs = mesh.vertices, mesh.faces, mesh.curvatures
    mesh_quants, patch_areas, num_elem_patch, avg_elem_area = np.zeros((len(quantities), k)), np.zeros(k), np.zeros(k), np.zeros(k)
    curvatures_func = {quantity: getattr(Curvatures, quantity) for quantity in quantities}

    for c in range(k):
        cluster_id = np.where(cluster_ids == c)[0]
        t, unique_triangle_nodes = triangle_indices_faces(cluster_id, v, f)
        cluster_areas = get_areas(v, t)
        
        ACluster = np.sum(cluster_areas)
        patch_areas[c], num_elem_patch[c], avg_elem_area[c] = ACluster, len(t), np.mean(cluster_areas)
        
        curv = Curvatures(pcs[unique_triangle_nodes], ACluster)
        mesh_quants[:, c] = [curvatures_func[q](curv) for q in quantities]

    return mesh_quants, patch_areas, num_elem_patch, avg_elem_area

def organize_data(scan_name, k, cluster_centers, mesh_quants, num_elem_patch, avg_elem_area, patch_areas, quantities):
    """Organize non-curvature data and concatenate."""
    return pd.concat([
        pd.DataFrame(cluster_centers, columns=['X', 'Y', 'Z']),
        pd.DataFrame(mesh_quants.T, columns=quantities),
        pd.DataFrame({'Num_Elem_per_Patch': num_elem_patch, 'Avg_Elem_per_Area': avg_elem_area, 'Patch_Area': patch_areas})
    ], axis=1)

def tol(pcs):
    """Calculate the xy-plane point removal tolerance."""
    rn = math.ceil(1 / np.median(pcs[:, 0][pcs[:, 0] != 0]) / 10) * 10
    return max(rn, 20)

def edge_cleanup(vertices, indices, pcs, tol1=0.4, tol2=20):
    """Used to remove the flat edges from the manifold."""
    #- getting the z locations of each of the two edges: 
    def get_edges(svs):
        zc = [v[2] for v in svs]
        zcr = np.round(zc, 1)
        z_count = Counter(zcr)
        most_common_z = z_count.most_common(1)
        if len(most_common_z) < 1:
            raise ValueError("Less than 1 ascending flat region found.")
        return most_common_z[0][0], 0
    
    z1, z2 = get_edges(vertices)
    
    #- identifying which points fall within the removal cylinders:
    def is_within_cylinder(x, y, z, cx, cy, cz, height, radius):
        return (np.abs(z - cz) <= height) and (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius)

    #- tol1 is the z-spacing tolerance and works well when set to 0.4 
    z1_points = vertices[np.abs(vertices[:, 2] - z1) < tol1]
    z2_points = vertices[np.abs(vertices[:, 2] - z2) < tol1]
    cx1, cy1 = np.mean(z1_points[:, 0]), np.mean(z1_points[:, 1])
    cx2, cy2 = np.mean(z2_points[:, 0]), np.mean(z2_points[:, 1])
    mask = []
    for vertex in vertices:
        x, y, z = vertex
        #- tol2 form the radius of the removal cylinder and scales with the overall radius of the aorta via the tol function 
        if is_within_cylinder(x, y, z, cx1, cy1, z1, tol1, tol2) or is_within_cylinder(x, y, z, cx2, cy2, z2, tol1, tol2):
            mask.append(False)
        else:
            mask.append(True)
    #- a mask of vertices to keep, meaning they did not exist within the removal cylinders
    mask = np.array(mask)
    filtered_vertices = vertices[mask]
    filtered_pcs = pcs[mask]
    old_to_new_index = np.full(vertices.shape[0], -1)
    old_to_new_index[mask] = np.arange(np.sum(mask))
    filtered_indices = []
    for triangle in indices:
        if all(old_to_new_index[triangle] != -1):
            filtered_indices.append(old_to_new_index[triangle])
    filtered_indices = np.array(filtered_indices)
    return filtered_vertices, filtered_indices, filtered_pcs, mask

def clean_mesh(mesh, mesh_name=None):

    def flat_edge_outlier_exclusion(pcs):
        mn = np.abs(np.mean(pcs))
        thresh = 1 / 2000
        removed = np.where((np.abs(pcs[:, 0]) < mn * thresh) & (np.abs(pcs[:, 1]) < mn * thresh))[0]
        return removed

    def point_removal(pcs, mesh_name):
        kg = pcs[:, 0] * pcs[:, 1]
        k1 = pcs[:, 0]
        if "KY" in mesh_name or "SA" in mesh_name:
            removed = np.where((np.abs(kg) > (np.mean(kg) + 1.1 * np.std(kg))) & 
                               (np.abs(k1) > (np.mean(k1) + 1.6 * np.std(k1))))[0]
        else:
            removed = np.where((np.abs(kg) > (np.mean(kg) + 2 * np.std(kg))) & 
                               (np.abs(k1) > (np.mean(k1) + 3 * np.std(k1))))[0]
        return removed
    
    v, f, pcs = mesh.vertices, mesh.faces, mesh.curvatures
    assert len(v) == len(pcs), "Vertex and curvature arrays have mismatched dimensions."
    
    removed_primary = flat_edge_outlier_exclusion(pcs)
    kept_vertices = np.setdiff1d(np.arange(len(v)), removed_primary)
    fp, vp, vs_keep_idcs = triangle_indices_vertices(kept_vertices, v, f)
    pcsp = pcs[vs_keep_idcs, :]
    
    removed_secondary = point_removal(pcsp, mesh_name)
    kept_vertices = np.setdiff1d(np.arange(len(vp)), removed_secondary)
    fs, vs, vs_keep_idcs = triangle_indices_vertices(kept_vertices, vp, fp)
    pcss = pcsp[vs_keep_idcs, :]
    mesh_clean = trimesh.Trimesh(vertices=vs, faces=fs)
    mesh_clean.curvatures = pcss
    return mesh_clean

def Manifold(mesh, scan_name, quantities, m, prm):
    """Core function for partitioning and per-patch curvature calculations."""
    if prm == 'curvature':
        mesh_clean = clean_mesh(mesh, mesh_name=scan_name)
    else:
        mesh_clean = mesh
    
    triangle_COMs = calcCOMs(mesh_clean)
    num_faces = len(mesh_clean.faces)
    
    #Fixed Patches
    if m > 10:
        if m >= num_faces:
            k = num_faces  # Ensure k does not exceed available points
            print(f'''For {scan_name}, num_patches overwritten to {k} to match {num_faces} elements due to insufficient points.''')
        else:
            k = m
            print(f'''For {scan_name}, num_patches set to {k} with {num_faces} elements.''')

    # Floating Patches
    else:
        calc_k = calc_num_patches(mesh_clean, m)
        if calc_k >= num_faces:
            k = num_faces  # Prevent exceeding available points
            print(f'''For {scan_name}, num_patches overwritten to {k} to match {num_faces} elements due to insufficient points.''')
        else:
            k = calc_k
            print(f'''For {scan_name}, num_patches set to {k} with {num_faces} elements.''')

    km = MiniBatchKMeans(n_clusters=k, max_iter=100, batch_size=1536).fit(triangle_COMs)
    mesh_quants, patch_areas, num_elem_patch, avg_elem_area = calculate_quantities(mesh_clean, quantities, k, km.labels_)
    return organize_data(scan_name, k, km.cluster_centers_, mesh_quants, num_elem_patch, avg_elem_area, patch_areas, quantities), k, mesh_clean, km.labels_

def ProcessManifold(path, quantities, m, progress_queue, prm):
    """Used to organize the results of Manifold."""
    scan_name, path_name = path[1], path[0]
    full_scan_path = os.path.join(path_name, scan_name)
    if 'SA' in scan_name: #special handling for pediatric scans 
        if ('M5.' not in scan_name) or (m != 1):
            print(f"Skipping {scan_name}: SA scan without M5. or m != 1 (m={m})")
            progress_queue.put(m)
            return None, None
    mesh = GetMeshFromParquet(full_scan_path)
    scan_name_no_ext = file_name_not_ext(scan_name)
    manifold_df, patches, mesh_clean, cluster_ids = Manifold(mesh, quantities=quantities, m=m, scan_name=scan_name, prm=prm)
    A, As, V, k1, k2 = mesh_clean.area, mesh_clean.area_faces, mesh_clean.volume, mesh_clean.curvatures[:,0], mesh_clean.curvatures[:,1]
    vertex_areas = np.zeros(len(mesh_clean.vertices))
    for vertex_idx in range(len(mesh_clean.vertices)):
        connected_faces = np.where(mesh_clean.faces == vertex_idx)[0]
        vertex_areas[vertex_idx] = As[connected_faces].sum() / len(connected_faces)
    As = vertex_areas
    scan_features = pd.concat([pd.DataFrame({
        'Scan_ID': [scan_name_no_ext],
        'AvElemPatch': [np.mean(manifold_df['Num_Elem_per_Patch'])],
        'AvElemArea': [np.mean(manifold_df['Avg_Elem_per_Area'])],
        'AvPatchArea': [np.mean(manifold_df['Patch_Area'])],
        'Num_Patches': [patches],
        'SurfaceArea': [A],
        'Volume': [V],
        'Flatness_Index': [A**3/V**2],
        'Sphericity_Index': [4.836 * V**(2/3) / A],
        'GLN': [(1/(4*np.pi)) * np.sqrt(A * np.sum((k1*k2)**2 * As))],
        'GAA': [np.sum(k1*k2*As)/A],
        'MLN': [(1/(4*np.pi)) * np.sqrt(np.sum((0.5*(k1+k2))**2 * As))],
        'MAA': [np.sum((0.5*(k1+k2))*As)/A],
        'MeanEdgeLength': [np.mean(mesh_clean.edges_unique_length)],
        'MeanFaceAngle': [np.mean(mesh_clean.face_adjacency_angles)],
        'MeanVertexAngle': [np.mean(mesh_clean.face_angles)],
        'EulerNumber': [mesh.euler_number],
        'MomentInertia': [np.linalg.norm(mesh_clean.moment_inertia)],
    }), GetStatFeatures(manifold_df, quantities)], axis=1)
    progress_queue.put(m)
    print(f"Processed manifold {scan_name}: {m}")
    return scan_features, manifold_df, cluster_ids

def BatchManifold(paths, quantities, m, progress_queue, progress_counts, progress_bars, prm):
    """Parent function to Manifold which handles doing many at once."""
    results = []
    for path in paths:
        scan_features, manifold_df, cluster_ids = ProcessManifold(path, quantities, m, progress_queue, prm)
        progress_counts[m] += 1
        if progress_bars[m]: progress_bars[m].update(progress(progress_counts[m], len(paths)))
        if scan_features is None:
            raise ValueError(f"ProcessManifold returned None for path: {path}, m: {m}")
        results.append((scan_features, manifold_df, cluster_ids))
    return results

def _store_result(m, result, file_names, m_group, m_dict):
    """Parses raw results and updates the dictionaries."""
    if not result or not isinstance(result, list):
        print(f"Unexpected result format for m={m}: {result}")
        return

    m_group[m] = [r[0] for r in result]
    for i, item in enumerate(result):
        m_dict[(m, file_names[i])] = item[1:]
    
    return m_group, m_dict

def _run_parallel(paths, quantities, m_set, prm, file_names):
    "Run the BatchManifold function in parallel"
    m_group, m_dict = {}, {}
    overall_bar = display(progress(0, len(m_set) * len(paths)), display_id=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        futures = {}
        for m in m_set:
            # Setup specific progress bar for this m
            p_bar = display(progress(0, len(paths)), display_id=True)
            futures[executor.submit(BatchManifold, paths, quantities, m, queue.Queue(), {m: 0}, {m: p_bar}, prm)] = m

        # Process as they complete
        progress_count = 0
        for future in concurrent.futures.as_completed(futures):
            m = futures[future]
            try:
                m_group, m_dict = _store_result(m, future.result(), file_names, m_group, m_dict)
                
                # Update overall progress
                progress_count += len(paths)
                if overall_bar: overall_bar.update(progress(progress_count, len(m_set) * len(paths)))
            except Exception as e:
                raise ValueError(f"Error processing m={m}: {e}")
        
    return m_group, m_dict

def _run_sequential(paths, quantities, m_set, prm, file_names):
    "Run the BatchManifold function sequentially"
    m_group, m_dict = {}, {}
    overall_bar = display(progress(0, len(m_set) * len(paths)), display_id=True)
    progress_count = 0

    for m in m_set:
        try:
            # Setup specific progress bar for this m
            p_bar = display(progress(0, len(paths)), display_id=True)
            
            # Run blocking call
            result = BatchManifold(paths, quantities, m, queue.Queue(), {m: 0}, {m: p_bar}, prm)
            m_group, m_dict = _store_result(m, result, file_names, m_group, m_dict)

            # Update overall progress
            progress_count += len(paths)
            if overall_bar: overall_bar.update(progress(progress_count, len(m_set) * len(paths)))
        except Exception as e:
            raise ValueError(f"Error processing m={m}: {e}")

    return m_group, m_dict

def MsetBatchManifold(paths, quantities, m_set, prm, parallel):
    """Processing function which allows multiple partitioning schemes to be executed"""
    file_names = [file_name_not_ext(path[1]) for path in paths]

    if parallel:
        manifold_group, manifold_dict = _run_parallel(paths, quantities, m_set, prm, file_names)
    else:
        manifold_group, manifold_dict = _run_sequential(paths, quantities, m_set, prm, file_names)

    return manifold_group, manifold_dict

def process_m_with_progress(m, manifold_group):
    """Progress tracking with multiple m_set values."""
    data = pd.concat(manifold_group[m], ignore_index=True)
    data['Partition_Prefactor'] = str(m)
    return data

def GetAnatoMeshResults(parent_folder, filter_strings, file_filter_strings, prm=None, quantities=['Gaussian'], m_set=[1.0], parallel=True):
    """Top most function."""
    print("Organizing paths and file names:")
    paths = GetFilteredMeshPaths(parent_folder, filter_strings, file_filter_strings, ext=".parquet")
    assert len(paths) > 0, "All available data was filtered out."
    print("Starting GetAortaMeshResults: the top most progress bar is for all calculations and the progress bars below are for parallel processes.")
    manifold_group, manifold_dict = MsetBatchManifold(paths, quantities, m_set, prm, parallel)
    results = [process_m_with_progress(m, manifold_group) for m in m_set]
    results_df = pd.concat(results, ignore_index=True)
    print("Finished.")
    return results_df, manifold_dict