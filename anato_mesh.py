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
import time
import concurrent.futures
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
import queue

class Curvatures:
    """Class to store all curvature metric functions."""
    def __init__(self, pcs, ta):
        self.pcs = pcs
        self.ta = ta
        self.k1 = pcs[:,0]
        self.k2 = pcs[:,1]
    def Gaussian(self):
        """Returns the average Gaussian curvature: mean(k1 * k2)."""
        return np.mean(self.k1 * self.k2)
    def Total(self):
        """Returns the total curvature: mean(k1 * k2) * area."""
        return np.mean(self.k1 * self.k2) * self.ta
    def Casorati(self):
        """Returns the Casorati curvature: mean(sqrt(0.5 * (k1**2 + k2**2)))."""
        return np.mean(np.sqrt(0.5 * (self.k1**2 + self.k2**2)))
    def Willmore(self):
        """Returns the Willmore functional: 0.25 * ((k1-k2)**2 * area)."""
        return 0.25 * np.mean((self.k1 - self.k2)**2) * self.ta
    def Willmore2(self): 
        """Returns the Willmore functional: 0.25 * ((k1**2+k2**2) * area)."""
        return 0.25 * np.mean(self.k1**2 + self.k2**2) * self.ta
    def IntMean(self):
        """Returns the integrated mean curvature: mean(((k1+k2)/2)**2)."""
        return np.mean(((self.k1 + self.k2) / 2)**2)
    def ShapeIndex(self):
        """Returns Koenderink's shape index: mean((2/pi)*arctan((k1+k2)/(k2-k1)))."""
        return np.mean((2 / np.pi) * np.arctan((self.k2 + self.k1) / (self.k2 - self.k1)))

def GetStatFeatures(partition_df, quantities):
    """Reducing each distribution of curvature quantities down to statistical scalar features."""
    stats = {}
    areas = partition_df['Patch_Area'].values
    total_area = np.sum(areas)
    for q in quantities:
        quant = partition_df[q].values
        stats[f"{q}_Mean"] = np.mean(quant)
        stats[f"{q}_Var"] = np.var(quant)
        weighted_sum = np.sum(quant * areas)
        weighted_sum_sq = np.sum((quant**2) * areas)
        stats[f"{q}_Fluct"] = (weighted_sum_sq / total_area) - (weighted_sum / total_area)**2
        stats[f"{q}_Sum"] = np.sum(quant)
    stats_df = pd.DataFrame([stats])
    return stats_df

def get_areas(svs,sts):
    """Returns the areas of each triangle."""
    P1 = svs[sts[:,0]]
    P2 = svs[sts[:,1]]
    P3 = svs[sts[:,2]]
    d1 = P2-P1
    d2 = P3-P1
    cps = np.cross(d1,d2)
    areas = 0.5 * np.linalg.norm(cps, axis=1)
    return areas
    
def calcCOMs(svs, sts):
    """Returns the triangles center of mass."""
    assert sts.shape[0] > sts.shape[1], "incorrect dimensions input matrix"
    assert svs.shape[0] > svs.shape[1], "incorrect dimensions input matrix"
    element_COM = svs[sts].mean(axis=1)
    return element_COM

def triangle_indices_faces(cluster_id, svs, sts):
    """Per cluster extraction of assoicated vertices and triangles."""
    triangles = sts[cluster_id]
    unique_triangle_nodes = np.unique(triangles)
    patch_vertices = svs[unique_triangle_nodes]
    return triangles, patch_vertices, unique_triangle_nodes

def triangle_indices_vertices(kept_vertices, svs, sts):
    """For a subset of vertices, returns the associated triangles."""
    sts_idx = np.all(np.isin(sts, kept_vertices), axis=1)
    triangles = sts[sts_idx]
    vsidx2, indices = np.unique(triangles, return_inverse=True)
    vertices = svs[vsidx2]
    if not np.array_equal(kept_vertices, np.arange(len(svs))):
        triangles = indices.reshape(triangles.shape)
    return triangles, vertices, vsidx2

def calc_num_patches(svs, sts, pcs, m):
    """The scaling law used to calculate the number of partitions per manifold."""
    areas = get_areas(svs,sts)
    k1 = pcs[:,0]
    R2 = 1 / k1
    median_R2_squared = np.median(R2)**2
    total_area = np.sum(areas)
    k = m * np.round(total_area / median_R2_squared)
    if k < len(sts):
        k = int(k)
    else:
        #- in the event of the smallest object gets overdefined by the partitioning scheme
        k = int(len(sts)) 
    return k

def calculate_quantities(svs, sts, pcs, quantities, k, cluster_ids):
    """Calculates per-patch curvature quantities and returns as manifold dataframe."""
    mesh_quants = np.zeros((len(quantities), k))
    patch_areas = np.zeros(k)
    num_elem_patch = np.zeros(k)
    avg_elem_area = np.zeros(k)
    
    function_dict = {
        'Gaussian': 'Gaussian',
        'Total': 'Total',
        'Casorati': 'Casorati',
        'Willmore': 'Willmore',
        'Willmore2':'Willmore2',
        'IntMean': 'IntMean',
        'ShapeIndex': 'ShapeIndex'
    }
   
    for c in range(k):
        cluster_id = np.where(cluster_ids == c)[0]
        triangles, patch_vertices, unique_triangle_nodes = triangle_indices_faces(cluster_id, svs, sts)
        cluster_principal_curvs = pcs[unique_triangle_nodes]
        cluster_areas = get_areas(svs, triangles)
        
        ACluster = np.sum(cluster_areas)
        patch_areas[c] = ACluster
        num_elem_patch[c] = len(triangles)
        avg_elem_area[c] = np.mean(cluster_areas)
        
        curvatures = Curvatures(cluster_principal_curvs, ACluster)
        
        for j, quantity in enumerate(quantities):
            method_name = function_dict.get(quantity)
            if method_name:
                mesh_quants[j, c] = getattr(curvatures, method_name)()
                
    return mesh_quants, patch_areas, num_elem_patch, avg_elem_area

def organize_data(scan_name, k, cluster_centers, mesh_quants, num_elem_patch, avg_elem_area, patch_areas, quantities):
    """Non-curvature data organization and concatenation."""
    manifold_df = pd.concat([
        pd.DataFrame(cluster_centers, columns=['X', 'Y', 'Z']),
        pd.DataFrame(mesh_quants.T, columns=quantities),
        pd.DataFrame(num_elem_patch, columns=['Num_Elem_per_Patch']),
        pd.DataFrame(avg_elem_area, columns=['Avg_Elem_per_Area']),
        pd.DataFrame(patch_areas, columns=['Patch_Area'])
    ], axis=1)
    return manifold_df

def tol(pcs):
    """Used to calculate the xy-plane point removal tolerance for the edge_cleanup function."""
    non_zero_values = pcs[:, 0][pcs[:, 0] != 0]
    n = 1 / np.median(non_zero_values)
    rn = math.ceil(n / 10) * 10
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

def flat_edge_outlier_exclusion(pcs):
    """Remove edges via curvature thresholding."""
    mn = np.abs(np.mean(pcs))
    thresh = 1/2000
    removed = np.where((np.abs(pcs[:,0]) < mn * thresh) & (np.abs(pcs[:,1]) < mn * thresh))
    return removed

def primary_outlier_exclusion(svs, sts, pcs, mesh_name):
    """Remove first set of outlier vertices via curvature thresholding."""
    nvertices = len(svs)
    assert nvertices == len(pcs), "incorrect dimensions"
    removed = flat_edge_outlier_exclusion(pcs)
    kept_vertices = np.setdiff1d(np.arange(1, nvertices), removed)
    sts, svs, vs_keep_idcs = triangle_indices_vertices(kept_vertices, svs, sts)
    pcs = pcs[vs_keep_idcs,:]
    return svs, sts, pcs

def point_removal(pcs, mesh_name):
    """Remove vertices within main anatomy mass via curvature thresholding. KY and SA distinguish the anatomy as normal (healthy)."""
    assert pcs.shape[1] < pcs.shape[0], "principal curvatures matrix has incorrect dimensions."
    kg = pcs[:, 0] * pcs[:, 1]
    k1 = pcs[:, 0]
    if "KY" in mesh_name or "SA" in mesh_name: #KY and SA distinguishes normal aortas in the Pocivavsek lab 
        removed = np.where((np.abs(kg) > (np.mean(kg) + 1.1 * np.std(kg))) & (np.abs(k1) > (np.mean(k1) + 1.6 * np.std(k1))))[0]
    else:
        removed = np.where((np.abs(kg) > (np.mean(kg) + 2 * np.std(kg))) & (np.abs(k1) > (np.mean(k1) + 3 * np.std(k1))))[0]
    return removed

def secondary_outlier_exclusion(svs, sts, pcs, mesh_name):
    """Remove second set of outlier vertices via curvature thresholding."""
    nvertices = len(svs)
    assert nvertices == len(pcs), "incorrect dimensions"
    removed = point_removal(pcs, mesh_name)
    kept_vertices = np.setdiff1d(np.arange(1, nvertices), removed)
    sts, svs, vs_keep_idcs = triangle_indices_vertices(kept_vertices, svs, sts)
    pcs = pcs[vs_keep_idcs,:]
    return svs, sts, pcs

def Manifold(pcs, svs, sts, scan_name, quantities, m, prm):
    """Core function for partitioning and per-patch curvautre calculations."""
    start_time = time.monotonic()
    if prm == 'thoracic':
        svsc, stsc, pcsc, mask = edge_cleanup(svs, sts, pcs, tol1=0.4, tol2=tol(pcs))
        print(f'''The {prm} point removal method was employed.''')
    else:
        svsp, stsp, pcsp = primary_outlier_exclusion(svs, sts, pcs, scan_name)
        svsc, stsc, pcsc = secondary_outlier_exclusion(svsp, stsp, pcsp, scan_name)
        print('The curvature point removal method was employed.')
    
    k = calc_num_patches(svsc, stsc, pcsc, m)
    triangle_COMs = calcCOMs(svsc, stsc)
    km = MiniBatchKMeans(n_clusters=k, max_iter=100, batch_size=1536).fit(triangle_COMs)
    cluster_centers, cluster_ids = km.cluster_centers_, km.labels_
    print('K-Means done.')
    mesh_quants, patch_areas, num_elem_patch, avg_elem_area = calculate_quantities(
        svsc, stsc, pcsc, quantities, k, cluster_ids)
    print('Quantities calculated.')
    manifold_df = organize_data(scan_name, k, cluster_centers, mesh_quants, num_elem_patch, avg_elem_area, patch_areas, quantities)
    end_time = time.monotonic()
    print(f'''{scan_name} with {k} patches using m={m} took: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}''')
    return manifold_df, k

def ProcessManifold(path, quantities, m, progress_queue, prm):
    """Used to organize the results of Manifold."""
    scan_name, path_name = path[1], path[0]
    full_scan_path = os.path.join(path_name, scan_name)
    svs, sts, pcs, mesh_features = GetMeshFromParquet(full_scan_path)
    scan_name_no_ext = file_name_not_ext(scan_name)
    manifold_df, patches = Manifold(pcs=pcs, svs=svs, sts=sts, quantities=quantities, m=m, scan_name=scan_name, prm=prm)
    scan_features = pd.concat([
        pd.DataFrame([scan_name_no_ext], columns=['Scan_Name']),
        pd.DataFrame([np.mean(manifold_df['Num_Elem_per_Patch'])], columns=['AvElemPatch']),
        pd.DataFrame([np.mean(manifold_df['Avg_Elem_per_Area'])], columns=['AvElemArea']),
        pd.DataFrame([np.mean(manifold_df['Patch_Area'])], columns=['AvPatchArea']),
        pd.DataFrame([patches], columns=['Num_Patches']),
        GetStatFeatures(manifold_df, quantities),
        mesh_features
    ], axis=1)
    progress_queue.put(m)
    return scan_features, svs, sts, pcs, manifold_df

def BatchManifold(paths, quantities, m, progress_queue, progress_counts, progress_bars, prm):
    """Parent funciton to Manifold which handles doing many at once."""
    results = []
    for path in paths:
        result = ProcessManifold(path, quantities, m, progress_queue, prm)
        results.append(result)
        progress_counts[m] += 1
        progress_bars[m].update(progress(progress_counts[m], len(paths)))
    return results

def MsetBatchManifold(paths, quantities, m_set, prm):
    """Parellel processing function which allows multiple partitioning schemes to be calculated simultaneously."""
    manifold_group, vertices_dict, triangles_dict, pcs_dict, manifold_dict = {}, {}, {}, {}, {}

    overall_progress = display(progress(0, len(m_set) * len(paths)), display_id=True)
    progress_bars = {m: display(progress(0, len(paths)), display_id=True) for m in m_set}
    progress_counts = {m: 0 for m in m_set}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(BatchManifold, paths, quantities, m, queue.Queue(), progress_counts, progress_bars, prm): m for m in m_set}

        total_tasks = len(m_set) * len(paths)
        overall_progress_value = 0

        for future in concurrent.futures.as_completed(futures):
            m = futures[future]
            try:
                result = future.result()
                if not result or not isinstance(result, list):
                    print(f"Unexpected result format for m={m}: {result}")
                    continue

                # Process the full tuple here
                manifold_group[m] = [r[0] for r in result]
                vertices_dict[m] = {file_name_not_ext(path[1]): pd.DataFrame(r[1], columns=['X', 'Y', 'Z']) for path, r in zip(paths, result)}
                triangles_dict[m] = {file_name_not_ext(path[1]): pd.DataFrame(r[2], columns=['T1', 'T2', 'T3']) for path, r in zip(paths, result)}
                pcs_dict[m] = {file_name_not_ext(path[1]): pd.DataFrame(r[3], columns=['K1', 'K2']) for path, r in zip(paths, result)}
                manifold_dict[m] = {file_name_not_ext(path[1]): r[4] for path, r in zip(paths, result)}

                overall_progress_value += len(paths)
                overall_progress.update(progress(overall_progress_value, total_tasks))
            except Exception as e:
                print(f"Error processing m={m}: {e} (paths={paths}, quantities={quantities})")
                continue

    return manifold_group, vertices_dict, triangles_dict, pcs_dict, manifold_dict

def process_m_with_progress(m, manifold_group):
    """Progress tracking with multiple m_set values."""
    data = pd.concat(manifold_group[m], ignore_index=True)
    data['Partition_Prefactor'] = str(m)
    return data

def GetAortaMeshResults(parent_folder, filter_strings, file_filter_strings, prm='thoracic', quantities=['Casorati', 'Total'], m_set=[1.0]):
    """Top most function."""
    print("Organizing paths and file names:")
    paths = GetFilteredParqMeshPaths(parent_folder, filter_strings, file_filter_strings)
    assert len(paths) > 0, "All available data was filtered out."
    print("Starting GetAortaMeshResults: the top most progress bar is for all calculations and the progress bars below are for parallel processes.")
    manifold_group, vertices_dict, triangles_dict, pcs_dict, manifold_dict = MsetBatchManifold(paths, quantities, m_set, prm)
    results = [process_m_with_progress(m, manifold_group) for m in m_set]
    results_df = pd.concat(results, ignore_index=True)
    print("Finished.")
    return results_df