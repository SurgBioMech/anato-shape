# -*- coding: utf-8 -*-
"""
The main functions for calculating the principal curvatures using python to reproduce
the algorithm originally published by S. Rusinkiewicz here: https://ieeexplore.ieee.org/abstract/document/1335277

@author: S. Rusinkiewicz
rewritten for python by joepugar
"""

from anato_utils import *
import pandas as pd
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import scipy.io
from IPython.display import HTML, display
from scipy.io import savemat


def RotateCoordinateSystem(up, vp, nf):
    npp = np.cross(up, vp) / np.linalg.norm(np.cross(up, vp))
    ndot = np.clip(np.dot(nf, npp), -1, 1)
    r_new_u, r_new_v = (up, vp) if ndot > -1 else (-up, -vp)
    perp = nf - ndot * npp
    dperp = (npp + nf) / (1 + ndot)
    return r_new_u - dperp * np.dot(perp, r_new_u), r_new_v - dperp * np.dot(
        perp, r_new_v
    )


def ProjectCurvatureTensor(uf, vf, nf, old_ku, old_kuv, old_kv, up, vp):
    r_new_u, r_new_v = RotateCoordinateSystem(up, vp, nf)
    OldTensor = np.array([[old_ku, old_kuv], [old_kuv, old_kv]])
    u1, v1, u2, v2 = (
        np.dot(r_new_u, uf),
        np.dot(r_new_u, vf),
        np.dot(r_new_v, uf),
        np.dot(r_new_v, vf),
    )
    new_ku = np.dot([u1, v1], np.dot(OldTensor, [u1, v1]))
    new_kuv = np.dot([u1, v1], np.dot(OldTensor, [u2, v2]))
    new_kv = np.dot([u2, v2], np.dot(OldTensor, [u2, v2]))
    return new_ku, new_kuv, new_kv


def CalcCurvature(mesh, VertexNormals, FaceNormals, Avertex, Acorner, up, vp):
    # - matrix of each face at each cell
    v, f = mesh.vertices, mesh.faces
    FaceSFM, VertexSFM = list(), list()
    for i in range(f.shape[0]):
        FaceSFM.append([[0, 0], [0, 0]])
    for i in range(v.shape[0]):
        VertexSFM.append([[0, 0], [0, 0]])
    Kn = np.zeros((1, f.shape[0]))
    # - get all edge vectors
    e0 = v[f[:, 2], :] - v[f[:, 1], :]
    e1 = v[f[:, 0], :] - v[f[:, 2], :]
    e2 = v[f[:, 1], :] - v[f[:, 0], :]
    e0_norm = normr(e0)
    wfp = np.array(np.zeros((f.shape[0], 3)))
    # - calculate curvature per face & set face coordinate frame
    for i in range(f.shape[0]):
        nf = FaceNormals[i, :]
        t = e0_norm[i, :]
        B = np.cross(nf, t)
        B = B / (np.linalg.norm(B))
        # - extract relevant normals in face vertices
        n0 = VertexNormals[f[i][0], :]
        n1 = VertexNormals[f[i][1], :]
        n2 = VertexNormals[f[i][2], :]
        # - solve least squares problem of th form Ax=b
        A = np.array(
            [
                [np.dot(e0[i, :], t), np.dot(e0[i, :], B), 0],
                [0, np.dot(e0[i, :], t), np.dot(e0[i, :], B)],
                [np.dot(e1[i, :], t), np.dot(e1[i, :], B), 0],
                [0, np.dot(e1[i, :], t), np.dot(e1[i, :], B)],
                [np.dot(e2[i, :], t), np.dot(e2[i, :], B), 0],
                [0, np.dot(e2[i, :], t), np.dot(e2[i, :], B)],
            ]
        )
        b = np.array(
            [
                np.dot(n2 - n1, t),
                np.dot(n2 - n1, B),
                np.dot(n0 - n2, t),
                np.dot(n0 - n2, B),
                np.dot(n1 - n0, t),
                np.dot(n1 - n0, B),
            ]
        )
        x = np.linalg.lstsq(A, b, None)
        FaceSFM[i] = np.array([[x[0][0], x[0][1]], [x[0][1], x[0][2]]])
        Kn[0][i] = np.dot(
            np.array([1, 0]), np.dot(FaceSFM[i], np.array([[1.0], [0.0]]))
        )
        # - calculate curvature per vertex
        wfp[i][0] = Acorner[i][0] / Avertex[f[i][0]]
        wfp[i][1] = Acorner[i][1] / Avertex[f[i][1]]
        wfp[i][2] = Acorner[i][2] / Avertex[f[i][2]]
        # - calculate new coordinate system and project the tensor
        for j in range(3):
            new_ku, new_kuv, new_kv = ProjectCurvatureTensor(
                t, B, nf, x[0][0], x[0][1], x[0][2], up[f[i][j], :], vp[f[i][j], :]
            )
            VertexSFM[f[i][j]] += np.dot(
                wfp[i][j], np.array([[new_ku, new_kuv], [new_kuv, new_kv]])
            )
    return FaceSFM, VertexSFM, wfp


def GetCurvaturesAndDerivatives(mesh):
    FaceNormals = CalcFaceNormals(mesh)
    (VertexNormals, Avertex, Acorner, up, vp) = CalcVertexNormals(mesh, FaceNormals)
    (FaceSFM, VertexSFM, wfp) = CalcCurvature(
        mesh, VertexNormals, FaceNormals, Avertex, Acorner, up, vp
    )
    [PrincipalCurvature, PrincipalDi1, PrincipalDi2] = getPrincipalCurvatures(
        mesh, VertexSFM, up, vp
    )
    return PrincipalCurvature, PrincipalDi1, PrincipalDi2


def CalcFaceNormals(mesh):
    v, f = mesh.vertices, mesh.faces
    e0, e1 = v[f[:, 2]] - v[f[:, 1]], v[f[:, 0]] - v[f[:, 2]]
    return normr(np.cross(e0, e1))


def normr0(X):
    if len(np.shape(X)) == 1:
        return X / np.abs(X)
    else:
        a = np.shape(X)[1]
        b = np.shape(X)[0]
        return (
            np.dot(
                np.reshape(
                    np.transpose(np.sqrt(1 / somme_colonnes(np.transpose(X**2)))),
                    (b, 1),
                ),
                np.ones((1, a)),
            )
            * X
        )


def normr(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def CalcVertexNormals(mesh, N):
    v, f = mesh.vertices, mesh.faces
    # - get all edge vectors
    e0 = np.array(v[f[:, 2], :] - v[f[:, 1], :])
    e1 = np.array(v[f[:, 0], :] - v[f[:, 2], :])
    e2 = np.array(v[f[:, 1], :] - v[f[:, 0], :])
    e0_norm = normr(e0)
    e1_norm = normr(e1)
    e2_norm = normr(e2)
    # - normalization procedure & calculate face area
    # - edge lengths
    de0 = np.sqrt((e0[:, 0]) ** 2 + (e0[:, 1]) ** 2 + (e0[:, 2]) ** 2)
    de1 = np.sqrt((e1[:, 0]) ** 2 + (e1[:, 1]) ** 2 + (e1[:, 2]) ** 2)
    de2 = np.sqrt((e2[:, 0]) ** 2 + (e2[:, 1]) ** 2 + (e2[:, 2]) ** 2)
    l2 = np.array([de0**2, de1**2, de2**2])
    l2 = np.transpose(l2)
    # - using ew to calculate the cot of the angles for the voronoi area
    # - ew is the triangle barycenter, I later check if its inside or outside
    # - the triangle
    ew = np.array(
        [
            l2[:, 0] * (l2[:, 1] + l2[:, 2] - l2[:, 0]),
            l2[:, 1] * (l2[:, 2] + l2[:, 0] - l2[:, 1]),
            l2[:, 2] * (l2[:, 0] + l2[:, 1] - l2[:, 2]),
        ]
    )
    s = (de0 + de1 + de2) / 2
    Af = np.sqrt(s * (s - de0) * (s - de1) * (s - de2))
    # - calculate weights
    Acorner = np.zeros((np.shape(f)[0], 3))
    Avertex = np.zeros((np.shape(v)[0], 1))
    # - calculate vertex normals
    VertexNormals, up, vp = (
        np.zeros((np.shape(v)[0], 3)),
        np.zeros((np.shape(v)[0], 3)),
        np.zeros((np.shape(v)[0], 3)),
    )
    # - calculate weights according to N.Max [1999]
    for i in range(np.shape(f)[0]):
        wfv1 = Af[i] / ((de1[i] ** 2) * (de2[i] ** 2))
        wfv2 = Af[i] / ((de0[i] ** 2) * (de2[i] ** 2))
        wfv3 = Af[i] / ((de1[i] ** 2) * (de0[i] ** 2))
        VertexNormals[f[i][0], :] += wfv1 * N[i, :]
        VertexNormals[f[i][1], :] += wfv2 * N[i, :]
        VertexNormals[f[i][2], :] += wfv3 * N[i, :]
        # - calculate areas for weights according to Meyer et al [2002]
        # - check if the triange is obtuse, right or acute
        if ew[0][i] <= 0:
            Acorner[i][1] = (
                -0.25 * l2[i][2] * Af[i] / (np.dot(e0[i, :], np.transpose(e2[i, :])))
            )
            Acorner[i][2] = (
                -0.25 * l2[i][1] * Af[i] / (np.dot(e0[i, :], np.transpose(e1[i, :])))
            )
            Acorner[i][0] = Af[i] - Acorner[i][2] - Acorner[i][1]
        elif ew[1][i] <= 0:
            Acorner[i][2] = (
                -0.25 * l2[i][0] * Af[i] / (np.dot(e1[i, :], np.transpose(e0[i, :])))
            )
            Acorner[i][0] = (
                -0.25 * l2[i][2] * Af[i] / (np.dot(e1[i, :], np.transpose(e2[i, :])))
            )
            Acorner[i][1] = Af[i] - Acorner[i][2] - Acorner[i][0]
        elif ew[2][i] <= 0:
            Acorner[i][0] = (
                -0.25 * l2[i][1] * Af[i] / (np.dot(e2[i, :], np.transpose(e1[i, :])))
            )
            Acorner[i][1] = (
                -0.25 * l2[i][0] * Af[i] / (np.dot(e2[i, :], np.transpose(e0[i, :])))
            )
            Acorner[i][2] = Af[i] - Acorner[i][1] - Acorner[i][0]
        else:
            ewscale = 0.5 * Af[i] / (ew[0][i] + ew[1][i] + ew[2][i])
            Acorner[i][0] = ewscale * (ew[1][i] + ew[2][i])
            Acorner[i][1] = ewscale * (ew[0][i] + ew[2][i])
            Acorner[i][2] = ewscale * (ew[1][i] + ew[0][i])
        Avertex[f[i][0]] += Acorner[i][0]
        Avertex[f[i][1]] += Acorner[i][1]
        Avertex[f[i][2]] += Acorner[i][2]
        # - calculate initial coordate system
        up[f[i][0], :] = e2_norm[i, :]
        up[f[i][1], :] = e0_norm[i, :]
        up[f[i][2], :] = e1_norm[i, :]
    VertexNormals = normr(VertexNormals)
    for i in range(np.shape(v)[0]):
        up[i, :] = np.cross(up[i, :], VertexNormals[i, :])
        up[i, :] = up[i, :] / np.linalg.norm(up[i, :])
        vp[i, :] = np.cross(VertexNormals[i, :], up[i, :])
    return VertexNormals, Avertex, Acorner, up, vp


def getPrincipalCurvatures(mesh, VertexSFM, up, vp):
    v, f = mesh.vertices, mesh.faces
    PrincipalCurvature = np.zeros((2, np.shape(v)[0]))
    PrincipalDi1, PrincipalDi2 = [
        np.zeros((np.shape(v)[0], 3)),
        np.zeros((np.shape(v)[0], 3)),
    ]
    for i in range(np.shape(v)[0]):
        npp = np.cross(up[i, :], vp[i, :])
        r_old_u, r_old_v = RotateCoordinateSystem(up[i, :], vp[i, :], npp)
        ku = VertexSFM[i][0][0]
        kuv = VertexSFM[i][0][1]
        kv = VertexSFM[i][1][1]
        c, s, tt = 1, 0, 0
        if kuv != 0:
            h = 0.5 * (kv - ku) / kuv
            if h < 0:
                tt = 1 / (h - np.sqrt(1 + h**2))
            else:
                tt = 1 / (h + np.sqrt(1 + h**2))
            c = 1 / np.sqrt(1 + tt**2)
            s = tt * c
        k1 = ku - tt * kuv
        k2 = kv + tt * kuv
        if abs(k1) >= abs(k2):
            PrincipalDi1[i, :] = c * r_old_u - s * r_old_v
        else:
            [k1, k2] = [k2, k1]
            PrincipalDi1[i, :] = c * r_old_u + s * r_old_v
        PrincipalDi2[i, :] = np.cross(npp, PrincipalDi1[i, :])
        PrincipalCurvature[0][i] = k1
        PrincipalCurvature[1][i] = k2
        if np.isnan(k1) or np.isnan(k2):
            print("Nan")
    return PrincipalCurvature, PrincipalDi1, PrincipalDi2


def somme_colonnes(X):
    xx = list()
    for i in range(np.shape(X)[1]):
        xx.append(sum(X[:, i]))
    return np.array(xx)


def ProcessMeshFile(mat_file):
    keys = []
    for key in mat_file:
        keys.append(key)
    svs_all = mat_file[keys[3]]
    sts_all = mat_file[keys[4]] - 1
    part_mask1 = mat_file[keys[5]] == 1
    part_mask2 = mat_file[keys[5]] == 2
    sts1 = sts_all[part_mask1.ravel()]
    sts2 = sts_all[part_mask2.ravel()]
    vertices_part1 = svs_all[np.unique(sts1)]
    vertices_part2 = svs_all[np.unique(sts2)]
    tree_part2 = cKDTree(vertices_part2)
    distances, _ = tree_part2.query(vertices_part1, distance_upper_bound=1e-6)
    mask = np.isinf(distances)
    filtered_vertices = vertices_part1[mask]
    vertex_map = {tuple(v): i for i, v in enumerate(filtered_vertices)}
    kept_vertex_indices = np.array([vertex_map[tuple(v)] for v in filtered_vertices])
    new_triangles = []
    for tri in sts1:
        if all(tuple(svs_all[v]) in vertex_map for v in tri):
            new_tri = [vertex_map[tuple(svs_all[v])] for v in tri]
            new_triangles.append(new_tri)
    new_triangles = np.array(new_triangles)
    mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=new_triangles)
    return mesh


def anato_clean_group(parent_path, group_str, file_str, ext_str):
    paths = GetFilteredMeshPaths(parent_path, group_str, file_str)
    total_files = len(paths)
    out = display(progress(0, total_files - 1), display_id=True)
    t = 0
    for i in range(total_files):
        progress(i, total_files)
        os.chdir(paths[i][0])
        mat_file = scipy.io.loadmat(paths[i][1])
        mesh = ProcessMeshFile(mat_file)
        new_file_name = paths[i][1][:-4] + ext_str + ".parquet"
        vertices = pd.DataFrame(mesh.vertices, columns=["X", "Y", "Z"])
        triangles = pd.DataFrame(mesh.faces, columns=["T1", "T2", "T3"])
        concatenated_data = pd.concat([vertices, triangles], axis=1)
        concatenated_data.to_parquet(new_file_name, compression="gzip", index=False)
        t += 1
        out.update(progress(t, total_files))
        print(f"""Saved {paths[i][1][:-4] + ext_str}.parquet to {paths[i][0]}""")


def anato_curv_group_wRemoval(parent_path, group_str, file_str, ext_str):
    paths = GetFilteredMeshPaths(parent_path, group_str, file_str)
    total_files = len(paths)
    out = display(progress(0, total_files - 1), display_id=True)
    t = 0
    for i in range(total_files):
        progress(i, total_files)
        os.chdir(paths[i][0])
        mat_file = scipy.io.loadmat(paths[i][1])
        mesh = ProcessMeshFile(mat_file)
        try:
            PrincipalCurvatures, PrincipalDi1, PrincipalDi2 = (
                GetCurvaturesAndDerivatives(mesh)
            )
        except np.linalg.LinAlgError as e:
            print(
                f"LinAlg Error: SVD did not converge for file {paths[i][1]}. Skipping this scan."
            )
            continue  # - skip this iteration if SVD did not converge
        PCS_array = np.array(PrincipalCurvatures.T)
        new_file_name = paths[i][1][:-4] + ext_str + ".parquet"
        vertices = pd.DataFrame(mesh.vertices, columns=["X", "Y", "Z"])
        triangles = pd.DataFrame(mesh.faces, columns=["T1", "T2", "T3"])
        curvatures = pd.DataFrame(PCS_array, columns=["K1", "K2"])
        concatenated_data = pd.concat([vertices, triangles, curvatures], axis=1)
        concatenated_data.to_parquet(new_file_name, compression="gzip", index=False)
        t += 1
        out.update(progress(t, total_files))
        print(f"""Saved {paths[i][1][:-4] + ext_str}.parquet to {paths[i][0]}""")


def anato_curv_group_old_woRemoval(parent_path, group_str, file_str, ext_str):
    paths = GetFilteredMeshPaths(parent_path, group_str, file_str)
    total_files = len(paths)
    out = display(progress(0, total_files - 1), display_id=True)
    t = 0
    for i in range(total_files):
        progress(i, total_files)
        os.chdir(paths[i][0])
        mat_file = scipy.io.loadmat(paths[i][1])
        for vkey in mat_file:
            if "_surface_vertices" in vkey:
                svs = mat_file[vkey]
        for tkey in mat_file:
            if "_surface_triangles" in tkey and "_surface_triangles_" not in tkey:
                sts = mat_file[tkey] - 1
        mesh = trimesh.Trimesh(vertices=svs, faces=sts)
        try:
            PrincipalCurvatures, PrincipalDi1, PrincipalDi2 = (
                GetCurvaturesAndDerivatives(mesh)
            )
        except np.linalg.LinAlgError as e:
            print(
                f"LinAlg Error: SVD did not converge for file {paths[i][1]}. Skipping this scan."
            )
            continue  # - skip this iteration if SVD did not converge
        PCS_array = np.array(PrincipalCurvatures.T)
        new_file_name = paths[i][1][:-4] + ext_str + ".parquet"
        vertices = pd.DataFrame(mesh.vertices, columns=["X", "Y", "Z"])
        triangles = pd.DataFrame(mesh.faces, columns=["T1", "T2", "T3"])
        curvatures = pd.DataFrame(PCS_array, columns=["K1", "K2"])
        concatenated_data = pd.concat([vertices, triangles, curvatures], axis=1)
        concatenated_data.to_parquet(new_file_name, compression="gzip", index=False)
        t += 1
        out.update(progress(t, total_files))
        print(f"""Saved {paths[i][1][:-4] + ext_str}.parquet to {paths[i][0]}""")


def convert_stl_folder_to_mat(stl_folder, mat_folder):
    """
    Convert all .stl meshes in `stl_folder` into .mat files in `mat_folder`,
    saving variables named:
      <basename>_surface_vertices    (Nx3 float)
      <basename>_surface_triangles   (Mx3 int, 1-based)

    Those .mat files can then be re-imported as:

        mat = scipy.io.loadmat(path)
        svs = mat[<basename>+'_surface_vertices']
        sts = mat[<basename>+'_surface_triangles'] - 1
        mesh = trimesh.Trimesh(vertices=svs, faces=sts)
    """
    os.makedirs(mat_folder, exist_ok=True)

    for fname in os.listdir(stl_folder):
        if not fname.lower().endswith(".stl"):
            continue

        base, _ = os.path.splitext(fname)
        stl_path = os.path.join(stl_folder, fname)
        mat_path = os.path.join(mat_folder, base + ".mat")

        # load the triangular mesh
        mesh = trimesh.load(stl_path, force="mesh")
        verts = mesh.vertices
        faces = mesh.faces + 1  # convert to 1‐based indexing

        mdict = {f"{base}_surface_vertices": verts, f"{base}_surface_triangles": faces}
        savemat(mat_path, mdict)
