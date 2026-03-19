# -*- coding: utf-8 -*-
"""
ODB geometric analysis pipeline.

Two stages, run on separate machines:

1. **Abaqus extraction** (randi / HPC via ``abaqus python``):
   Use the standalone ``abaqus_extract_odb.py`` script to read field data
   from ODB files and save numpy arrays. Cannot run locally — requires
   Abaqus Python.

2. **Local post-processing** (workstation):
   ``parse_inp_surface()``, ``extract_odb_geometry()``, ``smooth_mesh()``,
   ``odb_geometric_analysis()`` handle INP parsing, mesh deformation,
   smoothing, curvature computation, and parquet export for the
   anato-shape manifold pipeline.

@author: kameelkhabaz
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
import trimesh.smoothing
from scipy.spatial import cKDTree

from anato_curv import GetCurvaturesAndDerivatives


# ===========================================================================
# Abaqus extraction (runs on HPC via `abaqus python`)
# ===========================================================================


def extract_displacements(odb_path, step_name, instance_name, nset_name,
                          output_dir, max_frames=None):
    """Extract per-frame displacement arrays from an ODB file.

    **Requires the Abaqus Python environment** (``odbAccess``).
    Must be run on HPC (e.g. randi) via ``abaqus python``.

    .. note::
        Prefer the standalone ``abaqus_extract_odb.py`` script for HPC use.
        It supports additional options (``--field``, ``--coords``).

    Parameters
    ----------
    odb_path : str
        Path to the .odb file.
    step_name : str
        Analysis step name (e.g. ``"Step-1"``).
    instance_name : str
        Part instance name (e.g. ``"PART-1-1"``). Uppercased automatically.
    nset_name : str
        Node set name (e.g. ``"OUTER"``). Uppercased automatically.
    output_dir : str
        Directory to write output files.
    max_frames : int or None
        Extract only the first N frames. None extracts all.

    Outputs
    -------
    Writes to ``output_dir``:
        - ``node_labels.npy`` — (M,) int array of node labels
        - ``U_frame_XX.npy`` — (M, 3) float displacement per frame
        - ``metadata.json`` — extraction parameters and frame count
    """
    from odbAccess import openOdb  # only available in Abaqus Python

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    instance_name = instance_name.upper()
    nset_name = nset_name.upper()

    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step = odb.steps[step_name]
        instance = odb.rootAssembly.instances[instance_name]
        nset = instance.nodeSets[nset_name]

        frames = step.frames
        n_frames = len(frames)
        if max_frames is not None:
            n_frames = min(n_frames, max_frames)

        if n_frames == 0:
            print("Warning: no frames found in step '%s'." % step_name)
            odb.close()
            return

        pad = max(2, len(str(n_frames - 1)))
        fmt = 'U_frame_%0' + str(pad) + 'd.npy'

        print("Extracting %d frames..." % n_frames)

        node_labels = None
        n_nodes = 0
        for i in range(n_frames):
            field = frames[i].fieldOutputs['U']
            subset = field.getSubset(region=nset)

            frame_labels = np.array([v.nodeLabel for v in subset.values])
            frame_disp = np.array([v.dataDouble for v in subset.values])

            sort_idx = np.argsort(frame_labels)
            sorted_labels = frame_labels[sort_idx]
            sorted_disp = frame_disp[sort_idx]

            if node_labels is None:
                node_labels = sorted_labels
                n_nodes = len(node_labels)
                np.save(os.path.join(output_dir, 'node_labels.npy'), node_labels)

            np.save(os.path.join(output_dir, fmt % i), sorted_disp)
            print("  Frame %d/%d (%d nodes)" % (i + 1, n_frames, n_nodes))

        metadata = {
            'odb_path': os.path.abspath(odb_path),
            'step_name': step_name,
            'instance_name': instance_name,
            'nset_name': nset_name,
            'field_name': 'U',
            'n_frames': n_frames,
            'n_nodes': n_nodes,
            'frame_pad': pad,
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Done. Output written to %s" % output_dir)
    finally:
        odb.close()


# ===========================================================================
# Local post-processing (runs on workstation)
# ===========================================================================


def parse_inp_surface(inp_path, node_set_name="Outer"):
    """Parse an Abaqus INP file and extract the outer surface triangulation.

    For each C3D4/C3D10 tetrahedron, if exactly 3 of its 4 corner vertices
    belong to the specified node set, those 3 form a surface triangle.
    Normals are oriented outward using the 4th (inner) vertex.

    Parameters
    ----------
    inp_path : str or Path
        Path to the ``.inp`` file.
    node_set_name : str
        Name of the node set defining the outer surface (case-insensitive).

    Returns
    -------
    surface_vertices : ndarray, shape (V, 3)
    surface_triangles : ndarray, shape (F, 3)
        0-indexed into ``surface_vertices``.
    step_name : str
    instance_name : str
    total_frames : int
    """
    inp_path = Path(inp_path)
    with open(inp_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    node_start = None
    elem_start = None
    elem_type = None

    for i, line in enumerate(lines):
        low = line.strip().lower()
        if node_start is None and (low == "*node" or low.startswith("*node,")):
            node_start = i + 1
        elif elem_start is None and low.startswith("*element,"):
            elem_start = i + 1
            m = re.search(r"type\s*=\s*(\w+)", line, re.IGNORECASE)
            if m:
                elem_type = m.group(1).upper()

    if node_start is None:
        raise ValueError("Could not find *Node section in INP file")
    if elem_start is None:
        raise ValueError("Could not find *Element section in INP file")

    # Read nodes
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

    # Read elements
    elem_rows = []
    for i in range(elem_start, len(lines)):
        if lines[i].strip().startswith("*"):
            break
        parts = lines[i].split(",")
        vals = [int(p.strip()) for p in parts if p.strip()]
        elem_rows.append(vals)
    elem_arr = np.array(elem_rows)
    elem_connectivity = elem_arr[:, 1:]

    # Node-ID to 0-based index map
    if np.array_equal(node_ids, np.arange(1, len(node_ids) + 1)):
        id_to_idx = None
    else:
        id_to_idx = np.empty(node_ids.max() + 1, dtype=int)
        id_to_idx[:] = -1
        id_to_idx[node_ids] = np.arange(len(node_ids))

    def _remap(ids):
        if id_to_idx is None:
            return ids - 1
        return id_to_idx[ids]

    # Parse node set
    nset_target = node_set_name.lower()
    nset_start = None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("*nset") and f"nset={nset_target}" in low.replace(" ", ""):
            nset_start = i + 1
            break

    if nset_start is None:
        raise ValueError(f"Could not find *Nset with nset={node_set_name}")

    nset_ids = []
    for i in range(nset_start, len(lines)):
        if lines[i].strip().startswith("*"):
            break
        for p in lines[i].split(","):
            p = p.strip()
            if p:
                nset_ids.append(int(p))
    nset_set = set(nset_ids)

    # Parse metadata
    step_name = ""
    instance_name = ""
    total_frames = 0

    for line in lines:
        low = line.strip().lower()
        if not step_name and low.startswith("*step"):
            m = re.search(r"name\s*=\s*([^,\s]+)", line, re.IGNORECASE)
            if m:
                step_name = m.group(1)
        if not instance_name and low.startswith("*instance"):
            m = re.search(r"name\s*=\s*([^,\s]+)", line, re.IGNORECASE)
            if m:
                instance_name = m.group(1).upper()
        if (
            not total_frames
            and "*output" in low
            and "field" in low
            and "number interval" in low
        ):
            m = re.search(r"number\s+interval\s*=\s*(\d+)", line, re.IGNORECASE)
            if m:
                total_frames = int(m.group(1))

    # Extract surface triangles (corner nodes only for C3D10)
    corners = elem_connectivity[:, :4]

    triangles = []
    inner_points = []
    for row in corners:
        mask = np.array([nid in nset_set for nid in row])
        if mask.sum() == 3:
            triangles.append(row[mask])
            inner_points.append(row[~mask][0])

    triangles = np.array(triangles, dtype=int)
    inner_points = np.array(inner_points, dtype=int)

    # Orient normals outward (away from inner vertex)
    tri_idx = _remap(triangles)
    inner_idx = _remap(inner_points)

    v0 = node_coords[tri_idx[:, 0]]
    v1 = node_coords[tri_idx[:, 1]]
    v2 = node_coords[tri_idx[:, 2]]
    centers = (v0 + v1 + v2) / 3.0

    normals = np.cross(v2 - v1, v0 - v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms

    to_inner = node_coords[inner_idx] - centers
    flip = np.sum(normals * to_inner, axis=1) > 0
    tri_idx[flip] = tri_idx[flip][:, ::-1]

    # Re-index to surface-only vertices
    unique_ids = np.unique(tri_idx)
    surface_vertices = node_coords[unique_ids]
    surf_remap = np.empty(len(node_coords), dtype=int)
    surf_remap[:] = -1
    surf_remap[unique_ids] = np.arange(len(unique_ids))
    surface_triangles = surf_remap[tri_idx]

    return surface_vertices, surface_triangles, step_name, instance_name, total_frames


def extract_odb_geometry(inp_path, extraction_dir, output_dir=None):
    """Build per-frame deformed surface meshes from INP + extracted displacements.

    Parameters
    ----------
    inp_path : str or Path
        Path to the ``.inp`` file.
    extraction_dir : str or Path
        Directory containing ``node_labels.npy``, frame ``.npy`` files,
        and ``metadata.json`` produced by ``abaqus_extract_odb.py``.
    output_dir : str or Path or None
        Where to save per-frame STL files. Defaults to ``extraction_dir / stl``.

    Returns
    -------
    meshes : list of trimesh.Trimesh
    surface_triangles : ndarray
    """
    inp_path = Path(inp_path)
    extraction_dir = Path(extraction_dir)
    if output_dir is None:
        output_dir = extraction_dir / "stl"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_verts, surface_tris, _, _, _ = parse_inp_surface(inp_path)

    with open(extraction_dir / "metadata.json") as f:
        meta = json.load(f)
    n_frames = meta["n_frames"]
    field_name = meta["field_name"]
    pad = meta["frame_pad"]

    extracted_labels = np.load(extraction_dir / "node_labels.npy")

    # Re-parse INP node table for label-to-coordinate mapping
    with open(inp_path, "r") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    node_start = None
    for i, line in enumerate(raw_lines):
        low = line.strip().lower()
        if low == "*node" or low.startswith("*node,"):
            node_start = i + 1
            break

    all_node_rows = []
    for i in range(node_start, len(raw_lines)):
        if raw_lines[i].strip().startswith("*"):
            break
        parts = raw_lines[i].split(",")
        if len(parts) >= 4:
            all_node_rows.append([float(p.strip()) for p in parts[:4]])
    all_node_arr = np.array(all_node_rows)
    all_node_ids = all_node_arr[:, 0].astype(int)
    all_node_coords = all_node_arr[:, 1:4]

    label_to_idx = {int(label): idx for idx, label in enumerate(all_node_ids)}

    # Map extracted node labels to surface vertex ordering via nearest-neighbor
    full_coords_for_extracted = all_node_coords[
        np.array([label_to_idx[int(lb)] for lb in extracted_labels])
    ]
    tree = cKDTree(full_coords_for_extracted)
    _, indices = tree.query(surface_verts)

    meshes = []
    for frame_i in range(n_frames):
        disp_file = extraction_dir / f"{field_name}_frame_{frame_i:0{pad}d}.npy"
        displacements = np.load(disp_file)
        deformed_verts = surface_verts + displacements[indices]

        mesh = trimesh.Trimesh(
            vertices=deformed_verts, faces=surface_tris, process=False
        )
        mesh.export(str(output_dir / f"frame_{frame_i:0{pad}d}.stl"))
        meshes.append(mesh)

    return meshes, surface_tris


def smooth_mesh(
    mesh, target_elem_length=1.625, smooth_iterations=15, smooth_lambda=0.5
):
    """Decimate and Laplacian-smooth a triangle mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    target_elem_length : float
        Target edge length for equilateral triangles after decimation.
    smooth_iterations : int
    smooth_lambda : float

    Returns
    -------
    trimesh.Trimesh
    """
    target_area = (np.sqrt(3) / 4) * target_elem_length**2
    mean_area = mesh.area_faces.mean()

    if mean_area < target_area:
        ratio = mean_area / target_area
        target_faces = max(4, int(len(mesh.faces) * ratio))
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

    trimesh.smoothing.filter_laplacian(
        mesh, lamb=smooth_lambda, iterations=smooth_iterations
    )
    return mesh


def odb_geometric_analysis(
    inp_path, extraction_dir, output_dir=None, smooth_params=None
):
    """Extract deformed geometries, smooth, compute curvatures, save as parquet.

    Produces parquet files (``output_dir/mesh/frame_XX.parquet``) in the
    format expected by ``GetAnatoMeshResults``::

        all_results, results_dict = am.GetAnatoMeshResults(
            output_dir, group_str, file_str, point_removal,
            quantities, m_set=m_sets, parallel=True,
        )

    Parameters
    ----------
    inp_path : str or Path
        Path to the ``.inp`` file.
    extraction_dir : str or Path
        Directory with ODB extraction outputs (from ``extract_displacements()``).
    output_dir : str or Path or None
        Root directory for results. Defaults to ``extraction_dir``.
    smooth_params : dict or None
        Override defaults for :func:`smooth_mesh`.
    """
    extraction_dir = Path(extraction_dir)
    if output_dir is None:
        output_dir = extraction_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if smooth_params is None:
        smooth_params = {}

    with open(extraction_dir / "metadata.json") as f:
        meta = json.load(f)
    n_frames = meta["n_frames"]
    pad = meta["frame_pad"]

    mesh_dir = output_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Check if all parquet files already exist
    existing = [mesh_dir / f"frame_{i:0{pad}d}.parquet" for i in range(n_frames)]
    if all(p.exists() for p in existing):
        print(f"All {n_frames} parquet files already exist in {mesh_dir}, skipping.")
        return

    meshes, _ = extract_odb_geometry(
        inp_path, extraction_dir, output_dir=output_dir / "stl_raw"
    )

    for i, mesh in enumerate(meshes):
        print(f"Processing frame {i + 1}/{len(meshes)}...")

        smoothed = smooth_mesh(mesh.copy(), **smooth_params)
        principal_curvatures, _, _ = GetCurvaturesAndDerivatives(smoothed)
        pcs = principal_curvatures.T

        vertices = pd.DataFrame(smoothed.vertices, columns=["X", "Y", "Z"])
        triangles = pd.DataFrame(smoothed.faces, columns=["T1", "T2", "T3"])
        curvatures = pd.DataFrame(pcs, columns=["K1", "K2"])
        data = pd.concat([vertices, triangles, curvatures], axis=1)
        data.to_parquet(mesh_dir / f"frame_{i:0{pad}d}.parquet", compression="gzip", index=False)

    print(f"Saved {len(meshes)} parquet files to {mesh_dir}")
