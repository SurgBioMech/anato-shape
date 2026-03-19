"""
Extract field data from Abaqus ODB files.

Must be run via ``abaqus python`` (Python 2.7 environment).

Usage:
    abaqus python abaqus_extract_odb.py <odb_path> <step_name> <instance_name> \\
        <nset_name> <output_dir> [--field U] [--frames N] [--coords]

Arguments:
    odb_path       Path to the .odb file
    step_name      Name of the analysis step (e.g. "Step-1")
    instance_name  Name of the part instance (e.g. "PART-1-1")
    nset_name      Name of the node set (e.g. "OUTER")
    output_dir     Directory for output .npy files

Options:
    --field NAME   Field output to extract (default: U)
    --frames N     Maximum number of frames to extract
    --coords       Also extract initial node coordinates
"""

import sys
import os
import json

import numpy as np
from odbAccess import openOdb
from abaqusConstants import NODAL


def _get_field_data(subset):
    """Extract double-precision values array from a field output subset.

    Raises AttributeError if the ODB was not run with double precision.
    """
    return np.array([v.dataDouble for v in subset.values])


def extract_fields(odb_path, step_name, instance_name, nset_name,
                   output_dir, field_name='U', max_frames=None,
                   extract_coords=False):
    """Extract per-frame field arrays from an ODB file.

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
    field_name : str
        Field output variable name (default ``"U"`` for displacement).
    max_frames : int or None
        Extract only the first *N* frames.  ``None`` extracts all.
    extract_coords : bool
        If True, also save initial node coordinates as ``coordinates.npy``.

    Outputs
    -------
    Writes to *output_dir*:

    - ``node_labels.npy``             -- (M,) int, sorted node labels
    - ``{field}_frame_XXXX.npy``      -- (M, D) float, field values per frame
    - ``coordinates.npy``             -- (M, 3) float, initial coords (if ``--coords``)
    - ``metadata.json``               -- extraction parameters and file naming info
    """
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
            return

        # Dynamic zero-padding (minimum 2 digits)
        pad = max(2, len(str(n_frames - 1)))
        fmt = '%s_frame_%0' + str(pad) + 'd.npy'

        print("Extracting %d frames of '%s'..." % (n_frames, field_name))

        node_labels = None
        n_nodes = 0
        for i in range(n_frames):
            field = frames[i].fieldOutputs[field_name]
            subset = field.getSubset(region=nset, position=NODAL)

            frame_labels = np.array([v.nodeLabel for v in subset.values])
            frame_data = _get_field_data(subset)

            sort_idx = np.argsort(frame_labels)
            sorted_labels = frame_labels[sort_idx]
            sorted_data = frame_data[sort_idx]

            if node_labels is None:
                node_labels = sorted_labels
                n_nodes = len(node_labels)
                np.save(os.path.join(output_dir, 'node_labels.npy'), node_labels)

            np.save(os.path.join(output_dir, fmt % (field_name, i)), sorted_data)
            print("  Frame %d/%d (%d nodes)" % (i + 1, n_frames, n_nodes))

        if extract_coords:
            nodes = nset.nodes
            coord_labels = np.array([n.label for n in nodes])
            coords = np.array([n.coordinates for n in nodes])
            sort_idx = np.argsort(coord_labels)
            np.save(os.path.join(output_dir, 'coordinates.npy'), coords[sort_idx])
            print("  Saved initial coordinates (%d nodes)" % len(coords))

        metadata = {
            'odb_path': os.path.abspath(odb_path),
            'step_name': step_name,
            'instance_name': instance_name,
            'nset_name': nset_name,
            'field_name': field_name,
            'n_frames': n_frames,
            'n_nodes': n_nodes,
            'frame_pad': pad,
            'has_coordinates': extract_coords,
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Done. Output in %s" % output_dir)
    finally:
        odb.close()


def _pop_flag(args, flag, has_value=True, default=None):
    """Remove --flag [value] from args list in place and return the value."""
    if flag not in args:
        return default
    idx = args.index(flag)
    args.pop(idx)
    if not has_value:
        return True
    return args.pop(idx)


if __name__ == '__main__':
    args = list(sys.argv[1:])

    # Parse optional flags first (removes them from args list)
    max_frames = _pop_flag(args, '--frames')
    if max_frames is not None:
        max_frames = int(max_frames)
    field_name = _pop_flag(args, '--field', default='U')
    extract_coords = _pop_flag(args, '--coords', has_value=False, default=False)

    if len(args) != 5:
        print(__doc__)
        sys.exit(1)

    extract_fields(
        odb_path=args[0],
        step_name=args[1],
        instance_name=args[2],
        nset_name=args[3],
        output_dir=args[4],
        field_name=field_name,
        max_frames=max_frames,
        extract_coords=extract_coords,
    )
