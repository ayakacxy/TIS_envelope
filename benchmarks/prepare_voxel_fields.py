"""Reconstruct two carrier fields from a GNN_TIS voxel leadfield and CSV.

The leadfield is a C-order (3 * voxels, non-reference electrodes) array in
V/m per mA. Names include the reference in the first position; the SFS CSV
contains electrode,current_mA,frequency_hz columns. No source files are changed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io


def _name(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value.decode() if isinstance(value, bytes) else str(value)


def _digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leadfield", required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--currents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-nifti", help="previously calculated scalar envelope, optional")
    parser.add_argument("--grid-indices", help="flat indices corresponding to leadfield rows")
    args = parser.parse_args()
    if bool(args.reference_nifti) != bool(args.grid_indices):
        parser.error("--reference-nifti and --grid-indices must be given together")

    leadfield = np.load(args.leadfield, mmap_mode="r")
    names = [_name(v) for v in scipy.io.loadmat(args.names)["electrode_names"].ravel()]
    if leadfield.ndim != 2 or leadfield.shape[0] % 3 or len(names) != leadfield.shape[1] + 1:
        raise ValueError("leadfield shape and electrode names do not match")
    if len(set(names)) != len(names):
        raise ValueError("electrode names are not unique")
    lookup = {name: i for i, name in enumerate(names[1:])}
    with open(args.currents, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frequencies = sorted({row["frequency_hz"] for row in rows}, key=float)
    if len(frequencies) != 2:
        raise ValueError("the current CSV must contain exactly two carrier frequencies")
    currents = np.zeros((leadfield.shape[1], 2), dtype=np.float32)
    for row in rows:
        electrode = row["electrode"]
        if electrode not in lookup:
            raise ValueError(f"electrode {electrode!r} is not a non-reference leadfield column")
        currents[lookup[electrode], frequencies.index(row["frequency_hz"])] += float(row["current_mA"])
    active = np.flatnonzero(np.any(currents != 0, axis=1))
    if not len(active):
        raise ValueError("no active currents")
    # This construction matches run_voxel_modified_j.py: (field[:, pos] @
    # currents).reshape(-1, 3, 2). It is done once, outside benchmark timing.
    projected = (leadfield[:, active] @ currents[active]).reshape(-1, 3, 2)
    e1 = np.ascontiguousarray(projected[:, :, 0])
    e2 = np.ascontiguousarray(projected[:, :, 1])
    if not np.isfinite(e1).all() or not np.isfinite(e2).all():
        raise ValueError("reconstructed fields contain non-finite values")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p1, p2 = output / "e1.npy", output / "e2.npy"
    np.save(p1, e1)
    np.save(p2, e2)

    norm1, norm2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
    dot = np.sum(e1 * e2, axis=1)
    oriented = np.where((dot < 0)[:, None], -e1, e1)
    close = np.all(np.isclose(oriented, e2, rtol=1e-5, atol=1e-8), axis=1)
    exact = np.all(oriented == e2, axis=1)
    relative_gap = np.linalg.norm(oriented - e2, axis=1) / np.maximum(np.maximum(norm1, norm2), 1e-30)
    metadata = {
        "leadfield": str(Path(args.leadfield).resolve()),
        "electrode_names": str(Path(args.names).resolve()),
        "currents_csv": str(Path(args.currents).resolve()),
        "currents_csv_sha256": _digest(args.currents),
        "reference_electrode": names[0],
        "frequencies_hz": frequencies,
        "frequency_net_currents_mA": currents.sum(axis=0, dtype=np.float64).tolist(),
        "active_electrodes": int(len(active)),
        "voxels": int(len(e1)),
        "units": "V/m",
        "e1_sha256": _digest(p1),
        "e2_sha256": _digest(p2),
        "e1_magnitude_percentiles_V_per_m": np.percentile(norm1, [0, 50, 90, 99, 100]).tolist(),
        "e2_magnitude_percentiles_V_per_m": np.percentile(norm2, [0, 50, 90, 99, 100]).tolist(),
        "oriented_exact_equal_count": int(exact.sum()),
        "oriented_old_isclose_count": int(close.sum()),
        "oriented_relative_gap_below_1e-4_count": int((relative_gap < 1e-4).sum()),
        "oriented_relative_gap_percentiles": np.percentile(relative_gap, [0, 1, 50, 99, 100]).tolist(),
    }
    if args.reference_nifti:
        import nibabel as nib
        # Import locally so the script can run from the repository root.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from fast_tis_envelope.backends.numpy_backend import envelop_numpy
        indices = np.load(args.grid_indices)
        image = np.asarray(nib.load(args.reference_nifti).dataobj, dtype=np.float32).reshape(-1)
        if len(indices) != len(e1):
            raise ValueError("grid indices do not match the number of field vectors")
        previous = image[indices]
        actual = envelop_numpy(e1, e2)
        diff = np.abs(actual.astype(np.float64) - previous.astype(np.float64))
        metadata["reference_nifti"] = str(Path(args.reference_nifti).resolve())
        metadata["previous_envelope_comparison"] = {
            "max_abs_error_V_per_m": float(diff.max()),
            "mean_abs_error_V_per_m": float(diff.mean()),
            "p99_abs_error_V_per_m": float(np.percentile(diff, 99)),
        }
    metadata_path = output / "field_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
