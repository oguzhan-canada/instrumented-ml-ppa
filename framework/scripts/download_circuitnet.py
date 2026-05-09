#!/usr/bin/env python3
"""
scripts/download_circuitnet.py

Download and prepare CircuitNet dataset (N28 and N14).
- Clones the CircuitNet repository
- Runs fix_def_instances.py to correct DEF instance names
- Extracts congestion maps for spatial features

Usage:
  python scripts/download_circuitnet.py --output data/raw/circuitnet/

Reference: https://github.com/circuitnet/CircuitNet
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


CIRCUITNET_REPO = "https://github.com/circuitnet/CircuitNet.git"


def clone_repo(output_dir: Path):
    """Clone CircuitNet repo."""
    if (output_dir / ".git").exists():
        print(f"CircuitNet already cloned at {output_dir}")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning CircuitNet to {output_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", CIRCUITNET_REPO, str(output_dir)],
        check=True,
    )
    print("Clone complete.")


def run_def_fix(circuitnet_dir: Path):
    """Run fix_def_instances.py if it exists."""
    fix_script = Path("scripts/fix_def_instances.py")
    if fix_script.exists():
        print("Running DEF instance name fix...")
        subprocess.run(
            [sys.executable, str(fix_script), "--input-dir", str(circuitnet_dir)],
            check=True,
        )
        print("DEF fix complete.")
    else:
        print(f"WARNING: {fix_script} not found. DEF fixes may be needed manually.")


def extract_congestion_maps(circuitnet_dir: Path, output_dir: Path):
    """
    Extract congestion/density maps from CircuitNet data.
    Looks for pre-computed feature files in the dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # CircuitNet stores features as numpy arrays in organized directories
    feature_dirs = [
        circuitnet_dir / "Dataset" / "N28",
        circuitnet_dir / "Dataset" / "N14",
    ]

    for fdir in feature_dirs:
        if not fdir.exists():
            print(f"  Dataset directory not found: {fdir}")
            continue

        tech = fdir.name  # N28 or N14
        npy_files = list(fdir.rglob("*.npy"))
        print(f"  Found {len(npy_files)} feature files in {fdir}")

        for npy_file in tqdm(npy_files, desc=f"Processing {tech}"):
            try:
                data = np.load(npy_file, allow_pickle=True)
                # Resize to 100x100 if needed
                if data.ndim == 2 and data.shape[0] > 0:
                    from scipy.ndimage import zoom
                    if data.shape != (100, 100):
                        zy = 100.0 / data.shape[0]
                        zx = 100.0 / data.shape[1]
                        data = zoom(data, (zy, zx))
                    out_path = output_dir / f"{tech}_{npy_file.stem}.npy"
                    np.save(out_path, data.astype(np.float32))

                    manifest_rows.append({
                        "sample_id": f"circuitnet_{tech}_{npy_file.stem}",
                        "source": "circuitnet",
                        "design_name": npy_file.stem,
                        "tech_node": tech,
                        "spatial_path": str(out_path),
                        "has_delay": False,
                        "has_power": False,
                        "run_status": "success",
                        "tool_version": "circuitnet_v1",
                    })
            except Exception as e:
                print(f"    SKIP {npy_file.name}: {e}")

    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Download and prepare CircuitNet")
    parser.add_argument("--output", type=str, default="data/raw/circuitnet",
                        help="Output directory")
    parser.add_argument("--spatial-output", type=str, default="features/spatial/circuitnet",
                        help="Output for congestion maps")
    parser.add_argument("--skip-clone", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    spatial_output = Path(args.spatial_output)

    # Step 1: Clone
    if not args.skip_clone:
        clone_repo(output_dir)

    # Step 2: Fix DEF files
    run_def_fix(output_dir)

    # Step 3: Extract congestion maps
    rows = extract_congestion_maps(output_dir, spatial_output)
    if rows:
        import pandas as pd
        manifest_path = output_dir / "circuitnet_manifest_partial.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"Saved partial manifest: {manifest_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
