#!/usr/bin/env python3
"""
scripts/download_openabc.py

Download and prepare the OpenABC-D dataset.
- Downloads the 19 GB dataset from NYU archive (resumable)
- Extracts AIG graphs
- Converts to PyG format for our pipeline

Usage:
  python scripts/download_openabc.py --output data/raw/openabc_d/

Note: The full dataset is ~19 GB. Download may take significant time.
      The script supports resuming interrupted downloads.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# OpenABC-D dataset URL (Zenodo — ML-ready PyTorch files, ~18.6 GB)
OPENABC_URL = "https://zenodo.org/records/6399454/files/OPENABC2_DATASET.zip?download=1"
OPENABC_FILENAME = "OPENABC2_DATASET.zip"


def download_with_resume(url: str, output_path: Path):
    """Download a file with resume support using curl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {output_path} ...")

    cmd = [
        "curl", "-L", "-C", "-",  # resume
        "-o", str(output_path),
        "--progress-bar",
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        # Try wget as fallback
        cmd = [
            "wget", "-c",  # resume
            "-O", str(output_path),
            "--show-progress",
            url,
        ]
        subprocess.run(cmd, check=True)

    print(f"Download complete: {output_path}")


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract archive (.zip or .tar.gz)."""
    print(f"Extracting {archive_path} to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(archive_path).endswith('.zip'):
        subprocess.run(
            ["unzip", "-o", "-q", str(archive_path), "-d", str(output_dir)],
            check=True,
        )
    else:
        subprocess.run(
            ["tar", "xzf", str(archive_path), "-C", str(output_dir)],
            check=True,
        )
    print("Extraction complete.")


def convert_aig_to_pyg(aig_dir: Path, output_dir: Path, max_samples: int = 0):
    """
    Convert OpenABC-D AIG graphs to PyG Data objects.

    OpenABC-D stores each design as a PyTorch file with:
      - node_features: gate type, fan-in/fan-out counts
      - edge_index: net connectivity
      - target: delay (ns)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for .pt or .npz files in the AIG directory
    graph_files = list(aig_dir.rglob("*.pt")) + list(aig_dir.rglob("*.npz"))
    if not graph_files:
        print(f"WARNING: No graph files found in {aig_dir}")
        print("  Expected .pt or .npz files from OpenABC-D extraction.")
        print("  Check the OpenABC-D README for conversion steps.")
        return []

    if max_samples > 0:
        graph_files = graph_files[:max_samples]

    manifest_rows = []
    for gf in tqdm(graph_files, desc="Converting graphs"):
        try:
            if gf.suffix == ".pt":
                data = torch.load(gf, weights_only=False)
            else:
                npz = np.load(gf, allow_pickle=True)
                # Convert numpy arrays to PyG Data
                from torch_geometric.data import Data
                node_feat = torch.tensor(npz["node_features"], dtype=torch.float32)
                edge_index = torch.tensor(npz["edge_index"], dtype=torch.long)
                data = Data(x=node_feat, edge_index=edge_index)
                if "delay" in npz:
                    data.delay = float(npz["delay"])

            # Save as PyG .pt file
            design_name = gf.stem
            out_path = output_dir / f"{design_name}.pt"
            torch.save(data, out_path)

            delay = getattr(data, "delay", float("nan"))
            manifest_rows.append({
                "sample_id": design_name,
                "source": "openabc_d",
                "design_name": design_name,
                "tech_node": "mixed",
                "graph_path": str(out_path),
                "delay": delay,
                "node_count": data.x.shape[0] if data.x is not None else 0,
                "total_edges": data.edge_index.shape[1] if data.edge_index is not None else 0,
                "has_delay": not np.isnan(delay),
                "has_power": False,
                "run_status": "success",
                "tool_version": "openabc_d_v1",
            })
        except Exception as e:
            print(f"  SKIP {gf.name}: {e}")

    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Download and prepare OpenABC-D")
    parser.add_argument("--output", type=str, default="data/raw/openabc_d",
                        help="Output directory for raw data")
    parser.add_argument("--graph-output", type=str, default="features/graph/openabc",
                        help="Output directory for converted PyG graphs")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (data already present)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to convert (0 = all)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    graph_output = Path(args.graph_output)

    # Step 1: Download
    archive_path = output_dir / OPENABC_FILENAME
    if not args.skip_download and not archive_path.exists():
        download_with_resume(OPENABC_URL, archive_path)
    elif archive_path.exists():
        print(f"Archive already present: {archive_path}")
    else:
        print("Skipping download (--skip-download)")

    # Step 2: Extract
    extracted_dir = output_dir / "extracted"
    if archive_path.exists() and not extracted_dir.exists():
        extract_archive(archive_path, extracted_dir)
    elif extracted_dir.exists():
        print(f"Already extracted: {extracted_dir}")

    # Step 3: Convert to PyG
    if extracted_dir.exists():
        rows = convert_aig_to_pyg(extracted_dir, graph_output, args.max_samples)
        if rows:
            import pandas as pd
            manifest_path = output_dir / "openabc_manifest_partial.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            print(f"Saved partial manifest: {manifest_path} ({len(rows)} rows)")
    else:
        print("No extracted data found. Run without --skip-download first.")


if __name__ == "__main__":
    main()
