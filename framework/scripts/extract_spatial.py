#!/usr/bin/env python3
"""
scripts/extract_spatial.py

Extract 100×100 cell density maps from placed DEF files.

Usage:
  python scripts/extract_spatial.py \
    --input data/raw/openroad_runs/ \
    --output features/spatial/
"""

import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_def_to_density(def_path: str, grid_size: int = 100) -> np.ndarray:
    """
    Parse a DEF file and produce a cell density heatmap.

    Bins placed cell locations into a grid_size × grid_size grid.
    Returns normalised density map [0, 1].
    """
    with open(def_path) as f:
        content = f.read()

    # Extract die area
    die_match = re.search(
        r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)',
        content
    )
    if not die_match:
        raise ValueError(f"No DIEAREA found in {def_path}")

    die_x0, die_y0 = int(die_match.group(1)), int(die_match.group(2))
    die_x1, die_y1 = int(die_match.group(3)), int(die_match.group(4))

    die_w = die_x1 - die_x0
    die_h = die_y1 - die_y0

    if die_w <= 0 or die_h <= 0:
        raise ValueError(f"Invalid die area: ({die_x0},{die_y0}) to ({die_x1},{die_y1})")

    # Extract placed cell positions from COMPONENTS section
    density = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Pattern for placed components:
    # - inst_name cell_type ... PLACED ( x y ) orient ;
    placed_pattern = re.compile(
        r'(?:PLACED|FIXED)\s+\(\s*(\d+)\s+(\d+)\s*\)',
        re.MULTILINE
    )

    for m in placed_pattern.finditer(content):
        x = int(m.group(1)) - die_x0
        y = int(m.group(2)) - die_y0

        # Map to grid coordinates
        gx = min(int(x / die_w * grid_size), grid_size - 1)
        gy = min(int(y / die_h * grid_size), grid_size - 1)

        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            density[gy, gx] += 1.0

    # Normalise to [0, 1]
    dmax = density.max()
    if dmax > 0:
        density /= dmax

    return density


def compute_spatial_stats(density: np.ndarray) -> dict:
    """Compute summary statistics from a density map."""
    return {
        "mean_density": float(np.mean(density)),
        "std_density": float(np.std(density)),
        "peak_density": float(np.max(density)),
        "hotspot_tile_count": int((density > 0.8).sum()),
        "utilization_estimate": float((density > 0.0).sum() / density.size),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract spatial density maps")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory with DEF files")
    parser.add_argument("--output", type=str, default="features/spatial",
                        help="Output directory for .npy files")
    parser.add_argument("--grid-size", type=int, default=100,
                        help="Grid resolution (default: 100x100)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def_files = list(input_dir.rglob("*.def"))
    print(f"Found {len(def_files)} DEF files")

    stats_rows = []
    for df in tqdm(def_files, desc="Extracting spatial"):
        try:
            density = parse_def_to_density(str(df), args.grid_size)
            out_path = output_dir / f"{df.stem}.npy"
            np.save(out_path, density)

            stats = compute_spatial_stats(density)
            stats["spatial_path"] = str(out_path)
            stats["source_def"] = str(df)
            stats_rows.append(stats)
        except Exception as e:
            print(f"  SKIP {df.name}: {e}")

    if stats_rows:
        import pandas as pd
        stats_path = output_dir / "spatial_stats.csv"
        pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
        print(f"Saved spatial stats: {stats_path}")


if __name__ == "__main__":
    main()
