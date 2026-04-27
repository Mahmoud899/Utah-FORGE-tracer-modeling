from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CLUSTERING_CSV = "two_components/outputs/clustering_evaluation/best_clustering_assignments.csv"
OUTPUT_DIR = "two_components/outputs/phase_basin_identification"

PARAMS = ["MRT1", "MRT2", "Pe1", "Pe2", "fr1", "fr2"]
PARAM_COLS = [f"best_{name}" for name in PARAMS]


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CLUSTERING_CSV)
    x_all = df[PARAM_COLS].to_numpy(dtype=float)
    lower = np.min(x_all, axis=0)
    upper = np.max(x_all, axis=0)
    span = upper - lower
    z_all = (x_all - lower) / span
    labels = df["cluster_id"].to_numpy(dtype=int)

    basin_rows = []
    for basin_id in sorted(np.unique(labels)):
        basin_mask = labels == basin_id
        basin_df = df.loc[basin_mask]
        x_b = x_all[basin_mask]
        z_b = z_all[basin_mask]
        j_b = basin_df["J_m"].to_numpy(dtype=float)

        best_local = int(np.argmin(j_b))
        x_rep = x_b[best_local]
        z_rep = z_b[best_local]
        rep_row = basin_df.iloc[best_local]

        radius_norm = float(np.max(np.linalg.norm(z_b - z_rep, axis=1)))
        z_lo = np.maximum(0.0, z_rep - radius_norm)
        z_hi = np.minimum(1.0, z_rep + radius_norm)
        x_lo = lower + z_lo * span
        x_hi = lower + z_hi * span

        basin_rows.append(
            {
                "basin_id": int(basin_id),
                "n_points": int(np.sum(basin_mask)),
                "rep_run_idx": int(rep_row["run_idx"]),
                "rep_J": float(rep_row["J_m"]),
                "rep_MRT1": float(x_rep[0]),
                "rep_MRT2": float(x_rep[1]),
                "rep_Pe1": float(x_rep[2]),
                "rep_Pe2": float(x_rep[3]),
                "rep_fr1": float(x_rep[4]),
                "rep_fr2": float(x_rep[5]),
                "radius_norm": radius_norm,
                "rep_MRT1_norm": float(z_rep[0]),
                "rep_MRT2_norm": float(z_rep[1]),
                "rep_Pe1_norm": float(z_rep[2]),
                "rep_Pe2_norm": float(z_rep[3]),
                "rep_fr1_norm": float(z_rep[4]),
                "rep_fr2_norm": float(z_rep[5]),
                "MRT1_lo": float(x_lo[0]),
                "MRT1_hi": float(x_hi[0]),
                "MRT1_span": float(x_hi[0] - x_lo[0]),
                "MRT2_lo": float(x_lo[1]),
                "MRT2_hi": float(x_hi[1]),
                "MRT2_span": float(x_hi[1] - x_lo[1]),
                "Pe1_lo": float(x_lo[2]),
                "Pe1_hi": float(x_hi[2]),
                "Pe1_span": float(x_hi[2] - x_lo[2]),
                "Pe2_lo": float(x_lo[3]),
                "Pe2_hi": float(x_hi[3]),
                "Pe2_span": float(x_hi[3] - x_lo[3]),
                "fr1_lo": float(x_lo[4]),
                "fr1_hi": float(x_hi[4]),
                "fr1_span": float(x_hi[4] - x_lo[4]),
                "fr2_lo": float(x_lo[5]),
                "fr2_hi": float(x_hi[5]),
                "fr2_span": float(x_hi[5] - x_lo[5]),
            }
        )

    basin_df = pd.DataFrame(basin_rows).sort_values("basin_id")
    basin_df.to_csv(out_dir / "basins_summary.csv", index=False)

    print(f"Loaded ensemble rows: {len(df)}")
    print(f"Basins created: {len(basin_df)}")
    print(f"Wrote: {out_dir / 'basins_summary.csv'}")


if __name__ == "__main__":
    main()
