from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata

from single_component.problem_setup import GLOBAL_BOUNDS


INPUT_ENSEMBLE_CSV = "single_component/outputs/phase_exploration/exploration_ensemble.csv"
OUTPUT_DIR = "single_component/outputs/phase_basin_identification"

EPS = 0.05
DELTA = 0.10
CLUSTER_DISTANCE_THRESHOLD = 0.25

PARAMS = ["R", "MRT", "Pe"]
LOWER = np.array([GLOBAL_BOUNDS[p][0] for p in PARAMS], dtype=float)
UPPER = np.array([GLOBAL_BOUNDS[p][1] for p in PARAMS], dtype=float)
SPAN = UPPER - LOWER

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_ENSEMBLE_CSV)
    x_all = df[["best_R", "best_MRT", "best_Pe"]].to_numpy(dtype=float)
    j_all = df["J_m"].to_numpy(dtype=float)

    j_min = float(np.min(j_all))
    elite_mask = j_all <= j_min * (1.0 + EPS)
    elite_idx = np.where(elite_mask)[0]

    x_elite = x_all[elite_idx]
    j_elite = j_all[elite_idx]
    z_elite = (x_elite - LOWER) / SPAN

    labels = fclusterdata(
        z_elite,
        t=CLUSTER_DISTANCE_THRESHOLD,
        criterion="distance",
        method="single",
        metric="euclidean",
    ).astype(int)
    basin_labels = sorted(np.unique(labels))

    basin_rows = []
    assignment_rows = []

    for basin_label in basin_labels:
        basin_mask = labels == basin_label
        z_b = z_elite[basin_mask]
        x_b = x_elite[basin_mask]
        j_b = j_elite[basin_mask]
        elite_indices_b = elite_idx[basin_mask]

        z_min = np.min(z_b, axis=0)
        z_max = np.max(z_b, axis=0)
        z_lo = np.maximum(0.0, z_min - DELTA)
        z_hi = np.minimum(1.0, z_max + DELTA)

        x_lo = LOWER + z_lo * SPAN
        x_hi = LOWER + z_hi * SPAN

        best_local = int(np.argmin(j_b))
        x_rep = x_b[best_local]
        j_rep = float(j_b[best_local])
        rep_global_idx = int(elite_indices_b[best_local])

        basin_rows.append(
            {
                "basin_id": int(basin_label) + 1,
                "n_points": int(np.sum(basin_mask)),
                "rep_global_idx": rep_global_idx,
                "rep_R": float(x_rep[0]),
                "rep_MRT": float(x_rep[1]),
                "rep_Pe": float(x_rep[2]),
                "rep_J": j_rep,
                "R_lo": float(x_lo[0]),
                "R_hi": float(x_hi[0]),
                "MRT_lo": float(x_lo[1]),
                "MRT_hi": float(x_hi[1]),
                "Pe_lo": float(x_lo[2]),
                "Pe_hi": float(x_hi[2]),
            }
        )

        for i_local in np.where(basin_mask)[0]:
            assignment_rows.append(
                {
                    "basin_id": int(basin_label) + 1,
                    "global_idx": int(elite_idx[i_local]),
                    "J_m": float(j_elite[i_local]),
                    "R": float(x_elite[i_local, 0]),
                    "MRT": float(x_elite[i_local, 1]),
                    "Pe": float(x_elite[i_local, 2]),
                }
            )

    basin_df = pd.DataFrame(basin_rows).sort_values("basin_id")
    assignment_df = pd.DataFrame(assignment_rows).sort_values(["basin_id", "J_m"])
    elite_df = df.iloc[elite_idx].copy()
    elite_df["elite"] = True
    elite_df["J_min"] = j_min

    basin_df.to_csv(out_dir / "basins_summary.csv", index=False)
    assignment_df.to_csv(out_dir / "elite_assignments.csv", index=False)
    elite_df.to_csv(out_dir / "elite_subset.csv", index=False)

    print(f"Loaded ensemble rows: {len(df)}")
    print(f"J_min: {j_min:.6g}")
    print(f"Elite points: {len(elite_idx)} (EPS={EPS})")
    print(
        f"Basins created: {len(basin_df)} "
        f"(distance_threshold={CLUSTER_DISTANCE_THRESHOLD})"
    )
    print(f"Wrote: {out_dir / 'basins_summary.csv'}")
    print(f"Wrote: {out_dir / 'elite_assignments.csv'}")
    print(f"Wrote: {out_dir / 'elite_subset.csv'}")


if __name__ == "__main__":
    main()

