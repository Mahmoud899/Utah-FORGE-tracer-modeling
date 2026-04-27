import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from two_components.problem_setup import (
    GLOBAL_BOUNDS,
    build_constraints,
    load_data,
    objective_vector,
)


DATA_CSV = "data/data.csv"
DEFAULT_OUTPUT_DIR = "two_components/outputs/phase_exploration"

DEFAULT_MEXP = 256 * 4
DEFAULT_SEED_START = 101
DEFAULT_SHARD_SIZE = 64
DEFAULT_CPUS_PER_RUN = 8

DE_STRATEGY = "rand1bin"
DE_INIT = "sobol"
DE_POPSIZE = 40
DE_MUTATION = (0.5, 1.2)
DE_RECOMBINATION = 0.4
DE_UPDATING = "deferred"
DE_TOL = 0.01
DE_ATOL = 0.0
DE_MAXITER = 500
DE_POLISH = False

BOUNDS = [
    GLOBAL_BOUNDS["MRT1"],
    GLOBAL_BOUNDS["MRT2"],
    GLOBAL_BOUNDS["Pe1"],
    GLOBAL_BOUNDS["Pe2"],
    GLOBAL_BOUNDS["fr1"],
    GLOBAL_BOUNDS["fr2"],
]


def _run_one_exploration(run_idx, seed, t_obs, c_obs, cpus_per_run):
    result = differential_evolution(
        func=objective_vector,
        bounds=BOUNDS,
        args=(t_obs, c_obs),
        strategy=DE_STRATEGY,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        atol=DE_ATOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        init=DE_INIT,
        rng=seed,
        disp=False,
        polish=DE_POLISH,
        updating=DE_UPDATING,
        workers=cpus_per_run,
        vectorized=False,
        constraints=build_constraints(),
    )

    pop_final = result.population.tolist() if hasattr(result, "population") else []
    energies_final = (
        result.population_energies.tolist() if hasattr(result, "population_energies") else []
    )

    return {
        "run_idx": run_idx,
        "seed": seed,
        "best_MRT1": float(result.x[0]),
        "best_MRT2": float(result.x[1]),
        "best_Pe1": float(result.x[2]),
        "best_Pe2": float(result.x[3]),
        "best_fr1": float(result.x[4]),
        "best_fr2": float(result.x[5]),
        "best_J": float(result.fun),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "P_final": pop_final,
        "J_pop": [float(v) for v in energies_final],
    }


def _build_rows_subset(start_idx, end_idx, seed_start):
    return [(idx + 1, seed_start + idx) for idx in range(start_idx, end_idx)]


def _write_shard_outputs(shard_dir, rows, shard_id, start_idx, end_idx):
    shard_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = shard_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda r: r["run_idx"])
    for row in sorted_rows:
        artifact_file = artifacts_dir / f"run_{row['run_idx']:03d}.json"
        artifact_payload = {
            "run_idx": row["run_idx"],
            "seed": row["seed"],
            "x_best": [
                row["best_MRT1"],
                row["best_MRT2"],
                row["best_Pe1"],
                row["best_Pe2"],
                row["best_fr1"],
                row["best_fr2"],
            ],
            "best_J": row["best_J"],
            "P_final": row.pop("P_final"),
            "J_pop": row.pop("J_pop"),
        }
        artifact_file.write_text(json.dumps(artifact_payload))

    full_df = pd.DataFrame(sorted_rows)
    full_df.to_csv(shard_dir / "exploration_runs_full.csv", index=False)

    ensemble_df = full_df[
        [
            "run_idx",
            "seed",
            "best_MRT1",
            "best_MRT2",
            "best_Pe1",
            "best_Pe2",
            "best_fr1",
            "best_fr2",
            "best_J",
        ]
    ].rename(columns={"best_J": "J_m"})
    ensemble_df.to_csv(shard_dir / "exploration_ensemble.csv", index=False)

    manifest = {
        "shard_id": shard_id,
        "run_idx_start": start_idx + 1,
        "run_idx_end": end_idx,
        "n_runs": len(sorted_rows),
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def run_shard(args):
    shard_id = args.shard_id
    shard_size = args.shard_size
    mexp = args.mexp
    seed_start = args.seed_start
    cpus_per_run = args.cpus_per_run
    output_dir = Path(args.output_dir)

    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, mexp)
    if start_idx >= mexp:
        raise ValueError(
            f"Shard {shard_id} starts at run {start_idx + 1}, which exceeds MEXP={mexp}."
        )

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", args.total_cpus))
    max_parallel_runs = max(1, slurm_cpus // cpus_per_run)
    if slurm_cpus < cpus_per_run:
        raise ValueError(
            f"SLURM_CPUS_PER_TASK={slurm_cpus} is smaller than CPUS_PER_RUN={cpus_per_run}."
        )

    shard_dir = output_dir / "shards" / f"shard_{shard_id:03d}"
    rows_meta = _build_rows_subset(start_idx, end_idx, seed_start)

    t_obs, c_obs = load_data(DATA_CSV)
    start_time = time.time()
    rows = []

    print(f"Shard {shard_id}: run_idx {start_idx + 1}..{end_idx} ({len(rows_meta)} runs)")
    print(f"SLURM_CPUS_PER_TASK={slurm_cpus}")
    print(f"CPUS_PER_RUN={cpus_per_run}")
    print(f"MAX_PARALLEL_RUNS={max_parallel_runs}")
    print(f"Output dir: {shard_dir}")

    with ProcessPoolExecutor(max_workers=max_parallel_runs) as executor:
        futures = [
            executor.submit(_run_one_exploration, run_idx, seed, t_obs, c_obs, cpus_per_run)
            for run_idx, seed in rows_meta
        ]
        total = len(futures)
        for done_idx, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            elapsed = time.time() - start_time
            avg_per_run = elapsed / done_idx
            remaining = total - done_idx
            eta_sec = avg_per_run * remaining
            print(
                f"[{done_idx}/{total}] "
                f"run={row['run_idx']:03d} seed={row['seed']} "
                f"best_J={row['best_J']:.6g} "
                f"elapsed={elapsed/60:.1f}m "
                f"eta={eta_sec/60:.1f}m"
            )

    if len(rows) != len(rows_meta):
        raise RuntimeError("Completed run count does not match shard size.")
    if not all(np.isfinite(r["best_J"]) for r in rows):
        raise RuntimeError("Non-finite best_J detected in shard results.")

    _write_shard_outputs(shard_dir, rows, shard_id, start_idx, end_idx)
    elapsed = time.time() - start_time
    print(f"Shard {shard_id} complete in {elapsed/60:.1f} minutes")


def merge_shards(args):
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"Shards directory not found: {shards_dir}")

    shard_dirs = sorted(p for p in shards_dir.iterdir() if p.is_dir())
    if not shard_dirs:
        raise RuntimeError("No shard directories found to merge.")

    full_frames = []
    artifact_inputs = []
    for shard_dir in shard_dirs:
        csv_path = shard_dir / "exploration_runs_full.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing shard CSV: {csv_path}")
        full_frames.append(pd.read_csv(csv_path))
        artifact_inputs.append(shard_dir / "artifacts")

    full_df = pd.concat(full_frames, ignore_index=True).sort_values("run_idx")
    if len(full_df) != args.mexp:
        raise RuntimeError(f"Expected {args.mexp} merged runs, found {len(full_df)}.")
    if full_df["run_idx"].duplicated().any():
        dupes = full_df.loc[full_df["run_idx"].duplicated(), "run_idx"].tolist()
        raise RuntimeError(f"Duplicate run_idx values found: {dupes[:10]}")
    expected_run_idx = list(range(1, args.mexp + 1))
    actual_run_idx = full_df["run_idx"].astype(int).tolist()
    if actual_run_idx != expected_run_idx:
        raise RuntimeError("Merged run_idx values are incomplete or out of order.")
    if not np.all(np.isfinite(full_df["best_J"].to_numpy(dtype=float))):
        raise RuntimeError("Non-finite best_J detected in merged results.")

    artifacts_dir = output_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for artifact_input in artifact_inputs:
        for artifact_file in sorted(artifact_input.glob("run_*.json")):
            shutil.copy2(artifact_file, artifacts_dir / artifact_file.name)

    full_df.to_csv(output_dir / "exploration_runs_full.csv", index=False)
    ensemble_df = full_df[
        [
            "run_idx",
            "seed",
            "best_MRT1",
            "best_MRT2",
            "best_Pe1",
            "best_Pe2",
            "best_fr1",
            "best_fr2",
            "best_J",
        ]
    ].rename(columns={"best_J": "J_m"})
    ensemble_df.to_csv(output_dir / "exploration_ensemble.csv", index=False)

    summary = {
        "n_shards": len(shard_dirs),
        "mexp": args.mexp,
        "run_idx_min": int(full_df["run_idx"].min()),
        "run_idx_max": int(full_df["run_idx"].max()),
        "best_J_min": float(full_df["best_J"].min()),
        "best_J_max": float(full_df["best_J"].max()),
    }
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Merged {len(shard_dirs)} shard directories")
    print(f"Wrote: {output_dir / 'exploration_runs_full.csv'}")
    print(f"Wrote: {output_dir / 'exploration_ensemble.csv'}")
    print(f"Wrote: {output_dir / 'merge_summary.json'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run exploration shards and merge their results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-shard", help="Run one shard of exploration runs.")
    run_parser.add_argument("--shard-id", type=int, required=True)
    run_parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    run_parser.add_argument("--mexp", type=int, default=DEFAULT_MEXP)
    run_parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    run_parser.add_argument("--cpus-per-run", type=int, default=DEFAULT_CPUS_PER_RUN)
    run_parser.add_argument(
        "--total-cpus",
        type=int,
        default=DEFAULT_CPUS_PER_RUN,
        help="Fallback total CPU count when SLURM_CPUS_PER_TASK is unavailable.",
    )
    run_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run_parser.set_defaults(func=run_shard)

    merge_parser = subparsers.add_parser("merge-shards", help="Merge all completed shards.")
    merge_parser.add_argument("--mexp", type=int, default=DEFAULT_MEXP)
    merge_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    merge_parser.set_defaults(func=merge_shards)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
