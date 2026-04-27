import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from single_component.problem_setup import GLOBAL_BOUNDS, load_data, objective_vector


DATA_CSV = "data/data.csv"
OUTPUT_DIR = "single_component/outputs/phase_exploration"

MEXP = 32
SEED_START = 101
SEEDS = [SEED_START + i for i in range(MEXP)]

NUM_WORKERS = 128
CPUS_PER_RUN = 8
MAX_PARALLEL_RUNS = NUM_WORKERS // CPUS_PER_RUN

DE_STRATEGY = "rand1bin"
DE_INIT = "sobol"
DE_POPSIZE = 35
DE_MUTATION = (0.5, 1.2)
DE_RECOMBINATION = 0.4
DE_UPDATING = "deferred"
DE_WORKERS_PER_RUN = CPUS_PER_RUN
DE_TOL = 0.01
DE_ATOL = 0.0
DE_MAXITER = 500
DE_POLISH = False

BOUNDS = [
    GLOBAL_BOUNDS["R"],
    GLOBAL_BOUNDS["MRT"],
    GLOBAL_BOUNDS["Pe"],
]


def _run_one_exploration(run_idx, seed, t_obs, c_obs):
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
        workers=DE_WORKERS_PER_RUN,
        vectorized=False,
    )

    pop_final = result.population.tolist() if hasattr(result, "population") else []
    energies_final = (
        result.population_energies.tolist() if hasattr(result, "population_energies") else []
    )

    return {
        "run_idx": run_idx,
        "seed": seed,
        "best_R": float(result.x[0]),
        "best_MRT": float(result.x[1]),
        "best_Pe": float(result.x[2]),
        "best_J": float(result.fun),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "P_final": pop_final,
        "J_pop": [float(v) for v in energies_final],
    }


def main():
    if len(SEEDS) < MEXP:
        raise ValueError("SEEDS must have at least MEXP values.")
    if NUM_WORKERS < CPUS_PER_RUN or MAX_PARALLEL_RUNS < 1:
        raise ValueError("NUM_WORKERS must be >= CPUS_PER_RUN.")

    t_obs, c_obs = load_data(DATA_CSV)
    run_seeds = SEEDS[:MEXP]
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    rows = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_RUNS) as executor:
        futures = [
            executor.submit(_run_one_exploration, idx + 1, seed, t_obs, c_obs)
            for idx, seed in enumerate(run_seeds)
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

    rows.sort(key=lambda r: r["run_idx"])
    if len(rows) != MEXP:
        raise RuntimeError("Completed run count does not match MEXP.")
    if not all(np.isfinite(r["best_J"]) for r in rows):
        raise RuntimeError("Non-finite best_J detected in exploration results.")

    artifacts_path = output_path / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        artifact_file = artifacts_path / f"run_{row['run_idx']:03d}.json"
        artifact_payload = {
            "run_idx": row["run_idx"],
            "seed": row["seed"],
            "x_best": [row["best_R"], row["best_MRT"], row["best_Pe"]],
            "best_J": row["best_J"],
            "P_final": row.pop("P_final"),
            "J_pop": row.pop("J_pop"),
        }
        artifact_file.write_text(json.dumps(artifact_payload))

    full_df = pd.DataFrame(rows)
    full_df.to_csv(output_path / "exploration_runs_full.csv", index=False)

    ensemble_df = full_df[["run_idx", "seed", "best_R", "best_MRT", "best_Pe", "best_J"]].rename(
        columns={"best_J": "J_m"}
    )
    ensemble_df.to_csv(output_path / "exploration_ensemble.csv", index=False)
    total_elapsed = time.time() - start_time
    print(f"Done: {MEXP} runs finished in {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()

