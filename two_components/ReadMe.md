This case fits the corrected NDS tracer breakthrough curve with a two-component mixture model.

## 1. Problem setup

Forward model:

`C(t; MRT1, MRT2, Pe1, Pe2, fr1, fr2) = fr1 * RELAPModel(t; MRT1, Pe1) + fr2 * RELAPModel(t; MRT2, Pe2)`

where:
- `C`: concentration, mg/L
- `t`: time, hr
- `MRT1`, `MRT2`: component mean residence times, hr
- `Pe1`, `Pe2`: component Peclet numbers
- `fr1`, `fr2`: component mixture fractions

Constraints:
- `fr1 + fr2 <= 1.0`
- `MRT1 <= MRT2`

Objective function:

`J(x, t_true, c_true) = sqrt( Integrate{(C_true - C_pred)^2 dt} / Integrate{C_true^2 dt} )`

with:
- `x = [MRT1, MRT2, Pe1, Pe2, fr1, fr2]`
- `J` = whole-curve NRMSE

Global parameter bounds used in exploration:
- `MRT1` in `[1, 60]`
- `MRT2` in `[1, 60]`
- `Pe1` in `[0.1, 40]`
- `Pe2` in `[0.1, 40]`
- `fr1` in `[0, 1]`
- `fr2` in `[0, 1]`

## 2. Current workflow

The workflow now has four stages:
1. exploration
2. clustering evaluation
3. basin identification
4. exploitation

### 2.1 Exploration phase

Purpose:
- map the objective landscape over the full global bounds
- collect one best solution from each independent DE run
- store full-run artifacts for later analysis

Implementation:
- Python entrypoint: `two_components/exploration_phase.py`
- Batch wrapper: `two_components/exploration_phase.slurm`
- The current batch layout uses:
  - `MEXP = 1024`
  - `SHARD_SIZE = 64`
  - `CPUS_PER_RUN = 8`
  - 16 array shards

SciPy DE settings:
- `strategy='rand1bin'`
- `init='sobol'`
- `popsize=40`
- `mutation=(0.5, 1.2)`
- `recombination=0.4`
- `updating='deferred'`
- `tol=0.01`
- `atol=0`
- `maxiter=500`
- `polish=False`
- `rng=seed_i`
- `constraints = fr1 + fr2 <= 1` and `MRT1 <= MRT2`

Outputs:
- `two_components/outputs/phase_exploration/exploration_runs_full.csv`
- `two_components/outputs/phase_exploration/exploration_ensemble.csv`
- `two_components/outputs/phase_exploration/artifacts/run_XXX.json`
- shard-level outputs under `two_components/outputs/phase_exploration/shards/`

### 2.2 Clustering evaluation

Purpose:
- evaluate K-means clustering on the full exploration ensemble
- choose one clustering to define the basins used downstream

Implementation:
- Python entrypoint: `two_components/compare_clustering_algorithms.py`

Current behavior:
- reads `two_components/outputs/phase_exploration/exploration_ensemble.csv`
- uses all exploration rows, not an elite subset
- normalizes `best_MRT1`, `best_MRT2`, `best_Pe1`, `best_Pe2`, `best_fr1`, `best_fr2`
  by the min and max of those parameters in the exploration dataset itself
- runs `sklearn.cluster.KMeans` with:
  - `init='k-means++'`
  - `random_state=0`
  - `k = 1, 2, ..., 25`
- evaluates each clustering with:
  - `explained_fraction`
  - `normalized_entropy`
- defines:
  - `selection_score = 0.5 * (explained_fraction + normalized_entropy)` for `k >= 2`
- selects the best clustering as the row with the largest `selection_score`
  and breaks ties by choosing the smaller `k`

Outputs:
- `two_components/outputs/clustering_evaluation/kmeans_evaluation.csv`
- `two_components/outputs/clustering_evaluation/kmeans_evaluation.png`
- `two_components/outputs/clustering_evaluation/best_clustering_assignments.csv`
- `two_components/outputs/clustering_evaluation/best_clustering_pca_points.csv`
- `two_components/outputs/clustering_evaluation/best_clustering_pca.png`

### 2.3 Basin identification

Purpose:
- convert the selected clustering result into basin bounds for exploitation

Implementation:
- Python entrypoint: `two_components/basin_identification.py`
- Batch wrapper: `two_components/basin_identification.slurm`

Current behavior:
- reads `two_components/outputs/clustering_evaluation/best_clustering_assignments.csv`
- treats each `cluster_id` as a `basin_id`
- recomputes the same dataset-based normalization used in clustering evaluation
- for each basin:
  - uses the minimum-`J_m` point in that cluster as the representative point
  - computes `radius_norm` as the maximum Euclidean distance from that best point
    to the other points in the same cluster, in normalized coordinates
  - builds normalized bounds as `center +/- radius_norm`
  - clips those bounds to `[0, 1]`
  - scales them back to the original parameter coordinates

Outputs:
- `two_components/outputs/phase_basin_identification/basins_summary.csv`

`basins_summary.csv` contains:
- basin id and basin size
- representative run id and representative objective value
- representative parameter values in original coordinates
- representative parameter values in normalized coordinates
- `radius_norm`
- per-parameter regional bounds `*_lo`, `*_hi`
- per-parameter bound spans `*_span`

This stage no longer writes:
- `elite_subset.csv`
- `elite_assignments.csv`

### 2.4 Exploitation phase

Purpose:
- refine each basin independently with repeated DE runs inside the basin bounds
- compare the refined results across basins

Implementation:
- Python entrypoint: `two_components/exploitation_phase.py`
- Batch wrapper: `two_components/exploitation_phase.slurm`

Inputs:
- `two_components/outputs/phase_basin_identification/basins_summary.csv`
- forward model and objective function from `two_components/problem_setup.py`

Current refinement settings:
- `MREF = 15`
- seeds start at `201`
- `strategy='best1bin'`
- `init='latinhypercube'`
- `popsize=15`
- `mutation=(0.3, 0.8)`
- `recombination=0.6`
- `maxiter=500`
- `tol=0.01`
- `atol=0`
- `updating='deferred'`
- `polish=False`
- `workers=CPUS_PER_RUN`
- `constraints = fr1 + fr2 <= 1` and `MRT1 <= MRT2`

Outputs:
- `two_components/outputs/phase_exploitation/refinement_runs_full.csv`
- `two_components/outputs/phase_exploitation/best_refined_per_basin.csv`
- `two_components/outputs/phase_exploitation/overall_best_refined_solution.csv`
- `two_components/outputs/phase_exploitation/artifacts/basin_XX_run_YY.json`

## 3. Script order

The current intended order is:
1. run exploration
2. run clustering evaluation
3. run basin identification
4. run exploitation

In module form:
- `python -m two_components.exploration_phase ...`
- `python -m two_components.compare_clustering_algorithms`
- `python -m two_components.basin_identification`
- `python -m two_components.exploitation_phase`
