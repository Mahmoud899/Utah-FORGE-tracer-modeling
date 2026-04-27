## GeothermicsPaper

This project fits an NDS tracer breakthrough curve using semi-analytical transport models.
The main goal is to match the observed concentration response over time and recover a set of transport parameters that explain the data.

## Data

The input data is stored in `data/data.csv`.

The main columns used for fitting are:
- `Time_hr`: time in hours
- `C_corr_mgL`: corrected tracer concentration in mg/L
- `weight`: mask used to select rows for fitting

Both the single-component and two-component workflows read this same dataset and evaluate model fit with a whole-curve normalized root mean squared error (NRMSE).

## Modeling approaches

The repository contains two related model formulations.

### Single-component model

Location:
- `single_component/`

This workflow assumes the tracer response can be represented by one transport component.

The fitted parameters are:
- `R`: recovery factor
- `MRT`: mean residence time
- `Pe`: Peclet number

### Two-component model

Location:
- `two_components/`

This workflow assumes the tracer response is a weighted sum of two transport components.

The fitted parameters are:
- `MRT1`, `MRT2`
- `Pe1`, `Pe2`
- `fr1`, `fr2`

with the constraints:
- `fr1 + fr2 <= 1`
- `MRT1 <= MRT2`

## High-level fitting workflow

Both modeling tracks follow the same staged strategy for fitting a nonconvex objective function.

### 1. Exploration

The exploration stage searches broadly over the global parameter bounds.
It runs many independent differential evolution (DE) optimizations with different random seeds and collects an ensemble of candidate solutions.

Purpose:
- map the misfit landscape
- avoid committing too early to one local minimum
- generate candidate solutions for the next stage

### 2. Basin identification

The basin-identification stage finds regions of low misfit from the exploration results.

Purpose:
- group similar good solutions
- define promising regions in parameter space
- build regional bounds for refinement

In practice, these basins represent candidate attraction regions where the objective function is relatively low.

### 3. Exploitation / refinement

The exploitation stage refines each promising basin separately.
It runs repeated DE searches inside the basin-specific bounds and then compares the best refined solution from each basin.

Purpose:
- perform focused local improvement inside good regions
- reduce the chance of missing a strong solution found during exploration
- identify the overall best refined fit

## Project layout

- `single_component/`: single-component model setup and staged fitting workflow
- `two_components/`: two-component model setup and staged fitting workflow
- `reactive_transport/`: shared forward-model and metric utilities
- `data/`: tracer breakthrough data used by both workflows

For the current implementation details of each workflow, see:
- `single_component/ReadMe.md`
- `two_components/ReadMe.md`
