This case tries to fit the corrected NDS tracer breakthrough curve assuming a single porosity sinlge component system:

1. Problem setup.
Forward Pass Model:
C(t; R, MRT, Pe) = R * RELAPModel(t; MRT, Pe)

such that:
C: concentration, mg/L.
t: time, hr.
The model fitting parameters:
R: tracer mass recovery factor.
MRT: mean residence time, hr.
Pe: Peclet Number.


Objective Function
the objective function is the Normalized Root Mean Squared Error NRMSE:
J(x, t_true, c_true) = sqrt (Integrate {(C_true - C_pred)^2 dt} / Integrate {C_true^2 dt})
x = [R, MRT, Pe]


Parameter Initial Global Bounds:
R belongs to [0, 1.0]
MRT belongs to [4, 60]
Pe belongs to [1, 30]

Optimization Algorithm
Staged Differential Evolution (DE) optimization algorithm thta has the following phases:
i. Exploration phase.
ii. Basing identification.
iii. Exploitation Phase.

2. Exploration Phase: 
Map the objective-function landscape over the full initial global bounds and collect an ensemble of good candidate minima for the next phase, basin identification. This phase should favor broad search, not fast local convergence. In DE terms, random-base strategies are more exploratory, while best-based strategies converge faster but are more exploitative.

Mxp : number of independent exploration DE runs (M = 10)
seeds = [seed_1, seed_2, ..., seed_i, ...., seed_Mxp]

DE settings in SciPy:
strategy='rand1bin'
init='sobol'
popsize=25
mutation=(0.5, 1.2)
recombination=0.4
updating='deferred'
workers=-5
tol=0.01
atol=0
maxiter=500
polish=False
rng=seed_i

Run structure:
- Assign a unique seed per run.
- Run each Scipy DE once over the full blobal bounds.
- Store the run output.

x_m = results.x [for the mth run]
J_m = result.fun [for the mth run]

Results to store in a csv table:
run index m, random seed seed, best parameter vector x, best objective value J, number of iterations nit, number of functions evaluations nfev, termination flag success, termination message message, final population P_final, final population energies J_pop.

what to pass to basin identification: is an ensamble of best solutions across independent DE runs.
E_exp = {(x_m, J_m)} m = 1, 2, ..., Mexp.

3. Basin Identification
Take the ensemble of best solutions from the exploration phase and identify distinct candidate basins of attraction for the next phase, exploitation/refinement.
exploration ensemble → elite filtering → normalization → clustering → basin definition → regional bounds → exploitation

eps: elite subset tolerance
delta: normalized regional bounds constant
nB: number of basins

Inputs:
E_exp = {(x_m, J_m)} m = 1, 2, ..., Mexp.
Original global bounds.

a. Find the best objective value across all exploration runs
J_min = min( {J_m} )

b. Define the elite subset:
J_m <= J_min * (1 + eps)

c. Extract the parameter vectors: theta = [x_1, x_2, ..., x_E]: E is the number of elite solutions.

d. Normalize each elite parameter vecor using original global bounds.
z_j = (x_j - L_j)/(U_j - L_j) [x_j is the jth parameter in the parameter vector x]
normalized elite vectors z = [z_1, z_2, ..., z_E]

e. Cluster the normalized elite vectors into nB basins : 
elite solutions that lie close together in normalized parameter space are assigned the same cluster, each cluster represents one basin.

f. for each basin b, define the regional bounds:
a_j = max(0, min(z_j) - delta)
b_j = min(1, max(z_j) + delta)

z_j is the jth parameter in the vector j that belongs to the basin b.

convert the bounds to original parameter units.

g. define the representative starting point as the best vector (the one with minimum J) in the basin.

Outputs:
basins: B = [b_1, b_2, ..., b_nB]
Regional bounds for each basin in parameter original units.
best point for each basin.


4. Exploitation Phase
Take each candidate basin identified in the previous phase and refine the solution within that basin to obtain the best local minimum supported by repeated DE runs. This phase should favor focused search inside a narrowed region, not broad global exploration.

regional bounds per basin → repeated exploitative DE runs inside each basin → best refined solution per basin → compare across basins → final solution

Inputs:
basins: B = [b_1, b_2, ..., b_nB]
Regional bounds for each basin in parameter original units.
best point for each basin.
Forward model and objective function.

Mref: number refinement runs per basin
seeds = [seed_1, ..., seed_Mref]

for each basin b:
define the search domain

DE settings in SciPy:
popsize=15
mutation=(0.3, 0.8)
recombination=0.6
maxiter=500
tol=0.01
atol=0
updating='deferred'
polish=False

run Mref independent runs for each basin with different seeds.
store the best parameter and the objective function value for each refinement.
build a refinement ensemble for each basin from the independent runs and select the best refined solution in that ensemble.

compare the best solution across basins.
store:
basin index, random seed, best parameter vector, best objective value, nit, nfev, success, mesage, final population, final population energies.

Outputs:
best refined solution per basin.
refinement ensemble per basin.
overll best refined solution across all basins.
