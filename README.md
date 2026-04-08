# Eco_functions

Stochastic consumer-resource and Lotka-Volterra community assembly simulations,
with cavity-method theoretical predictions.

## Repository structure

```
Eco_functions/
├── Eco_function/               # Main Python package
│   ├── eco_lib.py              # Core ODE consumer-resource model (Ecology_simulation)
│   ├── eco_func.py             # Statistics utilities (KL divergence, I/O helpers)
│   ├── eco_plot.py             # Plotting: invasion richness, efficiency time series
│   ├── C_matrix.py             # Consumption matrix builders (block, circulant, binomial)
│   ├── D_matrix.py             # Metabolic/crossfeeding D-matrix builders
│   ├── Model_cavity_CRM.py     # CRM cavity simulation (CRM_Cavity_simulation)
│   ├── Model_cavity_LV.py      # LV cavity simulation (LV_Cavity_simulation, LVModel)
│   └── usertools.py            # High-level matrix builders with family structure
├── Examples/
│   ├── Richness_augmentation.py          # Sequential invasion experiments
│   ├── Richness_augmentation_mpirun.py   # MPI-parallelised invasion
│   ├── Indirect_f_correction.py          # Parameter correction utilities
│   ├── Transfer_Environment.py           # Environment transfer experiments
│   ├── Plot_low_dimensions.ipynb         # Phase-portrait visualisation
│   ├── Plot_high_dimensions.ipynb        # High-dimensional statistics plots
│   └── Plot_high_dimensions_II.ipynb     # Comparison / variant plots
├── Main.py                     # CRM cavity sweep (multiprocessing entry point)
├── LV_cavity.py                # LV cavity sweep (multiprocessing entry point)
├── Test_CRM_cavity.py          # Quick CRM sanity check
├── test/
│   └── test_eco_functions.py   # pytest test suite (33 tests)
├── config.json                 # Default simulation parameters
├── crossfeeding_D.pkl          # Pre-computed crossfeeding matrix
└── setup.py
```

## Models

### Consumer-Resource Model (CRM)
Species *i* grows on resources *j* according to:

```
dR_j/dt = renewal_j(R) − R_j · Σ_i N_i C_ij
dN_i/dt = N_i · g_i · (Σ_j C_ij e_j R_j − m_i)
```

Three resource dynamics are supported (`dynamics` parameter):

| `dynamics`     | Resource renewal                     |
|----------------|--------------------------------------|
| `linear`       | `(K − R) / τ`  (chemostat-like)     |
| `quadratic`    | `R(K − R) / τ` (logistic)           |
| `crossfeeding` | Metabolic byproduct recycling via D  |

### Lotka-Volterra (LV) Model
```
dN_i/dt = N_i · g_i · (k_i − Σ_j A_ij N_j)
```

Interaction matrix A can be sampled from Gaussian, gamma, binomial, or uniform
distributions with tunable symmetry (`sys_gamma`).

## Requirements

- Python ≥ 3.11
- numpy, scipy, pandas, matplotlib
- cvxopt, cvxpy (convex optimisation solvers)
- pytest (for running tests)

## Installation

```bash
# 1. Create and activate the conda environment
conda create -n eco_functions python=3.11 -y
conda activate eco_functions

# 2. Install dependencies
pip install numpy scipy pandas matplotlib cvxopt cvxpy pytest

# 3. Install the package in development mode
git clone https://github.com/Wenping-Cui/Eco_functions
cd Eco_functions
pip install -e .
```

## Minimal working example

```python
import numpy as np
from Eco_function.Model_cavity_CRM import CRM_Cavity_simulation

parameters = {
    'sample_size': 100,   # number of random realisations
    'S': 50,              # species
    'M': 50,              # resources
    'K': 1.0,             # resource carrying capacity (mean)
    'sigma_K': 0.1,
    'mu': 1.0,            # mean consumption rate
    'sigma_c': 2.0,       # consumption rate variance
    'm': 0.1,             # metabolic cost (mean)
    'sigma_m': 0.01,
    'loop_size': 50,
    't0': 0, 't1': 500, 'Nt': 1000,
}

model = CRM_Cavity_simulation(parameters)
model.B_type = 'null'       # no structured base matrix
model.gamma_flag = 'S/M'    # rescaling convention

results = model.ode_simulation(Dynamics='quadratic', Simulation_type='CVXOPT')
print(f"phi_N = {results['phi_N']:.3f}")   # fraction of surviving species
print(f"phi_R = {results['phi_R']:.3f}")   # fraction of nonzero resources
print(f"chi   = {results['chi']:.4f}")     # linear response susceptibility
```

## Running the cavity sweep scripts

```bash
# CRM sweep over sigma_c values
python Main.py --d quadratic --C gaussian --B null --s CVXOPT

# LV sweep
python LV_cavity.py --A gaussian --B identity

# Positional arguments:
#   --d   dynamics:  quadratic | linear | crossfeeding
#   --C   C_type:    gaussian  | binomial | gamma | uniform
#   --B   B_type:    null | identity | circulant | block
#   --s   solver:    CVXOPT | ODE
#   --A   A_type (LV): gaussian | binomial | gamma | uniform
```

## Key parameter table

| Parameter    | Location      | Default | Description                              |
|--------------|---------------|---------|------------------------------------------|
| `S`          | config.json   | 100     | Number of species                        |
| `M`          | config.json   | 100     | Number of resources                      |
| `K`          | config.json   | 1.0     | Resource carrying capacity (mean)        |
| `mu`         | config.json   | 1.0     | Mean consumption rate                    |
| `sigma_c`    | config.json   | 2.0     | Std dev of consumption matrix entries    |
| `m`          | config.json   | 0.1     | Mean metabolic cost                      |
| `sample_size`| config.json   | 1000    | Realisations per parameter point         |
| `t1`         | config.json   | 500     | ODE integration end time                 |
| `e`          | Model_cavity_CRM | 0.6  | Metabolic efficiency (crossfeeding)      |
| `gamma_flag` | Model_cavity_CRM | S/M  | Rescaling: `S/M` or `M/S`              |
| `sys_gamma`  | Model_cavity_LV  | 0    | Symmetry: 1=symmetric, 0=asymmetric     |

All defaults are also stored in `config.json` for easy modification.

## Running tests

```bash
conda activate eco_functions
python -m pytest test/test_eco_functions.py -v
```

Expected output: **33 passed**.

## Contact

Wenping Cui — wenpingcui@gmail.com  
GitHub: https://github.com/Wenping-Cui/Eco_functions
