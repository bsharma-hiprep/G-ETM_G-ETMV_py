This repository contains implementations of the G-ETM and G-ETMV methods
described in:

Robust point-process Granger causality analysis in presence of exogenous
temporal modulations and trial-by-trial variability in spike trains.

by Casile A., Faghih R. T. & Brown E. N.

published in PLoS Computational Biology

Both MATLAB and Python implementations are provided.

------------------ Python Implementation ------------------

The Python implementation is located in the `python/` directory and mirrors
the structure of the `Matlab/` directory. It uses NumPy, SciPy, statsmodels,
and Matplotlib.

### Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Running the Python Examples

#### G-ETM Example

```bash
cd python/G_ETM
python run_granger_examples_g_etm.py
```

Set the variable `data_set` at the top of the script to choose which example:
- `data_set = 1`: Simple network (2 units, one connection, non-stationary firing)
- `data_set = 2`: Monkey data (12 neurons, pre-motor cortex, Fig. 7 and S1)

#### G-ETMV Example

```bash
cd python/G_ETMV
python run_granger_examples_g_etmv.py
```

This runs the G-ETMV method and generates results equivalent to Fig. 8 of the paper.

### Python Dependencies

- **NumPy** (>=1.21.0): Array operations and mathematical functions
- **SciPy** (>=1.7.0): Statistical functions (chi2 distribution, signal processing)
- **statsmodels** (>=0.13.0): GLM fitting (replaces MATLAB's `glmfit`)
- **Matplotlib** (>=3.4.0): Plotting and visualization
- **joblib** (>=1.2.0): Parallel execution (replaces MATLAB's `parfor`)

### Parallel Execution

The Python code parallelises the same loops that used `parfor` in MATLAB using
**joblib**. Both `run_granger_g_etm` and `run_granger_g_etmv` accept an
`n_jobs` parameter:

| `n_jobs` value | Behaviour |
|----------------|-----------|
| `-1` (default) | Use all available CPUs |
| `1` | Serial execution (no parallelism) |
| `N > 1` | Use exactly N worker processes |

```python
from G_ETM.run_granger_g_etm import run_granger_g_etm

out_struct = run_granger_g_etm(
    spike_trains, global_regressor, history_regressor, history_regressor_n_bins,
    n_jobs=-1,   # use all CPUs
)
```

Two loops are parallelised, mirroring MATLAB's two `parfor` blocks:

1. **Round 1** – Inner loop over `(history_regressor_ind × neuron_ind)` for each
   global regressor configuration (mirrors `parfor currHistoryRegressorInd`).
2. **Causal step** – Loop over `trigger_neuron_ind` for each target neuron
   (mirrors `parfor triggerNeuronInd`).

### Differences Between MATLAB and Python Versions

- `parfor` is replaced by `joblib.Parallel` / `joblib.delayed` (process-based,
  bypasses Python's GIL for CPU-bound GLM fitting).
- Results are saved as `.npz` files (NumPy format) rather than `.mat` files.
- All array indices are 0-based (Python) vs 1-based (MATLAB).
- The GLM fitting uses `statsmodels.GLM` with binomial family and logit link,
  which is equivalent to MATLAB's `glmfit(..., 'binomial', 'link', 'logit')`.

### Python Directory Structure

```
python/
├── G_ETM/
│   ├── fdr.py                                  # False Discovery Rate
│   ├── compute_log_likelihood_spike_trains.py  # Log-likelihood (G-ETM)
│   ├── compute_raster_and_rate.py              # Raster and firing rate
│   ├── fit_glm_g_etm.py                        # GLM fitting (G-ETM)
│   ├── run_granger_g_etm.py                    # G-ETM Granger causality
│   ├── plot_results_granger.py                 # Plotting results
│   └── run_granger_examples_g_etm.py           # Example script
└── G_ETMV/
    ├── compute_log_likelihood_spike_trains_trial_by_trial_var.py
    ├── fit_glm_g_etmv.py                       # GLM fitting (G-ETMV)
    ├── run_granger_g_etmv.py                   # G-ETMV Granger causality
    └── run_granger_examples_g_etmv.py          # Example script
```

------------------ MATLAB Implementation ------------------

The original MATLAB code is in the `Matlab/` directory and remains unchanged.

For computational reasons, we used Matlab's parallel toolbox. If that is
not available the "parfor" commands in the routines fitGLM_G_ETM, fitGLM_G_ETMV,
runGranger_G_ETM and runGranger_G_ETMV must be substituted with a "for".

------------------ Directory Matlab/G-ETM ------------------

The file runGranger_Examples_G_ETM.m runs two examples. Which of the
two examples is run depends on how the user sets the variable "dataSet"
in the code.

The first example runs our G-ETM Granger-causality method on a simple
network consisting of two units with one functional connection from
unit 1 to unit 2

The second example runs G-ETM on a data set consisting of 12 neurons recorded
from the monkey pre-motor cortex (Figs. 7 and S1 in the paper). This is a
computation-intensive example and it will take quite some time to complete.


------------------ Directory Matlab/G-ETMV ------------------

The file runGranger_Examples_G_ETMV generates the results plotted in Fig. 8
of the paper.


