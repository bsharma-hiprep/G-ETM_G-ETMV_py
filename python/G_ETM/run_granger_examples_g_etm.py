"""
This script runs the G-ETM method as described in:

Robust point-process Granger causality analysis in presence of exogenous
temporal modulations and trial-by-trial variability in spike trains.

by Casile A., Faghih R. T. & Brown E. N.

Converted from MATLAB to Python.

Usage:
    python run_granger_examples_g_etm.py

Set the variable `data_set` to choose which example to run:
    1 - Simple network (2 units, one connection, non-stationary spike rate)
    2 - Monkey data (12 neurons from pre-motor cortex, Fig. 7 and S1 of the paper)
"""

import os
import sys
import numpy as np
import scipy.io

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from G_ETM.run_granger_g_etm import run_granger_g_etm
from G_ETM.plot_results_granger import plot_results_granger


# Data set to load
data_set = 1

# Base path to Results directory
results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'Results'
)

# Load spike trains
if data_set == 1:
    # Simple network: two units with one connection and non-stationarity
    mat_path = os.path.join(results_dir, 'ExampleDataSet_2Units_Exogenous.mat')
    D = scipy.io.loadmat(mat_path)
    topology = D['Topology']
    time_topology_ms = D['time_Topology_ms'].flatten()
elif data_set == 2:
    # Monkey data (Fig. 7 and S1)
    mat_path = os.path.join(results_dir, 'ExampleDataSet_MonkeyData.mat')
    D = scipy.io.loadmat(mat_path)
else:
    raise ValueError(f"Unknown data_set value: {data_set}")

spike_trains = D['SpikeTrains']
sample_hz = float(D['sample_Hz'].flatten()[0])
sample_time_ms = 1000.0 / sample_hz
del D

# Get information about spike trains
n_neurons, len_trial_samples, n_trials = spike_trains.shape
len_trial_ms = len_trial_samples * sample_time_ms
print('Done with the spike trains! Now I fit GLM models')

# ---- Define global regressor ----
# Number of windows used to divide each trial
global_regressor_n_bins = np.array([1, 5, 10, 15, 20, 25, 30])
# Uncomment the following line and comment out the line above
# to run Kim et al. method:
# global_regressor_n_bins = np.array([1])

global_regressor = {
    'nBins': global_regressor_n_bins,
    'binDuration_samples': np.round(len_trial_samples / global_regressor_n_bins).astype(int),
    'binDuration_ms': np.round(len_trial_samples / global_regressor_n_bins).astype(int) * sample_time_ms,
}

# ---- Define history regressor ----
history_regressor = {
    'binDuration_samples': 3,
    'binDuration_ms': 3 * sample_time_ms,
    'maxNBins': 30,
    'winHistory_samples': np.ones(3),
    'winHistory_ms': np.ones(int(3 * sample_time_ms)),
}

# Number of steps for the history regressor to test
history_regressor_n_bins = np.arange(2, history_regressor['maxNBins'] + 1, 2)

# Run the Granger causality method
out_struct = run_granger_g_etm(
    spike_trains, global_regressor, history_regressor, history_regressor_n_bins
)

# Save all results as a .npz file
f_name = './Out.npz'
print(f'Saving {f_name}')
save_dict = {
    'SpikeTrains': spike_trains,
    'sample_Hz': sample_hz,
    'OutStruct': np.array(out_struct, dtype=object),
    'globalRegressor': np.array(global_regressor, dtype=object),
    'historyRegressor': np.array(history_regressor, dtype=object),
    'historyRegressorNBins': history_regressor_n_bins,
}
if data_set == 1:
    save_dict['Topology'] = topology
    save_dict['time_Topology_ms'] = time_topology_ms

np.savez(f_name, **save_dict)

# Plot the results (.npz suffix is added automatically by np.savez if not present)
f_name_npz = f_name if f_name.endswith('.npz') else f_name + '.npz'
plot_results_granger(f_name_npz)
