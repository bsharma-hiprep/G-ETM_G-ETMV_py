"""
This module runs the G-ETMV method as described in:

Robust point-process Granger causality analysis in presence of exogenous
temporal modulations and trial-by-trial variability in spike trains.

by Casile A., Faghih R. T. & Brown E. N.

Converted from MATLAB to Python.
"""

import numpy as np
from scipy.stats import chi2

from .fit_glm_g_etmv import fit_glm_g_etmv
from .compute_log_likelihood_spike_trains_trial_by_trial_var import (
    compute_log_likelihood_spike_trains_trial_by_trial_var,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from G_ETM.fdr import fdr


def run_granger_g_etmv(spike_trains, global_regressor_in, history_regressor, history_regressor_n_bins):
    """Run the G-ETMV Granger causality analysis (with trial-by-trial variability).

    Parameters
    ----------
    spike_trains : np.ndarray
        Spike train data, shape (nNeurons, nSamples, nTrials).
    global_regressor_in : dict
        Dictionary with keys 'nBins' (array), 'binDuration_samples', 'binDuration_ms'.
    history_regressor : dict
        Dictionary with keys 'binDuration_samples', 'winHistory_samples', etc.
    history_regressor_n_bins : array-like
        Array of history regressor bin counts to test.

    Returns
    -------
    granger_res : dict
        Dictionary containing all results of the Granger causality analysis.
    """
    n_neurons, n_samples, n_trials = spike_trains.shape

    history_regressor_n_bins = np.asarray(history_regressor_n_bins)
    n_history_regressor_n_bins = len(history_regressor_n_bins)
    n_global_regressor_n_bins = len(global_regressor_in['nBins'])

    # Storage for fitted parameters
    beta_trials = [[[ None for _ in range(n_global_regressor_n_bins)]
                    for _ in range(n_history_regressor_n_bins)]
                   for _ in range(n_neurons)]
    beta_trials_p_vals = [[[ None for _ in range(n_global_regressor_n_bins)]
                            for _ in range(n_history_regressor_n_bins)]
                           for _ in range(n_neurons)]
    beta_global = [[[ None for _ in range(n_global_regressor_n_bins)]
                    for _ in range(n_history_regressor_n_bins)]
                   for _ in range(n_neurons)]
    beta_global_p_vals = [[[ None for _ in range(n_global_regressor_n_bins)]
                            for _ in range(n_history_regressor_n_bins)]
                           for _ in range(n_neurons)]
    beta_history = [[[ None for _ in range(n_global_regressor_n_bins)]
                     for _ in range(n_history_regressor_n_bins)]
                    for _ in range(n_neurons)]
    beta_history_p_vals = [[[ None for _ in range(n_global_regressor_n_bins)]
                             for _ in range(n_history_regressor_n_bins)]
                            for _ in range(n_neurons)]

    # Deviance, log-likelihood, and AIC arrays
    glm_dev = np.zeros((n_neurons, n_history_regressor_n_bins, n_global_regressor_n_bins))
    spikes_ll = np.zeros((n_neurons, n_history_regressor_n_bins, n_global_regressor_n_bins))
    aic = np.zeros((n_neurons, n_history_regressor_n_bins, n_global_regressor_n_bins))

    # First round: fit full model for all combinations of regressor sizes
    for curr_global_regressor_ind in range(n_global_regressor_n_bins):
        curr_n_global_bins = int(global_regressor_in['nBins'][curr_global_regressor_ind])

        global_regressor = {
            'nBins': curr_n_global_bins,
            'binDuration_samples': int(round(n_samples / curr_n_global_bins)),
        }

        for curr_history_regressor_ind in range(n_history_regressor_n_bins):
            curr_n_regressor_steps = int(history_regressor_n_bins[curr_history_regressor_ind])

            for curr_neuron_ind in range(n_neurons):
                print(
                    f"Processing neuron #{curr_neuron_ind + 1} -- "
                    f"history regressor steps = {curr_n_regressor_steps} -- "
                    f"global regressor steps = {curr_n_global_bins}"
                )

                (tmp_trials, tmp_global, tmp_history, dev,
                 tmp_trials_p_vals, tmp_global_p_vals, tmp_history_p_vals) = fit_glm_g_etmv(
                    spike_trains, global_regressor,
                    history_regressor, curr_neuron_ind, curr_n_regressor_steps
                )

                # Save GLM fitting results
                beta_trials[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind] = tmp_trials
                beta_trials_p_vals[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind] = tmp_trials_p_vals
                beta_global[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind] = tmp_global
                beta_history[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind] = tmp_history
                beta_global_p_vals[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind] = tmp_global_p_vals
                beta_history_p_vals[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind] = tmp_history_p_vals

                glm_dev[curr_neuron_ind, curr_history_regressor_ind, curr_global_regressor_ind] = dev

                # Compute log-likelihood of spike trains
                spikes_ll[curr_neuron_ind, curr_history_regressor_ind, curr_global_regressor_ind] = \
                    compute_log_likelihood_spike_trains_trial_by_trial_var(
                        spike_trains,
                        beta_trials[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind],
                        beta_global[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind],
                        beta_history[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind],
                        global_regressor, history_regressor, curr_neuron_ind
                    )

                # Compute AIC
                aic[curr_neuron_ind, curr_history_regressor_ind, curr_global_regressor_ind] = (
                    -2 * spikes_ll[curr_neuron_ind, curr_history_regressor_ind, curr_global_regressor_ind]
                    + 2 * (
                        beta_history[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind].size
                        + beta_global[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind].size
                        + beta_trials[curr_neuron_ind][curr_history_regressor_ind][curr_global_regressor_ind].size
                        + 1
                    )
                )

    # ------------------------------------------------------------------
    # Second round: re-fit model excluding the effect of the trigger neuron
    # ------------------------------------------------------------------
    beta_trials_causal = [[None for _ in range(n_neurons)] for _ in range(n_neurons)]
    beta_global_causal = [[None for _ in range(n_neurons)] for _ in range(n_neurons)]
    beta_history_causal = [[None for _ in range(n_neurons)] for _ in range(n_neurons)]
    glm_dev_ratio = np.zeros((n_neurons, n_neurons))

    # Find optimal regressor sizes for each neuron using AIC
    inds_n_bins_history_neuron = np.zeros(n_neurons, dtype=int)
    inds_n_bins_global_neuron = np.zeros(n_neurons, dtype=int)

    for curr_neuron_ind in range(n_neurons):
        tmp_aic = aic[curr_neuron_ind, :, :]
        # Handle edge case where one dimension is 1 (squeeze behavior)
        if tmp_aic.shape[0] != aic.shape[1]:
            tmp_aic = tmp_aic.T
        flat_ind = np.argmin(tmp_aic)
        inds_n_bins_history_neuron[curr_neuron_ind], inds_n_bins_global_neuron[curr_neuron_ind] = \
            np.unravel_index(flat_ind, tmp_aic.shape)

    print(f"----- nBinsHistory_neuron = {history_regressor_n_bins[inds_n_bins_history_neuron]}")
    print(f"----- nBinsGlobal_neuron = {global_regressor_in['nBins'][inds_n_bins_global_neuron]}")

    print("Refitting model excluding the effect of the triggering neuron")
    for target_neuron_ind in range(n_neurons):
        n_history_bins = int(history_regressor_n_bins[inds_n_bins_history_neuron[target_neuron_ind]])
        global_regressor = {
            'nBins': int(global_regressor_in['nBins'][inds_n_bins_global_neuron[target_neuron_ind]]),
        }
        global_regressor['binDuration_samples'] = int(round(n_samples / global_regressor['nBins']))

        for trigger_neuron_ind in range(n_neurons):
            print(
                f"Causal Step - Processing target neuron #{target_neuron_ind + 1} "
                f"-- trigger neuron #{trigger_neuron_ind + 1}"
            )
            print(
                f"Target Neuron: nHistoryBins {n_history_bins} -- "
                f"nGlobalBins {global_regressor['nBins']}"
            )

            (tmp_trials, tmp_global, tmp_history, dev,
             tmp_trials_p_vals, tmp_global_p_vals, tmp_history_p_vals) = fit_glm_g_etmv(
                spike_trains,
                global_regressor, history_regressor,
                target_neuron_ind, n_history_bins, trigger_neuron_ind
            )

            beta_trials_causal[target_neuron_ind][trigger_neuron_ind] = tmp_trials
            beta_global_causal[target_neuron_ind][trigger_neuron_ind] = tmp_global
            beta_history_causal[target_neuron_ind][trigger_neuron_ind] = tmp_history
            glm_dev_ratio[target_neuron_ind, trigger_neuron_ind] = (
                dev - glm_dev[
                    target_neuron_ind,
                    inds_n_bins_history_neuron[target_neuron_ind],
                    inds_n_bins_global_neuron[target_neuron_ind]
                ]
            )

    # ==== Significance Testing ====
    D = glm_dev_ratio
    alpha = 0.05
    temp1 = np.zeros_like(D)
    neuron_history_regressor_n_bins = np.zeros(n_neurons, dtype=int)
    for target_neuron_ind in range(n_neurons):
        neuron_history_regressor_n_bins[target_neuron_ind] = int(
            history_regressor_n_bins[inds_n_bins_history_neuron[target_neuron_ind]]
        )
        temp1[target_neuron_ind, :] = (
            D[target_neuron_ind, :] >
            chi2.ppf(1 - alpha, neuron_history_regressor_n_bins[target_neuron_ind])
        )
    psi1 = temp1

    # Causal connectivity matrix with FDR correction
    fdr_v = 0.05
    psi2 = fdr(D, fdr_v, neuron_history_regressor_n_bins)

    # Package results
    granger_res = {
        'Psi1': psi1,
        'Psi2': psi2,
        'alpha': alpha,
        'beta_Trials': beta_trials,
        'beta_Trials_pVals': beta_trials_p_vals,
        'beta_Global': beta_global,
        'beta_Global_pVals': beta_global_p_vals,
        'beta_History': beta_history,
        'beta_History_pVals': beta_history_p_vals,
        'glm_dev': glm_dev,
        'glm_dev_ratio': glm_dev_ratio,
        'spikes_LL': spikes_ll,
        'aic': aic,
        'beta_Trials_causal': beta_trials_causal,
        'beta_Global_causal': beta_global_causal,
        'beta_History_causal': beta_history_causal,
        'historyRegressorNBins': history_regressor_n_bins,
        'neuronHistoryRegressorNBins': neuron_history_regressor_n_bins,
        'indsNBinsHistory_neuron': inds_n_bins_history_neuron,
        'globalRegressorNBins': global_regressor_in['nBins'],
        'indsNBinsGlobal_neuron': inds_n_bins_global_neuron,
    }

    return granger_res
