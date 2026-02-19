"""
This module computes the log likelihood given the regressors for the
method G-ETM described in:

Robust point-process Granger causality analysis in presence of exogenous
temporal modulations and trial-by-trial variability in spike trains.

by Casile A., Faghih R. T. & Brown E. N.

Converted from MATLAB to Python.
"""

import numpy as np


def compute_log_likelihood_spike_trains(
    spikes, beta_global, beta_history, global_regressor, history_regressor, target_neuron
):
    """Compute log-likelihood of spike trains given GLM parameters (G-ETM).

    Parameters
    ----------
    spikes : np.ndarray
        Spike train data, shape (nNeurons, lenTrial_samples, nTrials).
    beta_global : np.ndarray
        Global regressor coefficients, shape (nBins,).
    beta_history : np.ndarray
        History regressor coefficients, shape (nNeurons, nBinsHistory).
    global_regressor : dict
        Dictionary with keys 'nBins' and 'binDuration_samples'.
    history_regressor : dict
        Dictionary with key 'binDuration_samples'.
    target_neuron : int
        Index of the target neuron (0-based).

    Returns
    -------
    llk : float
        Log-likelihood value.
    """
    n_neurons, len_trial_samples, n_trials = spikes.shape

    # Compute parameters of the history regressor
    n_bins_history = beta_history.shape[1]
    total_history_samples = history_regressor['binDuration_samples'] * n_bins_history

    # Expand global regressor to the size of a trial:
    # Each beta_global value is repeated binDuration_samples times
    global_lambda = np.repeat(beta_global.flatten(), global_regressor['binDuration_samples'])

    # Initialize log-likelihood
    llk = 0.0

    for curr_trial_ind in range(n_trials):

        # Compute the effect of history regressors
        history_lambda = np.zeros(len_trial_samples)

        # Compute the effect of each neuron on own spiking history
        for curr_neuron_ind in range(n_neurons):
            # Get regressor for this neuron, replicate bins to match time bins
            # (bins for the local regressor are bigger than the time bins)
            current_regressor = np.repeat(
                beta_history[curr_neuron_ind, :],
                history_regressor['binDuration_samples']
            )

            # Convolve spike train with regressor
            tmp = np.convolve(spikes[curr_neuron_ind, :, curr_trial_ind], current_regressor)
            # Prepend 0 (no influence of past history at first bin)
            history_lambda += np.concatenate([[0.0], tmp[:len_trial_samples - 1]])

        # Compute probability using logistic function
        total_lambda = global_lambda + history_lambda
        P = np.exp(total_lambda) / (1.0 + np.exp(total_lambda))

        # Consider only from totalHistory_samples onward for consistency
        start_ind = total_history_samples
        tmp_spikes = spikes[target_neuron, start_ind:, curr_trial_ind]

        # Compute log-likelihood (Bernoulli)
        llk += np.sum(
            tmp_spikes * np.log(P[start_ind:]) + (1 - tmp_spikes) * np.log(1 - P[start_ind:])
        )

    return llk
