"""
This module fits a GLM that takes into account the spiking history
of a neuron as well as a global component that takes care of
non-stationarity in the firing of that neuron (G-ETM method).

Robust point-process Granger causality analysis in presence of exogenous
temporal modulations and trial-by-trial variability in spike trains.

by Casile A., Faghih R. T. & Brown E. N.

Converted from MATLAB to Python.
"""

import numpy as np
import statsmodels.api as sm
import warnings


def fit_glm_g_etm(
    spikes_data_in,
    global_regressor,
    history_regressor,
    target_neuron,
    n_bins_history,
    trigger_neuron=None,
):
    """Fit a GLM for the G-ETM Granger causality method.

    Parameters
    ----------
    spikes_data_in : np.ndarray
        Spike train data, shape (nNeurons, lenTrial_samples, nTrials).
    global_regressor : dict
        Dictionary with keys 'nBins' and 'binDuration_samples'.
    history_regressor : dict
        Dictionary with keys 'binDuration_samples' and 'winHistory_samples'.
    target_neuron : int
        Index of the target neuron (0-based).
    n_bins_history : int
        Number of history bins to use.
    trigger_neuron : int or None, optional
        Index of the trigger neuron to exclude (0-based). If None, all
        neurons are included (non-causal model).

    Returns
    -------
    beta_global : np.ndarray
        Fitted global regressor coefficients, shape (nBins,).
    beta_history : np.ndarray
        Fitted history regressor coefficients, shape (nNeurons_in, nBinsHistory).
        (nNeurons_in includes a zero row for the excluded trigger neuron if given.)
    dev : float
        Deviance of the fitted GLM model.
    beta_global_p_vals : np.ndarray
        P-values for global regressor coefficients, shape (nBins,).
    beta_history_p_vals : np.ndarray
        P-values for history regressor coefficients, shape (nNeurons_in, nBinsHistory).
    """
    # Remove trigger neuron from spike data if specified
    if trigger_neuron is not None:
        n_neurons_in = spikes_data_in.shape[0]
        if trigger_neuron == 0:
            spikes_data = spikes_data_in[1:, :, :]
        elif trigger_neuron == n_neurons_in - 1:
            spikes_data = spikes_data_in[:trigger_neuron, :, :]
        else:
            spikes_data = np.concatenate(
                [spikes_data_in[:trigger_neuron, :, :],
                 spikes_data_in[trigger_neuron + 1:, :, :]],
                axis=0
            )
    else:
        spikes_data = spikes_data_in

    n_neurons, len_trial_samples, n_trials = spikes_data.shape

    # Total duration of the history in samples
    total_history_samples = history_regressor['binDuration_samples'] * n_bins_history

    # Final size of the design matrix X and response vector Y
    dim2 = global_regressor['nBins'] + n_neurons * n_bins_history
    dim1 = (len_trial_samples - n_bins_history * history_regressor['binDuration_samples']) * n_trials

    X_neuron = np.zeros((dim1, dim2))
    Y_neuron = np.zeros(dim1)

    win_history = history_regressor['winHistory_samples']  # shape (binDuration_samples,)

    # Fill in the design matrix
    curr_row_ind = 0
    for curr_trial_ind in range(n_trials):
        for curr_time_ind in range(total_history_samples, len_trial_samples):

            # Global regressor: indicator for the current global time bin
            if global_regressor['nBins']:
                parms_global = np.zeros(global_regressor['nBins'])
                non_zero_ind = int(curr_time_ind // global_regressor['binDuration_samples'])
                non_zero_ind = min(non_zero_ind, global_regressor['nBins'] - 1)
                parms_global[non_zero_ind] = 1.0
            else:
                parms_global = np.array([1.0])

            # History regressor: binned spiking history for each neuron
            parms_history = np.zeros((n_bins_history, n_neurons))
            for curr_ind in range(n_neurons):
                tmp = spikes_data[
                    curr_ind,
                    curr_time_ind - total_history_samples:curr_time_ind,
                    curr_trial_ind
                ].astype(float)
                # Reshape into (binDuration_samples, nBinsHistory), column-major order
                tmp_reshaped = tmp.reshape(n_bins_history,
                                           history_regressor['binDuration_samples']).T
                # Equivalent to MATLAB: reshape(TMP, [binDuration_samples, nBinsHistory])
                parms_history[:, curr_ind] = win_history @ tmp_reshaped

            # Assemble row: [global params, history params (column-major flatten)]
            X_neuron[curr_row_ind, :] = np.concatenate(
                [parms_global, parms_history.flatten(order='F')]
            )
            curr_row_ind += 1

    # Fill in the response vector
    curr_row_ind = 0
    for curr_trial_ind in range(n_trials):
        for curr_time_ind in range(total_history_samples, len_trial_samples):
            Y_neuron[curr_row_ind] = spikes_data_in[target_neuron, curr_time_ind, curr_trial_ind]
            curr_row_ind += 1

    # Sanity check
    assert curr_row_ind == dim1, "Unexpected number of rows in design matrix"

    # Remove regressors where a 1 is never followed by a spike (Y=1),
    # as the likelihood would diverge in those cases
    n_betas = X_neuron.shape[1]
    inds_to_del = []
    for curr_regr_ind in range(n_betas):
        inds1 = np.where(X_neuron[:, curr_regr_ind] == 1)[0]
        if len(inds1) > 0 and np.sum(Y_neuron[inds1] == 1) == 0:
            inds_to_del.append(curr_regr_ind)

    inds_to_keep = np.setdiff1d(np.arange(n_betas), inds_to_del)
    X = X_neuron[:, inds_to_keep]

    # Fit the GLM (binomial family with logit link, no constant term)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(
            Y_neuron, X,
            family=sm.families.Binomial(link=sm.families.links.Logit())
        )
        result = model.fit(disp=False)

    par_est1 = result.params
    p_vals1 = result.pvalues
    dev = result.deviance

    # Reconstruct full parameter vector; set excluded regressors to -20
    par_est = np.zeros(n_betas)
    par_est_p_vals = np.zeros(n_betas)
    par_est[inds_to_keep] = par_est1
    par_est_p_vals[inds_to_keep] = p_vals1
    if len(inds_to_del) > 0:
        par_est[inds_to_del] = -20.0
        par_est_p_vals[inds_to_del] = 1.0

    # Split parameter vector into global and history components
    beta_global = par_est[:global_regressor['nBins']]
    beta_global_p_vals = par_est_p_vals[:global_regressor['nBins']]

    # Reshape history parameters: (nBinsHistory, nNeurons) column-major → (nNeurons, nBinsHistory)
    tmp = par_est[global_regressor['nBins']:].reshape(
        n_bins_history, n_neurons, order='F'
    ).T
    tmp1 = par_est_p_vals[global_regressor['nBins']:].reshape(
        n_bins_history, n_neurons, order='F'
    ).T

    # Re-insert zero row for the excluded trigger neuron
    if trigger_neuron is not None:
        zero_row = np.zeros((1, n_bins_history))
        one_row = np.ones((1, n_bins_history))
        if trigger_neuron == 0:
            tmp = np.vstack([zero_row, tmp])
            tmp1 = np.vstack([one_row, tmp1])
        elif trigger_neuron == n_neurons_in - 1:
            tmp = np.vstack([tmp, zero_row])
            tmp1 = np.vstack([tmp1, one_row])
        else:
            tmp = np.vstack([tmp[:trigger_neuron, :], zero_row, tmp[trigger_neuron:, :]])
            tmp1 = np.vstack([tmp1[:trigger_neuron, :], one_row, tmp1[trigger_neuron:, :]])

    # Flip columns (time 0 at index 0, matching MATLAB's fliplr)
    beta_history = np.fliplr(tmp)
    beta_history_p_vals = np.fliplr(tmp1)

    return beta_global, beta_history, dev, beta_global_p_vals, beta_history_p_vals
