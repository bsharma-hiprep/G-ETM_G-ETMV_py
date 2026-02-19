"""
Given the spike trains across trials (spike_trains_in), this function returns
a matrix containing the raster plot and one curve containing the smoothed spike
rate averaged across trials.

Converted from MATLAB to Python.
"""

import numpy as np
from scipy.signal import filtfilt


def compute_raster_and_rate(spike_trains_in, samp_freq_hz, n_pixels, filter_type=None):
    """Compute raster plots and smoothed spike rates from spike train data.

    Parameters
    ----------
    spike_trains_in : np.ndarray
        Spike train data, shape (nNeurons, nSamples, nTrials).
    samp_freq_hz : float
        Sampling frequency in Hz.
    n_pixels : int
        Number of pixels (time bins) for the output raster/rate.
    filter_type : dict, optional
        Dictionary specifying the smoothing filter with keys:
          - 'type': 'triangle', 'boxcar', or 'gaussian'
          - 'causal': bool, whether to use a causal filter
          - 'pars': list of parameters (e.g., [sigma_in_seconds])
        Defaults to a triangular kernel with sigma=0.1 s.

    Returns
    -------
    spike_trains : np.ndarray
        Raster plots, shape (nNeurons, nTrials, n_pixels).
    spike_trains_smoothed : np.ndarray
        Smoothed (expanded) raster plots, shape (nNeurons, nTrials, n_pixels).
    spike_rates : np.ndarray
        Smoothed spike rates, shape (nNeurons, nTrials, n_pixels).
    """
    if filter_type is None:
        # Default: triangular kernel with sigma=0.1 s (100 ms)
        filter_type = {
            'type': 'triangle',
            'causal': False,
            'pars': [0.1]
        }

    samp_time_s = 1.0 / samp_freq_hz

    # Build smoothing kernel based on filter type
    ft = filter_type['type']
    causal = filter_type.get('causal', False)

    if ft == 'triangle':
        sig = filter_type['pars'][0]
        T2 = np.arange(0, np.sqrt(6) * sig + samp_time_s, samp_time_s)
        T1 = np.arange(0, -(np.sqrt(6) * sig) - samp_time_s, -samp_time_s)
        T1 = T1[::-1]
        T = np.concatenate([T1[:-1], T2])
        KS = (1.0 / (6 * sig ** 2)) * (np.sqrt(6) * sig - np.abs(T))
        if causal:
            ind_max = np.argmax(KS)
            KS[:ind_max] = 0
            KS = KS / (np.sum(KS) * samp_time_s)

    elif ft == 'boxcar':
        sig = filter_type['pars'][0]
        T2 = np.arange(0, sig / 2 + samp_time_s, samp_time_s)
        T1 = np.arange(0, -(sig / 2) - samp_time_s, -samp_time_s)
        T1 = T1[::-1]
        T = np.concatenate([T1[:-1], T2])
        KS = np.ones_like(T)
        if causal:
            ind_zero = np.argmin(np.abs(T))
            KS[:ind_zero + 1] = 0
        KS = KS / (np.sum(KS) * samp_time_s)

    elif ft == 'gaussian':
        sig = filter_type['pars'][0]
        T2 = np.arange(0, 3 * sig + samp_time_s, samp_time_s)
        T1 = np.arange(0, -3 * sig - samp_time_s, -samp_time_s)
        T1 = T1[::-1]
        T = np.concatenate([T1[:-1], T2])
        KS = (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-(T ** 2) / (2 * sig ** 2))
        if causal:
            ind_max = np.argmax(KS)
            KS[:ind_max] = 0
            KS = KS / (np.sum(KS) * samp_time_s)

    else:
        raise ValueError(f"Unknown filter type: '{ft}'")

    # Get dimensions of the spike trains
    n_neurons, n_samples, n_trials = spike_trains_in.shape

    # Output arrays
    spike_trains = np.zeros((n_neurons, n_trials, n_pixels))
    spike_trains_smoothed = np.zeros((n_neurons, n_trials, n_pixels))
    spike_rates = np.zeros((n_neurons, n_trials, n_pixels))

    # Kernel for expanding spikes in raster (to improve visibility)
    K = np.array([1.0, 1.0])

    from_x = np.linspace(0, 1, n_samples)
    to_x = np.linspace(0, 1, n_pixels)

    for curr_neuron_ind in range(n_neurons):
        for curr_trial_ind in range(n_trials):
            from_spikes = spike_trains_in[curr_neuron_ind, :, curr_trial_ind].astype(float)

            # Map spike positions to pixel positions
            to_spikes = np.zeros(n_pixels)
            inds = np.where(from_spikes)[0]
            if len(inds) > 0:
                pixel_inds = np.ceil(inds * (n_pixels / n_samples)).astype(int)
                pixel_inds = np.clip(pixel_inds, 0, n_pixels - 1)
                to_spikes[pixel_inds] = 1.0
            spike_trains[curr_neuron_ind, curr_trial_ind, :] = to_spikes

            # Smooth raster with small kernel to improve visibility
            tmp1 = filtfilt(K, [1.0], to_spikes)
            tmp1[tmp1 > 1] = 1.0
            spike_trains_smoothed[curr_neuron_ind, curr_trial_ind, :] = tmp1

            # Compute spike rates by convolution then interpolate.
            # Use full convolution and extract central part (same length as
            # from_spikes), matching MATLAB's conv(a, b, 'same') behavior.
            full_conv = np.convolve(from_spikes, KS)
            start_idx = (len(KS) - 1) // 2
            curr_rates = full_conv[start_idx:start_idx + n_samples]
            spike_rates[curr_neuron_ind, curr_trial_ind, :] = np.interp(to_x, from_x, curr_rates)

    return spike_trains, spike_trains_smoothed, spike_rates
