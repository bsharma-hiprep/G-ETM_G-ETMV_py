"""
This module plots the results obtained from the G-ETM Granger causality method.

Converted from MATLAB to Python.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import scipy.io

from .fdr import fdr
from .compute_raster_and_rate import compute_raster_and_rate


def plot_results_granger(data_file, pars_for_plotting=None, figures_dir=None):
    """Plot results from the G-ETM Granger causality analysis.

    Parameters
    ----------
    data_file : str
        Path to the .npz file containing the saved analysis results.
    pars_for_plotting : dict, optional
        Dictionary of plotting parameters. May include:
          - 'offsetTAxis_s': float, time axis offset in seconds (default 0)
          - 'plotRateAllTrials': bool, whether to plot rates for all trials
          - 'plotGridGlobal': [sideLenX, sideLenY] for grid layout
          - 'plotGridBetas': [sideLenX, sideLenY] for beta grid layout
          - 'fontSizeAxisBetas': int
          - 'fontSizeTitleBetas': int
          - 'limBeta': float, y-axis limit for beta plots
    figures_dir : str, optional
        Directory to save figure files. If None, figures are displayed interactively.
    """
    if pars_for_plotting is None:
        pars_for_plotting = {}

    offset_t_axis_s = pars_for_plotting.get('offsetTAxis_s', 0.0)

    # Load the saved data
    D = np.load(data_file, allow_pickle=True)

    # Extract file base name for figure labeling
    base_name = os.path.splitext(os.path.basename(data_file))[0]

    spike_trains_data = D['SpikeTrains']
    sample_hz = float(D['sample_Hz'])
    out_struct = D['OutStruct'].item()

    # Get information about the data
    n_neurons, len_trial_samples, n_trials = spike_trains_data.shape
    len_trial_s = len_trial_samples / sample_hz

    # Compute raster plots and firing rates
    filt = {
        'type': 'gaussian',
        'causal': False,
        'pars': [0.05]
    }
    spike_trains, spike_trains_smoothed, spike_rates = compute_raster_and_rate(
        spike_trains_data, sample_hz, 1000, filt
    )

    # Compute max rate for scaling
    tmp = np.mean(spike_rates, axis=1)
    max_rate = np.ceil(np.max(tmp) / 5) * 5
    max_max_rate = np.ceil(np.max(spike_rates) / 5) * 5

    p_level = 0.05

    # Compute recovered connectivity matrix with FDR
    fdr_v = p_level
    psi2 = fdr(out_struct['glm_dev_ratio'], fdr_v, out_struct['neuronHistoryRegressorNBins'])

    # ------------------------------------------------------------------
    # Plot spike trains for each neuron
    # ------------------------------------------------------------------
    not_plotted_s = 1.5 * filt['pars'][0]

    for curr_neuron_ind in range(n_neurons):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        fig.canvas.manager.set_window_title(f'spike train - neuron {curr_neuron_ind + 1}')

        # Raster plot
        tmp_smooth = spike_trains_smoothed[curr_neuron_ind, :, :]
        X = np.linspace(0, len_trial_s, tmp_smooth.shape[1]) + offset_t_axis_s
        axes[0].imshow(
            tmp_smooth, aspect='auto', cmap='gray_r',
            extent=[X[0], X[-1], 1, n_trials],
            origin='upper'
        )
        axes[0].set_ylabel('trial #')
        axes[0].axis('off')
        axes[0].set_title(f'neuron {curr_neuron_ind + 1}', fontsize=26)

        # Spike rate plot
        tmp_rate = np.mean(spike_rates[curr_neuron_ind, :, :], axis=0)
        not_plotted_inds = int(round((not_plotted_s / len_trial_s) * len(X)))
        current_max_rate = np.max(tmp_rate[not_plotted_inds:len(X) - not_plotted_inds])

        axes[1].plot(
            X[not_plotted_inds:len(X) - not_plotted_inds],
            tmp_rate[not_plotted_inds:len(X) - not_plotted_inds],
            linewidth=2, color='k'
        )
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('rate (spikes/s)')
        axes[1].set_ylim([0, 1.2 * current_max_rate if current_max_rate > 0 else 1])
        axes[1].set_xlim([X[0], X[-1]])
        axes[1].tick_params(labelsize=20)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        if figures_dir is not None:
            out_name = os.path.join(figures_dir, f'{base_name}_rasters_{curr_neuron_ind + 1}.eps')
            print(f'Printing {out_name}')
            plt.savefig(out_name, format='eps')
        plt.show()

    # ------------------------------------------------------------------
    # Plot all spike trains in a square grid
    # ------------------------------------------------------------------
    if 'plotGridGlobal' not in pars_for_plotting:
        side_len_x = int(np.ceil(np.sqrt(n_neurons)))
        side_len_y = side_len_x
    else:
        side_len_x = pars_for_plotting['plotGridGlobal'][0]
        side_len_y = pars_for_plotting['plotGridGlobal'][1]

    fig, axes_grid = plt.subplots(side_len_x, side_len_y,
                                  figsize=(4 * side_len_y, 4 * side_len_x))
    fig.canvas.manager.set_window_title('spike trains - ALL neurons')

    if n_neurons == 1:
        axes_grid = np.array([[axes_grid]])
    elif side_len_x == 1 or side_len_y == 1:
        axes_grid = np.array(axes_grid).reshape(side_len_x, side_len_y)

    curr_plot_ind = 0
    for x_ind in range(side_len_x):
        for y_ind in range(side_len_y):
            curr_neuron_ind = curr_plot_ind
            if curr_neuron_ind >= n_neurons:
                axes_grid[x_ind, y_ind].axis('off')
                curr_plot_ind += 1
                continue

            ax = axes_grid[x_ind, y_ind]
            spike_rate = np.mean(spike_rates[curr_neuron_ind, :, :], axis=0)
            x_rates_s = np.linspace(0, len_trial_s, len(spike_rate)) + offset_t_axis_s
            ax.plot(x_rates_s, spike_rate, color='k', linewidth=2)
            ax.set_xlim([x_rates_s[0], x_rates_s[-1]])
            ax.set_ylim([0, max_rate if max_rate > 0 else 1])
            ax.tick_params(labelsize=14)
            ax.set_title(f'neuron {curr_neuron_ind + 1}', fontsize=16)

            if y_ind == 0:
                ax.set_ylabel('rate (spikes/s)', fontsize=14)
            if x_ind == side_len_x - 1:
                ax.set_xlabel('time (s)')

            curr_plot_ind += 1

    plt.tight_layout()
    if figures_dir is not None:
        out_name = os.path.join(figures_dir, f'{base_name}_global_AllSquare.eps')
        print(f'Printing {out_name}')
        plt.savefig(out_name, format='eps')
    plt.show()

    # ------------------------------------------------------------------
    # Plot significant interaction functions
    # ------------------------------------------------------------------
    inds_sel = np.argwhere(psi2)
    if 'fontSizeAxisBetas' in pars_for_plotting:
        font_size_axis = pars_for_plotting['fontSizeAxisBetas']
    else:
        font_size_axis = 18
    if 'fontSizeTitleBetas' in pars_for_plotting:
        font_size_title = pars_for_plotting['fontSizeTitleBetas']
    else:
        font_size_title = 20

    if len(inds_sel) > 0:
        inds_x = inds_sel[:, 0]
        inds_y = inds_sel[:, 1]

        if 'plotGridBetas' not in pars_for_plotting:
            side_len_x = int(np.ceil(np.sqrt(len(inds_x))))
            side_len_y = side_len_x
        else:
            side_len_x = pars_for_plotting['plotGridBetas'][0]
            side_len_y = pars_for_plotting['plotGridBetas'][1]

        if 'limBeta' in pars_for_plotting:
            lim_beta = pars_for_plotting['limBeta']
        else:
            lim_beta = 0.0
            for x_ind_val, y_ind_val in zip(inds_x, inds_y):
                sel_ind = out_struct['indsNBinsHistory_neuron'][x_ind_val]
                betas_vals = out_struct['beta_History'][x_ind_val][sel_ind][
                    out_struct['indsNBinsGlobal_neuron'][x_ind_val]
                ]
                betas_vals_row = betas_vals[y_ind_val, :]
                inds_valid = np.where(betas_vals_row != -20)[0]
                if len(inds_valid) > 0:
                    cur_max = np.max(np.abs(betas_vals_row[inds_valid]))
                    if cur_max > lim_beta:
                        lim_beta = cur_max
            lim_beta = np.ceil(lim_beta)

        fig, axes_betas = plt.subplots(
            side_len_x, max(side_len_y, 1),
            figsize=(4 * max(side_len_y, 1), 4 * side_len_x)
        )
        fig.canvas.manager.set_window_title('SIGNIFICANT betas')

        if side_len_x == 1 and side_len_y == 1:
            axes_betas = np.array([[axes_betas]])
        elif side_len_x == 1:
            axes_betas = np.array([axes_betas])
        elif side_len_y == 1:
            axes_betas = np.array([[ax] for ax in axes_betas])

        history_regressor_n_bins = out_struct['historyRegressorNBins']
        history_regressor_bin_dur_ms = float(D['historyRegressor'].item()['binDuration_ms'])

        curr_plot_ind = 0
        for sub_plot_ind in range(len(inds_x)):
            x_ind_val = inds_x[sub_plot_ind]
            y_ind_val = inds_y[sub_plot_ind]

            grid_y = curr_plot_ind % side_len_y
            grid_x = curr_plot_ind // side_len_y

            if grid_x >= side_len_x:
                break

            optimal_global_regressor_ind = out_struct['indsNBinsGlobal_neuron'][x_ind_val]
            sel_ind = out_struct['indsNBinsHistory_neuron'][x_ind_val]
            betas_vals = out_struct['beta_History'][x_ind_val][sel_ind][optimal_global_regressor_ind]
            betas_p = out_struct['beta_History_pVals'][x_ind_val][sel_ind][optimal_global_regressor_ind]
            x_ms = np.arange(history_regressor_n_bins[sel_ind]) * history_regressor_bin_dur_ms
            max_x_ms = (history_regressor_n_bins[-1] - 1) * history_regressor_bin_dur_ms

            ax = axes_betas[grid_x, grid_y]
            if len(x_ms) > 0:
                ax.plot(x_ms, betas_vals[y_ind_val, :], linewidth=3)
                inds_sig = np.where(betas_p[y_ind_val, :] < p_level)[0]
                if len(inds_sig) > 0:
                    ax.plot(x_ms[inds_sig], betas_vals[y_ind_val, inds_sig],
                            'ro', markerfacecolor='r', markersize=8)

            ax.set_ylim([-lim_beta, lim_beta])
            ax.set_xlim([0, max_x_ms])

            if 'Topology' in D.files:
                topology = D['Topology']
                time_topology_ms = D['time_Topology_ms']
                ax.plot(time_topology_ms, topology[x_ind_val, y_ind_val, :], 'k', linewidth=2)

            ax.tick_params(labelsize=font_size_axis)
            ax.set_title(f'{y_ind_val + 1} → {x_ind_val + 1}', fontsize=font_size_title)

            if grid_x == side_len_x - 1:
                ax.set_xlabel('time (ms)', fontsize=font_size_axis)

            curr_plot_ind += 1

        plt.tight_layout()
        if figures_dir is not None:
            out_name = os.path.join(figures_dir, f'{base_name}_selectedBetas.eps')
            print(f'Printing {out_name}')
            plt.savefig(out_name, format='eps')
        plt.show()

    # ------------------------------------------------------------------
    # Plot global regressors
    # ------------------------------------------------------------------
    global_regressor_data = D['globalRegressor'].item()

    # First compute max beta for scaling
    max_beta = -np.inf
    for curr_neuron_ind in range(n_neurons):
        optimal_global_idx = out_struct['indsNBinsGlobal_neuron'][curr_neuron_ind]
        optimal_history_idx = out_struct['indsNBinsHistory_neuron'][curr_neuron_ind]
        y_trial = sample_hz * np.exp(
            out_struct['beta_Global'][curr_neuron_ind][optimal_history_idx][optimal_global_idx]
        )
        max_beta = max(max_beta, np.max(y_trial))

    if len(global_regressor_data['nBins']) > 1:
        for curr_neuron_ind in range(n_neurons):
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.canvas.manager.set_window_title(
                f'global regressors - neuron {curr_neuron_ind + 1}'
            )

            optimal_global_idx = out_struct['indsNBinsGlobal_neuron'][curr_neuron_ind]
            optimal_history_idx = out_struct['indsNBinsHistory_neuron'][curr_neuron_ind]
            bin_duration_ms = global_regressor_data['binDuration_ms'][optimal_global_idx]
            x_trial = (
                np.arange(0, len_trial_s * 1000, bin_duration_ms)
                + bin_duration_ms / 2
                + offset_t_axis_s * 1000
            )
            x_trial_s = x_trial / 1000.0
            y_trial = sample_hz * np.exp(
                out_struct['beta_Global'][curr_neuron_ind][optimal_history_idx][optimal_global_idx]
            )

            ax.plot(x_trial_s[:len(y_trial)], y_trial, linewidth=3)
            inds_sig = np.where(
                out_struct['beta_Global_pVals'][curr_neuron_ind][optimal_history_idx][optimal_global_idx]
                < p_level
            )[0]
            if len(inds_sig) > 0:
                ax.plot(x_trial_s[inds_sig], y_trial[inds_sig], 'ro',
                        markerfacecolor='r', markersize=8)

            # Overlay spike rates
            spike_rate = np.mean(spike_rates[curr_neuron_ind, :, :], axis=0)
            x_rates_s = np.linspace(0, len_trial_s, len(spike_rate)) + offset_t_axis_s
            ax.plot(x_rates_s, spike_rate, color='k', linewidth=2)

            ax.set_xlim([x_rates_s[0], x_rates_s[-1]])
            ax.set_ylim([0, max(max_rate, np.ceil(max_beta / 5) * 5) if max_rate > 0 else 1])
            ax.set_xlabel('time (s)')
            ax.set_ylabel('rate (spikes/s)')
            ax.tick_params(labelsize=22)
            ax.set_title(f'neuron {curr_neuron_ind + 1}', fontsize=28)

            plt.tight_layout()
            if figures_dir is not None:
                out_name = os.path.join(figures_dir,
                                        f'{base_name}_global_{curr_neuron_ind + 1}.eps')
                print(f'Printing {out_name}')
                plt.savefig(out_name, format='eps')
            plt.show()

        # Plot global regressors as a square grid
        if 'plotGridGlobal' not in pars_for_plotting:
            side_len_x = int(np.ceil(np.sqrt(n_neurons)))
            side_len_y = side_len_x
        else:
            side_len_x = pars_for_plotting['plotGridGlobal'][0]
            side_len_y = pars_for_plotting['plotGridGlobal'][1]

        fig, axes_grid = plt.subplots(
            side_len_x, side_len_y, figsize=(4 * side_len_y, 4 * side_len_x)
        )
        fig.canvas.manager.set_window_title('global regressor - ALL neurons')

        if n_neurons == 1:
            axes_grid = np.array([[axes_grid]])
        elif side_len_x == 1 or side_len_y == 1:
            axes_grid = np.array(axes_grid).reshape(side_len_x, side_len_y)

        curr_plot_ind = 0
        for x_ind in range(side_len_x):
            for y_ind in range(side_len_y):
                curr_neuron_ind = curr_plot_ind
                if curr_neuron_ind >= n_neurons:
                    axes_grid[x_ind, y_ind].axis('off')
                    curr_plot_ind += 1
                    continue

                ax = axes_grid[x_ind, y_ind]
                optimal_global_idx = out_struct['indsNBinsGlobal_neuron'][curr_neuron_ind]
                optimal_history_idx = out_struct['indsNBinsHistory_neuron'][curr_neuron_ind]
                bin_duration_ms = global_regressor_data['binDuration_ms'][optimal_global_idx]
                x_trial = (
                    np.arange(0, len_trial_s * 1000, bin_duration_ms)
                    + bin_duration_ms / 2
                    + offset_t_axis_s * 1000
                )
                x_trial_s = x_trial / 1000.0
                y_trial = sample_hz * np.exp(
                    out_struct['beta_Global'][curr_neuron_ind][optimal_history_idx][optimal_global_idx]
                )
                max_beta = max(max_beta, np.max(y_trial))

                ax.plot(x_trial_s[:len(y_trial)], y_trial, linewidth=2)
                inds_sig = np.where(
                    out_struct['beta_Global_pVals'][curr_neuron_ind][optimal_history_idx][optimal_global_idx]
                    < p_level
                )[0]
                if len(inds_sig) > 0:
                    ax.plot(x_trial_s[inds_sig], y_trial[inds_sig], 'ro',
                            markerfacecolor='r', markersize=6)

                spike_rate = np.mean(spike_rates[curr_neuron_ind, :, :], axis=0)
                x_rates_s = np.linspace(0, len_trial_s, len(spike_rate)) + offset_t_axis_s
                ax.plot(x_rates_s, spike_rate, color='k', linewidth=2)

                ax.set_xlim([x_rates_s[0], x_rates_s[-1]])
                ax.set_ylim([0, max(max_rate, np.ceil(max_beta / 5) * 5) if max_rate > 0 else 1])
                ax.tick_params(labelsize=16)
                ax.set_title(f'neuron {curr_neuron_ind + 1}', fontsize=16)

                if y_ind == 0:
                    ax.set_ylabel('rate (spikes/s)', fontsize=16)
                if x_ind == side_len_x - 1:
                    ax.set_xlabel('time (s)', fontsize=18)

                curr_plot_ind += 1

        plt.tight_layout()
        if figures_dir is not None:
            out_name = os.path.join(figures_dir, f'{base_name}_global_AllSquare.eps')
            print(f'Printing {out_name}')
            plt.savefig(out_name, format='eps')
        plt.show()

    # ------------------------------------------------------------------
    # Plot estimates of spike rate coefficients (if available)
    # ------------------------------------------------------------------
    if 'beta_Trials' in out_struct:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title("estimates of spike rates' coefficients")

        col_ini = np.array([1.0, 0.0, 0.0])
        col_end = np.array([0.0, 0.0, 1.0])
        lambdas = np.linspace(0, 1, n_neurons)
        legend_labels = []

        for curr_neuron_ind in range(n_neurons):
            tmp = out_struct['beta_Trials'][curr_neuron_ind][
                out_struct['indsNBinsHistory_neuron'][curr_neuron_ind]
            ][out_struct['indsNBinsGlobal_neuron'][curr_neuron_ind]]
            curr_col = lambdas[curr_neuron_ind] * col_ini + (1 - lambdas[curr_neuron_ind]) * col_end
            legend_labels.append(f'neuron {curr_neuron_ind + 1}')
            ax.plot(np.exp(tmp), color=curr_col, linewidth=2)

        if 'coeffsSpkRate' in D.files:
            ax.plot(D['coeffsSpkRate'][0, :], 'k', linewidth=2)
            legend_labels.append('ground-truth')

        ax.tick_params(labelsize=22)
        ax.set_ylim([0, 4])
        ax.set_xlabel('trial #')
        ax.set_ylabel('resp. magnitude')
        ax.legend(legend_labels)

        plt.tight_layout()
        if figures_dir is not None:
            out_name = os.path.join(figures_dir, f'{base_name}_estimatedCorrVar.eps')
            plt.savefig(out_name, format='eps')
        plt.show()

    # ------------------------------------------------------------------
    # Plot ground-truth spike rate coefficients (if available)
    # ------------------------------------------------------------------
    if 'coeffsSpkRate' in D.files:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title("spike rates' coefficients")
        ax.plot(D['coeffsSpkRate'].T, color='k', linewidth=3)
        ax.set_xlabel('trial #')
        ax.set_ylabel('resp. magnitude')
        ax.tick_params(labelsize=22)
        ax.set_ylim([0, 3])

        plt.tight_layout()
        if figures_dir is not None:
            out_name = os.path.join(figures_dir, f'{base_name}_groundTruthCorrVar.eps')
            print(f'Printing {out_name}')
            plt.savefig(out_name, format='eps')
        plt.show()

    # ------------------------------------------------------------------
    # Plot recovered topology
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.canvas.manager.set_window_title('Recovered Topology')
    cmap = matplotlib.colors.ListedColormap([[0, 0, 0], [0, 1, 0]])
    ax.imshow(psi2, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(np.arange(n_neurons))
    ax.set_xticklabels(np.arange(1, n_neurons + 1))
    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(np.arange(1, n_neurons + 1))
    ax.set_xlabel('source', fontsize=42)
    ax.set_ylabel('target', fontsize=42)
    ax.tick_params(labelsize=22)

    plt.tight_layout()
    if figures_dir is not None:
        out_name = os.path.join(figures_dir, f'{base_name}_recoveredConnectivity.eps')
        print(f'Printing {out_name}')
        plt.savefig(out_name, format='eps')
    plt.show()

    # ------------------------------------------------------------------
    # Plot ground-truth topology (if available)
    # ------------------------------------------------------------------
    if 'Topology' in D.files:
        topology = D['Topology']
        tmp = np.sum(np.abs(topology), axis=2)

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.canvas.manager.set_window_title('Ground-truth Topology')
        ax.imshow((tmp != 0).astype(float), cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(np.arange(topology.shape[0]))
        ax.set_xticklabels(np.arange(1, topology.shape[0] + 1))
        ax.set_yticks(np.arange(topology.shape[0]))
        ax.set_yticklabels(np.arange(1, topology.shape[0] + 1))
        ax.set_xlabel('source', fontsize=42)
        ax.set_ylabel('target', fontsize=42)
        ax.tick_params(labelsize=22)

        plt.tight_layout()
        if figures_dir is not None:
            out_name = os.path.join(figures_dir, f'{base_name}_realConnectivity.eps')
            plt.savefig(out_name, format='eps')
        plt.show()
