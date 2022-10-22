"""
Plots to show accuracy, and uncertainty by measure of calibration, sharpness, and dispersion
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uncertainty_toolbox as uct
from uqlab import metrics


def parity_plot(y_true, y_pred, plot_type='scatter', hist_bins=40, hexbin_size=40):
    """
    Plots parity plot to represent accuracy
    :param y_true:
    :param y_pred:
    :param plot_type:
    :param hist_bins:
    :param hexbin_size:
    """
    if plot_type == 'scatter':
        fig = sns.jointplot(x=y_true, y=y_pred, kind="scatter", color="#4CB391",
                            marginal_kws=dict(fill=True))
        fig.ax_joint.plot(y_true, y_true, 'r', alpha=0.5, linestyle='--')
        fig.set_axis_labels(xlabel='true values', ylabel='predicted values')
        # plt.show()

    elif plot_type == 'hex':
        fig = sns.jointplot(x=y_true, y=y_pred, kind="hex", color="#4CB391", joint_kws=dict(gridsize=hexbin_size),
                            marginal_kws=dict(bins=hist_bins, fill=True))
        fig.ax_joint.plot(y_true, y_true, 'r', alpha=0.5, linestyle='--')
        fig.set_axis_labels(xlabel='true values', ylabel='predicted values')
        # plt.show()

    else:
        print("plot_type arg must be 'hex' or 'scatter'.")


def calibration_plot(y_pred_mean, y_pred_std, y_test, calibrator=None, title=''):
    """

    :param y_pred_mean:
    :param y_pred_std:
    :param y_test:
    :param calibrator:
    """
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()
    y_test = y_test.flatten()

    if calibrator is None:
        uct.viz.plot_calibration(y_pred_mean, y_pred_std, y_test)

    else:
        exp_prop, obs_prop = metrics.get_proportion_lists(y_test, y_pred_mean, y_pred_std, calibrator)
        uct.viz.plot_calibration(y_pred_mean, y_pred_std, y_test, exp_props=exp_prop, obs_props=obs_prop)

    # plt.title(title)
    # plt.show()


def sharpness_plot(y_pred_mean, y_pred_std, calibrator=None, bins=30, title=''):
    """

    :param y_pred_mean:
    :param y_pred_std:
    :param calibrator:
    :param bins:
    """
    # for uncalibrated uncertainty
    if calibrator is None:
        sharp_vals = 3.92 * y_pred_std

    # for calibrated uncertainty
    else:
        lower_bnd, upper_bnd = calibrator.calibrate_interval(y_pred_mean, y_pred_std)
        sharp_vals = upper_bnd - lower_bnd

    fig = sns.histplot(sharp_vals, bins=bins, stat='probability')
    plt.title(title)
    plt.show()


def s_curve(y_true, y_pred_mean):

    y_pred_mean = y_pred_mean.flatten()
    y_true = y_true.flatten()

    residuals = np.abs(y_true - y_pred_mean)
    sorted_residuals = np.sort(residuals)

    n_samples = len(sorted_residuals)
    percentile = np.linspace(0, 100, num=n_samples)

    plt.plot(sorted_residuals, percentile)
    plt.xscale('log')
    plt.xlabel('Prediction Error')
    plt.ylabel('Percentile')
    plt.show()


def all_plots(y_pred_mean, y_pred_std, y_true, title='', calibrator=None):
    # ensure data in correct format
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()
    y_true = y_true.flatten()

    # define subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # s-curve
    residuals = np.abs(y_true - y_pred_mean)
    sorted_residuals = np.sort(residuals)
    n_samples = len(sorted_residuals)
    percentile = np.linspace(0, 100, num=n_samples)

    axs[0].plot(sorted_residuals, percentile)
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Prediction Error')
    axs[0].set_ylabel('Percentile')
    axs[0].set_title('Prediction Error S-Curve')

    # parity plot
    axs[1] = sns.scatterplot(x=y_true, y=y_pred_mean, color="#4CB391", ax=axs[1])
    axs[1].plot(y_true, y_true, 'r', alpha=0.7, linestyle='--')
    axs[1].set_xlabel('True Values')
    axs[1].set_ylabel('Predicted Values')
    axs[1].set_title('Accuracy')

    # for uncalibrated uncertainty
    if calibrator is None:

        # calibration plot
        axs[2] = uct.plot_calibration(y_pred_mean, y_pred_std, y_true, ax=axs[2])

        # sharpness plot
        sharp_vals = 3.92 * y_pred_std
        axs[3] = sns.histplot(sharp_vals, bins=30, stat='probability', ax=axs[3])
        axs[3].set_xlabel('95% Confidence Interval Width')
        axs[3].set_title('Sharpness')

    # for calibrated uncertainty
    else:

        # calibration plot
        exp_prop, obs_prop = metrics.get_proportion_lists(y_true, y_pred_mean, y_pred_std, calibrator)
        axs[2] = uct.viz.plot_calibration(y_pred_mean, y_pred_std, y_true,
                                          exp_props=exp_prop, obs_props=obs_prop, ax=axs[2])

        # sharpness plot
        lower_bnd, upper_bnd = calibrator.calibrate_interval(y_pred_mean, y_pred_std)
        sharp_vals = upper_bnd - lower_bnd
        axs[3] = sns.histplot(sharp_vals, bins=30, stat='probability', ax=axs[3])
        axs[3].set_xlabel('95% Confidence Interval Width')
        axs[3].set_title('Sharpness')

    fig.suptitle(title)
    plt.show()
