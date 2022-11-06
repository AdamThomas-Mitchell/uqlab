"""
Metrics to quantify accuracy and uncertainty for probabilistic regression models
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union


def mean_absolute_error(y_true, y_pred_mean):
    """
    Calculate mean absolute error for test samples

    :param y_true:          true values for test samples
    :param y_pred_mean:     predicted mean for test samples
    :return mae:            mean absolute error
    """
    y_true = y_true.flatten()
    y_pred_mean.flatten()

    mae = np.mean(np.abs(y_true - y_pred_mean))
    return mae


def median_absolute_error(y_true, y_pred_mean):
    """
    Calculate median absolute error for test samples

    :param y_true:          true values for test samples
    :param y_pred_mean:     predicted mean for test samples
    :return mdae:           median absolute error
    """
    y_true = y_true.flatten()
    y_pred_mean.flatten()

    mdae = np.median(np.abs(y_true - y_pred_mean))
    return mdae


def root_mean_sq_error(y_true, y_pred_mean):
    """
    Calculate root mean squared error for test samples

    :param y_true:          true values for test samples
    :param y_pred_mean:     predicted mean for test samples
    :return rmse:           root mean squared error
    """
    y_true = y_true.flatten()
    y_pred_mean.flatten()

    mse = np.mean(np.square(y_true - y_pred_mean))
    rmse = np.sqrt(mse)
    return rmse


def mean_abs_rel_percent_diff(y_true, y_pred_mean):
    """
    Calculate mean absolute relative percent difference for test samples

    :param y_true:          true values for test samples
    :param y_pred_mean:     predicted mean for test samples
    :return marpd:          mean absolute relative percent difference
    """
    y_true = y_true.flatten()
    y_pred_mean.flatten()

    rpd = 100 * (y_pred_mean - y_true) / (0.5 * (np.abs(y_pred_mean) + np.abs(y_true)))
    marpd = np.mean(np.abs(rpd))
    return marpd


def r_squared(y_true, y_pred_mean):
    """
    Calculate r squared  test samples

    :param y_true:          true values for test samples
    :param y_pred_mean:     predicted mean for test samples
    :return marpd:          mean absolute relative percent difference
    """
    y_true = y_true.flatten()
    y_pred_mean.flatten()

    mean_y_true = np.mean(y_true)
    mse = np.mean(np.square(y_true - y_pred_mean))
    r_sq = 1 - mse / np.mean(np.square(y_true - mean_y_true * np.ones_like(y_true)))
    return r_sq


def accuracy_metrics(y_true, y_pred_mean):
    """
    Return all accuracy metrics as pandas data frame

    :param y_true:          true values for test samples
    :param y_pred_mean:     predicted mean for test samples
    :return:                all accuracy metrics
    """
    mae = mean_absolute_error(y_true, y_pred_mean)
    mdae = median_absolute_error(y_true, y_pred_mean)
    rmse = root_mean_sq_error(y_true, y_pred_mean)
    marpd = mean_abs_rel_percent_diff(y_true, y_pred_mean)
    r_sq = r_squared(y_true, y_pred_mean)

    metrics = {
        "mae": mae,
        "mdae": mdae,
        "rmse": rmse,
        "marpd": marpd,
        "r2": r_sq
    }
    metrics_df = pd.DataFrame(metrics, index=["Accuracy"])

    return metrics_df


def proportion_in_interval(y_true, y_pred_mean, y_pred_std, quantile_level, calibrator=None):
    """
    This may need changed
    """
    # ensure arrays are correct shape
    y_true = y_true.flatten()
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()

    # for uncalibrated uncertainty
    if calibrator is None:

        # calculate p values
        p_lower = 0.5 * (1.0 - quantile_level)
        p_upper = 1.0 - p_lower

        p_lower_arr = np.full_like(y_pred_mean, p_lower)
        p_upper_arr = np.full_like(y_pred_mean, p_upper)

        lower_bnd_arr = norm.ppf((p_lower_arr, y_pred_mean, y_pred_std))
        upper_bnd_arr = norm.ppf((p_upper_arr, y_pred_mean, y_pred_std))

        num_in_range = np.sum(np.logical_and(lower_bnd_arr <= y_true, y_true <= upper_bnd_arr))

    # for calibrated uncertainty
    else:
        # check tuple type
        lower_bnd, upper_bnd = calibrator.calibrate_interval(y_pred_mean, y_pred_std, quantile_level)
        num_in_range = np.sum(np.logical_and(lower_bnd <= y_true, y_true <= upper_bnd))

    prop_in_range = num_in_range / len(y_true)

    return prop_in_range


def get_proportion_lists(y_true, y_pred_mean, y_pred_std, calibrator):
    # initialise arrays for expected and observed proportion of samples in interval
    exp_prop = np.arange(0.0, 1.0, 0.01)
    obs_prop = np.zeros_like(exp_prop)

    for i in range(len(exp_prop)):
        conf_lvl = exp_prop[i]  # set confidence level
        prop_in_range = proportion_in_interval(y_true, y_pred_mean, y_pred_std, conf_lvl, calibrator)
        obs_prop[i] = prop_in_range

    return exp_prop, obs_prop


def root_mean_sq_calibration_error(y_true, y_pred_mean, y_pred_std, calibrator=None):
    # ensure arrays are correct shape
    y_true = y_true.flatten()
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()

    exp_prop, obs_prop = get_proportion_lists(y_true, y_pred_mean, y_pred_std, calibrator)

    # calculate root mean sq calibration error
    sq_err = np.square(exp_prop - obs_prop)
    rmsce = np.sqrt(np.mean(sq_err))

    return rmsce


def miscalibration_area(y_true, y_pred_mean, y_pred_std, calibrator=None):
    # ensure arrays are correct shape
    y_true = y_true.flatten()
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()

    exp_prop, obs_prop = get_proportion_lists(y_true, y_pred_mean, y_pred_std, calibrator)

    # THIS SECTION IS PLAGIARISED - ADAPT
    # Compute approximation to area between curves
    polygon_points = []
    for point in zip(exp_prop, obs_prop):
        polygon_points.append(point)
    for point in zip(reversed(exp_prop), reversed(exp_prop)):
        polygon_points.append(point)
    polygon_points.append((exp_prop[0], obs_prop[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy
    ls = LineString(np.c_[x, y])
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscal_area = np.asarray(polygon_area_list).sum()
    #####

    return miscal_area


def sharpness(y_pred_mean, y_pred_std, calibrator=None):
    """
    Sharpness as a measure of width of 95% confidence interval

    :param y_pred_mean:
    :param y_pred_std:          predicted standard deviation for test samples
    :param calibrator:          quantile levels of calibrated 95% confidence interval
    :return:                    Single number representing sharpness of confidence intervals
    """
    y_pred_mean = y_pred_mean.flatten()
    y_pred_std = y_pred_std.flatten()

    # calculate average width of 95% confidence interval for uncalibrated models
    if calibrator is None:
        sha = np.mean(3.92 * y_pred_std)

    else:
        lower_quantile, upper_quantile = calibrator.calibrate_interval(y_pred_mean, y_pred_std, 0.95)
        interval_width = upper_quantile - lower_quantile
        sha = np.mean(interval_width)

    return sha


def uq_metrics(y_true, y_pred_mean, y_pred_std, calibrator=None):
    """
    Return all metrics relating to uncertainty quantification as pandas dataframe
    """
    rmsce = root_mean_sq_calibration_error(y_true, y_pred_mean, y_pred_std, calibrator)
    miscal_area = miscalibration_area(y_true, y_pred_mean, y_pred_std, calibrator)
    sha = sharpness(y_pred_mean, y_pred_std, calibrator)

    metrics = {
        "rms_cal": rmsce,
        "miscal_area": miscal_area,
        "sharp": sha
    }
    metrics_df = pd.DataFrame(metrics, index=["UQ"])

    return metrics_df


def all_metrics(y_true, y_pred_mean, y_pred_std, calibrator=None):
    """
    Return all metrics for accuracy, calibration, and sharpness as pandas DataFrame
    """
    mae = mean_absolute_error(y_true, y_pred_mean)
    mdae = median_absolute_error(y_true, y_pred_mean)
    rmse = root_mean_sq_error(y_true, y_pred_mean)
    marpd = mean_abs_rel_percent_diff(y_true, y_pred_mean)
    r_sq = r_squared(y_true, y_pred_mean)
    rmsce = root_mean_sq_calibration_error(y_true, y_pred_mean, y_pred_std, calibrator)
    miscal_area = miscalibration_area(y_true, y_pred_mean, y_pred_std, calibrator)
    sha = sharpness(y_pred_mean, y_pred_std, calibrator)

    metrics = {
        "mae": mae,
        "mdae": mdae,
        "rmse": rmse,
        "marpd": marpd,
        "r2": r_sq,
        "rms_cal": rmsce,
        "miscal_area": miscal_area,
        "sharp": sha
    }
    metrics_df = pd.DataFrame(metrics, index=["all_metrics"])

    return metrics_df
