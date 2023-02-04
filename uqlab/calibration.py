"""
Class to perform post-hoc calibration using CRUDE algorithm
"""
import numpy as np


class Crude:
    def __init__(self, x_cal, y_cal, mu_cal, sigma_cal):
        """
        :param x_cal:           X values for calibration set
        :param y_cal:           y values for calibration set
        :param mu_cal:                predicted mean for values in calibration set
        :param sigma_cal:             predicted standard deviation for values in calibration set
        """
        self.x_cal = x_cal  # could probably remove this
        self.y_cal = y_cal.flatten()
        self.mu_cal = mu_cal.flatten()
        self.sigma_cal = sigma_cal.flatten()
        self.epsilon_estimate = self.estimate_noise_distribution()
        self.epsilon_expected_val = self.calculate_epsilon_expected_val()
        self.epsilon_variance = self.calculate_epsilon_variance()

    def estimate_noise_distribution(self):
        """
        Estimate cdf of Z scores - see CRUDE paper for details
        """
        unsorted_noise_vals = (self.y_cal - self.mu_cal) / self.sigma_cal
        epsilon_estimate = np.sort(unsorted_noise_vals)

        return epsilon_estimate

    def calibrate_interval(self, mu_test, sigma_test, confidence_level=0.95):
        """
        Produce calibrated upper and lower quantile levels for given confidence level using
        CRUDE algorithm - algorithm 1 in CRUDE paper

        :param mu_test:             predicted mean for test set
        :param sigma_test:          predicted standard deviations for test set
        :param confidence_level:    given confidence level for confidence interval

        """
        p_lower = 0.5 * (1.0 - confidence_level)
        p_upper = 1.0 - p_lower

        # set lower and upper Z-score
        z_length = len(self.epsilon_estimate)
        z_lower = self.epsilon_estimate[int(p_lower * z_length)]
        z_upper = self.epsilon_estimate[int(p_upper * z_length)]

        # calculate lower and upper quantile
        lower_quantile = mu_test + (sigma_test * z_lower)
        upper_quantile = mu_test + (sigma_test * z_upper)  # check if variance or std required

        return lower_quantile, upper_quantile

    def calculate_epsilon_expected_val(self):
        epsilon_expected_val = np.mean(self.epsilon_estimate)
        return epsilon_expected_val

    def calculate_epsilon_variance(self):

        epsilon = self.epsilon_estimate
        epsilon_exp_val = self.epsilon_expected_val

        summation = np.square(epsilon - np.full_like(epsilon, epsilon_exp_val))
        epsilon_var = np.mean(summation)

        return epsilon_var

    def calibrate_mean(self, mu_test, sigma_test):
        mean_recal = mu_test + sigma_test*self.epsilon_expected_val
        return mean_recal

    def calibrate_variance(self, sigma_test):
        var_recal = np.square(sigma_test) * self.epsilon_variance
        return var_recal
