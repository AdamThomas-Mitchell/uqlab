import math
import numpy as np
import pandas as pd
import torch
import gpytorch

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import uqlab
from uqlab.preprocessing import DataLoader, WaterDimerLoader
from uqlab.models import manchester_kernel
from uqlab.calibration import Crude

from tptorch.distributions import MultivariateStudentT
from tptorch.likelihoods import StudentTLikelihood
from tptorch.mlls import ExactStudentTMarginalLogLikelihood
from tptorch.models import ExactTP

torch.set_default_dtype(torch.float64)


class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = manchester_kernel(train_x.shape[1], base_kernel='matern52')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class StudentTProcess(ExactTP):
    def __init__(self, train_x, train_y, nu=5):
        likelihood = StudentTLikelihood()
        super(StudentTProcess, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = manchester_kernel(train_x.shape[1], base_kernel='matern52')

        nu = torch.tensor(nu, dtype=torch.float64)
        self.nu = torch.nn.Parameter(nu)
        self.data_num = torch.tensor(self.train_targets.shape[0], dtype=torch.float64)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x_train_data = self.covar_module(self.train_inputs[0])
        inv_quad, _ = covar_x_train_data.inv_quad_logdet(
            inv_quad_rhs=self.train_targets - self.train_targets.mean(), logdet=False
        )

        tp_var_scale = (self.nu + inv_quad - 2) / (self.nu + self.data_num - 2)

        covar_x = tp_var_scale.float() * covar_x

        return MultivariateStudentT(mean_x, covar_x, self.nu, self.data_num)


def GP_atom_predictions(atom):
    X_train = water_dimer_data[atom]['train'][0]
    y_train = water_dimer_data[atom]['train'][1]
    X_cal = water_dimer_data[atom]['cal'][0]
    y_cal = water_dimer_data[atom]['cal'][1]
    X_test = water_dimer_data[atom]['test'][0]
    y_test = water_dimer_data[atom]['test'][1]

    # instantiate model
    model = GaussianProcess(X_train, y_train)

    # use gpu if available - may need to extend to multiple
    if torch.cuda.is_available():
        model = model.cuda()

    # prepare for training
    model.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.9)

    # training loop
    loss_list = []
    lr_list = []
    with gpytorch.settings.max_cholesky_size(0):
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss = loss.mean()

            # record loss and learning rate
            loss_list.append(loss.detach().numpy())
            lr_list.append(optimizer.param_groups[0]["lr"])

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

    # Get into evaluation (predictive posterior) mode
    model.eval()

    # predict on test set - get predicted mean and std
    with torch.no_grad(),\
            gpytorch.settings.max_cholesky_size(0), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.max_preconditioner_size(200):

        f_pred = model(X_test)
        y_pred = model.likelihood(f_pred)

        y_pred_mean = y_pred.mean
        y_pred_var = y_pred.variance
        y_pred_std = y_pred_var.sqrt()

    # reshape tensors to np arrays
    y_pred_mean = y_pred_mean.cpu().detach().numpy()
    y_pred_std = y_pred_std.cpu().detach().numpy()
    y_test_final = y_test.cpu().detach().numpy()

    # predict on calibration set, initialise calibrator and access empirical distribution
    with torch.no_grad(), \
            gpytorch.settings.max_cholesky_size(0), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.max_preconditioner_size(200):

        f_pred_cal = model(X_cal)
        y_pred_cal = model.likelihood(f_pred_cal)

        # take off GPU
        mu_cal = y_pred_cal.mean
        sigmaSqr_cal = y_pred_cal.variance
        sigma_cal = sigmaSqr_cal.sqrt()

    # reshape tensors to np arrays
    mu_cal = mu_cal.cpu().detach().numpy()
    sigma_cal = sigma_cal.cpu().detach().numpy()
    y_cal_final = y_cal.cpu().detach().numpy()

    return y_test_final, y_pred_mean, y_pred_std, y_cal_final, mu_cal, sigma_cal


def TP_atom_predictions(atom):
    X_train = water_dimer_data[atom]['train'][0]
    y_train = water_dimer_data[atom]['train'][1]
    X_cal = water_dimer_data[atom]['cal'][0]
    y_cal = water_dimer_data[atom]['cal'][1]
    X_test = water_dimer_data[atom]['test'][0]
    y_test = water_dimer_data[atom]['test'][1]

    # instantiate model
    model = StudentTProcess(X_train, y_train)

    # use gpu if available - may need to extend to multiple
    if torch.cuda.is_available():
        model = model.cuda()

    # prepare for training
    model.train()
    mll = ExactStudentTMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.9)

    # training loop
    loss_list = []
    lr_list = []
    with gpytorch.settings.max_cholesky_size(0):
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss = loss.mean()

            # record loss and learning rate
            loss_list.append(loss.detach().numpy())
            lr_list.append(optimizer.param_groups[0]["lr"])

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            with torch.no_grad():
                model.nu.clamp_(3.0)

    # Get into evaluation (predictive posterior) mode
    model.eval()

    # predict on test set - get predicted mean and std
    with torch.no_grad(), \
            gpytorch.settings.max_cholesky_size(0), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.max_preconditioner_size(200):

        f_pred = model(X_test)
        y_pred = model.likelihood(f_pred)

        y_pred_mean = y_pred.mean
        y_pred_var = y_pred.variance
        y_pred_std = y_pred_var.sqrt()

    # reshape tensors to np arrays
    y_pred_mean = y_pred_mean.cpu().detach().numpy()
    y_pred_std = y_pred_std.cpu().detach().numpy()
    y_test_final = y_test.cpu().detach().numpy()

    # predict on calibration set, initialise calibrator
    with torch.no_grad(), \
            gpytorch.settings.max_cholesky_size(0), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.max_preconditioner_size(200):

        f_pred_cal = model(X_cal)
        y_pred_cal = model.likelihood(f_pred_cal)

        # take off GPU
        mu_cal = y_pred_cal.mean
        sigmaSqr_cal = y_pred_cal.variance
        sigma_cal = sigmaSqr_cal.sqrt()

    # reshape tensors to np arrays
    mu_cal = mu_cal.cpu().detach().numpy()
    sigma_cal = sigma_cal.cpu().detach().numpy()
    y_cal_final = y_cal.cpu().detach().numpy()

    return y_test_final, y_pred_mean, y_pred_std, y_cal_final, mu_cal, sigma_cal

# define variables
training_iter = int(300)
n_train = int(4000)
n_cal = int(0.5*n_train)
n_test = int(4000)

# initialise dataloader and get data dictionary
data_loader = WaterDimerLoader(
    n_train=n_train,
    n_cal=n_cal,
    n_test=n_test,
    scale_output=True,
    as_tensor=True,
    random_state=None
)
water_dimer_data = data_loader.data_dict

# initialise dataframes
results_df_GP = None
recal_df_GP = None
results_df_TP = None
recal_df_TP = None

for atom in water_dimer_data.keys():

    # get GP predictions
    y_true_GP, y_pred_mean_GP, y_pred_std_GP, y_cal_GP, mu_cal_GP, sigma_cal_GP = GP_atom_predictions(atom)
    y_true_GP, y_pred_mean_GP, y_pred_std_GP = data_loader.unscale_targets(y_true_GP, y_pred_mean_GP, y_pred_std_GP, atom)
    y_cal_GP, mu_cal_GP, sigma_cal_GP = data_loader.unscale_targets(y_cal_GP, mu_cal_GP, sigma_cal_GP, atom)
    calibrator_GP = Crude(0, y_cal_GP, mu_cal_GP, sigma_cal_GP)

    # get TP predictions
    y_true_TP, y_pred_mean_TP, y_pred_std_TP, y_cal_TP, mu_cal_TP, sigma_cal_TP = TP_atom_predictions(atom)
    y_true_TP, y_pred_mean_TP, y_pred_std_TP = data_loader.unscale_targets(y_true_TP, y_pred_mean_TP, y_pred_std_TP, atom)
    y_cal_TP, mu_cal_TP, sigma_cal_TP = data_loader.unscale_targets(y_cal_TP, mu_cal_TP, sigma_cal_TP, atom)
    calibrator_TP = Crude(0, y_cal_TP, mu_cal_TP, sigma_cal_TP)

    # record results to dataframe
    atom_results_GP = pd.DataFrame({
        f'{atom}_y_true': y_true_GP,
        f'{atom}_pred_mean': y_pred_mean_GP,
        f'{atom}_pred_std': y_pred_std_GP
    })
    atom_results_TP = pd.DataFrame({
        f'{atom}_y_true': y_true_TP,
        f'{atom}_pred_mean': y_pred_mean_TP,
        f'{atom}_pred_std': y_pred_std_TP
    })

    # get empirical noise distribution and record to separate dataframe
    epsilon_GP = calibrator_GP.epsilon_estimate
    recal_results_GP = pd.DataFrame(
        {f'{atom}_noise_dist': epsilon_GP},
    )
    epsilon_TP = calibrator_TP.epsilon_estimate
    recal_results_TP = pd.DataFrame(
        {f'{atom}_noise_dist': epsilon_TP},
    )

    # GP results to dataframes
    if results_df_GP is None:
        results_df_GP = atom_results_GP
    else:
        results_df_GP = pd.concat([results_df_GP, atom_results_GP], axis=1)

    if recal_df_GP is None:
        recal_df_GP = recal_results_GP
    else:
        recal_df_GP = pd.concat([recal_df_GP, recal_results_GP], axis=1)

    # results to dataframes
    if results_df_TP is None:
        results_df_TP = atom_results_TP
    else:
        results_df_TP = pd.concat([results_df_TP, atom_results_TP], axis=1)

    if recal_df_TP is None:
        recal_df_TP = recal_results_TP
    else:
        recal_df_TP = pd.concat([recal_df_TP, recal_results_TP], axis=1)

# results to csv files
results_df_GP.to_csv('waterDimer_GP_matern_4000pts_results.csv')
recal_df_GP.to_csv('waterDimer_GP_matern_4000pts_recal.csv')
results_df_TP.to_csv('waterDimer_TP_matern_4000pts_results.csv')
recal_df_TP.to_csv('waterDimer_TP_matern_4000pts_recal.csv')
