import math
import numpy as np
import torch
import gpytorch
import tptorch
from .kernels import RBFManchesterKernel, MaternManchesterKernel


class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel='MaternManchesterKernel'):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'MaternManchesterKernel':
            self.covar_module = MaternManchesterKernel(train_x.shape[1])
        elif kernel == 'RBFManchesterKernel':
            self.covar_module = RBFManchesterKernel(train_x.shape[1])
        else:
            raise Exception("kernel arg must be 'RBFManchesterKernel' or 'MaternManchesterKernel'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class StudentTProcess(tptorch.models.ExactTP):
    def __init__(self, train_x, train_y, nu=5, kernel='MaternManchesterKernel'):
        likelihood = tptorch.likelihoods.StudentTLikelihood()
        super(StudentTProcess, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'MaternManchesterKernel':
            self.covar_module = MaternManchesterKernel(train_x.shape[1])
        elif kernel == 'RBFManchesterKernel':
            self.covar_module = RBFManchesterKernel(train_x.shape[1])
        else:
            raise Exception("kernel arg must be 'RBFManchesterKernel' or 'MaternManchesterKernel'.")

        nu = torch.tensor(nu, dtype=torch.float64)
        self.nu = torch.nn.Parameter(nu)
        self.data_num = torch.tensor(self.train_x.shape[0], dtype=torch.float64)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x_train_data = self.covar_module(self.train_inputs[0])
        inv_quad, _ = covar_x_train_data.inv_quad_logdet(
            inv_quad_rhs=self.train_targets - self.train_targets.mean(),
            logdet=False
        )

        tp_var_scale = (self.nu + inv_quad - 2) / (self.nu + self.data_num - 2)

        covar_x = tp_var_scale.float() * covar_x

        return tptorch.distributions.MultivariateStudentT(mean_x, covar_x, self.nu, self.data_num)

