import math
import numpy as np
import torch
import gpytorch


def manchester_kernel(X_dim, base_kernel='RBF'):
    """
    Custom kernel for GPyTorch models following manchester convention
    non-cyclic features have an RBF kernel, cyclic features have periodic kernel with period set to 2pi
    :param base_kernel:
    :param X_dim:
    :return:
    """
    # determine number of cyclic and non-cyclic dimensions
    noncyclic_dim_idx = [dim - 1 for dim in range(1, X_dim + 1) if not (dim > 3 and dim % 3 == 0)]
    cyclic_dim_idx = [dim - 1 for dim in range(1, X_dim + 1) if (dim > 3 and dim % 3 == 0)]
    n_noncyclic = len(noncyclic_dim_idx)
    n_cyclic = len(cyclic_dim_idx)

    # define kernel for non-cyclic features
    if base_kernel == 'RBF' or base_kernel == 'rbf':
        noncyclic_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=n_noncyclic,
            active_dims=np.arange(X_dim)[:n_noncyclic]
        )

    elif base_kernel == 'matern52' or base_kernel == 'Matern52':
        noncyclic_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=n_noncyclic,
            active_dims=np.arange(X_dim)[:n_noncyclic]
        )

    # define kernel for cyclic features
    cyclic_kernel = gpytorch.kernels.PeriodicKernel(
        ard_num_dims=n_cyclic,
        active_dims=np.arange(X_dim)[n_noncyclic:],
        period_length_constraint=gpytorch.constraints.Interval(
            lower_bound=(2.0 * math.pi - 1e-3),
            upper_bound=(2.0 * math.pi + 1e-3)
        )
    )

    # combine non-cyclic and cyclic kernels
    kernel = noncyclic_kernel * cyclic_kernel

    return kernel


class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = manchester_kernel(train_x.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class StudentTProcess(ExactTP):
#     def __init__(self, train_x, train_y, nu=5):
#         likelihood = StudentTLikelihood()
#         super(TPModel, self).__init__(train_x, train_y, likelihood)
#
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = manchester_kernel(train_x.shape[1])
#
#         nu = torch.tensor(nu, dtype=torch.float64)
#         self.nu = torch.nn.Parameter(nu)
#         self.data_num = torch.tensor(self.train_targets.shape[0], dtype=torch.float64)
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#
#         covar_x_train_data = self.covar_module(self.train_inputs[0])
#         inv_quad, _ = covar_x_train_data.inv_quad_logdet(
#             inv_quad_rhs=self.train_targets - self.train_targets.mean(), logdet=False
#         )
#
#         tp_var_scale = (self.nu + inv_quad - 2) / (self.nu + self.data_num - 2)
#
#         covar_x = tp_var_scale.float() * covar_x
#
#         return MultivariateStudentT(mean_x, covar_x, self.nu, self.data_num)


def gp_train_and_predict(X_train_torch, y_train_torch, X_test_torch):
    # instantiate model
    model = GaussianProcess(X_train_torch, y_train_torch)

    # use gpu if available - may need to extend to multiple
    if torch.cuda.is_available():
        model = model.cuda()

    # put model in train mode
    model.train()

    # define the appropriate marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # define optimisation method and learning rate adjustments
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.9)

    # training loop
    training_iter = 2000

    with gpytorch.settings.max_cholesky_size(0):  # Ensure we don't try to use Cholesky
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(X_train_torch)
            loss = -mll(output, y_train_torch)
            loss = loss.mean()
            loss.backward()

            if (i == 0) or ((i + 1) % 10 == 0):
                print(f'Iteration {i + 1} - loss = {loss:.2f} - noise = {model.likelihood.noise.item():e}')

            optimizer.step()
            scheduler.step(loss)

    # Get into evaluation (predictive posterior) mode
    model.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(0), gpytorch.settings.fast_pred_var():

        f_pred = model(X_test_torch)
        y_pred = model.likelihood(f_pred)

        # get mean and standard deviation predictions
        y_pred_mean = y_pred.mean
        y_pred_var = y_pred.variance
        y_pred_std = y_pred_var.sqrt()

    # reshape tensors to np arrays and take off GPU
    y_pred_mean_GP = y_pred_mean.cpu().detach().numpy()
    y_pred_std_GP = y_pred_std.cpu().detach().numpy()

    return y_pred_mean_GP, y_pred_std_GP


def train_model(model, mll, epochs, X_train_torch, y_train_torch, verbose=False):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.9)

    with gpytorch.settings.max_cholesky_size(0):
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(X_train_torch)
            loss = -mll(output, y_train_torch)
            loss = loss.mean()
            loss.backward()

            if verbose:
                if (i == 0) or ((i + 1) % 10 == 0):
                    print(f'Iteration {i + 1} - loss = {loss:.2f}')

            optimizer.step()
            scheduler.step(loss)

    return model
