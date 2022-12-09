"""
Define Gaussian Process kernels used by Manchester research group
"""
import math
import numpy as np
import torch
import gpytorch


def RBFManchesterKernel(X_dim, batch_shape=None):
    """
    Custom kernel for GPyTorch models following manchester convention,
    using RBF kernel for non-cyclic features

    :param X_dim:        Dimension of feature array
    :param batch_shape:  batch shape for use when minibatching
    :return:
    """
    # determine cyclic and non-cyclic dimensions
    noncyclic_dim_idx = [dim - 1 for dim in range(1, X_dim + 1) if not (dim > 3 and dim % 3 == 0)]
    cyclic_dim_idx = [dim - 1 for dim in range(1, X_dim + 1) if (dim > 3 and dim % 3 == 0)]

    # for minibatch functionality
    if batch_shape is not None:
        # define matern kernel for non-cyclic features
        noncyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=len(noncyclic_dim_idx),
                active_dims=tuple(noncyclic_dim_idx),
                batch_shape=batch_shape
            )
        )
        # define periodic kernel for cyclic features
        cyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                ard_num_dims=len(cyclic_dim_idx),
                active_dims=tuple(cyclic_dim_idx),
                batch_shape=batch_shape,
                period_length_constraint=gpytorch.constraints.Interval(
                    lower_bound=(2.0 * math.pi - 1e-5),
                    upper_bound=(2.0 * math.pi + 1e-5)
                )
            )
        )

    else:
        # define matern kernel for non-cyclic features
        noncyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=len(noncyclic_dim_idx),
                active_dims=tuple(noncyclic_dim_idx)
            )
        )
        # define periodic kernel for cyclic features
        cyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                ard_num_dims=len(cyclic_dim_idx),
                active_dims=tuple(cyclic_dim_idx),
                period_length_constraint=gpytorch.constraints.Interval(
                    lower_bound=(2.0 * math.pi - 1e-5),
                    upper_bound=(2.0 * math.pi + 1e-5)
                )
            )
        )

    # combine non-cyclic and cyclic kernels
    kernel = gpytorch.kernels.ProductKernel(noncyclic_kernel, cyclic_kernel)

    return kernel


def MaternManchesterKernel(X_dim, batch_shape=None):
    """
    Custom kernel for GPyTorch models following manchester convention,
    using Matern52 kernel for non-cyclic features

    :param X_dim:        Dimension of feature array
    :param batch_shape:  batch shape for use when minibatching
    :return:
    """
    # determine cyclic and non-cyclic dimensions
    noncyclic_dim_idx = [dim - 1 for dim in range(1, X_dim + 1) if not (dim > 3 and dim % 3 == 0)]
    cyclic_dim_idx = [dim - 1 for dim in range(1, X_dim + 1) if (dim > 3 and dim % 3 == 0)]

    # for minibatch functionality
    if batch_shape is not None:
        # define matern kernel for non-cyclic features
        noncyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=len(noncyclic_dim_idx),
                active_dims=tuple(noncyclic_dim_idx),
                batch_shape=batch_shape
            )
        )
        # define periodic kernel for cyclic features
        cyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                ard_num_dims=len(cyclic_dim_idx),
                active_dims=tuple(cyclic_dim_idx),
                batch_shape=batch_shape,
                period_length_constraint=gpytorch.constraints.Interval(
                    lower_bound=(2.0 * math.pi - 1e-5),
                    upper_bound=(2.0 * math.pi + 1e-5)
                )
            )
        )

    else:
        # define matern kernel for non-cyclic features
        noncyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=len(noncyclic_dim_idx),
                active_dims=tuple(noncyclic_dim_idx)
            )
        )
        # define periodic kernel for cyclic features
        cyclic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                ard_num_dims=len(cyclic_dim_idx),
                active_dims=tuple(cyclic_dim_idx),
                period_length_constraint=gpytorch.constraints.Interval(
                    lower_bound=(2.0 * math.pi - 1e-5),
                    upper_bound=(2.0 * math.pi + 1e-5)
                )
            )
        )

    # combine non-cyclic and cyclic kernels
    kernel = gpytorch.kernels.ProductKernel(noncyclic_kernel, cyclic_kernel)

    return kernel
