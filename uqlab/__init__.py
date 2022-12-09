"""uqlab: a python package for to perform post-hoc calibration and measure accuracy and uncertainty quantification."""

__version__ = '0.1.1'

from .preprocessing import (
    DataLoader,
    WaterDimerLoader,
    GlycineLoader
)

from .kernels import (
    RBFManchesterKernel,
    MaternManchesterKernel
)

from .models import (
    GaussianProcess,
    StudentTProcess
)

from .calibration import (
    Crude
)

from .metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_sq_error,
    mean_abs_rel_percent_diff,
    r_squared,
    accuracy_metrics,
    proportion_in_interval,
    get_proportion_lists,
    root_mean_sq_calibration_error,
    miscalibration_area,
    sharpness,
    uq_metrics,
    all_metrics
)

from .plots import (
    parity_plot,
    calibration_plot,
    sharpness_plot,
    s_curve,
    all_plots
)
