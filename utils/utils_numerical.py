# Inspired by https://github.com/probabll/mixed-rv-vae/blob/fe809beb42f3c4d0d388ccd534cdba800d4d0a72/torch_log_ndtr.py

import numpy as np
import torch

_LOGNDTR_FLOAT32_LOWER = torch.tensor(-10, dtype=torch.float32)
_LOGNDTR_FLOAT32_UPPER = torch.tensor(5, dtype=torch.float32)
DBL_FAC = [1, 1, 2, 3, 8, 15, 48, 105, 384, 945]
HALF_SQRT2 = np.sqrt(2) / 2
LOG2PI = np.log(2 * np.pi)

def _nrm_logpdf(x):
    """
    Log probability density function of a standard normal distribution.
    """
    return -(LOG2PI + (x**2)) / 2


def _ndtr(x):
    """
    Compute the normal cumulative distribution function (CDF).

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: CDF of the input tensor.
    """
    w = x * HALF_SQRT2
    z = torch.abs(w)
    y = torch.where(
        torch.lt(z, HALF_SQRT2),
        torch.erf(w) + 1,
        torch.where(torch.gt(w, 0), -torch.erfc(z) + 2, torch.erfc(z)),
    )
    ndtr = y / 2
    return ndtr


def _log_ndtr_lower(x, series_order):
    """
    Compute the log of the normal CDF for very small values of x using an asymptotic expansion.

    Args:
        x (torch.Tensor): Input tensor.
        series_order (int): Order of the asymptotic series expansion.

    Returns:
        torch.Tensor: Log of the normal CDF for small values of x.
    """
    x_2 = x.square()
    log_scale = -(x_2 / 2) - torch.log(-x) - 0.5 * np.log(2.0 * np.pi)
    return log_scale + torch.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
    """
    Calculate the asymptotic series used in the log of the normal CDF.

    Args:
        x (torch.Tensor): Input tensor.
        series_order (int): Order of the asymptotic series expansion.

    Returns:
        torch.Tensor: Asymptotic series expansion for the log of the normal CDF.
    """
    dtype = x.dtype
    if series_order <= 0:
        return torch.tensor(1, dtype)
    x_2 = x.square()
    even_sum = torch.zeros_like(x)
    odd_sum = torch.zeros_like(x)
    x_2n = x_2
    for n in range(1, series_order + 1):
        y = DBL_FAC[2 * n - 1] / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n *= x_2
    return 1 + even_sum - odd_sum


def _log_ndtr(x, series_order=3):
    """
    Compute the log of the normal CDF over different ranges of x for numerical stability.

    Args:
        x (torch.Tensor): Input tensor.
        series_order (int, optional): Order of the asymptotic series expansion. Defaults to 3.

    Returns:
        torch.Tensor: Log of the normal CDF of the input tensor.
    """
    lower_segment = _LOGNDTR_FLOAT32_LOWER
    upper_segment = _LOGNDTR_FLOAT32_UPPER

    return torch.where(
        torch.gt(x, upper_segment),
        -_ndtr(-x),
        torch.where(
            torch.gt(x, lower_segment),
            torch.log(_ndtr(torch.maximum(x, lower_segment))),
            _log_ndtr_lower(torch.minimum(x, lower_segment), series_order),
        ),
    )


def log_survival_function(x, loc, scale):
    """
    Compute the log survival function (log of the complementary CDF) for a normal distribution.

    Args:
        x (torch.Tensor): Input tensor.
        loc (torch.Tensor): Mean of the distribution.
        scale (torch.Tensor): Standard deviation of the distribution.

    Returns:
        torch.Tensor: Log survival function of the input tensor.
    """
    normalized_x = (x - loc) / scale
    normalized_x = normalized_x
    return _log_ndtr(-normalized_x)
