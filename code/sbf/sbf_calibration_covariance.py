"""Ковариации калибровки SBF / Explicit covariance propagation for SBF fits.

Only documented shared terms are correlated. Population/image correlations
between different filters are not inferred from the small overlap sample.
"""

import numpy as np


def shared_anchor_covariance(labels, sigma):
    """Один общий сдвиг на группу / One shared anchor error per named group."""
    labels = np.asarray(labels, dtype=str)
    sigma = np.asarray(sigma, dtype=float)
    if labels.shape != sigma.shape or sigma.ndim != 1:
        raise ValueError("Anchor labels and errors must be equally sized vectors")
    if not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("Shared anchor errors must be finite and non-negative")
    if np.any((labels == "") & (sigma != 0)):
        raise ValueError("A nonzero shared anchor error needs a group label")
    same_group = (labels[:, None] == labels[None, :]) & (labels[:, None] != "")
    return np.outer(sigma, sigma) * same_group


def validate_color_covariance(sigma_y, sigma_x, covariance_xy, shared_y=None):
    """Проверка PSD до фита / Reject impossible local error covariance."""
    sy, sx, cxy = np.broadcast_arrays(sigma_y, sigma_x, covariance_xy)
    if not np.isfinite([sy, sx, cxy]).all() or (sy <= 0).any() or (sx < 0).any():
        raise ValueError("Measurement errors and covariance must be finite and valid")
    local_y_variance = sy**2
    if shared_y is not None:
        local_y_variance = local_y_variance - np.diag(shared_y)
    # Tolerance protects round-off only, not inconsistent physical budgets.
    tolerance = 1e-12 * np.maximum(sy**2, 1e-12)
    if np.any(local_y_variance < -tolerance):
        raise ValueError("Shared anchor variance exceeds the marginal variance")
    bound = sx * np.sqrt(np.maximum(local_y_variance, 0.0))
    if np.any(np.abs(cxy) > bound + 1e-12):
        raise ValueError("Non-PSD color/SBF errors: abs(covariance) > sigma_color * sigma_SBF")


def constant_loo_distance_covariance(
    sigma_measurement, sigma_trgb_no_reddening, sigma_ebv,
    sigma_calibration_point, loo_intrinsic_scatter, population_scatter,
    r_sbf, r_trgb, common_trgb_scale,
):
    """Условная ковариация LOO / Fixed-weight covariance for a constant law.

    A[i, j] is calibrator j's weight in target i's leave-one-out zero point.
    H=I-A includes target--training correlations, not merely covariance of
    the fitted zero points. One global population variance is used so that
    the same galaxy has one random population offset in every LOO fit.
    Weights and the fitted scatter are held fixed: this is a first-order
    conditional calculation, distinct from the primary bootstrap budget.
    """
    sm, st, se, sy, si = [np.asarray(value, dtype=float) for value in (
        sigma_measurement, sigma_trgb_no_reddening, sigma_ebv,
        sigma_calibration_point, loo_intrinsic_scatter,
    )]
    if len(sm) < 3 or any(value.shape != sm.shape for value in (st, se, sy, si)):
        raise ValueError("LOO covariance needs matching vectors for at least three galaxies")
    if not np.isfinite([sm, st, se, sy, si]).all() or np.any(np.array([sm, st, se, sy, si]) < 0):
        raise ValueError("LOO uncertainty inputs must be finite and non-negative")
    if (sy <= 0).any() or population_scatter < 0 or common_trgb_scale < 0:
        raise ValueError("Calibration errors must be positive and scale errors non-negative")
    weights = 1 / (sy[None, :]**2 + si[:, None]**2)
    np.fill_diagonal(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    response = np.eye(len(sm)) - weights
    reddening_response = -r_sbf * np.eye(len(sm)) - (r_trgb - r_sbf) * weights
    covariance = (
        (response * (sm**2 + population_scatter**2)) @ response.T
        + (weights * st**2) @ weights.T
        + (reddening_response * se**2) @ reddening_response.T
        + common_trgb_scale**2 * np.ones((len(sm), len(sm)))
    )
    return covariance, weights
