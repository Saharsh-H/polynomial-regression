"""Linear models. All take standardized features Z and centered targets y.

Objective (n = number of rows):
    (1/2n)||y - Zb||^2 + alpha * ( rho*||b||_1 + (1-rho)/2*||b||_2^2 )

    rho = 0 -> Ridge (alpha is called lam), rho = 1 -> Lasso, in between -> Elastic Net.
"""
import numpy as np
import torch


def ols_fit(Z, y):
    return torch.linalg.lstsq(Z, y).solution


def ridge_fit(Z, y, lam):
    # Closed form: (Z^T Z + n*lam*I) b = Z^T y
    n, p = Z.shape
    return torch.linalg.solve(Z.T @ Z + n * lam * torch.eye(p, dtype=Z.dtype), Z.T @ y)


def soft_threshold(x, t):
    # Move every value t closer to 0; anything smaller than t becomes exactly 0
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)


def enet_fit(Z, y, alpha, rho, beta=None, L=None, n_iter=2000, tol=1e-5):
    """Elastic net / Lasso with proximal gradient descent (FISTA).

    Each step: gradient step on the smooth part (squared loss + L2),
               then soft-threshold for the L1 part.
    beta: starting point (warm start), L: largest eigenvalue of Z^T Z / n (step size).
    """
    n, p = Z.shape
    if beta is None:
        beta = torch.zeros(p, dtype=Z.dtype)
    if L is None:
        L = torch.linalg.matrix_norm(Z, 2).item() ** 2 / n

    step = 1.0 / (L + alpha * (1 - rho))
    z, t = beta.clone(), 1.0

    for _ in range(n_iter):
        grad = Z.T @ (Z @ z - y) / n + alpha * (1 - rho) * z
        beta_new = soft_threshold(z - step * grad, step * alpha * rho)

        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2               # momentum
        z = beta_new + ((t - 1) / t_new) * (beta_new - beta)

        converged = (beta_new - beta).abs().max() < tol
        beta, t = beta_new, t_new
        if converged:
            break

    return beta


def enet_path_fit(Z, y, alpha, rho, n_steps=15):
    """Walk alpha down from 1 to the target, warm-starting each fit from the previous one."""
    L = torch.linalg.matrix_norm(Z, 2).item() ** 2 / len(y)
    beta = None
    for a in np.geomspace(max(1.0, alpha), alpha, n_steps):
        beta = enet_fit(Z, y, a, rho, beta, L)
    return beta


def fit_model(cfg, Z, y):
    """Fit the model described by a config dict, e.g. {"model": "lasso", "alpha": 0.01}."""
    model = cfg["model"]
    if model == "ols":
        return ols_fit(Z, y)
    if model == "ridge":
        return ridge_fit(Z, y, cfg["lam"])
    if model == "lasso":
        return enet_path_fit(Z, y, cfg["alpha"], rho=1.0)
    if model == "elasticnet":
        return enet_path_fit(Z, y, cfg["alpha"], cfg["rho"])
    raise ValueError(f"Unknown model: {model}")


def describe(cfg):
    """Human-readable one-liner for printing."""
    model, d = cfg["model"], cfg["degree"]
    if model == "ols":
        return f"OLS (no penalty), degree {d}"
    if model == "ridge":
        return f"Ridge (L2 penalty), degree {d}, lambda = {cfg['lam']:g}"
    if model == "lasso":
        return f"Lasso (L1 penalty), degree {d}, alpha = {cfg['alpha']:g}"
    return f"Elastic Net (L1 + L2), degree {d}, alpha = {cfg['alpha']:g}, rho = {cfg['rho']:g}"
