"""Data helpers: polynomial features, scaling, CV folds and metrics."""
from collections import Counter
from itertools import combinations_with_replacement

import torch


def make_polynomial_features(X, degree):
    """All monomials of total degree 1..degree.

    No constant column: the intercept is handled by centering y instead,
    so it never gets penalised.
    """
    n_features = X.shape[1]
    columns = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            columns.append(X[:, list(combo)].prod(dim=1))
    return torch.stack(columns, dim=1)


def term_names(n_features, degree):
    """Readable names in the same column order as make_polynomial_features, e.g. 'x1^2·x3'."""
    names = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            names.append("·".join(f"x{i + 1}" + (f"^{c}" if c > 1 else "")
                                  for i, c in Counter(combo).items()))
    return names


def standardize(X_fit, X_apply):
    """Scale both matrices with the mean/std of X_fit only (no leakage)."""
    mean = X_fit.mean(dim=0)
    std = X_fit.std(dim=0)
    std[std == 0] = 1.0
    return (X_fit - mean) / std, (X_apply - mean) / std


def make_folds(n, k=5, seed=42):
    """Same split as the notebook: shuffle once with a fixed seed, cut into k chunks."""
    torch.manual_seed(seed)
    return torch.chunk(torch.randperm(n), k)


def train_val_split(folds, i):
    val_idx = folds[i]
    train_idx = torch.cat([folds[j] for j in range(len(folds)) if j != i])
    return train_idx, val_idx


def mse_r2(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    r2 = 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - y_true.mean()) ** 2)
    return mse.item(), r2.item()
