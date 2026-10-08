"""Train the final models chosen in model.ipynb and write test predictions.

Usage:
    python train.py          # both problems
    python train.py var1     # one problem

Outputs:
    predictions/IMT2024008_pred_varX.csv   test predictions (submission format)
    plots/varX_pred_vs_actual.png          out-of-fold CV predictions vs true y
    plots/varX_coefficient_sizes.png       sorted |coefficient| sizes (shows Lasso's zeros)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import plots
from models import describe, fit_model
from utils import make_folds, make_polynomial_features, mse_r2, standardize, term_names, train_val_split

torch.set_default_dtype(torch.float64)

ROLL = "IMT2024008"
DATA_DIR = Path("dataset")
PRED_DIR = Path("predictions")
PLOT_DIR = Path("plots")

# Chosen in model.ipynb: lowest 5-fold CV MSE after the coarse -> fine sweeps
CONFIG = {
    # Sparse true polynomial: Lasso drops useless terms and clearly beats Ridge (~0.347 vs ~0.467)
    "var1": {"model": "lasso", "degree": 5, "alpha": 0.00975},
    # Ridge / Lasso / Elastic Net all tie (~0.24), so take the simplest: Ridge
    "var2": {"model": "ridge", "degree": 10, "lam": 0.000975},
}


def load(problem):
    train = pd.read_csv(DATA_DIR / f"{ROLL}_train_{problem}.csv")
    test = pd.read_csv(DATA_DIR / f"{ROLL}_test_{problem}.csv")
    features = [c for c in train.columns if c != "y"]
    X = torch.tensor(train[features].values)
    y = torch.tensor(train["y"].values)
    X_test = torch.tensor(test[features].values)
    return X, y, X_test


def fit_and_predict(cfg, X_train, y_train, X_eval):
    """Scale with training stats, center y, fit, predict. Used by both CV and the final fit."""
    Z_train, Z_eval = standardize(X_train, X_eval)
    y_mean = y_train.mean()
    beta = fit_model(cfg, Z_train, y_train - y_mean)
    return Z_eval @ beta + y_mean, beta


def cross_validate(cfg, X_poly, y, k=5):
    """5-fold CV of ONE config: a sanity check that the refactor matches the notebook.

    Returns per-fold (mse, r2) and the out-of-fold prediction for every training row
    (each row is predicted by the model that did NOT train on it).
    """
    folds = make_folds(len(y), k)
    scores = []
    oof_pred = torch.zeros_like(y)
    for i in range(k):
        train_idx, val_idx = train_val_split(folds, i)
        y_pred, _ = fit_and_predict(cfg, X_poly[train_idx], y[train_idx], X_poly[val_idx])
        oof_pred[val_idx] = y_pred
        scores.append(mse_r2(y[val_idx], y_pred))
    return np.array(scores), oof_pred


def run(problem):
    cfg = CONFIG[problem]
    name = describe(cfg)
    print(f"\n=== {problem}: {name} ===")

    X, y, X_test = load(problem)
    X_poly = make_polynomial_features(X, cfg["degree"])
    X_test_poly = make_polynomial_features(X_test, cfg["degree"])
    print(f"{len(y)} training rows, {X.shape[1]} inputs -> {X_poly.shape[1]} polynomial terms")

    # 1. Sanity check with 5-fold CV
    scores, oof_pred = cross_validate(cfg, X_poly, y)
    mse, r2 = scores.mean(axis=0)
    print(f"5-fold CV:  MSE = {mse:.4f} (fold std {scores[:, 0].std(ddof=1):.4f}), R² = {r2:.4f}")
    print(f"Baseline (always predict the mean): MSE = {y.var().item():.4f}")

    # 2. Final fit on all training rows
    y_pred, beta = fit_and_predict(cfg, X_poly, y, X_test_poly)

    names = term_names(X.shape[1], cfg["degree"])
    top = torch.argsort(beta.abs(), descending=True)[:5]
    print(f"Non-zero coefficients: {int((beta != 0).sum())} of {len(beta)}")
    print("Largest terms:", ", ".join(f"{names[i]} ({beta[i]:+.2f})" for i in top))

    # 3. Plots
    y_np, oof_np = y.numpy(), oof_pred.numpy()
    plots.pred_vs_actual(y_np, oof_np, f"{problem}: {name}\nCV MSE = {mse:.3f}, R² = {r2:.3f}",
                         PLOT_DIR / f"{problem}_pred_vs_actual.png")
    plots.coefficient_sizes(beta.numpy(), f"{problem}: coefficient sizes",
                            PLOT_DIR / f"{problem}_coefficient_sizes.png")

    # 4. Test predictions
    pred_path = PRED_DIR / f"{ROLL}_pred_{problem}.csv"
    pd.DataFrame({"y": y_pred.numpy()}).to_csv(pred_path, index=False)
    print(f"Test predictions in [{y_pred.min():.2f}, {y_pred.max():.2f}] "
          f"(train y in [{y.min():.2f}, {y.max():.2f}])")
    print(f"Saved {pred_path} + plots in {PLOT_DIR}/")


if __name__ == "__main__":
    PRED_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    for problem in sys.argv[1:] or CONFIG:
        run(problem)