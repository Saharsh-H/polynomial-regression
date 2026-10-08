"""Plots for the final model, saved as PNGs (e.g. plots/var1_pred_vs_actual.png)."""
import matplotlib

matplotlib.use("Agg")            # save to files, never open a window
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pred_vs_actual(y_true, y_pred, title, path):
    """Out-of-fold CV predictions vs true y. Points on the diagonal = perfect predictions."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.5)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, label="perfect prediction")
    ax.set(xlabel="true y", ylabel="predicted y (out-of-fold)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, path)


def coefficient_sizes(beta, title, path):
    """Sorted |coefficient| on a log scale. With Lasso the line stops where coefficients hit exactly 0."""
    mags = np.sort(np.abs(beta))[::-1]
    mags = mags[mags > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(mags) + 1), mags, marker=".")
    ax.set(xlabel=f"coefficient rank ({len(mags)} non-zero of {len(beta)})",
           ylabel="|β| (standardized scale, log)", title=title)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, path)