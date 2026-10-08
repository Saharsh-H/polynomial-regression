# ML Assignment 1: Polynomial Regression

Predicting a continuous target `y` from a few input features using **polynomial regression only**, for two problems:

| Problem | Inputs | Max degree allowed |
|---|---|---|
| **var1**: steam turbine power score | 6 (`x1`–`x6`) | 10 |
| **var2**: thermal anomaly score | 3 (`x1`–`x3`) | 20 |

Each problem has 1000 training rows and 1000 test rows (test labels are hidden).

## Final models

| Problem | Model | Degree | Penalty | 5-fold CV MSE | CV R² |
|---|---|---|---|---|---|
| var1 | Lasso (L1) | 5 | α = 0.00975 | 0.347 | 0.965 |
| var2 | Ridge (L2) | 10 | λ = 0.000975 | 0.237 | 0.995 |

For comparison, always predicting the mean of `y` gives an MSE of 9.95 (var1) and 46.5 (var2).

## Approach

**Features.** Every monomial of the inputs up to degree *d* (for example `x1`, `x1·x5`, `x2³·x3`). The intercept is handled by centering `y`, so it is never penalised.

**Evaluation.** 5-fold cross-validation. Inside each fold the features are standardized using the training part only, so nothing leaks from the validation part.

**Models, in order:**
1. **OLS** for every degree up to the limit, to see which degrees make sense.
2. **Ridge (L2)**: shrinks all coefficients. Tuned with a coarse sweep of λ, then a finer one around the best value.
3. **Lasso (L1)**: can set coefficients to exactly zero, which drops terms. Tuned the same coarse → fine way.
4. **Elastic Net**: a mix of L1 and L2. Only a small sweep, narrowed down using the Ridge and Lasso results.

All four use the same loss, scaled by `1/n` so the penalty strength doesn't depend on the number of rows:

```
(1/2n)·||y − Zβ||²  +  α·( ρ·||β||₁ + (1−ρ)/2·||β||₂² )      ρ = 0 → Ridge,  ρ = 1 → Lasso
```

Ridge has a closed-form solution. Lasso and Elastic Net are solved with proximal gradient descent (FISTA).

**Degree cutoff.** Once a degree has more terms than training rows, OLS can fit the training data perfectly in infinitely many ways and its validation error explodes. For var1 this happens from degree 6 (924 terms vs 800 rows), for var2 from degree 15 (816 vs 800). OLS is still run up to the full limit to show this. The regularized models are searched below the cutoff.

## Why these models

- **var1 → Lasso.** Lasso keeps only 118 of 461 terms and beats Ridge clearly (0.347 vs 0.467). This suggests the true polynomial uses only some of the terms. Ridge has to keep a small weight on every term, which partly fits noise. Lasso drops the useless ones. Elastic Net got better the closer it was to pure Lasso.
- **var2 → Ridge.** Ridge, Lasso and Elastic Net all score about 0.24, so the differences are just noise. Ridge is the simplest of the three. Here regularization mainly keeps the high powers (which look almost identical on [-1, 1]) from making the fit unstable.

## Files

```
├── dataset/              training and test CSVs
├── exploratory_model.ipynb     exploration: all sweeps, plots and reasoning
├── train.py              trains the final models and writes predictions + plots
├── models.py             OLS, Ridge, Lasso / Elastic Net
├── utils.py              polynomial features, scaling, CV folds, metrics
├── plots.py              plots for the final models
└── requirements.txt
```

The notebook is where the models were **chosen**. `train.py` only **trains** the chosen ones; their settings are in the `CONFIG` dictionary at the top of the file.

## How to run

```bash
pip install -r requirements.txt

python3 train.py          # train both final models
python3 train.py var1     # or just one problem
```

To redo the exploration, open `exploratory_model.ipynb` and run all cells.

Run both from the project folder so the scripts can find `dataset/`.

## Outputs

| What | Where |
|---|---|
| Test predictions | `predictions/IMT2024008_pred_var1.csv`, `predictions/IMT2024008_pred_var2.csv` |
| Final model plots (from `train.py`) | `plots/var1_*.png`, `plots/var2_*.png` |
| Exploration plots (from the notebook) | `plots/explore_var1_*.png`, `plots/explore_var2_*.png` |