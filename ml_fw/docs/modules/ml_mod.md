# ML Model — Training and Hyperparameter Tuning

A lightweight wrapper around scikit-learn for model training and hyperparameter optimization. This module simplifies the common workflow of fitting an estimator, optionally performing grid search, and handling multi-scorer model selection.

---

## Features

- Unified training interface (`train`)
  - Direct estimator fitting
  - Optional hyperparameter grid search via scikit-learn's `GridSearchCV`
  - Fractional grid-search subsampling for faster tuning on large datasets
  - Automatic random-state management
  - Multi-scorer parameter selection via min-max normalized RMS combination
  
- Raw grid-search wrapper (`tune`)
  - Thin wrapper around `GridSearchCV`
  - Forwards custom keyword arguments to `GridSearchCV`

---

## Importing

```python
from ml_fw import ml_mod
```

Or import individual functions:

```python
from ml_fw.ml_mod import train, tune
```

---

## Main API

### `train`

```python
train(
    f_dat,
    y_dat,
    estimator,
    grid_params=None,
    grid_kwargs=None,
    grid_ratio=0.3,
    random_state=17
)
```

Fit a scikit-learn estimator, with optional hyperparameter tuning via grid search.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `f_dat` | pd.DataFrame | Feature matrix for training. Shape: (n_samples, n_features). |
| `y_dat` | pd.DataFrame \| pd.Series | Target values for training. Shape: (n_samples,) or (n_samples, n_outputs). |
| `estimator` | estimator object | A scikit-learn estimator instance (e.g., `RandomForestRegressor()`, `LogisticRegression()`). Must implement the scikit-learn estimator interface with `fit()` and (optionally) `score()` methods. |
| `grid_params` | dict \| list[dict], optional | Dictionary of parameter names → lists of parameter values to search over, or a list of such dictionaries (see scikit-learn `GridSearchCV`). If None, the estimator is fit without grid search. Default is None. |
| `grid_kwargs` | dict, optional | Additional keyword arguments to pass to `GridSearchCV` (e.g., `cv=5`, `n_jobs=-1`, `scoring='accuracy'`). Default is None. |
| `grid_ratio` | float \| int, optional | Fraction or absolute number of training samples to use for grid search. If 0 < `grid_ratio` < 1, samples are drawn as a fraction. If >= 1, treated as absolute sample count. If None or >= 1.0, uses full dataset. Useful for large datasets where grid search would be slow. Default is 0.3. |
| `random_state` | int, optional | Random seed. Applied to the estimator if it has a `random_state` parameter and one wasn't already set. Also used for train/test splitting in grid-search subsampling. Default is 17. |

#### Returns

| Type | Description |
|---|---|
| estimator object | Fitted estimator (scikit-learn model instance). If grid search was performed, this is the fitted model with best parameters; otherwise, it is the original estimator fitted on full data. |

#### Behavior

- **No grid search** (if `grid_params=None`): Fits the estimator directly on `(f_dat, y_dat)`.
- **Grid search with subsampling** (if `grid_params` is set and 0 < `grid_ratio` < 1):
  - Splits data into train (size `grid_ratio * n_samples`) and holdout set via `train_test_split`.
  - Runs `GridSearchCV` on the train split only (faster, useful for large datasets).
  - Extracts best parameters or best estimator from the grid-search result.
  - Fits the final estimator on the **full** dataset (train + holdout) using the best parameters.
- **Grid search with full data** (if `grid_params` is set and `grid_ratio >= 1.0` or `grid_ratio=None`):
  - Runs `GridSearchCV` on all data.
  - Extracts and returns the fitted best estimator directly.
- **Multi-scorer handling**: If `GridSearchCV` result does not expose `best_estimator_` (occurs when multiple scorers are passed), the function:
  - Retrieves `mean_train_{scorer}` and `mean_test_{scorer}` scores for each scorer.
  - Min-max normalizes all scores.
  - Computes RMS across normalized scores for each parameter set.
  - Selects the parameter set with the highest RMS (best balanced performance).
  - Refits the estimator with these parameters on the full data.
- **Random state management**: If the estimator has a `random_state` parameter and it is not set or is None, it is automatically set to the provided `random_state` value (or the estimator's existing value is preserved).

#### Example: Simple Training

```python
from ml_fw import ml_mod
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Prepare data
X_train, y_train = ...  # your training data

# Train a model
model = ml_mod.train(
    f_dat=X_train,
    y_dat=y_train,
    estimator=RandomForestRegressor(random_state=42),
    grid_params=None  # No grid search
)

# Use the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

#### Example: Grid Search with Subsampling

```python
from ml_fw import ml_mod
from sklearn.ensemble import GradientBoostingRegressor

# Large dataset — grid search may be slow
X_large, y_large = ...  # 100,000+ samples

# Use 30% of data for grid search (speed up), then refit on full data
model = ml_mod.train(
    f_dat=X_large,
    y_dat=y_large,
    estimator=GradientBoostingRegressor(),
    grid_params={
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7]
    },
    grid_kwargs={'cv': 5, 'n_jobs': -1},
    grid_ratio=0.3  # Use 30% of data for grid search
)
```

#### Example: Multi-Scorer Selection

```python
from ml_fw import ml_mod
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Grid search with multiple scoring metrics
model = ml_mod.train(
    f_dat=X_train,
    y_dat=y_train,
    estimator=SVC(),
    grid_params={
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    grid_kwargs={
        'cv': 3,
        'scoring': {'accuracy': 'accuracy', 'f1': 'f1_macro'}
    }
)
# Best parameters selected via RMS of min-max normalized scores
```

---

### `tune`

```python
tune(
    estimator,
    grid_param,
    f_dat,
    y_dat,
    **kwargs
)
```

Perform raw grid search using scikit-learn's `GridSearchCV`. This is a thin wrapper for advanced users who need direct access to the full `GridSearchCV` interface.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `estimator` | estimator object | Scikit-learn estimator instance. |
| `grid_param` | dict \| list[dict] | Parameter grid(s) for `GridSearchCV`. |
| `f_dat` | pd.DataFrame | Feature matrix. |
| `y_dat` | pd.DataFrame \| pd.Series | Target values (flattened if needed). |
| `**kwargs` | dict | Additional keyword arguments passed directly to `GridSearchCV` (e.g., `cv`, `scoring`, `n_jobs`, `verbose`). |

#### Returns

| Type | Description |
|---|---|
| GridSearchCV object | Fitted `GridSearchCV` result. Access `cv_results_`, `best_params_`, `best_estimator_` (if single scorer) as needed. |

#### Example

```python
from ml_fw import ml_mod
from sklearn.tree import DecisionTreeClassifier

# Direct grid search access
grid_result = ml_mod.tune(
    estimator=DecisionTreeClassifier(),
    grid_param={'max_depth': [3, 5, 7], 'min_samples_split': [2, 5]},
    f_dat=X_train,
    y_dat=y_train,
    cv=5,
    n_jobs=-1,
    verbose=1
)

print(f"Best parameters: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")
best_model = grid_result.best_estimator_
```

---

## Use Cases

### Fast Prototyping with Hyperparameter Optimization

Use `train()` with `grid_params` to quickly explore hyperparameter combinations and select the best model without manual iteration.

### Large-Scale Datasets

Use `grid_ratio < 1.0` in `train()` to speed up grid search on datasets too large to search exhaustively in a reasonable time. The model is then retrained on all data with the optimal parameters.

### Multi-Objective Model Selection

When optimizing for multiple performance metrics (e.g., accuracy and F1 score), pass multiple scorers to `grid_kwargs['scoring']` and let `train()` select parameters via balanced RMS combination.

### Simple Model Training

Call `train()` with `grid_params=None` for straightforward estimator fitting with automatic random-state management.

---

## Notes

- **Estimator interface**: The estimator must follow scikit-learn conventions: `fit(X, y)` method, and optionally `score(X, y)` for validation. Most sklearn estimators (tree, ensemble, linear, SVM, etc.) are compatible.
- **y_dat flattening**: If `y_dat` is 2-D with one column, it is squeezed to 1-D before fitting. Multi-output targets are supported by sklearn estimators that implement multi-output logic.
- **Grid-search subsampling note**: When `0 < grid_ratio < 1`, the subsampling uses `train_test_split()` with the provided `random_state` for reproducibility. The holdout set is not evaluated but included in the final refit.
- **Best-estimator selection**: If `GridSearchCV` exposes `best_estimator_`, it is used directly (already fitted). For multi-scorer cases without `best_estimator_`, custom parameter selection via RMS is applied.
- **Output**: The returned model is fitted and ready to use (call `.predict()` immediately). For grid-search cases, it has been retrained on the full dataset.

---

## See Also

- **`data_io.create()`** — Prepare feature matrices for training.
- **`profile.cor_matrix()`** — Understand feature-target relationships before training.
- **`inspect.boxplot_metvx()`** — Evaluate model performance binned by feature values.
- **Scikit-learn docs**: https://scikit-learn.org/stable/modules/grid_search.html
