"""Single source of truth for feature preprocessing.

Why this module exists (v2.2 fix): preprocessing used to be re-implemented in
four places — both trainers, the serving path, and each audit script — with
three drift-prone quirks:

  1. The top-10 country list was recomputed FROM THE DATA BEING SCORED, so a
     drifting country mix silently changed how the same user was encoded.
  2. LabelEncoder mapped unknown categories to ``classes_[0]`` — i.e. the
     alphabetically first real category, not a neutral "Other".
  3. Missing model columns were silently zero-filled at serve time, so schema
     drift produced garbage predictions instead of an error.

The fix: preprocessing is a sklearn ColumnTransformer that is FIT ONCE at
training time and shipped INSIDE the served Pipeline in the model bundle.
Train, audit, and serving all call the same fitted object; there is nothing
left to re-implement or drift.

Encoding notes:
  - OneHotEncoder(max_categories=N) reproduces the old "top-N + Other"
    behaviour, but the bucket membership is learned at fit time and frozen.
  - handle_unknown="infrequent_if_exist" sends unseen categories to the
    infrequent bucket (the honest "Other"), never to an arbitrary class.
  - Missing values: constant-impute — 0 for numerics, "(missing)" for
    categoricals — matching the old fillna(0) semantics for numerics.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def _to_float64(X):
    """Cast numerics to float64 before imputation. Must be a module-level
    named function (not a lambda) so the fitted pipeline stays picklable.
    SimpleImputer(fill_value=0.0) refuses int64 input columns otherwise."""
    return np.asarray(X, dtype=np.float64)

# Top-N + infrequent bucket per categorical feature. 11 = old top-10 country
# behaviour + one "Other" bucket; low-cardinality features (platform, channel)
# are unaffected because they never exceed the cap.
MAX_CATEGORIES = 11


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build the (unfitted) shared preprocessor for a feature set."""
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="(missing)")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=MAX_CATEGORIES,
            sparse_output=False,
        )),
    ])
    num_pipe = Pipeline([
        ("cast",   FunctionTransformer(_to_float64, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
    ])
    return ColumnTransformer(
        [
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        verbose_feature_names_out=False,
    )


def select_raw_inputs(df, features: list[str]):
    """Select the raw model-input columns, failing LOUDLY on schema drift.

    The old serving path zero-filled any missing column, which turned a
    renamed/dropped feature into silently garbage predictions. A missing
    column is a contract violation and must surface as an error.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(
            f"feature table is missing model input columns: {missing} — "
            f"feature schema drifted since training; rebuild features or retrain"
        )
    return df[features]
