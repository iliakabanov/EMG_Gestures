"""Window classifiers: LDA and LightGBM, each behind a scaler fitted on the training split only."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

AMPLITUDE = ("MAV", "RMS", "WL")
LOG_EPS = 1e-6     # mV; far below the rest noise, only guards against log(0)


def _log_columns(X: np.ndarray, cols: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float64, copy=True)
    X[:, cols] = np.log(X[:, cols] + LOG_EPS)
    return X


def amplitude_columns(feat_cols: list[str]) -> np.ndarray:
    """Positions of the MAV / RMS / WL columns."""
    return np.flatnonzero([c.split("_")[0] in AMPLITUDE for c in feat_cols])


def log_amplitude(X: np.ndarray, feat_cols: list[str]) -> np.ndarray:
    """Copy of X with the log of the amplitude columns: a common gain k becomes the shift log k."""
    return _log_columns(X, amplitude_columns(feat_cols))


def make_model(name: str, params: dict | None = None, seed: int = 42,
               feat_cols: list[str] | None = None) -> Pipeline:
    """[log of amplitude features] + StandardScaler + classifier.

    Args:
        name: model name from the config; also its type unless `params["type"]` is given.
        params: classifier constructor arguments, plus optional keys
            type: "lda" or "lgbm" (lets one config hold several variants of the same model);
            log_amp: take the log of MAV / RMS / WL columns (needs `feat_cols`).
        seed: random state for LightGBM.
        feat_cols: feature column names, in the order of X.
    """
    params = dict(params or {})
    kind = params.pop("type", name)
    log_amp = params.pop("log_amp", False)
    if kind == "lda":
        clf = LinearDiscriminantAnalysis(**params)
    elif kind == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(random_state=seed, verbose=-1, **params)
    else:
        raise ValueError(kind)
    steps = []
    if log_amp:
        if feat_cols is None:
            raise ValueError("log_amp needs feat_cols")
        steps.append(FunctionTransformer(_log_columns, kw_args={"cols": amplitude_columns(feat_cols)}))
    return make_pipeline(*steps, StandardScaler(), clf)
