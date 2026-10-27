"""Hudgins time-domain features per window and channel, and the cached feature table for a set of trials.

Features (per channel, x = filtered signal in mV inside one window):
    MAV = mean |x|                       amplitude
    RMS = sqrt(mean x^2)                 amplitude, more weight on peaks
    WL  = sum |x[i] - x[i-1]|            waveform length: amplitude and frequency together
    ZC  = zero crossings with |x[i] - x[i+1]| >= zc_thr               frequency
    SSC = slope sign changes with a step >= ssc_thr on either side    frequency
Thresholds keep the noise floor from producing crossings.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from data import FS, ROOT, load_trial
from preprocess import design_filter, filter_causal, ms_to_samples, sliding_windows

FEATURES = ["MAV", "RMS", "WL", "ZC", "SSC"]
CACHE_DIR = ROOT / "data" / "cache"


def hudgins(w: np.ndarray, zc_thr: float, ssc_thr: float) -> np.ndarray:
    """Hudgins features of windows w [N, L, C] -> [N, 5 * C], grouped by feature (all MAV, then all RMS, ...)."""
    d = np.diff(w, axis=1)                       # x[i+1] - x[i]
    mav = np.abs(w).mean(axis=1)
    rms = np.sqrt((w ** 2).mean(axis=1))
    wl = np.abs(d).sum(axis=1)
    zc = ((w[:, :-1] * w[:, 1:] < 0) & (np.abs(d) >= zc_thr)).sum(axis=1)
    d1, d2 = d[:, :-1], d[:, 1:]                 # slope before and after each inner sample
    ssc = ((d1 * d2 < 0) & ((np.abs(d1) >= ssc_thr) | (np.abs(d2) >= ssc_thr))).sum(axis=1)
    return np.concatenate([mav, rms, wl, zc, ssc], axis=1).astype(np.float32)


def feature_names(channels: list[str]) -> list[str]:
    """Column names matching the order of `hudgins` output."""
    return [f"{f}_{c}" for f in FEATURES for c in channels]


def trial_features(x: np.ndarray, sos: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Filter one trial, drop the filter start-up, cut windows and compute features.

    Args:
        x: raw trial [T, C], mV.
        sos: filter from `design_filter`.
        cfg: dict with window_ms, step_ms, warmup_ms, zc_thr_mv, ssc_thr_mv.

    Returns:
        Features [N, 5 * C] and the end time of each window in seconds from the trial start.
    """
    y, _ = filter_causal(x, sos)
    warm = ms_to_samples(cfg["warmup_ms"])
    win, step = ms_to_samples(cfg["window_ms"]), ms_to_samples(cfg["step_ms"])
    w = sliding_windows(y[warm:], win, step)
    t_end = (warm + np.arange(len(w)) * step + win) / FS
    return hudgins(w, cfg["zc_thr_mv"], cfg["ssc_thr_mv"]), t_end


def _cfg_key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def feature_table(index: pd.DataFrame, cfg: dict, channels: list[str]) -> pd.DataFrame:
    """Features of all windows of all trials in `index`, cached per participant.

    Args:
        index: rows of `data.build_index()`.
        cfg: preprocessing + feature settings (band_hz, order, notch_hz, window_ms, step_ms, warmup_ms,
            zc_thr_mv, ssc_thr_mv); the cache key is derived from it.
        channels: channel names for the feature columns.

    Returns:
        One row per window: participant, session, gesture, trial, t_end, then the feature columns.
    """
    sos = design_filter(band_hz=tuple(cfg["band_hz"]), order=cfg["order"], notch_hz=tuple(cfg["notch_hz"]))
    cache = CACHE_DIR / f"features_{_cfg_key(cfg)}"
    cache.mkdir(parents=True, exist_ok=True)
    names = feature_names(channels)
    parts = []
    for p, sub in index.groupby("participant"):
        path = cache / f"p{p}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
            continue
        rows = []
        for r in sub.itertuples():
            f, t = trial_features(load_trial(r.session, r.participant, r.gesture, r.trial), sos, cfg)
            meta = pd.DataFrame({"participant": r.participant, "session": r.session, "gesture": r.gesture,
                                 "trial": r.trial, "t_end": t})
            rows.append(pd.concat([meta, pd.DataFrame(f, columns=names)], axis=1))
        df = pd.concat(rows, ignore_index=True)
        df.to_parquet(path)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
