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
from preprocess import design_antialias, design_filter, filter_causal, ms_to_samples, sliding_windows

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


def rate_band(cfg: dict) -> tuple[float, float]:
    """Band-pass edges at the working rate FS / downsample: the upper edge is lowered to the anti-alias cutoff
    (antialias_ratio * new rate) when it lies above, e.g. 0.4 * 512 = 205 Hz instead of 450 Hz."""
    lo, hi = cfg["band_hz"]
    k = cfg.get("downsample", 1)
    return (lo, min(hi, cfg["antialias_ratio"] * FS / k)) if k > 1 else (lo, hi)


def rate_filters(cfg: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Band-pass designed for the working rate FS / downsample and, when downsampling, the anti-alias low-pass
    applied at FS before every k-th sample is kept (None otherwise)."""
    k = cfg.get("downsample", 1)
    band = rate_band(cfg)
    sos = design_filter(fs=FS / k, band_hz=band, order=cfg["order"], notch_hz=tuple(cfg["notch_hz"]))
    return sos, (design_antialias(band[1], cfg["antialias_order"]) if k > 1 else None)


def trial_features(x: np.ndarray, sos: np.ndarray, cfg: dict,
                   aa_sos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Filter one trial, drop the filter start-up, cut windows and compute features.

    Args:
        x: raw trial [T, C], mV, at FS.
        sos: band-pass from `design_filter` for the working rate FS / downsample.
        cfg: dict with window_ms, step_ms, warmup_ms, zc_thr_mv, ssc_thr_mv and optional downsample (k).
        aa_sos: anti-alias low-pass at FS, needed when downsample > 1.

    Returns:
        Features [N, 5 * C] and the end time of each window in seconds from the trial start.
    """
    k = cfg.get("downsample", 1)
    if k > 1:
        x = filter_causal(x, aa_sos)[0][::k]
    fs = FS / k
    y, _ = filter_causal(x, sos)
    warm = ms_to_samples(cfg["warmup_ms"], fs)
    win, step = ms_to_samples(cfg["window_ms"], fs), ms_to_samples(cfg["step_ms"], fs)
    w = sliding_windows(y[warm:], win, step)
    t_end = (warm + np.arange(len(w)) * step + win) / fs
    return hudgins(w, cfg["zc_thr_mv"], cfg["ssc_thr_mv"]), t_end


def _cfg_key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def feature_table(index: pd.DataFrame, cfg: dict, channels: list[str]) -> pd.DataFrame:
    """Features of all windows of all trials in `index`, cached per participant.

    Args:
        index: rows of `data.build_index()`.
        cfg: preprocessing + feature settings (band_hz, order, notch_hz, window_ms, step_ms, warmup_ms,
            zc_thr_mv, ssc_thr_mv); optional record_channels (record columns, default the 12 wrist channels) and
            downsample with antialias_ratio, antialias_order (see `rate_filters`). The cache key is derived from it.
        channels: channel names for the feature columns.

    Returns:
        One row per window: participant, session, gesture, trial, t_end, then the feature columns.
    """
    sos, aa_sos = rate_filters(cfg)
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
            x = load_trial(r.session, r.participant, r.gesture, r.trial, cfg.get("record_channels"))
            f, t = trial_features(x, sos, cfg, aa_sos)
            meta = pd.DataFrame({"participant": r.participant, "session": r.session, "gesture": r.gesture,
                                 "trial": r.trial, "t_end": t})
            rows.append(pd.concat([meta, pd.DataFrame(f, columns=names)], axis=1))
        df = pd.concat(rows, ignore_index=True)
        df.to_parquet(path)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
