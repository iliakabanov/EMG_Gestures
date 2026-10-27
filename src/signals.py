"""Filtered raw signal per trial for neural networks: causal band-pass, start-up dropped, decimated, cached.

Same filter and start-up handling as the feature pipeline, so a network and the feature models see the same signal.
Decimation keeps every `decimate`-th sample after the band-pass; the band-pass doubles as the anti-alias filter
(at 2048 / 2 = 1024 Hz the new Nyquist is 512 Hz, where the 20-450 Hz filter is already at about -8 dB).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from data import FS, ROOT, load_trial
from preprocess import design_filter, filter_causal, ms_to_samples

CACHE_DIR = ROOT / "data" / "cache"
META_COLS = ["participant", "session", "gesture", "trial"]


def trial_signal(x: np.ndarray, sos: np.ndarray, warmup_ms: float, decimate: int) -> np.ndarray:
    """Raw trial [T, C] -> filtered, start-up dropped, decimated signal [C, T'] as float32."""
    y, _ = filter_causal(x, sos)
    return np.ascontiguousarray(y[ms_to_samples(warmup_ms):][::decimate].T, dtype=np.float32)


def _cfg_key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def signal_table(index: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Signals of all trials in `index`, cached per participant.

    Args:
        index: rows of `data.build_index()`.
        cfg: band_hz, order, notch_hz, warmup_ms, decimate.

    Returns:
        meta: one row per trial (participant, session, gesture, trial), in the order of the array.
        X: signals [n_trials, C, T'] in mV.
    """
    keys = ["band_hz", "order", "notch_hz", "warmup_ms", "decimate"]
    cfg = {k: cfg[k] for k in keys}
    sos = design_filter(band_hz=tuple(cfg["band_hz"]), order=cfg["order"], notch_hz=tuple(cfg["notch_hz"]))
    cache = CACHE_DIR / f"signals_{_cfg_key(cfg)}"
    cache.mkdir(parents=True, exist_ok=True)
    metas, arrays = [], []
    for p, sub in index.groupby("participant"):
        path = cache / f"p{p}.npz"
        if not path.exists():
            sub = sub.sort_values(META_COLS)
            X = np.stack([trial_signal(load_trial(r.session, r.participant, r.gesture, r.trial), sos,
                                       cfg["warmup_ms"], cfg["decimate"]) for r in sub.itertuples()])
            np.savez(path, X=X, meta=sub[META_COLS].to_numpy())
        with np.load(path) as z:
            arrays.append(z["X"])
            metas.append(pd.DataFrame(z["meta"], columns=META_COLS))
    return pd.concat(metas, ignore_index=True), np.concatenate(arrays)


def effective_fs(decimate: int) -> float:
    """Sampling rate after decimation, Hz."""
    return FS / decimate
