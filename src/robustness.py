"""Stage 7: what recognition needs from the sensor. Bracelet rotation, channel sets, sampling rate, channel failure.

LDA (log + shrink on Hudgins features; cheap, so every configuration is retrained):
    channels  E1 (within session) on named channel sets: all wrist, each wrist ring, forearm, forearm + wrist
    greedy    E1, backward elimination over the wrist channels: repeatedly drop the channel whose removal costs
              least (mean over participants). The order is chosen on dev and replayed on test (`order`)
    rates     E1 with the signal brought to 2048 / k Hz: causal anti-alias low-pass, every k-th sample kept,
              band-pass recomputed for the new rate (features.rate_filters)
    rotation  E2 without calibration (own session-1 model), sessions 2-3 seen through a bracelet turned by
              -1 / 0 / +1 electrode positions; trained as is or on rotated copies of session 1 (augmentation)
    failure   E1, 1 or 2 wrist channels fail on the test trials (every channel and every pair): `zero` = flat line
              (broken lead), `noise` = band-limited noise of a normal amplitude without EMG (lost contact); the
              model is trained on healthy channels, or retrained without the failed ones (`removed`: what a
              detector of bad channels would give back)
TCN (E3 + the stage-5 recipe: pretrained on other people, AdaBN + fine-tuning on the 3x1 calibration):
    rotation  the whole target session (calibration and test) turned; base trained as is or with rotations
    failure   channels fail after the calibration (test trials only)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

import adapt
import signals
import tcn
from data import FS, ring_rotation
from evaluate import score_rows
from features import FEATURES, feature_names, rate_filters, trial_features
from models import log_amplitude, make_model
from preprocess import design_filter, filter_causal, ms_to_samples

METRICS = ["bal_acc", "macro_f1"]
NOT_KEYS = ["participant", "session", "split", "failed"] + METRICS

WRIST = [f"W{k}" for k in range(1, 13)]
FOREARM = [f"F{k}" for k in range(1, 17)]
CHANNEL_SETS = {"wrist": WRIST, "ring3": WRIST[:6], "ring4": WRIST[6:], "forearm": FOREARM,
                "forearm_wrist": FOREARM + WRIST}
SET_LABELS = {"wrist": "запястье", "ring3": "кольцо 3\n(к локтю)", "ring4": "кольцо 4\n(к кисти)",
              "forearm": "предплечье", "forearm_wrist": "предпл. +\nзапястье"}
NAMES = {"plain": "обучение как есть", "rot_aug": "обучение с поворотами ±1", "none": "без калибровки",
         "full": "AdaBN + дообучение, 3×1", "zero": "канал = 0 (обрыв)", "noise": "канал = шум без ЭМГ",
         "removed": "модель обучена без канала"}


# ---------------------------------------------------------------- LDA

def cols_of(feat_cols: list[str], channels: list[str]) -> np.ndarray:
    """Positions of the feature columns of the given channels."""
    keep = set(channels)
    return np.flatnonzero([c.split("_", 1)[1] in keep for c in feat_cols])


def lda_e1(F: np.ndarray, meta: pd.DataFrame, y: np.ndarray, cols: np.ndarray, params: dict,
           tags: dict) -> list[dict]:
    """E1 (trials 1-5 -> 6-7 of the same session) LDA on feature columns `cols`, for every session in `meta`.

    Args:
        F: features with the amplitude columns already in log, one row per window.
        meta: participant, session, trial of every row.
        y: gesture of every row.
    """
    trial = meta.trial.to_numpy()
    rows = []
    for idx in meta.groupby(["participant", "session"]).indices.values():
        tr, te = idx[trial[idx] <= 5], idx[trial[idx] >= 6]
        m = make_model("lda", params).fit(F[np.ix_(tr, cols)], y[tr])
        rows += score_rows(meta.iloc[te], y[te], m.predict(F[np.ix_(te, cols)]), tags)
    return rows


def greedy_order(F: np.ndarray, meta: pd.DataFrame, y: np.ndarray, feat_cols: list[str], channels: list[str],
                 params: dict) -> list[str]:
    """Backward elimination: repeatedly drop the channel whose removal leaves the highest E1 balanced accuracy
    (mean over participants of their session means). Returns all channels in the order they are dropped; the last
    one is the single best channel left."""
    left, order = list(channels), []
    while len(left) > 1:
        score = {}
        for c in left:
            rows = lda_e1(F, meta, y, cols_of(feat_cols, [k for k in left if k != c]), params, {})
            score[c] = pd.DataFrame(rows).groupby("participant").bal_acc.mean().mean()
        order.append(max(score, key=score.get))
        left.remove(order[-1])
        print(f"greedy: drop {order[-1]} -> {len(left)} channels, {100 * score[order[-1]]:.1f} %", flush=True)
    return order + left


def rotate_cols(feat_cols: list[str], shift: int) -> np.ndarray:
    """Column order that turns wrist features into those of a bracelet rotated by `shift` positions:
    F[:, rotate_cols(cols, s)] (channel order as in `data.ring_rotation`)."""
    pos = {c: i for i, c in enumerate(feat_cols)}
    perm = ring_rotation(shift)
    out = []
    for c in feat_cols:
        f, ch = c.split("_", 1)
        out.append(pos[f"{f}_W{perm[int(ch[1:]) - 1] + 1}"])
    return np.array(out)


def lda_rotation(F: np.ndarray, meta: pd.DataFrame, y: np.ndarray, feat_cols: list[str], params: dict,
                 shifts: list[int], train_shifts: dict[str, list[int]]) -> list[dict]:
    """E2 without calibration: an LDA on session 1 of each participant, sessions 2-3 seen through a bracelet
    turned by each of `shifts`. `train_shifts` names the training variants: {name: shifts of the session-1 copies
    stacked for training} ([0] = as is)."""
    p, s = meta.participant.to_numpy(), meta.session.to_numpy()
    rows = []
    for q in np.unique(p):
        tr, te = np.flatnonzero((p == q) & (s == 1)), np.flatnonzero((p == q) & (s > 1))
        for name, ts in train_shifts.items():
            F_tr = np.concatenate([F[np.ix_(tr, rotate_cols(feat_cols, k))] for k in ts])
            m = make_model("lda", params).fit(F_tr, np.tile(y[tr], len(ts)))
            for sh in shifts:
                pred = m.predict(F[np.ix_(te, rotate_cols(feat_cols, sh))])
                rows += score_rows(meta.iloc[te], y[te], pred, {"part": "rotation", "model": "lda", "experiment": "E2",
                                                                "base": name, "method": "none", "shift": sh})
        print(f"rotation: p{q} done", flush=True)
    return rows


def band_noise(n_samples: int, n_traces: int, sos: np.ndarray, warm: int, rms_mv: float,
               rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise [n_samples, n_traces] at FS, scaled so that after the band-pass `sos` (first `warm`
    samples dropped) every trace has RMS `rms_mv`: a channel with a normal amplitude and no EMG in it."""
    x = rng.standard_normal((n_samples, n_traces))
    y, _ = filter_causal(x, sos)
    return x * (rms_mv / y[warm:].std(axis=0))


def noise_feature_bank(feat_cfg: dict, rms_mv: float, seed: int, seconds: float = 60.0) -> np.ndarray:
    """Hudgins features (amplitude ones in log) of the windows of one noise channel [N, 5], to replace the features
    of a channel that has lost contact."""
    sos, _ = rate_filters(feat_cfg)
    x = band_noise(int(seconds * FS), 1, sos, ms_to_samples(feat_cfg["warmup_ms"]), rms_mv,
                   np.random.default_rng(seed))
    f, _ = trial_features(x, sos, feat_cfg)
    return log_amplitude(f, feature_names(["N"]))


def failed_features(F: np.ndarray, feat_cols: list[str], channels: tuple[str, ...], mode: str, bank: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Copy of log features F with the given channels failed: `zero` = features of a flat line (log of ~0 amplitude,
    no crossings), `noise` = random rows of the noise bank."""
    zero = log_amplitude(np.zeros((1, len(FEATURES))), feature_names(["N"]))
    out = F.copy()
    for ch in channels:
        cols = [feat_cols.index(f"{f}_{ch}") for f in FEATURES]
        out[:, cols] = zero if mode == "zero" else bank[rng.integers(0, len(bank), size=len(out))]
    return out


def lda_failure(F: np.ndarray, meta: pd.DataFrame, y: np.ndarray, feat_cols: list[str], channels: list[str],
                params: dict, modes: list[str], n_failed: list[int], bank: np.ndarray, seed: int) -> list[dict]:
    """E1 with failed channels on the test trials: every combination of `n` channels for each n in `n_failed`.

    Rows: mode `none` (no failure), each of `modes` (model trained on healthy channels, unaware of the failure),
    and `removed` (model retrained without the failed channels: the failure is detected and the channel dropped).
    """
    rng = np.random.default_rng(seed)
    trial = meta.trial.to_numpy()
    rows = []
    for (p, s), idx in meta.groupby(["participant", "session"]).indices.items():
        tr, te = idx[trial[idx] <= 5], idx[trial[idx] >= 6]
        tags = {"part": "failure", "model": "lda", "experiment": "E1", "method": "none"}
        full = make_model("lda", params).fit(F[tr], y[tr])
        rows += score_rows(meta.iloc[te], y[te], full.predict(F[te]), {**tags, "mode": "none", "n_failed": 0})
        for n in n_failed:
            for combo in combinations(channels, n):
                t = {**tags, "n_failed": n, "failed": "+".join(combo)}
                for mode in modes:
                    pred = full.predict(failed_features(F[te], feat_cols, combo, mode, bank, rng))
                    rows += score_rows(meta.iloc[te], y[te], pred, {**t, "mode": mode})
                keep = cols_of(feat_cols, [c for c in channels if c not in combo])
                m = make_model("lda", params).fit(F[np.ix_(tr, keep)], y[tr])
                rows += score_rows(meta.iloc[te], y[te], m.predict(F[np.ix_(te, keep)]), {**t, "mode": "removed"})
        print(f"failure: p{p}/s{s} done", flush=True)
    return rows


# ---------------------------------------------------------------- TCN

def noise_signal_bank(pre: dict, rms_mv: float, n: int, seed: int) -> np.ndarray:
    """[n, T'] noise traces processed like a trial for the TCN (band-pass, start-up dropped, decimated), with RMS
    `rms_mv` after the band-pass."""
    sos = design_filter(band_hz=tuple(pre["band_hz"]), order=pre["order"], notch_hz=tuple(pre["notch_hz"]))
    x = band_noise(int(adapt.TRIAL_S * FS), n, sos, ms_to_samples(pre["warmup_ms"]), rms_mv,
                   np.random.default_rng(seed))
    return signals.trial_signal(x, sos, pre["warmup_ms"], pre["decimate"])


def failed_signals(X: np.ndarray, channels: tuple[int, ...], mode: str, bank: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Copy of signals X [N, C, T'] with the given channel positions flat (`zero`) or replaced by noise traces."""
    out = X.copy()
    for c in channels:
        out[:, c] = 0.0 if mode == "zero" else bank[rng.integers(0, len(bank), size=len(out))]
    return out


def tcn_split(base: tcn.TCN, base_name: str, X: np.ndarray, meta: pd.DataFrame, split, cfg: dict,
              ends: np.ndarray, fs: float, device: str, bank: np.ndarray | None) -> list[dict]:
    """Rotation and channel failure for the target sessions of one split, with the stage-5 protocol: trials
    1..calib_trials give the calibration (`calibration.budget`), the rest is the test.

    Rotation: calibration and test of the session are both turned (the bracelet stays turned for the whole
    wearing), then every method of `methods` adapts the base model. Failure (base `failure.base` only, no rotation):
    the adapted model meets failed channels on the test trials.
    """
    y = meta.gesture.to_numpy() - 1
    trial = meta.trial.to_numpy()
    budget = adapt.parse_budget(cfg["calibration"]["budget"])
    warmup_s, seed = cfg["preprocess"]["warmup_ms"] / 1000, cfg.get("seed", 42)
    fail = cfg.get("failure")
    rng = np.random.default_rng(seed)
    rows = []
    for key in meta[split.test].groupby(["participant", "session"]).indices:
        sess = split.test & (meta.participant.to_numpy() == key[0]) & (meta.session.to_numpy() == key[1])
        te = np.flatnonzero(sess & (trial > cfg["calib_trials"]))
        win_meta = meta.iloc[np.repeat(te, len(ends))].reset_index(drop=True)
        y_true = np.repeat(y[te], len(ends))
        X_cal, y_cal = adapt.calib_signals(X, meta, sess, budget, warmup_s, fs)
        for shift in cfg["rotation"]["shifts"]:
            perm = ring_rotation(shift)
            X_te = X[te][:, perm]
            for name, spec in cfg["methods"].items():
                m = adapt.adapted(base, X_cal[:, perm], y_cal, spec, cfg["crop"], seed, device)
                tags = {"model": "tcn", "experiment": split.experiment, "base": base_name, "method": name}
                pred = tcn.predict_windows(m, X_te, ends, device).ravel()
                rows += score_rows(win_meta, y_true, pred, {**tags, "part": "rotation", "shift": shift})
                if not fail or shift != 0 or base_name != fail["base"]:
                    continue
                rows += score_rows(win_meta, y_true, pred, {**tags, "part": "failure", "mode": "none", "n_failed": 0})
                for n in fail["n_failed"]:
                    for combo in combinations(range(X.shape[1]), n):
                        t = {**tags, "part": "failure", "n_failed": n, "failed": "+".join(f"W{c + 1}" for c in combo)}
                        for mode in fail["modes"]:
                            pred = tcn.predict_windows(m, failed_signals(X_te, combo, mode, bank, rng), ends, device)
                            rows += score_rows(win_meta, y_true, pred.ravel(), {**t, "mode": mode})
        print(f"{split.experiment} {split.name} {base_name} p{key[0]}/s{key[1]} done", flush=True)
    return rows


# ---------------------------------------------------------------- summary and figures

def per_participant(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One table per part: mean over a participant's sessions (and over the failed-channel choices)."""
    out = {}
    for part, g in scores.groupby("part", sort=False):
        g = g.dropna(axis=1, how="all")
        keys = [c for c in g.columns if c not in NOT_KEYS] + ["participant"]
        out[part] = g.groupby(keys, as_index=False, sort=False)[METRICS].mean()
    return out


def summary(pp: pd.DataFrame) -> pd.DataFrame:
    """Per setting: mean, SD, median, min, max of balanced accuracy over participants and mean macro-F1."""
    keys = [c for c in pp.columns if c not in ["participant"] + METRICS]
    return (pp.groupby(keys, sort=False)
            .agg(mean=("bal_acc", "mean"), sd=("bal_acc", "std"), median=("bal_acc", "median"),
                 min=("bal_acc", "min"), max=("bal_acc", "max"), macro_f1=("macro_f1", "mean"),
                 n=("bal_acc", "size"))
            .reset_index())


def failure_by_channel(scores: pd.DataFrame) -> pd.DataFrame:
    """Single-channel failures: mean balanced accuracy over all test sessions for every failed channel."""
    g = scores[(scores.part == "failure") & (scores.n_failed == 1)].dropna(axis=1, how="all")
    keys = [c for c in ["model", "experiment", "base", "method", "mode", "failed"] if c in g.columns]
    return g.groupby(keys, as_index=False, sort=False).bal_acc.mean()


def _points(ax, x: float, v: np.ndarray, color: str) -> None:
    ax.scatter(np.full(len(v), x), 100 * v, s=12, color=color, alpha=0.35, lw=0)


def _mean_sd(ax, xs: list, vals: list[np.ndarray], color: str, label: str | None = None,
             line: bool = True) -> list[float]:
    """Mean ± SD over participants at every x, annotated with the mean; returns the means in %."""
    m = [100 * np.mean(v) for v in vals]
    sd = [100 * np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals]
    ax.errorbar(xs, m, yerr=sd, marker="o", ms=5, capsize=3, color=color, label=label, ls="-" if line else "none")
    for x, v in zip(xs, m):
        ax.annotate(f"{v:.0f}", (x, v), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
    return m


def plot_sensor(pp: dict[str, pd.DataFrame], tag: str, order: list[str] | None = None) -> str | None:
    """Channel sets, greedy channel removal and sampling rate (E1), one panel each.
    Point = participant, marker and bar = mean ± SD over participants."""
    import matplotlib.pyplot as plt
    import viz

    parts = [p for p in ("channels", "greedy", "rate") if p in pp]
    if not parts:
        return None
    viz.setup()
    fig, axes = plt.subplots(1, len(parts), figsize=(5 * len(parts), 4.2), squeeze=False)
    c = viz.SERIES[0]
    for ax, part in zip(axes[0], parts):
        d = pp[part]
        if part == "channels":
            sets = list(dict.fromkeys(d.setting))
            vals = [d[d.setting == s].bal_acc.to_numpy() for s in sets]
            for i, v in enumerate(vals):
                _points(ax, i, v, c)
            _mean_sd(ax, list(range(len(sets))), vals, c, line=False)
            n = d.groupby("setting", sort=False).n_channels.first()
            ax.set_xticks(range(len(sets)), [f"{SET_LABELS.get(s, s)}\n{n[s]:.0f} к." for s in sets], fontsize=7)
            ax.set_title("Наборы каналов")
        elif part == "greedy":
            ns = sorted(d.n_channels.astype(int).unique(), reverse=True)
            vals = [d[d.n_channels == n].bal_acc.to_numpy() for n in ns]
            for n, v in zip(ns, vals):
                _points(ax, n, v, c)
            means = _mean_sd(ax, ns, vals, c, label="жадное удаление")
            if order:
                for n, m in zip(ns, means):
                    if n < len(order):
                        ax.annotate(f"−{order[len(order) - 1 - n]}", (n, m), xytext=(-4, -14),
                                    textcoords="offset points", fontsize=6.5, color=viz.MUTED, ha="center")
            if "channels" in pp:
                ch = pp["channels"]
                for i, s in enumerate(("ring3", "ring4")):
                    v = ch[ch.setting == s].bal_acc.to_numpy()
                    if len(v):
                        ax.scatter([6], [100 * v.mean()], marker="D", s=30, color=viz.SERIES[1 + i], zorder=3,
                                   label=SET_LABELS[s].replace("\n", " "))
            ax.invert_xaxis()
            ax.set_xticks(ns)
            ax.set_xlabel("каналов запястья")
            ax.legend(fontsize=7, loc="lower left")
            ax.set_title("Жадное удаление каналов")
        else:
            d = d.sort_values("fs_hz", ascending=False)
            fss = list(dict.fromkeys(d.fs_hz))
            vals = [d[d.fs_hz == f].bal_acc.to_numpy() for f in fss]
            for f, v in zip(fss, vals):
                _points(ax, f, v, c)
            _mean_sd(ax, fss, vals, c)
            hi = d.groupby("fs_hz").band_hi_hz.first()
            ax.set_xscale("log", base=2)
            ax.set_xticks(fss, [f"{f:g}\n20–{hi[f]:.0f} Гц" for f in fss], fontsize=7)
            ax.minorticks_off()
            ax.invert_xaxis()
            ax.set_xlabel("частота дискретизации, Гц (под ней — полоса)")
            ax.set_title("Частота дискретизации")
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("balanced accuracy, %")
    fig.suptitle("Датчик: E1, LDA; точка = участник, маркер и черта = среднее ± SD", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"07_sensor_{tag}")
    plt.close(fig)
    return str(path)


def _panels(d: pd.DataFrame) -> list[tuple]:
    return list(dict.fromkeys(zip(d.model, d.experiment, d.method)))


def _panel_title(model: str, exp: str, method: str) -> str:
    if exp == "E1":                      # within session: nothing to calibrate
        return f"{model.upper()}, {exp}"
    return f"{model.upper()}, {exp}: {NAMES.get(method, method)}"


def plot_rotation(pp: pd.DataFrame, tag: str) -> str:
    """Balanced accuracy vs bracelet rotation, one panel per model x experiment x method, one line per training
    variant. Point = participant, line = mean ± SD."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    panels = _panels(pp)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4), sharey=True, squeeze=False)
    for ax, (model, exp, method) in zip(axes[0], panels):
        d = pp[(pp.model == model) & (pp.experiment == exp) & (pp.method == method)]
        for i, b in enumerate(dict.fromkeys(d.base)):
            g = d[d.base == b]
            shifts = sorted(g["shift"].unique())
            vals = [g[g["shift"] == s].bal_acc.to_numpy() for s in shifts]
            off = (i - 0.5) * 0.08
            for s, v in zip(shifts, vals):
                _points(ax, s + off, v, viz.SERIES[i])
            _mean_sd(ax, [s + off for s in shifts], vals, viz.SERIES[i], label=NAMES.get(b, b))
            ax.set_xticks(shifts)
        ax.set_ylim(0, 100)
        ax.set_xlabel("поворот, позиций электрода (1 = 60°)")
        ax.set_title(_panel_title(model, exp, method), fontsize=9)
        ax.legend(fontsize=7, loc="lower center")
    axes[0][0].set_ylabel("balanced accuracy, %")
    fig.suptitle("Поворот браслета: точка = участник, линия = среднее ± SD", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"07_rotation_{tag}")
    plt.close(fig)
    return str(path)


def plot_failure(pp: pd.DataFrame, tag: str) -> str:
    """Balanced accuracy vs number of failed channels, one panel per model x experiment x method, one line per
    failure mode (0 failed = the same model on healthy channels). Mean over participants ± SD."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    panels = _panels(pp)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4), sharey=True, squeeze=False)
    for ax, (model, exp, method) in zip(axes[0], panels):
        d = pp[(pp.model == model) & (pp.experiment == exp) & (pp.method == method)]
        clean = d[d["mode"] == "none"].bal_acc.to_numpy()
        modes = [m for m in ("zero", "noise", "removed") if m in set(d["mode"])]
        for i, mode in enumerate(modes):
            g = d[d["mode"] == mode]
            ns = sorted(g.n_failed.unique())
            vals = [clean] + [g[g.n_failed == n].bal_acc.to_numpy() for n in ns]
            off = (i - (len(modes) - 1) / 2) * 0.06
            _mean_sd(ax, [x + off for x in [0] + ns], vals, viz.SERIES[i], label=NAMES.get(mode, mode))
            ax.set_xticks([0] + ns)
        ax.set_ylim(0, 100)
        ax.set_xlabel("отказавших каналов запястья")
        ax.set_title(_panel_title(model, exp, method), fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylabel("balanced accuracy, %")
    fig.suptitle("Отказ каналов: среднее по участникам ± SD (усреднено по всем каналам / парам)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"07_failure_{tag}")
    plt.close(fig)
    return str(path)
