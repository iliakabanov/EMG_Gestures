"""Stage 6: streaming recognition. A continuous stream is assembled from trials, filtered causally in one pass,
classified every 50 ms, turned into events by post-processing and scored as events.

Stream. GRABMyo has no continuous recordings, so a stream is spliced from the test trials of one session:
    rest (lead) -> gesture -> rest -> gesture -> ... -> rest
Gestures: every non-rest gesture of every test trial once, in random order, a random piece of `gesture_s` seconds.
Rest: random pieces of `rest_s` seconds from the session's rest trials, taken cyclically (there are only ~20 s of
unique rest per session, so rest pieces repeat; the number of unique rest seconds is reported).

Joins are equal-power crossfades: over L samples the outgoing piece is weighted by cos(pi/2 u) and the incoming one
by sin(pi/2 u), u = 0..1. For independent noise-like signals the power then moves smoothly from one level to the
other, with no step (a step would make the band-pass filter ring) and no dip. Rest -> gesture uses `rise_ms`
(a model of the contraction build-up), gesture -> rest `fall_ms` (release), and the joins inside the rest pool
(between rest trials and where the pool wraps around) `join_ms`. A gesture lasts from the start of its rise
(the decision to contract) to the end of its fall (the muscle has relaxed). The trials contain held contractions,
not the real movement onset (stage 1), so the rise is an assumption and latencies are measured from its start.

Post-processing of the class probabilities p_t (one row per decision, every `step_ms`):
    smoothing      p~_t = a p_t + (1 - a) p~_{t-1},  a = 1 - exp(-step / tau)       (tau = 0: none)
    activation     the best non-rest class c_t is active when p~_t(c_t) >= theta
    firing         an event (t, c) fires when c has been active for `min_ms` in a row
    refractory     after an event no new one fires until the activation has dropped at least once
                   AND `refractory_ms` have passed (one held gesture gives one command)

Event scoring per session. A gesture occupies [onset, offset + tol), onset = start of the rise, offset = end of
the fall, tol = grace time for decisions that lag behind the signal (window, smoothing, minimum duration);
the first event inside decides: correct class -> correct, other class -> confused, none -> missed;
further events inside are `extra`. Events outside every gesture interval are false activations; their rate is per
minute of rest (rest time = stream after the lead minus the gesture intervals). Latency = first correct event - onset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from data import FS, N_GESTURES, REST_GESTURE, load_trial

GESTURES = [g for g in range(1, N_GESTURES + 1) if g != REST_GESTURE]


@dataclass
class Stream:
    x: np.ndarray            # raw wrist signal [T, 12], mV, at FS
    events: pd.DataFrame     # one row per gesture: gesture, trial, piece_start (sample in the trial),
                             # onset_s (start of the rise), offset_s (end of the fall)
    lead_s: float            # rest at the start (filter and model warm-up), not scored
    unique_rest_s: float     # seconds of distinct rest data the rest pieces were cut from


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """a followed by b, the last n samples of a overlapped with the first n of b by an equal-power crossfade."""
    if n <= 0:
        return np.concatenate([a, b])
    u = (np.arange(n) + 0.5) / n
    mix = a[-n:] * np.cos(np.pi / 2 * u)[:, None] + b[:n] * np.sin(np.pi / 2 * u)[:, None]
    return np.concatenate([a[:-n], mix, b[n:]])


def rest_pool(trials: list[np.ndarray], join: int) -> np.ndarray:
    """Rest trials joined by crossfades into a loop: reading it cyclically never meets a step, also at the wrap."""
    pool = trials[0]
    for x in trials[1:]:
        pool = crossfade(pool, x, join)
    # Fade the end into the beginning and drop that beginning: index -1 is now followed smoothly by index 0.
    return crossfade(pool, pool[:join], join)[join:] if join > 0 else pool


def build_stream(participant: int, session: int, trials: list[int], seed: int, gesture_s: tuple[float, float],
                 rest_s: tuple[float, float], lead_s: float, rise_ms: float = 50.0, fall_ms: float = 50.0,
                 join_ms: float = 20.0) -> Stream:
    """Splice one session's trials into a stream (see the module docstring).

    Args:
        trials: trials to use (the test trials, e.g. 4-7; the calibration trials stay out of the stream).
        seed: the order and the pieces depend only on (seed, participant, session), so every model sees
            the same stream.
        gesture_s, rest_s: ranges of the gesture and rest piece durations, seconds (each includes its crossfades).
        lead_s: rest at the start.
        rise_ms, fall_ms, join_ms: crossfade lengths rest -> gesture, gesture -> rest, inside the rest pool.
    """
    rng = np.random.default_rng([seed, participant, session])
    ms = lambda v: int(round(v * FS / 1000))
    rise, fall, join = ms(rise_ms), ms(fall_ms), ms(join_ms)
    pool = rest_pool([load_trial(session, participant, REST_GESTURE, t) for t in trials], join)

    def rest_piece(seconds: float) -> np.ndarray:
        start = rng.integers(len(pool))
        return np.take(pool, (start + np.arange(int(round(seconds * FS)))) % len(pool), axis=0)

    x = rest_piece(lead_s)
    rows = []
    order = rng.permutation([(g, tr) for g in GESTURES for tr in trials])
    for g, tr in order:
        trial = load_trial(session, participant, int(g), int(tr))
        n = int(round(rng.uniform(*gesture_s) * FS))
        start = rng.integers(len(trial) - n + 1)
        onset = len(x) - rise                                   # the rise (decision to contract) starts here
        x = crossfade(x, trial[start:start + n], rise)
        offset = onset + n                                      # the fall (release) ends here
        x = crossfade(x, rest_piece(rng.uniform(*rest_s)), fall)
        rows.append({"gesture": int(g), "trial": int(tr), "piece_start": int(start), "onset_s": onset / FS,
                     "offset_s": offset / FS})
    return Stream(x, pd.DataFrame(rows), lead_s, len(pool) / FS)


def decision_times(n_samples: int, win: int, step: int, fs: float) -> np.ndarray:
    """End time (s) of every decision window over a signal of `n_samples`."""
    return (np.arange((n_samples - win) // step + 1) * step + win) / fs


def lda_probs(model, x_filt: np.ndarray, win: int, step: int, feat_cfg: dict,
              feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class probabilities of a feature model on every window of a filtered stream [T, C] at FS.

    Returns decision times (s), probabilities [n, classes] and the class labels (gestures).
    """
    from features import hudgins
    from models import log_amplitude
    from preprocess import sliding_windows

    F = hudgins(sliding_windows(x_filt, win, step), feat_cfg["zc_thr_mv"], feat_cfg["ssc_thr_mv"])
    P = model.predict_proba(log_amplitude(F, feat_cols))
    return decision_times(len(x_filt), win, step, FS), P, model.classes_


def tcn_probs(model, x_filt: np.ndarray, win: int, step: int, decimate: int,
              device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class probabilities of a TCN on a filtered stream [T, C] at FS, at the same moments as `lda_probs`.

    The whole stream goes through the causal network in one pass; the window that ends at sample k*step + win - 1
    (at FS) ends at sample (k*step + win - 1) // decimate after decimation.
    """
    import torch

    t = decision_times(len(x_filt), win, step, FS)
    ends = (np.arange(len(t)) * step + win - 1) // decimate
    x = torch.from_numpy(np.ascontiguousarray(x_filt[::decimate].T, dtype=np.float32))[None].to(device)
    model.eval()
    with torch.no_grad():
        P = torch.softmax(model(x)[0][:, ends], dim=0).T.cpu().numpy()
    return t, P, np.arange(1, P.shape[1] + 1)


# ---------------------------------------------------------------- post-processing

def smooth(P: np.ndarray, step_s: float, tau_ms: float) -> np.ndarray:
    """Causal exponential smoothing of probabilities along time (rows), started from the first row."""
    if tau_ms <= 0:
        return P
    a = 1 - np.exp(-step_s / (tau_ms / 1000))
    zi = (1 - a) * P[0]
    out, _ = signal.lfilter([a], [1, -(1 - a)], P, axis=0, zi=zi[None, :])
    return out


def detect(P: np.ndarray, t: np.ndarray, theta: float, min_ms: float, refractory_ms: float,
           classes: np.ndarray) -> list[tuple[float, int]]:
    """Events (time, gesture) from smoothed probabilities P [n, n_classes] (see the module docstring).

    Works on runs of equal (active, class) decisions instead of single decisions, so a sweep over thresholds
    stays fast.
    """
    rest = classes == REST_GESTURE
    Pg = np.where(rest[None, :], -1.0, P)
    best = Pg.argmax(axis=1)
    active = Pg[np.arange(len(P)), best] >= theta
    key = np.where(active, best, -1)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    step = t[1] - t[0]
    n_min = max(1, int(round(min_ms / 1000 / step)) + 1) if min_ms > 0 else 1

    events, last, released = [], -np.inf, True
    for s, e in zip(starts, ends):
        if key[s] < 0:
            released = True
            continue
        if e - s < n_min:
            continue
        fire = t[s + n_min - 1]
        if released and fire - last >= refractory_ms / 1000:
            events.append((float(fire), int(classes[key[s]])))
            last, released = fire, False
    return events


def score_events(events: list[tuple[float, int]], truth: pd.DataFrame, lead_s: float, end_s: float,
                 tol_s: float) -> dict:
    """Event metrics of one stream (see the module docstring)."""
    on, off = truth.onset_s.to_numpy(), truth.offset_s.to_numpy() + tol_s
    first = np.full(len(truth), -1)            # index of the first event inside each gesture interval
    extra = fa = 0
    for i, (te, ce) in enumerate(events):
        if te < lead_s:
            continue
        k = np.flatnonzero((on <= te) & (te < off))
        if len(k) == 0:
            fa += 1
        elif first[k[0]] < 0:
            first[k[0]] = i
        else:
            extra += 1
    got = first >= 0
    cls = np.array([events[i][1] if i >= 0 else 0 for i in first])
    correct = got & (cls == truth.gesture.to_numpy())
    lat = np.array([events[i][0] for i in first[correct]]) - on[correct]
    rest_min = (end_s - lead_s - np.sum(np.minimum(off, end_s) - on)) / 60
    n = len(truth)
    return {"n": n, "correct": correct.sum() / n, "confused": (got & ~correct).sum() / n, "missed": (~got).sum() / n,
            "extra_per_gesture": extra / n, "fa_per_min": fa / rest_min, "fa": fa, "rest_min": rest_min,
            "latency_ms_median": 1000 * float(np.median(lat)) if len(lat) else np.nan}


def sweep(probs: dict, truths: dict, grid: dict, step_s: float, tol_s: float) -> pd.DataFrame:
    """Event metrics for every stream x post-processing setting.

    Args:
        probs: {stream key: (t [n], P [n, classes], classes, lead_s, end_s)}.
        truths: {stream key: events table of the stream}.
        grid: lists tau_ms, min_ms, refractory_ms, theta.
    """
    rows = []
    for key, (t, P, classes, lead_s, end_s) in probs.items():
        for tau in grid["tau_ms"]:
            Ps = smooth(P, step_s, tau)
            for mn in grid["min_ms"]:
                for ref in grid["refractory_ms"]:
                    for th in grid["theta"]:
                        ev = detect(Ps, t, th, mn, ref, classes)
                        rows.append({**dict(zip(["model", "experiment", "participant", "session"], key)),
                                     "tau_ms": tau, "min_ms": mn, "refractory_ms": ref, "theta": th,
                                     **score_events(ev, truths[key[2:]], lead_s, end_s, tol_s)})
    return pd.DataFrame(rows)


SETTING = ["model", "experiment", "tau_ms", "min_ms", "refractory_ms", "theta"]
METRICS = ["correct", "confused", "missed", "extra_per_gesture", "fa_per_min", "latency_ms_median"]


def per_participant(scores: pd.DataFrame) -> pd.DataFrame:
    """Average over a participant's streams (sessions)."""
    return scores.groupby(SETTING + ["participant"], as_index=False)[METRICS].mean()


def operating_points(pp: pd.DataFrame, fa_target: float, extra_max: float | None = None) -> pd.DataFrame:
    """Per model x experiment: the setting with the highest mean share of correct events among those whose
    mean false-activation rate is at most `fa_target` per minute and, if given, whose mean number of extra commands
    per gesture is at most `extra_max` (means over participants). An extra command is an unwanted second press
    for the user, so without this limit the rule favours fast settings that chatter during a held gesture."""
    m = pp.groupby(SETTING, as_index=False)[METRICS].mean()
    ok = m[m.fa_per_min <= fa_target]
    if extra_max is not None:
        ok = ok[ok.extra_per_gesture <= extra_max]
    return ok.loc[ok.groupby(["model", "experiment"]).correct.idxmax()].reset_index(drop=True)


def plot_stream_example(stream: Stream, x_filt: np.ndarray, models: dict, t0: float, t1: float, tag: str,
                        tol_s: float = 0.5) -> str:
    """A piece of a stream: signal envelope with the true gestures, and for every model the smoothed probability
    of the best gesture vs the threshold, with the events it fired.

    Args:
        stream, x_filt: the stream and its causally filtered signal.
        models: {label: (t, smoothed P, classes, theta, events)}.
        t0, t1: time range, seconds.
    """
    import matplotlib.pyplot as plt
    import viz
    from data import GESTURE_SHORT

    viz.setup()
    fig, axes = plt.subplots(1 + len(models), 1, figsize=(12, 2.0 + 1.9 * len(models)), sharex=True)
    ev = stream.events[(stream.events.offset_s > t0) & (stream.events.onset_s < t1)]
    n = int(0.05 * FS)                                       # envelope: RMS over 50 ms, mean over channels
    k = np.arange(int(t0 * FS) // n, int(t1 * FS) // n)
    env = np.sqrt((x_filt[k[0] * n:(k[-1] + 1) * n].reshape(len(k), n, -1) ** 2).mean(axis=1)).mean(axis=1)
    axes[0].plot((k + 1) * n / FS, 1000 * env, color=viz.INK_2, lw=0.9)
    axes[0].set_ylabel("огибающая, мкВ")
    for ax in axes:
        for r in ev.itertuples():
            ax.axvspan(r.onset_s, r.offset_s, color=viz.SERIES[2], alpha=0.10, lw=0)
            ax.axvspan(r.offset_s, r.offset_s + tol_s, color=viz.SERIES[2], alpha=0.04, lw=0)
    for r in ev.itertuples():
        axes[0].text((r.onset_s + r.offset_s) / 2, axes[0].get_ylim()[1] * 0.92, GESTURE_SHORT[r.gesture - 1],
                     ha="center", fontsize=8, color=viz.INK)
    on, off = ev.onset_s.to_numpy(), ev.offset_s.to_numpy() + tol_s
    for ax, (label, (t, P, classes, theta, events)) in zip(axes[1:], models.items()):
        sel = (t >= t0) & (t <= t1)
        rest = classes == REST_GESTURE
        ax.plot(t[sel], P[sel][:, ~rest].max(axis=1), color=viz.SERIES[0], lw=1, label="лучший жест")
        ax.plot(t[sel], P[sel][:, rest].ravel(), color=viz.REFERENCE, lw=0.8, label="покой")
        ax.axhline(theta, color=viz.SERIES[1], lw=0.8, ls="--", label=f"порог {theta:g}")
        for te, ce in events:
            if not t0 <= te <= t1:
                continue
            k_ = np.flatnonzero((on <= te) & (te < off))
            ok = len(k_) and ev.gesture.to_numpy()[k_[0]] == ce
            ax.scatter([te], [1.04], marker="v", s=36, color=viz.SERIES[2] if ok else viz.SERIES[7], zorder=3, clip_on=False)
            ax.text(te, 1.10, GESTURE_SHORT[ce - 1], ha="center", fontsize=7,
                    color=viz.SERIES[2] if ok else viz.SERIES[7])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("вероятность")
        ax.set_title(label, fontsize=9, loc="left", pad=14)
    axes[1].legend(fontsize=7, loc="lower right", ncol=3)
    axes[-1].set_xlabel("время, с")
    fig.suptitle("Пример потока: зелёная полоса = жест (+0,5 с допуска), треугольник = команда (зелёный — верная)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"06_stream_example_{tag}")
    plt.close(fig)
    return str(path)


def plot_tradeoff(pp: pd.DataFrame, chosen: pd.DataFrame, tag: str, fa_target: float,
                  labels: dict[str, str] | None = None) -> str:
    """False activations per minute vs share of gestures without a correct command, along the threshold,
    for the chosen smoothing / duration / refractory setting of each model; median latency at a few points.
    Band = interquartile range over participants of the missed + confused share."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    labels = labels or {}
    exps = sorted(pp.experiment.unique())
    fig, axes = plt.subplots(1, len(exps), figsize=(6 * len(exps), 4.2), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], exps):
        for i, r in enumerate(chosen[chosen.experiment == e].itertuples()):
            g = pp[(pp.model == r.model) & (pp.experiment == e) & (pp.tau_ms == r.tau_ms) & (pp.min_ms == r.min_ms)
                   & (pp.refractory_ms == r.refractory_ms)].copy()
            g["fail"] = 100 * (1 - g.correct)
            c = g.groupby("theta").agg(fa=("fa_per_min", "mean"), fail=("fail", "mean"),
                                       q1=("fail", lambda v: v.quantile(0.25)), q3=("fail", lambda v: v.quantile(0.75)),
                                       lat=("latency_ms_median", "median")).reset_index()
            col = viz.SERIES[i]
            ax.plot(c.fa, c.fail, marker="o", ms=3, color=col, label=labels.get(r.model, r.model))
            ax.fill_between(c.fa, c.q1, c.q3, color=col, alpha=0.12, lw=0)
            for row in c.iloc[::3].itertuples():
                ax.annotate(f"{row.lat:.0f} мс", (row.fa, row.fail), xytext=(4, 3), textcoords="offset points",
                            fontsize=6.5, color=col)
            ax.scatter([r.fa_per_min], [100 * (1 - r.correct)], s=60, facecolors="none", edgecolors=col, lw=1.5, zorder=3)
        ax.axvline(fa_target, color=viz.REFERENCE, lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(left=0)
        ax.set_xlabel("ложные срабатывания в минуту покоя")
        ax.set_ylim(0, 100)
        ax.set_title({"E2": "E2: новый день", "E3": "E3: новый человек"}.get(e, e))
    axes[0][0].set_ylabel("жестов без верной команды, %")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Компромисс по порогу: точка = порог, кружок = рабочая точка, подпись = медианная задержка",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"06_tradeoff_{tag}")
    plt.close(fig)
    return str(path)
