"""Per-channel signal quality metrics for GRABMyo wrist channels.

All metrics are computed per (participant, session, channel) from Welch PSDs of individual trials.
Band-limited quantities are obtained by integrating the PSD over a band; this is an offline analysis,
not a preprocessing step, so it does not violate the causal-filtering rule.

Notes from stage 1 and the ADC check (see reports/notes.md):
- 60 Hz is already notched in the published data, so line interference is measured on harmonics 120..420 Hz.
- Original 12-bit ADC codes are not recoverable (each file is rescaled to the full 16-bit range and the
  data were post-processed), so saturation cannot be measured as "samples at full scale".
  It is approximated by amplitude outliers: fraction of samples with |x| above a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from data import FS, GESTURE_SHORT, REST_GESTURE, load_trial

LINE_HZ = 60.0
SIGNAL_BAND = (20.0, 450.0)
LOW_BAND = (0.0, 20.0)
HARMONICS = range(2, 8)       # 120..420 Hz, inside the signal band
HARMONIC_HALF_WIDTH = 1.0     # Hz


def trial_spectral_stats(x: np.ndarray, fs: int = FS, nperseg: int = 2048,
                         amp_threshold_mv: float = 5.0) -> dict[str, np.ndarray]:
    """Per-channel spectral and amplitude statistics of one trial.

    Args:
        x: trial signal [T, C] in mV.
        fs: sampling rate, Hz.
        nperseg: Welch segment length (2048 at 2048 Hz -> 1 Hz resolution).
        amp_threshold_mv: |x| above this counts as a possible saturation / artifact sample.

    Returns:
        Dict of arrays of shape [C]: powers in mV^2 (band, low, total, harmonics, 60 Hz residual),
        sample variance, max |x| and the fraction of samples above the amplitude threshold.
    """
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=0)
    df = f[1] - f[0]

    def band_power(lo: float, hi: float) -> np.ndarray:
        m = (f >= lo) & (f < hi)
        return p[m].sum(axis=0) * df

    harm = sum(band_power(k * LINE_HZ - HARMONIC_HALF_WIDTH, k * LINE_HZ + HARMONIC_HALF_WIDTH + 1e-9)
               for k in HARMONICS)
    xc = x - x.mean(axis=0)
    return {
        "p_band": band_power(*SIGNAL_BAND),
        "p_low": band_power(*LOW_BAND),
        "p_total": p.sum(axis=0) * df,
        "p_harm": harm,
        "p_60": band_power(LINE_HZ - HARMONIC_HALF_WIDTH, LINE_HZ + HARMONIC_HALF_WIDTH + 1e-9),
        "var": xc.var(axis=0),
        "max_abs": np.abs(xc).max(axis=0),
        "frac_over": (np.abs(xc) > amp_threshold_mv).mean(axis=0),
    }


def channel_quality(index: pd.DataFrame, channel_names: list[str],
                    amp_threshold_mv: float = 5.0, dead_std_uv: float = 0.5) -> pd.DataFrame:
    """Quality table: one row per (participant, session, channel).

    Columns:
        noise_rms_uv: RMS in 20-450 Hz during rest (noise floor).
        signal_rms_uv: RMS in 20-450 Hz during gestures 1-16.
        snr_db: 20*log10(signal_rms / noise_rms).
        harm_frac_rest_pct: share of rest power in 60 Hz harmonics (120..420 Hz, +-1 Hz). Relative: a clean,
            low-noise channel shows a high share even for ordinary interference.
        harm_rms_rest_uv: absolute RMS of those harmonics during rest.
        res60_frac_rest_pct: share of rest power left at 60 +-1 Hz (expected ~0: data are notched).
        low_frac_rest_pct: share of rest power below 20 Hz (motion / contact drift indicator).
        max_abs_mv, over_thr_pct: largest |x| and share of samples with |x| > threshold, all trials.
        n_rejected_trials: trials of this participant-session excluded from the power averages because
            some channel exceeded the amplitude threshold (artifact / possible saturation).
        dead: rest standard deviation below dead_std_uv.
    """
    rows = []
    for (p, s), sub in index.groupby(["participant", "session"]):
        acc: dict[bool, list[dict]] = {True: [], False: []}
        for r in sub.itertuples():
            acc[r.gesture == REST_GESTURE].append(
                trial_spectral_stats(load_trial(r.session, r.participant, r.gesture, r.trial),
                                     amp_threshold_mv=amp_threshold_mv))
        all_stats = acc[True] + acc[False]
        # Artifact rejection: a trial with |x| above the threshold on any channel is excluded from the
        # power averages (one such trial can shift a session's SNR by ~2 dB), but still counted below.
        clean = {is_rest: [d for d in lst if d["max_abs"].max() <= amp_threshold_mv] for is_rest, lst in acc.items()}
        n_rejected = sum(len(acc[k]) - len(clean[k]) for k in acc)
        mean = {is_rest: {k: np.mean([d[k] for d in lst], axis=0) for k in lst[0]}
                for is_rest, lst in clean.items()}
        max_abs = np.max([d["max_abs"] for d in all_stats], axis=0)
        over = np.mean([d["frac_over"] for d in all_stats], axis=0)
        rest, gest = mean[True], mean[False]
        for c, name in enumerate(channel_names):
            noise = np.sqrt(rest["p_band"][c])
            sig = np.sqrt(gest["p_band"][c])
            rows.append({
                "participant": p, "session": s, "channel": name,
                "noise_rms_uv": 1e3 * noise,
                "signal_rms_uv": 1e3 * sig,
                "snr_db": 20 * np.log10(sig / noise),
                "harm_frac_rest_pct": 100 * rest["p_harm"][c] / rest["p_total"][c],
                "harm_rms_rest_uv": 1e3 * np.sqrt(rest["p_harm"][c]),
                "res60_frac_rest_pct": 100 * rest["p_60"][c] / rest["p_total"][c],
                "low_frac_rest_pct": 100 * rest["p_low"][c] / rest["p_total"][c],
                "max_abs_mv": max_abs[c],
                "over_thr_pct": 100 * over[c],
                "n_rejected_trials": n_rejected,
                "dead": bool(np.sqrt(rest["var"][c]) * 1e3 < dead_std_uv),
            })
    return pd.DataFrame(rows)


def flag_outliers(q: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Flag channel-sessions whose metric deviates from the dataset median by more than k robust SDs.

    Robust SD = 1.4826 * MAD, computed on log scale for positive heavy-tailed metrics.
    Low SNR, high noise, high harmonic share and high low-frequency share are flagged.
    """
    out = q.copy()
    checks = {"snr_db": ("low", False), "noise_rms_uv": ("high", True),
              "harm_frac_rest_pct": ("high", True), "low_frac_rest_pct": ("high", True)}
    reasons = [[] for _ in range(len(out))]
    for col, (side, use_log) in checks.items():
        v = np.log10(out[col].clip(lower=1e-6)) if use_log else out[col]
        med = v.median()
        sd = 1.4826 * (v - med).abs().median()
        z = (v - med) / sd
        bad = z < -k if side == "low" else z > k
        for i in np.flatnonzero(bad.to_numpy()):
            reasons[i].append(f"{col} {side} (z={z.iloc[i]:+.1f})")
    out["flag_reasons"] = ["; ".join(r) for r in reasons]
    out.loc[out.dead, "flag_reasons"] = out.loc[out.dead, "flag_reasons"] + "; dead"
    return out


def session_drift(q: pd.DataFrame, value: str = "snr_db") -> pd.DataFrame:
    """How much a per-channel dB quantity changes between sessions of the same participant.

    For each participant: the split of the variation of `value` into a common part (all channels of a session
    shift together) and a channel-specific part (individual electrode placement / contact), via
    value[s, c] = mu[s] + alpha[c] + eps[s, c]. Computed within each participant only: session numbers of
    different people are unrelated, so pooling participants before the split would mix their drifts.

    Note that SNR is invariant to a common gain; a pure amplitude-scale drift shows up only in a level
    quantity such as 20*log10(signal_rms).
    """
    rows = []
    for p, sub in q.groupby("participant"):
        m = sub.pivot(index="session", columns="channel", values=value)
        common = m.mean(axis=1)                      # per-session mean over channels
        resid = m.sub(common, axis=0)                # channel-specific deviations
        rows.append({
            "participant": p,
            "session_means_db": [round(v, 1) for v in common.tolist()],
            "common_shift_range_db": common.max() - common.min(),
            "median_channel_range_db": (m.max() - m.min()).median(),
            "max_channel_range_db": (m.max() - m.min()).max(),
            "channel_specific_sd_db": resid.sub(resid.mean(), axis=1).stack().std(),
        })
    return pd.DataFrame(rows)


def channel_profiles(q: pd.DataFrame, value: str = "snr_db") -> pd.DataFrame:
    """Per-session channel profiles: value[s, c] - mean over channels of session s.

    Returns a table with index (participant, session) and one column per channel. Averaging the rows of one
    participant over sessions gives alpha_c from `session_drift`.
    """
    chans = list(dict.fromkeys(q.channel))
    m = q.pivot_table(index=["participant", "session"], columns="channel", values=value)[chans]
    return m.sub(m.mean(axis=1), axis=0)


def profile_correlations(prof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlations of channel profiles; observations are the channels.

    Returns:
        within: one row per (participant, session pair) — profile of session a vs profile of session b.
        between: one row per participant pair — session-averaged profile of A vs that of B.
    """
    from itertools import combinations

    within = []
    for p, sub in prof.groupby(level="participant"):
        sub = sub.droplevel("participant")
        for a, b in combinations(sub.index, 2):
            within.append({"participant": p, "sessions": f"{a}-{b}",
                           "r": np.corrcoef(sub.loc[a], sub.loc[b])[0, 1]})
    alpha = prof.groupby(level="participant").mean()
    between = [{"pair": f"{a}-{b}", "r": np.corrcoef(alpha.loc[a], alpha.loc[b])[0, 1]}
               for a, b in combinations(alpha.index, 2)]
    return pd.DataFrame(within), pd.DataFrame(between)


def plot_quality(q: pd.DataFrame, tag: str) -> list[str]:
    """Save the stage-2 figures and return their paths."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    paths = []
    chans = list(dict.fromkeys(q.channel))
    parts = sorted(q.participant.unique())
    sessions = sorted(q.session.unique())

    # 1) SNR heatmap: channels x participants, one panel per session, one shared colour scale
    vmin, vmax = np.floor(q.snr_db.min()), np.ceil(q.snr_db.max())
    fig, axes = plt.subplots(1, len(sessions), figsize=(4.2 * len(sessions), 4.6), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), sessions):
        m = q[q.session == s].pivot(index="channel", columns="participant", values="snr_db").loc[chans, parts]
        im = ax.imshow(m.values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > (vmin + vmax) / 2 else viz.INK)
        ax.set_xticks(range(len(parts)), parts)
        ax.set_yticks(range(len(chans)), chans)
        ax.set_xlabel("участник")
        ax.set_title(f"сессия {s}")
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="SNR, дБ", shrink=0.8)
    fig.suptitle("SNR каналов запястья: RMS жестов / RMS покоя в полосе 20–450 Гц",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    paths.append(str(viz.save(fig, f"02_snr_heatmap_{tag}")))
    plt.close(fig)

    # 2) SNR per channel across sessions, one panel per participant
    fig, axes = plt.subplots(1, len(parts), figsize=(3.2 * len(parts), 3.4), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), parts):
        for i, s in enumerate(sessions):
            v = q[(q.participant == p) & (q.session == s)].set_index("channel").loc[chans, "snr_db"]
            ax.plot(range(len(chans)), v.values, marker="o", ms=3, color=viz.SERIES[i], label=f"сессия {s}")
        ax.axvline(5.5, color=viz.AXIS, lw=0.8)
        ax.set_xticks(range(len(chans)), [c[1:] for c in chans], fontsize=7)
        ax.set_xlabel("канал W (1–6 кольцо 1, 7–12 кольцо 2)")
        ax.set_title(f"участник {p}")
    np.atleast_1d(axes)[0].set_ylabel("SNR, дБ")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Как SNR каждого канала меняется от сессии к сессии",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    paths.append(str(viz.save(fig, f"02_snr_sessions_{tag}")))
    plt.close(fig)

    # 3) Outlier map: noise floor vs harmonic share (relative), harmonic RMS (absolute), low-frequency share
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    flagged = q["flag_reasons"] != ""
    for ax, col, label in [(axes[0], "harm_frac_rest_pct", "доля гармоник сети в покое, %"),
                           (axes[1], "harm_rms_rest_uv", "RMS гармоник сети в покое, мкВ"),
                           (axes[2], "low_frac_rest_pct", "доля мощности < 20 Гц в покое, %")]:
        ax.scatter(q.loc[~flagged, "noise_rms_uv"], q.loc[~flagged, col], s=12, color=viz.SERIES[0],
                   alpha=0.6, lw=0, label="в норме")
        ax.scatter(q.loc[flagged, "noise_rms_uv"], q.loc[flagged, col], s=26, color=viz.SERIES[1],
                   edgecolor=viz.SURFACE, lw=0.8, label="выброс (3 робастных SD)")
        # one label per flagged participant-session, placed at its most extreme point
        for (p, s), g in q[flagged].groupby(["participant", "session"]):
            r = g.loc[g[col].idxmax()]
            ax.annotate(f"{p}/s{s} ({len(g)} кан.)", (r.noise_rms_uv, r[col]), fontsize=7,
                        color=viz.INK_2, xytext=(4, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("шум покоя (RMS 20–450 Гц), мкВ")
        ax.set_ylabel(label)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Канал × сессия: шумовой уровень против помех и низкочастотных артефактов",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    paths.append(str(viz.save(fig, f"02_outliers_{tag}")))
    plt.close(fig)
    return paths
