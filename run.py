"""Experiment entry point: python run.py --config configs/<exp>.yaml

The config's `experiment` key selects the runner; results go to results/<name>.json, figures to reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import build_index  # noqa: E402


def resolve_participants(spec: str | list[int]) -> list[int]:
    """Turn a group name from splits.yaml ('dev', 'test', ...), 'all', a sum like 'dev+test' or an explicit list
    into participant ids."""
    if isinstance(spec, list):
        return spec
    if spec == "all":
        return list(range(1, 44))
    if "+" in spec:
        return [p for part in spec.split("+") for p in resolve_participants(part)]
    splits = yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())
    return splits[spec]


def run_quality(cfg: dict) -> dict:
    """Stage 2: per-channel signal quality table, outliers, session drift, figures."""
    import quality
    from data import WRIST_IDX

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    names = [f"W{i + 1}" for i in range(len(WRIST_IDX))]
    q = quality.channel_quality(idx, names, amp_threshold_mv=cfg["amp_threshold_mv"],
                                dead_std_uv=cfg["dead_std_uv"])
    q = quality.flag_outliers(q, k=cfg["outlier_k"])
    drift = quality.session_drift(q)
    figures = quality.plot_quality(q, cfg["name"])

    csv_path = ROOT / "results" / f"{cfg['name']}_channels.csv"
    q.to_csv(csv_path, index=False)

    def summary(col: str) -> dict:
        v = q[col]
        return {"median": v.median(), "p5": v.quantile(0.05), "p95": v.quantile(0.95),
                "min": v.min(), "max": v.max()}

    return {
        "participants": participants,
        "n_channel_sessions": len(q),
        "metrics": {c: summary(c) for c in ["noise_rms_uv", "signal_rms_uv", "snr_db", "harm_frac_rest_pct",
                                            "res60_frac_rest_pct", "low_frac_rest_pct", "max_abs_mv",
                                            "over_thr_pct"]},
        "per_participant_session_snr_median_db":
            {f"{p}/s{s}": round(v, 2) for (p, s), v in q.groupby(["participant", "session"]).snr_db.median().items()},
        "n_dead": int(q.dead.sum()),
        "rejected_trials": {f"{p}/s{s}": int(n) for (p, s), n in
                            q.groupby(["participant", "session"]).n_rejected_trials.first().items() if n},
        "outliers": q[q["flag_reasons"] != ""][["participant", "session", "channel", "flag_reasons"]].to_dict("records"),
        "session_drift": drift.to_dict("records"),
        "channel_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures],
    }


def run_baseline(cfg: dict) -> dict:
    """Stage 3: causal preprocessing, Hudgins features, LDA / LightGBM on splits E1-E4."""
    import evaluate
    from data import WRIST_IDX
    from features import feature_names, feature_table

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    channels = [f"W{i + 1}" for i in range(len(WRIST_IDX))]
    table = feature_table(idx, {**cfg["preprocess"], **cfg["features"]}, channels)
    feat_cols = feature_names(channels)
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None
    scores, conf = evaluate.evaluate(table, feat_cols, cfg["experiments"], cfg["models"],
                                     cfg["cross_subject"], test_ps, cfg.get("seed", 42), cfg.get("n_folds", 5))
    pp = evaluate.per_participant(scores)

    csv_path = ROOT / "results" / f"{cfg['name']}_scores.csv"
    scores.to_csv(csv_path, index=False)
    figures = [evaluate.plot_e1_e4(pp, cfg["name"])]
    figures += [evaluate.plot_confusion(conf, m, cfg["name"]) for m in cfg["models"]]

    summary = (pp.groupby(["experiment", "model"]).bal_acc
               .agg(["median", "min", "max"]).round(4).reset_index().to_dict("records"))
    return {
        "participants": participants,
        "n_windows": len(table),
        "n_features": len(feat_cols),
        "bal_acc_summary": summary,
        "per_participant": pp.round(4).to_dict("records"),
        "confusion": {k: v.tolist() for k, v in conf.items()},
        "scores_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures],
    }


def run_tcn(cfg: dict) -> dict:
    """Stage 4: causal TCN on the filtered signal, splits E1-E4, decisions at the same moments as the features.

    With `select_epochs: true` (validation participants) every split is trained for `max_epochs` and its test part
    is scored every `eval_every` epochs; the epoch with the best median balanced accuracy is reported per experiment.
    Otherwise each experiment trains for `epochs[experiment]`, fixed beforehand on the validation run.
    `only_split` (a split name, e.g. "p22/s1" in E1, "p22" in E2/E3) trains that one split only, for quick trials.
    """
    import time

    import pandas as pd
    import torch
    from sklearn.metrics import confusion_matrix

    import evaluate
    import signals
    import tcn
    from preprocess import ms_to_samples
    from splits import make_splits

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.get("seed", 42)

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    meta, X = signals.signal_table(idx, cfg["preprocess"])
    y = meta.gesture.to_numpy() - 1
    fs = signals.effective_fs(cfg["preprocess"]["decimate"])
    win = ms_to_samples(cfg["windows"]["window_ms"], fs)
    step = ms_to_samples(cfg["windows"]["step_ms"], fs)
    n_win = (X.shape[2] - win) // step + 1
    ends = tcn.window_ends(win, step, n_win)
    train_cfg = {**cfg["train"], "crop": ms_to_samples(cfg["train"]["crop_ms"], fs)}
    select = cfg.get("select_epochs", False)
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None
    name = cfg.get("model_name", "tcn")

    rows, conf, curves, rf, size = [], {}, [], None, None
    for exp in cfg["experiments"]:
        epochs = cfg["train"]["max_epochs"] if select else cfg["epochs"][exp]
        exp_splits = make_splits(meta, exp, cfg["cross_subject"], test_ps, cfg.get("n_folds", 5), seed)
        if cfg.get("only_split") is not None:
            names = [s.name for s in exp_splits]
            exp_splits = [s for s in exp_splits if s.name == cfg["only_split"]]
            if not exp_splits:
                raise ValueError(f"no split {cfg['only_split']!r} in {exp}; available: {names}")
        for split in exp_splits:
            te = np.flatnonzero(split.test)
            win_meta = meta.iloc[np.repeat(te, n_win)].reset_index(drop=True)
            y_true = np.repeat(y[te], n_win)

            def monitor(model):
                pred = tcn.predict_windows(model, X[te], ends, device).ravel()
                return {"scores": evaluate.score_rows(win_meta, y_true, pred, {})}

            t0 = time.time()
            model, hist = tcn.fit(X[split.train], y[split.train], cfg["model"], train_cfg, epochs, seed, device,
                                  monitor if select else None, cfg["train"].get("eval_every", 0))
            rf, size = model.receptive_field, tcn.n_params(model)
            for h in hist:
                curves += [{"experiment": exp, "epoch": h["epoch"], **r} for r in h["scores"]]
            pred = tcn.predict_windows(model, X[te], ends, device).ravel()
            key = f"{exp}/{name}"
            conf[key] = conf.get(key, 0) + confusion_matrix(y_true, pred, labels=np.arange(len(evaluate.LABELS)))
            rows += evaluate.score_rows(win_meta, y_true, pred, {"experiment": exp, "model": name, "split": split.name})
            print(f"{exp} {split.name} {name}: {epochs} epochs, {time.time() - t0:.0f} s", flush=True)

    scores = pd.DataFrame(rows)
    pp = evaluate.per_participant(scores)
    csv_path = ROOT / "results" / f"{cfg['name']}_scores.csv"
    scores.to_csv(csv_path, index=False)

    plot_pp = pp
    for ref, spec in (cfg.get("compare_with") or {}).items():
        other = json.loads((ROOT / spec["result"]).read_text(encoding="utf-8"))
        ref_pp = pd.DataFrame(other["per_participant"])
        plot_pp = pd.concat([plot_pp, ref_pp[ref_pp.model == spec["model"]].assign(model=ref)], ignore_index=True)
    figures = [evaluate.plot_e1_e4(plot_pp, cfg["name"], stage="04"),
               evaluate.plot_confusion(conf, name, cfg["name"], stage="04")]

    out = {}
    if select:
        c = pd.DataFrame(curves)
        med = (c.groupby(["experiment", "epoch", "participant"]).bal_acc.mean()
               .groupby(["experiment", "epoch"]).median().reset_index())
        best = med.loc[med.groupby("experiment").bal_acc.idxmax()]
        out["best_epochs"] = {r.experiment: int(r.epoch) for r in best.itertuples()}
        out["best_epoch_bal_acc"] = {r.experiment: round(r.bal_acc, 4) for r in best.itertuples()}
        out["curve_median"] = {e: g[["epoch", "bal_acc"]].round(4).values.tolist() for e, g in med.groupby("experiment")}
        figures.append(evaluate.plot_curves(med, out["best_epochs"], cfg["name"]))

    summary = (pp.groupby(["experiment", "model"]).bal_acc
               .agg(["median", "min", "max"]).round(4).reset_index().to_dict("records"))
    return {
        "participants": participants,
        "n_trials": len(meta),
        "receptive_field_samples": rf,
        "receptive_field_ms": round(1000 * rf / fs, 1),
        "n_params": size,
        "decisions_per_trial": n_win,
        **out,
        "bal_acc_summary": summary,
        "per_participant": pp.round(4).to_dict("records"),
        "confusion": {k: v.tolist() for k, v in conf.items()},
        "scores_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures],
    }


def _adapt_outputs(rows: list[dict], cfg: dict) -> dict:
    """Shared tail of the stage-5 runners: score table, summary, main figure."""
    import pandas as pd

    import adapt

    scores = pd.DataFrame(rows)
    csv_path = ROOT / "results" / f"{cfg['name']}_scores.csv"
    scores.to_csv(csv_path, index=False)
    pp = adapt.per_participant(scores)
    plot = adapt.plot_passes if cfg.get("figure") == "passes" else adapt.plot_calibration
    figures = [plot(pp, cfg["name"], cfg.get("labels"))]
    return {
        "summary": adapt.summary(pp).round(4).to_dict("records"),
        "per_participant": pp.round(4).to_dict("records"),
        "scores_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures],
    }


def _target_splits(meta, exp: str, cfg: dict, test_ps: list[int] | None) -> list:
    """Splits of one experiment for stage 5. With `holdout` both groups are loaded (dev trains the E3 model),
    but only the test participants are adapted to: within-person splits (E1/E2) of dev people are dropped."""
    from splits import make_splits

    splits = make_splits(meta, exp, cfg["cross_subject"], test_ps, cfg.get("n_folds", 5), cfg.get("seed", 42))
    if test_ps is None:
        return splits
    return [s for s in splits if set(meta.participant[s.test].unique()) <= set(test_ps)]


def run_adapt_lda(cfg: dict) -> dict:
    """Stage 5, LDA: unlabeled normalisation and labelled calibration on new sessions (E2) and new people (E3)."""
    import adapt
    from data import WRIST_IDX
    from features import feature_names, feature_table

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    channels = [f"W{i + 1}" for i in range(len(WRIST_IDX))]
    table = feature_table(idx, {**cfg["preprocess"], **cfg["features"]}, channels)
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None
    rows = []
    for exp in cfg["experiments"]:
        rows += adapt.evaluate_lda(table, feature_names(channels), _target_splits(table, exp, cfg, test_ps), cfg)
    return {"participants": participants, **_adapt_outputs(rows, cfg)}


def run_adapt_tcn(cfg: dict) -> dict:
    """Stage 5, TCN: input rescaling, AdaBN and fine-tuning on new sessions (E2) and new people (E3).

    Base models use the stage-4 recipe and are cached in data/cache/models_*, so a second run with other
    adaptation settings does not retrain them.
    """
    import torch

    import adapt
    import signals
    import tcn
    from preprocess import ms_to_samples

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    meta, X = signals.signal_table(idx, cfg["preprocess"])
    y = meta.gesture.to_numpy() - 1
    fs = signals.effective_fs(cfg["preprocess"]["decimate"])
    win = ms_to_samples(cfg["windows"]["window_ms"], fs)
    step = ms_to_samples(cfg["windows"]["step_ms"], fs)
    ends = tcn.window_ends(win, step, (X.shape[2] - win) // step + 1)
    cfg = {**cfg, "crop": ms_to_samples(cfg["train"]["crop_ms"], fs)}
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None

    rows, scratch_cache = [], {}
    for exp in cfg["experiments"]:
        for split in _target_splits(meta, exp, cfg, test_ps):
            model = adapt.base_tcn(X, y, cfg, participants, split, cfg["epochs"][exp], device)
            rows += adapt.evaluate_tcn(model, X, meta, split, cfg, ends, fs, device, scratch_cache)
    return {"participants": participants, **_adapt_outputs(rows, cfg)}


def stream_cache_path(cfg: dict) -> Path:
    """Cache file of the stage-6 probabilities: keyed by everything that defines the streams and the models
    (not by the post-processing grid, the operating-point rule or the frozen settings)."""
    import hashlib

    keep = ["participants", "seed", "cross_subject", "n_folds", "experiments", "epochs", "preprocess", "windows",
            "features", "model", "train", "stream", "calibration", "lda", "tcn_adapt"]
    digest = hashlib.sha1(json.dumps({k: cfg.get(k) for k in keep}, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return ROOT / "data" / "cache" / f"stream_probs_{digest}.pkl"


def _stream_probs(cfg: dict) -> tuple[dict, dict, dict]:
    """Stage 6, expensive part: build the streams, calibrate both models on every target session (budget from
    the calibration trials) and compute their probabilities on the streams. Cached (`stream_cache_path`) by
    everything that defines streams and models, so the post-processing sweep can be re-run cheaply.

    Returns probs {(model, experiment, participant, session): (t, P, classes, lead_s, end_s)},
    truths {(participant, session): events table} and stream info {(participant, session): unique rest seconds}.
    """
    import copy
    import pickle

    import torch

    import adapt
    import signals
    import streaming
    import tcn
    from data import FS, WRIST_IDX
    from features import feature_names, feature_table
    from preprocess import design_filter, filter_causal, ms_to_samples

    cache = stream_cache_path(cfg)
    if cache.exists():
        print(f"probabilities from {cache.relative_to(ROOT)}", flush=True)
        return pickle.loads(cache.read_bytes())

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    participants = resolve_participants(cfg["participants"])
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]

    pre, win_cfg, sc = cfg["preprocess"], cfg["windows"], cfg["stream"]
    feat_cfg = {**{k: pre[k] for k in ["band_hz", "order", "notch_hz", "warmup_ms"]}, **win_cfg, **cfg["features"]}
    channels = [f"W{i + 1}" for i in range(len(WRIST_IDX))]
    feat_cols = feature_names(channels)
    table = feature_table(idx, feat_cfg, channels)
    meta, X = signals.signal_table(idx, pre)
    y = meta.gesture.to_numpy() - 1
    fs_dec = signals.effective_fs(pre["decimate"])
    win, step = ms_to_samples(win_cfg["window_ms"]), ms_to_samples(win_cfg["step_ms"])
    sos = design_filter(band_hz=tuple(pre["band_hz"]), order=pre["order"], notch_hz=tuple(pre["notch_hz"]))
    budget = adapt.parse_budget(cfg["calibration"]["budget"])
    tcfg = {**cfg, "crop": ms_to_samples(cfg["train"]["crop_ms"], fs_dec)}

    streams, probs = {}, {}

    def stream_of(p: int, s: int) -> tuple:
        if (p, s) not in streams:
            st = streaming.build_stream(p, s, sc["trials"], cfg.get("seed", 42), tuple(sc["gesture_s"]),
                                        tuple(sc["rest_s"]), sc["lead_s"], sc["rise_ms"], sc["fall_ms"], sc["join_ms"])
            x_filt, _ = filter_causal(st.x, sos)
            streams[(p, s)] = (st, x_filt)
        return streams[(p, s)]

    lda_cache = {}
    for exp in cfg["experiments"]:
        for split in _target_splits(meta, exp, cfg, test_ps):
            base = adapt.base_tcn(X, y, tcfg, participants, split, cfg["epochs"][exp], device)
            for p, s in meta[split.test].groupby(["participant", "session"]).indices:
                st, x_filt = stream_of(p, s)
                end_s = len(st.x) / FS
                if (p, s) not in lda_cache:      # personal LDA: the same for E2 and E3
                    rows = ((table.participant == p) & (table.session == s)).to_numpy() & adapt.window_mask(table, budget)
                    lda = make_lda(table, rows, feat_cols, cfg["lda"])
                    lda_cache[(p, s)] = streaming.lda_probs(lda, x_filt, win, step, cfg["features"], feat_cols)
                probs[("lda", exp, p, s)] = (*lda_cache[(p, s)], st.lead_s, end_s)

                sess = split.test & (meta.participant.to_numpy() == p) & (meta.session.to_numpy() == s)
                X_cal, y_cal = adapt.calib_signals(X, meta, sess, budget, pre["warmup_ms"] / 1000, fs_dec)
                m = copy.deepcopy(base)
                adapt.set_input_scale(m, X_cal)
                adapt.adabn(m, X_cal, device)
                adapt.finetune(m, X_cal, y_cal, cfg["tcn_adapt"], tcfg["crop"], cfg.get("seed", 42), device)
                probs[("tcn", exp, p, s)] = (*streaming.tcn_probs(m, x_filt, win, step, pre["decimate"], device),
                                             st.lead_s, end_s)
                print(f"{exp} {split.name} p{p}/s{s}: stream {end_s:.0f} s, {len(st.events)} gestures", flush=True)

    truths = {k: v[0].events for k, v in streams.items()}
    info = {k: v[0].unique_rest_s for k, v in streams.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((probs, truths, info)))
    return probs, truths, info


def make_lda(table, rows: np.ndarray, feat_cols: list[str], params: dict):
    """LDA (log of amplitude features + scaler + LDA) fitted on the given rows of a feature table."""
    from models import log_amplitude, make_model

    F = log_amplitude(table.loc[rows, feat_cols].to_numpy(), feat_cols)
    return make_model("lda", params).fit(F, table.gesture.to_numpy()[rows])


def run_stream(cfg: dict) -> dict:
    """Stage 6: streaming recognition with event metrics. Post-processing is swept over `grid`; on dev the
    operating point is chosen per model x experiment (most correct events at <= fa_target false activations per
    minute and <= extra_max extra commands per gesture); on test the dev choice is given in `frozen`
    ({model: {experiment: setting}}) and only reported (the threshold sweep is kept for the trade-off curve)."""
    import pandas as pd

    import streaming

    probs, truths, info = _stream_probs(cfg)
    step_s = cfg["windows"]["step_ms"] / 1000
    frozen = cfg.get("frozen")
    scores = []
    for model, exp in sorted({k[:2] for k in probs}):
        sub = {k: v for k, v in probs.items() if k[:2] == (model, exp)}
        grid = dict(cfg["grid"])
        if frozen:                               # keep the frozen smoothing / duration / refractory, sweep theta
            f = frozen[model][exp]
            grid.update(tau_ms=[f["tau_ms"]], min_ms=[f["min_ms"]], refractory_ms=[f["refractory_ms"]])
            grid["theta"] = sorted(set(grid["theta"]) | {f["theta"]})
        scores.append(streaming.sweep(sub, truths, grid, step_s, cfg["stream"]["tol_ms"] / 1000))
    scores = pd.concat(scores, ignore_index=True)
    csv_path = ROOT / "results" / f"{cfg['name']}_scores.csv"
    scores.to_csv(csv_path, index=False)
    pp = streaming.per_participant(scores)

    if frozen:
        sel = [pp[(pp.model == m) & (pp.experiment == e) & (pp.tau_ms == f["tau_ms"]) & (pp.min_ms == f["min_ms"])
                  & (pp.refractory_ms == f["refractory_ms"]) & np.isclose(pp.theta, f["theta"])]
               for m, by_exp in frozen.items() for e, f in by_exp.items()]
        chosen = (pd.concat(sel).groupby(streaming.SETTING, as_index=False)[streaming.METRICS].mean())
    else:
        chosen = streaming.operating_points(pp, cfg["fa_target_per_min"], cfg.get("extra_max_per_gesture"))
    at_point = pp.merge(chosen[streaming.SETTING], on=streaming.SETTING)
    figures = [streaming.plot_tradeoff(pp, chosen, cfg["name"], cfg["fa_target_per_min"], cfg.get("labels"))]
    curves = pp.groupby(streaming.SETTING, as_index=False)[streaming.METRICS].mean()
    return {
        "participants": sorted({int(k[2]) for k in probs}),
        "n_streams": len(truths),
        "stream_minutes_mean": round(float(np.mean([v[4] for v in probs.values()])) / 60, 2),
        "unique_rest_s_per_session": round(float(np.mean(list(info.values()))), 1),
        "operating_points": chosen.round(4).to_dict("records"),
        "at_operating_point": at_point.round(4).to_dict("records"),
        "curves_mean": curves.round(4).to_dict("records"),
        "scores_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures],
    }


def _robust_outputs(rows: list[dict], name: str, greedy_order: list[str] | None = None) -> dict:
    """Shared tail of the stage-7 runners: score table, per-participant tables, summaries, figures."""
    import pandas as pd

    import robustness

    scores = pd.DataFrame(rows)
    csv_path = ROOT / "results" / f"{name}_scores.csv"
    scores.to_csv(csv_path, index=False)
    pp = robustness.per_participant(scores)
    figures = [robustness.plot_sensor(pp, name, greedy_order)]
    if "rotation" in pp:
        figures.append(robustness.plot_rotation(pp["rotation"], name))
    out = {}
    if "failure" in pp:
        figures.append(robustness.plot_failure(pp["failure"], name))
        out["failure_by_channel"] = robustness.failure_by_channel(scores).round(4).to_dict("records")
    return {
        "summary": {k: robustness.summary(v).round(4).to_dict("records") for k, v in pp.items()},
        **out,
        "per_participant": {k: v.round(4).to_dict("records") for k, v in pp.items()},
        "scores_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures if f],
    }


def run_robust_lda(cfg: dict) -> dict:
    """Stage 7, LDA: channel sets, greedy channel removal and sampling rate (E1), bracelet rotation (E2),
    channel failure (E1). Each part runs if its section is in the config; see src/robustness.py."""
    import robustness
    from data import FOREARM_IDX, FS, WRIST_IDX, channel_names
    from features import feature_names, feature_table, rate_band
    from models import log_amplitude

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    feat_cfg = {**cfg["preprocess"], **cfg["features"]}
    params, seed = cfg["lda"], cfg.get("seed", 42)
    meta_cols = ["participant", "session", "trial"]

    def load(fcfg: dict, record: list[int]) -> tuple:
        """Feature table -> (meta, gestures, log features, feature column names)."""
        names = feature_names(channel_names(record))
        t = feature_table(idx, fcfg, channel_names(record))
        return t[meta_cols], t.gesture.to_numpy(), log_amplitude(t[names].to_numpy(), names), names

    meta, y, F, cols = load(feat_cfg, WRIST_IDX)    # the wrist table of stages 3-6 (same cache)
    wrist = channel_names(WRIST_IDX)
    rows, out = [], {}

    if "channel_sets" in cfg:
        rec = FOREARM_IDX + WRIST_IDX
        m28, y28, F28, c28 = load({**feat_cfg, "record_channels": rec}, rec)
        for name in cfg["channel_sets"]:
            chs = robustness.CHANNEL_SETS[name]
            rows += robustness.lda_e1(F28, m28, y28, robustness.cols_of(c28, chs), params,
                                      {"part": "channels", "model": "lda", "experiment": "E1", "setting": name,
                                       "n_channels": len(chs)})
            print(f"channels: {name} done", flush=True)
        del m28, y28, F28

    if "greedy" in cfg:
        order = cfg["greedy"].get("order") or robustness.greedy_order(F, meta, y, cols, wrist, params)
        out.update(greedy_order=order, greedy_frozen=bool(cfg["greedy"].get("order")))
        for n in range(len(order), 0, -1):
            rows += robustness.lda_e1(F, meta, y, robustness.cols_of(cols, order[len(order) - n:]), params,
                                      {"part": "greedy", "model": "lda", "experiment": "E1", "n_channels": n})

    if "rates" in cfg:
        rc = cfg["rates"]
        for k in rc["downsample"]:
            fcfg = feat_cfg if k == 1 else {**feat_cfg, "downsample": k, "antialias_ratio": rc["antialias_ratio"],
                                            "antialias_order": rc["antialias_order"]}
            mk, yk, Fk, ck = load(fcfg, WRIST_IDX)
            rows += robustness.lda_e1(Fk, mk, yk, np.arange(len(ck)), params,
                                      {"part": "rate", "model": "lda", "experiment": "E1", "fs_hz": FS / k,
                                       "band_hi_hz": rate_band(fcfg)[1]})
            print(f"rates: {FS / k:g} Hz done", flush=True)

    if "rotation" in cfg:
        rows += robustness.lda_rotation(F, meta, y, cols, params, cfg["rotation"]["shifts"],
                                        cfg["rotation"]["train_shifts"])

    if "failure" in cfg:
        fc = cfg["failure"]
        bank = robustness.noise_feature_bank(feat_cfg, fc["noise_rms_mv"], seed)
        rows += robustness.lda_failure(F, meta, y, cols, wrist, params, fc["modes"], fc["n_failed"], bank, seed)

    return {"participants": participants, **out, **_robust_outputs(rows, cfg["name"], out.get("greedy_order"))}


def run_robust_tcn(cfg: dict) -> dict:
    """Stage 7, TCN with the stage-5 recipe (E3 base + AdaBN + fine-tuning on the 3x1 calibration): bracelet
    rotation for base networks trained as is and with rotation augmentation (`bases`: overrides of `train`),
    channel failure after the calibration. Base networks are cached as in stage 5 (the plain ones are reused)."""
    import torch

    import adapt
    import robustness
    import signals
    import tcn
    from preprocess import ms_to_samples

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    meta, X = signals.signal_table(idx, cfg["preprocess"])
    y = meta.gesture.to_numpy() - 1
    fs = signals.effective_fs(cfg["preprocess"]["decimate"])
    win = ms_to_samples(cfg["windows"]["window_ms"], fs)
    step = ms_to_samples(cfg["windows"]["step_ms"], fs)
    ends = tcn.window_ends(win, step, (X.shape[2] - win) // step + 1)
    cfg = {**cfg, "crop": ms_to_samples(cfg["train"]["crop_ms"], fs)}
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None
    fc = cfg.get("failure")
    bank = robustness.noise_signal_bank(cfg["preprocess"], fc["noise_rms_mv"], fc["bank"], cfg.get("seed", 42)) \
        if fc else None

    rows = []
    for base_name, over in cfg["bases"].items():
        bcfg = {**cfg, "train": {**cfg["train"], **over}}
        for exp in cfg["experiments"]:
            for split in _target_splits(meta, exp, cfg, test_ps):
                base = adapt.base_tcn(X, y, bcfg, participants, split, cfg["epochs"][exp], device)
                rows += robustness.tcn_split(base, base_name, X, meta, split, cfg, ends, fs, device, bank)
    return {"participants": participants, **_robust_outputs(rows, cfg["name"])}


def run_export(cfg: dict) -> dict:
    """Stage 8: the E3 TCN adapted to every target session (stage-5 recipe) is exported to ONNX and quantized to
    int8 by every variant; accuracy on the session's test trials, size, CPU latency, operations, latency budget.
    See src/export.py."""
    import pandas as pd
    import torch

    import adapt
    import export
    import signals
    import tcn
    from data import FS
    from preprocess import design_filter, ms_to_samples

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    participants = resolve_participants(cfg["participants"])
    idx = build_index()
    idx = idx[idx.participant.isin(participants)]
    missing = sorted(set(participants) - set(idx.participant))
    if missing:
        raise FileNotFoundError(f"no local data for participants {missing}; run src/data.py first")

    meta, X = signals.signal_table(idx, cfg["preprocess"])
    y = meta.gesture.to_numpy() - 1
    fs = signals.effective_fs(cfg["preprocess"]["decimate"])
    win = ms_to_samples(cfg["windows"]["window_ms"], fs)
    step = ms_to_samples(cfg["windows"]["step_ms"], fs)
    ends = tcn.window_ends(win, step, (X.shape[2] - win) // step + 1)
    cfg = {**cfg, "crop": ms_to_samples(cfg["train"]["crop_ms"], fs)}
    test_ps = resolve_participants("test") if cfg["cross_subject"] == "holdout" else None
    work = ROOT / "data" / "cache" / "onnx" / cfg["name"]
    seed = cfg.get("seed", 42)

    bc, pre = cfg["budget"], cfg["preprocess"]
    sos = design_filter(band_hz=tuple(pre["band_hz"]), order=pre["order"], notch_hz=tuple(pre["notch_hz"]))
    delays = export.group_delay_ms(sos, FS, bc["filter_at_hz"])
    read = lambda name: json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
    sweep, total = read(bc["sweep"]), read(bc["stream"])
    budget = lambda infer: export.latency_budget(sweep, "tcn", "E3", delays[0], cfg["windows"]["step_ms"], infer,
                                                 total)
    budget(0.0)                            # fail now, not after the long loop

    rows, bench, info = [], None, {}
    for split in _target_splits(meta, "E3", cfg, test_ps):
        base = adapt.base_tcn(X, y, cfg, participants, split, cfg["epochs"]["E3"], device)
        if base.receptive_field != win:
            raise ValueError(f"receptive field {base.receptive_field} != window {win} samples")
        W_train = export.random_windows(X, np.flatnonzero(split.train), cfg["quant_calib"]["n_windows"], win, seed)
        for key in meta[split.test].groupby(["participant", "session"]).indices:
            sess = split.test & (meta.participant.to_numpy() == key[0]) & (meta.session.to_numpy() == key[1])
            r, art = export.evaluate_session(base, X, meta, sess, cfg, ends, fs, device, W_train, work,
                                             {"experiment": "E3", "model": "tcn", "split": split.name})
            rows += r
            if bench is None:              # one architecture: time and count it once, on the first session
                bench = export.benchmark(art["model"], art["paths"], cfg)
                info = {**export.operations(art["model"], step),
                        "window_vs_stream_max_abs": export.check_window(art["model"], X[sess], ends)}
            print(f"E3 {split.name} p{key[0]}/s{key[1]} export done", flush=True)

    scores = pd.DataFrame(rows)
    csv_path = ROOT / "results" / f"{cfg['name']}_scores.csv"
    scores.to_csv(csv_path, index=False)
    pp = export.per_participant(scores)
    bench = pd.DataFrame(bench)

    infer = float(bench.set_index("variant").loc[bc["variant"], "latency_ms_median"])
    figures = [export.plot_quant(pp, bench, cfg["labels"], cfg["name"])]
    return {
        "participants": participants,
        "model_info": info,
        "filter_group_delay_ms": dict(zip(map(str, bc["filter_at_hz"]), delays)),
        "summary": export.summary(pp).round(4).to_dict("records"),
        "benchmark": bench.round(4).to_dict("records"),
        "latency_budget": budget(infer),
        "per_participant": pp.round(4).to_dict("records"),
        "scores_table": str(csv_path.relative_to(ROOT)),
        "figures": [str(Path(f).relative_to(ROOT)) for f in figures],
    }


RUNNERS = {"quality": run_quality, "baseline": run_baseline, "tcn": run_tcn,
           "adapt_lda": run_adapt_lda, "adapt_tcn": run_adapt_tcn, "stream": run_stream,
           "robust_lda": run_robust_lda, "robust_tcn": run_robust_tcn, "export": run_export}


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    np.random.seed(cfg.get("seed", 42))

    result = RUNNERS[cfg["experiment"]](cfg)
    out = {"config": cfg, "finished": datetime.now().isoformat(timespec="seconds"), **result}
    path = ROOT / "results" / f"{cfg['name']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
