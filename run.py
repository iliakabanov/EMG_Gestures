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


RUNNERS = {"quality": run_quality, "baseline": run_baseline, "tcn": run_tcn,
           "adapt_lda": run_adapt_lda, "adapt_tcn": run_adapt_tcn}


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
