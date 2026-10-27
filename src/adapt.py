"""Stage 5: adapting a trained model to a new session (new day or new person), without labels and with a short
labelled calibration.

Protocol. Every target session (one the base model has not seen) is split by trial:
    calibration pool  trials 1..calib_trials   (recorded first: the beginning of the new wearing)
    test              the remaining trials      (the same for every method and budget, so all numbers compare)
A budget "PxS" takes the first S seconds of holding every gesture in each of the first P trials (passes through
all gestures). A plain number s means s seconds per gesture gathered as fast as possible: s <= 5 -> "1xs",
s = 5m -> "mx5". What the user spends is 17 * P * S seconds of holding.

Methods are given in the config as {name: spec}. Spec keys for LDA (log + shrink on Hudgins features):
    center: none | rest | unlabeled
        rest       every session (training ones too) is shifted by the mean features of its first `rest_s`
                   seconds of rest (trial 1); costs rest_s seconds of rest, no labels
        unlabeled  shifted by the mean of the calibration windows of all gestures, labels not used
    calibrate: none | means | scratch
        means      class means replaced by the calibration class means, shrunk to the training means:
                   mean = a * calibration + (1 - a) * training, a = s / (s + tau_s) with s = P * S seconds
                   per gesture; covariance from training
        scratch    LDA trained on the calibration windows only (a personal model from zero)
Spec keys for the TCN:
    unlabeled: none | input_scale | adabn
        input_scale  per-channel input SD re-estimated on the calibration signal
        adabn        input_scale + batch-norm statistics re-estimated on the calibration signal
    finetune: none | head | full, with lr, steps (and optional batch, weight_decay)
        labelled fine-tuning of the classification head or of all weights, after `unlabeled`
    scratch: true, with lr, steps (and optional batch, weight_decay)
        a new TCN from random weights trained on the calibration only (no other people, no earlier day):
        isolates what the pretrained base adds over the architecture itself
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from sklearn.pipeline import Pipeline

import tcn
from data import N_GESTURES, REST_GESTURE, ROOT
from evaluate import score_rows
from models import log_amplitude, make_model
from splits import Split

TRIAL_S = 5.0                      # 10240 samples at 2048 Hz
MODEL_CACHE = ROOT / "data" / "cache"


# ---------------------------------------------------------------- calibration budget

@dataclass(frozen=True)
class Budget:
    """Calibration budget: the first `per_pass_s` seconds of every gesture in each of the first `passes` trials."""
    passes: int
    per_pass_s: float

    @property
    def per_gesture_s(self) -> float:
        return self.passes * self.per_pass_s

    @property
    def label(self) -> str:
        return f"{self.passes}x{self.per_pass_s:g}"

    def tags(self) -> dict:
        """Columns that identify the budget in the score table."""
        return {"budget": self.label, "passes": self.passes, "per_pass_s": self.per_pass_s,
                "budget_s": self.per_gesture_s}


NO_BUDGET = {"budget": "0", "passes": 0, "per_pass_s": 0.0, "budget_s": 0.0}


def parse_budget(b: float | str) -> Budget:
    """"PxS" -> P passes x S seconds; a number s -> s seconds per gesture, as few passes as possible."""
    if isinstance(b, str):
        p, s = b.lower().split("x")
        return Budget(int(p), float(s))
    if b <= TRIAL_S:
        return Budget(1, float(b))
    if b % TRIAL_S:
        raise ValueError(f"budget {b} s: above 5 s use a multiple of 5 or the 'PxS' form")
    return Budget(int(b // TRIAL_S), TRIAL_S)


def config_budgets(cfg: dict) -> list[Budget]:
    """Budgets of a config (`budgets_s`: numbers or "PxS" strings), checked against the calibration pool."""
    out = [parse_budget(b) for b in cfg["budgets_s"]]
    for b in out:
        if not (1 <= b.passes <= cfg["calib_trials"] and 0 < b.per_pass_s <= TRIAL_S):
            raise ValueError(f"budget {b.label}: needs 1..{cfg['calib_trials']} passes of 0..{TRIAL_S:g} s")
    return out


def window_mask(meta: pd.DataFrame, budget: Budget) -> np.ndarray:
    """Windows inside the budget: trial <= passes and ending within per_pass_s of the trial start."""
    return (meta.trial.to_numpy() <= budget.passes) & (meta.t_end.to_numpy() <= budget.per_pass_s + 1e-9)


def cost_seconds(spec: dict, budget: Budget | None) -> float:
    """User time a method needs: rest seconds for `rest`, 17 x P x S for anything that uses the gesture pool."""
    if uses_pool(spec):
        return N_GESTURES * budget.per_gesture_s
    return float(spec.get("rest_s", 0)) if spec.get("center") == "rest" else 0.0


def uses_pool(spec: dict) -> bool:
    """Does the method use the calibration pool (and so depend on the budget)?"""
    return (spec.get("center") == "unlabeled" or spec.get("calibrate", "none") != "none"
            or spec.get("unlabeled", "none") != "none" or spec.get("finetune", "none") != "none"
            or bool(spec.get("scratch")))


# ---------------------------------------------------------------- LDA

def centred(F: np.ndarray, meta: pd.DataFrame, ref_mask: np.ndarray) -> np.ndarray:
    """F minus, session by session, the mean of F over that session's reference windows (`ref_mask`).

    Each session uses only its own reference windows, so no information moves between sessions.
    """
    out = F.copy()
    for key, rows in meta.groupby(["participant", "session"]).indices.items():
        ref = rows[ref_mask[rows]]
        if len(ref) == 0:
            raise ValueError(f"no reference windows in participant/session {key}")
        out[rows] -= F[ref].mean(axis=0)
    return out


def adapt_means(model: Pipeline, F_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> Pipeline:
    """Copy of a fitted [StandardScaler, LDA(lsqr)] pipeline with class means moved towards the calibration data.

    LDA decides by distance to class means measured with one shared covariance. The covariance needs a lot of data
    and stays from training; the 17 means need little data. On dev this wins at small budgets, but with 15 s per
    gesture an LDA fitted on the calibration alone is 8-9 points better: the covariance changes between sessions too.
    """
    m = copy.deepcopy(model)
    scaler, lda = m[-2], m[-1]
    Z = scaler.transform(F_cal)
    cal = np.stack([Z[y_cal == c].mean(axis=0) for c in lda.classes_])
    lda.means_ = alpha * cal + (1 - alpha) * lda.means_
    lda.coef_ = linalg.lstsq(lda.covariance_, lda.means_.T)[0].T
    lda.intercept_ = -0.5 * np.diag(lda.means_ @ lda.coef_.T) + np.log(lda.priors_)
    return m


def evaluate_lda(table: pd.DataFrame, feat_cols: list[str], splits: list[Split], cfg: dict) -> list[dict]:
    """Score every LDA method x budget on the test trials of every target session.

    Args:
        table: feature table (participant, session, gesture, trial, t_end, features).
        feat_cols: feature columns; amplitude features are log-transformed here.
        splits: E2/E3 splits; their test part defines the target sessions.
        cfg: config with lda (classifier params), methods, budgets_s, calib_trials.

    Returns:
        One row per (method, budget, target participant, session).
    """
    F = log_amplitude(table[feat_cols].to_numpy(), feat_cols)
    y = table.gesture.to_numpy()
    trial = table.trial.to_numpy()
    pool, lda_params = cfg["calib_trials"], cfg["lda"]
    budgets = config_budgets(cfg)

    rows = []
    for split in splits:
        fitted = {}

        def base(center: str, budget: Budget | None, spec: dict) -> tuple[Pipeline, np.ndarray]:
            """Model on the training part with the given centring, cached per split; also the centred features."""
            key = (center, budget if center == "unlabeled" else spec.get("rest_s"))
            if key not in fitted:
                if center == "none":
                    Fc = F
                elif center == "rest":
                    ref = (table.gesture.to_numpy() == REST_GESTURE) & window_mask(table, Budget(1, spec["rest_s"]))
                    Fc = centred(F, table, ref)
                elif center == "unlabeled":
                    Fc = centred(F, table, window_mask(table, budget))
                else:
                    raise ValueError(center)
                fitted[key] = (make_model("lda", lda_params).fit(Fc[split.train], y[split.train]), Fc)
            return fitted[key]

        targets = table[split.test].groupby(["participant", "session"]).indices
        for name, spec in cfg["methods"].items():
            for b in (budgets if uses_pool(spec) else [None]):
                tags = {"experiment": split.experiment, "model": "lda", "method": name,
                        **(b.tags() if b else NO_BUDGET), "cost_s": cost_seconds(spec, b), "split": split.name}
                calibrate = spec.get("calibrate", "none")
                if calibrate != "scratch":
                    model, Fc = base(spec.get("center", "none"), b, spec)
                for key in targets:
                    sess = split.test & (table.participant.to_numpy() == key[0]) & (table.session.to_numpy() == key[1])
                    test = np.flatnonzero(sess & (trial > pool))
                    cal = np.flatnonzero(sess & window_mask(table, b)) if b else None
                    if calibrate == "scratch":
                        m, Fx = make_model("lda", lda_params).fit(F[cal], y[cal]), F
                    elif calibrate == "means":
                        s = b.per_gesture_s
                        m, Fx = adapt_means(model, Fc[cal], y[cal], s / (s + spec.get("tau_s", 0.0))), Fc
                    else:
                        m, Fx = model, Fc
                    rows += score_rows(table.iloc[test], y[test], m.predict(Fx[test]), tags)
            print(f"{split.experiment} {split.name} lda {name} done", flush=True)
    return rows


# ---------------------------------------------------------------- TCN

def calib_signals(X: np.ndarray, meta: pd.DataFrame, rows: np.ndarray, budget: Budget, warmup_s: float,
                  fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Calibration signal of one target session: the first per_pass_s of every gesture in each of the first passes.

    Args:
        X: signals [N, C, T'] that start `warmup_s` after the trial start.
        meta: one row per trial of X.
        rows: boolean mask of the target session's trials.

    Returns:
        Signals [M, C, n] of equal length and their 0-based classes [M].
    """
    sel = np.flatnonzero(rows & (meta.trial.to_numpy() <= budget.passes))
    n = min(X.shape[2], int(round((budget.per_pass_s - warmup_s) * fs)))
    return X[sel, :, :n], meta.gesture.to_numpy()[sel] - 1


@torch.no_grad()
def set_input_scale(model: tcn.TCN, X_cal: np.ndarray) -> None:
    """Re-estimate the per-channel input SD on the new session's signal (a gain correction per channel)."""
    model.scale.copy_(torch.from_numpy(X_cal.std(axis=(0, 2))).reshape(1, -1, 1))


@torch.no_grad()
def adabn(model: tcn.TCN, X_cal: np.ndarray, device: str, batch: int = 64) -> None:
    """AdaBN: replace the batch-norm running mean / variance by those of the new session's signal.

    Only batch-norm layers run in training mode (so they collect statistics); weights do not change.
    momentum=None makes the running statistics a plain average over the batches.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum, bn._saved_momentum = None, bn.momentum
        bn.train()
    for i in range(0, len(X_cal), batch):
        model(torch.from_numpy(X_cal[i:i + batch]).to(device))
    for bn in bns:
        bn.momentum = bn._saved_momentum
        del bn._saved_momentum
    model.eval()


def finetune(model: tcn.TCN, X_cal: np.ndarray, y_cal: np.ndarray, spec: dict, crop: int, seed: int,
             device: str) -> None:
    """Fine-tune the head or all weights on the labelled calibration signal, in place.

    The model stays in eval mode: batch norm keeps the statistics set by `adabn` (a batch of a few crops of one
    session would give noisy ones), dropout is off. Crops are random, as in training, and no longer than the
    calibration segments.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for p in model.parameters():
        p.requires_grad_(spec["finetune"] == "full")
    for p in model.head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=spec["lr"],
                            weight_decay=spec.get("weight_decay", 0.0))
    model.eval()
    Xd = torch.from_numpy(X_cal).to(device)
    yd = torch.from_numpy(y_cal.astype(np.int64)).to(device)
    crop = min(crop, X_cal.shape[2])
    batch = spec.get("batch", 64)
    for _ in range(spec["steps"]):
        ti = rng.integers(0, len(X_cal), size=batch)
        starts = rng.integers(0, X_cal.shape[2] - crop + 1, size=batch)
        loss = tcn.crop_loss(model, Xd, yd, ti, starts, crop)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()


def scratch_tcn(X_cal: np.ndarray, y_cal: np.ndarray, n_classes: int, cfg: dict, spec: dict, seed: int,
                device: str) -> tcn.TCN:
    """A TCN from random weights trained on one session's labelled calibration only.

    Same architecture and loss as the base networks; the input scale is the calibration SD. Every step draws
    `batch` random crops (with replacement) no longer than the calibration segments, so the number of steps is the
    same for every budget; batch norm runs in training mode, as in normal training.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = tcn.build(X_cal.shape[1], n_classes, cfg["model"]).to(device)
    model.scale.copy_(torch.from_numpy(X_cal.std(axis=(0, 2))).reshape(1, -1, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec.get("weight_decay", 0.01))
    Xd = torch.from_numpy(X_cal).to(device)
    yd = torch.from_numpy(y_cal.astype(np.int64)).to(device)
    crop = min(cfg["crop"], X_cal.shape[2])
    batch = spec.get("batch", 64)
    model.train()
    for _ in range(spec["steps"]):
        ti = rng.integers(0, len(X_cal), size=batch)
        starts = rng.integers(0, X_cal.shape[2] - crop + 1, size=batch)
        loss = tcn.crop_loss(model, Xd, yd, ti, starts, crop)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def base_tcn(X: np.ndarray, y: np.ndarray, cfg: dict, participants: list[int], split: Split, epochs: int,
             device: str) -> tcn.TCN:
    """TCN trained on the split's training trials, cached on disk: adaptation methods can be re-run without
    retraining. The cache key covers everything that defines the training set and the recipe."""
    key = {k: cfg[k] for k in ["cross_subject", "preprocess", "model", "train"]}
    key.update(participants=sorted(participants), n_folds=cfg.get("n_folds", 5), seed=cfg.get("seed", 42),
               epochs=epochs)
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]
    safe_name = re.sub(r"[^\w]+", "_", split.name)
    path = MODEL_CACHE / f"models_{digest}" / f"{split.experiment}_{safe_name}.pt"
    model = tcn.build(X.shape[1], int(y.max()) + 1, cfg["model"]).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    tr = np.flatnonzero(split.train)
    train_cfg = {**cfg["train"], "crop": cfg["crop"]}
    model, _ = tcn.fit(X[tr], y[tr], cfg["model"], train_cfg, epochs, cfg.get("seed", 42), device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def evaluate_tcn(model: tcn.TCN, X: np.ndarray, meta: pd.DataFrame, split: Split, cfg: dict, ends: np.ndarray,
                 fs: float, device: str, scratch_cache: dict | None = None) -> list[dict]:
    """Score every TCN method x budget on the test trials of the split's target sessions.

    Args:
        model: base model trained on the split's training part (not modified).
        X, meta: signals [N, C, T'] and one row per trial.
        cfg: config with methods, budgets_s, calib_trials, preprocess.warmup_ms, crop (samples), seed.
        ends: last sample of every decision window.
        scratch_cache: predictions of `scratch` networks by (method, participant, session, budget). A scratch
            network does not depend on the base model, so a session that is a target in both E2 and E3 trains once.
    """
    scratch_cache = {} if scratch_cache is None else scratch_cache
    y = meta.gesture.to_numpy() - 1
    trial = meta.trial.to_numpy()
    pool, warmup_s = cfg["calib_trials"], cfg["preprocess"]["warmup_ms"] / 1000
    budgets = config_budgets(cfg)

    rows = []
    targets = meta[split.test].groupby(["participant", "session"]).indices
    for key in targets:
        sess = split.test & (meta.participant.to_numpy() == key[0]) & (meta.session.to_numpy() == key[1])
        te = np.flatnonzero(sess & (trial > pool))
        win_meta = meta.iloc[np.repeat(te, len(ends))].reset_index(drop=True)
        y_true = np.repeat(y[te], len(ends))
        for name, spec in cfg["methods"].items():
            for b in (budgets if uses_pool(spec) else [None]):
                if spec.get("scratch"):
                    ck = (name, key[0], key[1], b.label)
                    if ck not in scratch_cache:
                        X_cal, y_cal = calib_signals(X, meta, sess, b, warmup_s, fs)
                        net = scratch_tcn(X_cal, y_cal, int(y.max()) + 1, cfg, spec, cfg.get("seed", 42), device)
                        scratch_cache[ck] = tcn.predict_windows(net, X[te], ends, device).ravel()
                    rows += score_rows(win_meta, y_true, scratch_cache[ck],
                                       {"experiment": split.experiment, "model": "tcn", "method": name,
                                        **b.tags(), "cost_s": cost_seconds(spec, b), "split": split.name})
                    continue
                m = copy.deepcopy(model)
                if b:
                    X_cal, y_cal = calib_signals(X, meta, sess, b, warmup_s, fs)
                    if spec.get("unlabeled", "none") in ("input_scale", "adabn"):
                        set_input_scale(m, X_cal)
                    if spec.get("unlabeled", "none") == "adabn":
                        adabn(m, X_cal, device)
                    if spec.get("finetune", "none") != "none":
                        finetune(m, X_cal, y_cal, spec, cfg["crop"], cfg.get("seed", 42), device)
                pred = tcn.predict_windows(m, X[te], ends, device).ravel()
                rows += score_rows(win_meta, y_true, pred,
                                   {"experiment": split.experiment, "model": "tcn", "method": name,
                                    **(b.tags() if b else NO_BUDGET), "cost_s": cost_seconds(spec, b),
                                    "split": split.name})
        print(f"{split.experiment} {split.name} p{key[0]}/s{key[1]} tcn done", flush=True)
    return rows


# ---------------------------------------------------------------- summary and figure

GROUP = ["experiment", "model", "method", "budget", "passes", "per_pass_s", "budget_s", "cost_s"]


def per_participant(scores: pd.DataFrame) -> pd.DataFrame:
    """Average over a participant's target sessions: one row per (method, budget, participant)."""
    return scores.groupby(GROUP + ["participant"], as_index=False)[["bal_acc", "macro_f1"]].mean()


def summary(pp: pd.DataFrame) -> pd.DataFrame:
    """Per method x budget: median / min / max balanced accuracy over participants, and the paired gain over
    `none` (median over participants of method - none, and in how many participants it is positive)."""
    none = pp[pp.method == "none"].set_index(["experiment", "participant"]).bal_acc
    rows = []
    for k, g in pp.groupby(GROUP, sort=False):
        acc = g.set_index(["experiment", "participant"]).bal_acc
        gain = acc - none.loc[acc.index]
        rows.append({**dict(zip(GROUP, k)), "median": g.bal_acc.median(), "min": g.bal_acc.min(),
                     "max": g.bal_acc.max(), "gain_median": gain.median(),
                     "n_better": int((gain > 0).sum()), "n": len(gain)})
    return pd.DataFrame(rows)


def plot_calibration(pp: pd.DataFrame, tag: str, labels: dict[str, str] | None = None) -> str:
    """Main figure of stage 5: balanced accuracy vs user time spent on the new session, one panel per experiment.

    Line = median over participants, band = their interquartile range. `none` is a horizontal reference;
    a method that does not depend on the budget (e.g. `rest`) is one point at its cost.
    `labels` gives legend names and the order of the methods.
    """
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    labels = labels or {}
    exps = sorted(pp.experiment.unique())
    methods = [m for m in labels if m in set(pp.method)] + [m for m in dict.fromkeys(pp.method) if m not in labels]
    names = {"E2": "E2: новый день", "E3": "E3: новый человек"}
    fig, axes = plt.subplots(1, len(exps), figsize=(6 * len(exps), 4.2), sharey=True, squeeze=False)
    xmax = pp.cost_s.max()
    for ax, e in zip(axes[0], exps):
        sub = pp[pp.experiment == e]
        color_i = 0
        for m in methods:
            g = sub[sub.method == m].groupby("cost_s").bal_acc
            q = (100 * g.quantile([0.25, 0.5, 0.75])).unstack()
            label = labels.get(m, m)
            if m == "none":
                ax.axhline(q[0.5].iloc[0], color=viz.REFERENCE, lw=1.2, label=label)
                ax.axhspan(q[0.25].iloc[0], q[0.75].iloc[0], color=viz.REFERENCE, alpha=0.12, lw=0)
                continue
            c = viz.SERIES[color_i % len(viz.SERIES)]
            color_i += 1
            if len(q) == 1:
                ax.errorbar(q.index, q[0.5], yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]], fmt="D", ms=6, color=c,
                            capsize=3, label=label)
            else:
                ax.plot(q.index, q[0.5], marker="o", ms=4, color=c, label=label)
                ax.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.12, lw=0)
        ax.set_xlim(-0.02 * xmax, 1.03 * xmax)
        ax.set_ylim(0, 100)
        ax.set_xticks(sorted(sub.cost_s.unique()))
        ax.set_xlabel("время пользователя в новой сессии, с")
        ax.set_title(names.get(e, e))
    axes[0][0].set_ylabel("balanced accuracy, %")
    fig.suptitle("Адаптация: линия = медиана по участникам, полоса = 25–75 %", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    handles, names_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    path = viz.save(fig, f"05_calibration_{tag}")
    plt.close(fig)
    return str(path)


def plot_passes(pp: pd.DataFrame, tag: str, labels: dict[str, str] | None = None) -> str:
    """Repetitions vs seconds: one panel per experiment x method, one line per number of passes.

    Along a line the seconds per pass grow (marked at the points); at equal user time, a line above another means
    that splitting the same seconds into more passes is better. Line = median over participants,
    band = interquartile range, grey = no adaptation.
    """
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    labels = labels or {}
    exps = sorted(pp.experiment.unique())
    methods = [m for m in (list(labels) or list(dict.fromkeys(pp.method))) if m in set(pp.method) and m != "none"]
    names = {"E2": "E2", "E3": "E3"}
    fig, axes = plt.subplots(len(exps), len(methods), figsize=(3.6 * len(methods), 3.4 * len(exps)),
                             sharex=True, sharey=True, squeeze=False)
    for i, e in enumerate(exps):
        sub = pp[pp.experiment == e]
        none = 100 * sub[sub.method == "none"].bal_acc.median()
        for j, m in enumerate(methods):
            ax = axes[i][j]
            ax.axhline(none, color=viz.REFERENCE, lw=1, label="без адаптации")
            for k, (p, g) in enumerate(sub[sub.method == m].groupby("passes")):
                q = (100 * g.groupby("cost_s").bal_acc.quantile([0.25, 0.5, 0.75])).unstack()
                c = viz.SERIES[k]
                ax.plot(q.index, q[0.5], marker="o", ms=4, color=c, label=f"{p} прох.")
                ax.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.12, lw=0)
                per_pass = g.groupby("cost_s").per_pass_s.first()
                for x, s in per_pass.items():
                    ax.annotate(f"{s:g} с", (x, q.loc[x, 0.5]), xytext=(3, -10), textcoords="offset points",
                                fontsize=6.5, color=c)
            ax.set_ylim(0, 100)
            ax.set_xscale("log")
            ticks = sorted(pp[pp.method != "none"].cost_s.unique())
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks], fontsize=7)
            ax.minorticks_off()
            ax.set_title(f"{names.get(e, e)}: {labels.get(m, m)}", fontsize=9)
            if i == len(exps) - 1:
                ax.set_xlabel("время пользователя, с")
        axes[i][0].set_ylabel("balanced accuracy, %")
    fig.suptitle("Повторения или секунды: линия = число проходов, подпись = секунд на жест за проход",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    handles, names_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8)
    path = viz.save(fig, f"05_passes_{tag}")
    plt.close(fig)
    return str(path)
