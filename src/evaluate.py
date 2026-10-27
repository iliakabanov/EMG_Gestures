"""Run experiments E1-E4 for several models; per-participant metrics, confusion matrices and stage-3 figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from data import GESTURE_SHORT, N_GESTURES
from models import make_model
from splits import make_splits

LABELS = list(range(1, N_GESTURES + 1))


def evaluate(table: pd.DataFrame, feat_cols: list[str], experiments: list[str], models: dict[str, dict],
             cross_subject: str = "lopo", test_participants: list[int] | None = None,
             seed: int = 42, n_folds: int = 5) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit and test every model on every split.

    Returns:
        scores: one row per (experiment, model, test participant, test session) with balanced accuracy and macro-F1.
        confusion: {"<experiment>/<model>": counts [17, 17]} summed over all splits (rows = true gesture).
    """
    X = table[feat_cols].to_numpy()
    y = table.gesture.to_numpy()
    rows, conf = [], {}
    for exp in experiments:
        for split in make_splits(table, exp, cross_subject, test_participants, n_folds, seed):
            test_idx = np.flatnonzero(split.test)
            for name, params in models.items():
                model = make_model(name, params, seed, feat_cols).fit(X[split.train], y[split.train])
                pred = model.predict(X[test_idx])
                key = f"{exp}/{name}"
                conf[key] = conf.get(key, 0) + confusion_matrix(y[test_idx], pred, labels=LABELS)
                rows += score_rows(table.iloc[test_idx], y[test_idx], pred,
                                   {"experiment": exp, "model": name, "split": split.name})
                print(f"{exp} {split.name} {name} done", flush=True)
    return pd.DataFrame(rows), conf


def score_rows(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, tags: dict) -> list[dict]:
    """Balanced accuracy and macro-F1 per (participant, session) of the test windows.

    Args:
        meta: one row per test window with participant and session.
        y_true, y_pred: gesture of each window (same labels in both).
        tags: fields copied into every row (experiment, model, split).
    """
    rows = []
    for (p, s), pos in meta.groupby(["participant", "session"]).indices.items():
        rows.append({**tags, "participant": p, "session": s,
                     "bal_acc": balanced_accuracy_score(y_true[pos], y_pred[pos]),
                     "macro_f1": f1_score(y_true[pos], y_pred[pos], average="macro")})
    return rows


def per_participant(scores: pd.DataFrame) -> pd.DataFrame:
    """Average over test sessions: one row per (experiment, model, participant)."""
    return scores.groupby(["experiment", "model", "participant"], as_index=False)[["bal_acc", "macro_f1"]].mean()


def plot_e1_e4(pp: pd.DataFrame, tag: str, stage: str = "03") -> str:
    """Main figure: balanced accuracy E1 -> E4, one point per participant, line through the medians."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    exps = sorted(pp.experiment.unique())
    models = list(dict.fromkeys(pp.model))
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.18
    for i, m in enumerate(models):
        off = (i - (len(models) - 1) / 2) * width
        med = []
        for j, e in enumerate(exps):
            v = pp[(pp.model == m) & (pp.experiment == e)].bal_acc.to_numpy()
            ax.scatter(np.full(len(v), j + off), 100 * v, s=14, color=viz.SERIES[i], alpha=0.45, lw=0)
            med.append(100 * np.median(v))
        ax.plot(np.arange(len(exps)) + off, med, marker="o", ms=6, color=viz.SERIES[i], label=f"{m}: медиана")
        for j, v in enumerate(med):
            ax.annotate(f"{v:.0f}", (j + off, v), xytext=(5, 3), textcoords="offset points", fontsize=8,
                        color=viz.SERIES[i])
    ax.axhline(100 / len(LABELS), color=viz.REFERENCE, lw=0.8, ls="--")
    ax.text(len(exps) - 0.5, 100 / len(LABELS) + 1, "случайное угадывание", fontsize=7, color=viz.MUTED, ha="right")
    names = {"E1": "E1\nвнутри сессии", "E2": "E2\nмежду днями", "E3": "E3\nмежду людьми", "E4": "E4\nлюди + дни"}
    ax.set_xticks(range(len(exps)), [names.get(e, e) for e in exps])
    ax.set_ylabel("balanced accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Точность по окнам: точка = участник, линия = медиана")
    path = viz.save(fig, f"{stage}_e1_e4_{tag}")
    plt.close(fig)
    return str(path)


def plot_confusion(conf: dict[str, np.ndarray], model: str, tag: str, stage: str = "03") -> str:
    """Row-normalised confusion matrices of one model for every experiment."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    keys = [k for k in conf if k.endswith(f"/{model}")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        c = conf[k] / conf[k].sum(axis=1, keepdims=True)
        ax.imshow(100 * c, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(LABELS)), GESTURE_SHORT, rotation=90, fontsize=6)
        ax.set_yticks(range(len(LABELS)), GESTURE_SHORT, fontsize=6)
        ax.set_xlabel("предсказано")
        ax.set_title(k.split("/")[0])
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("истинный жест")
    fig.suptitle(f"Матрицы ошибок, {model}: доля окон истинного жеста, %", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"{stage}_confusion_{model}_{tag}")
    plt.close(fig)
    return str(path)


def plot_curves(med: pd.DataFrame, best: dict[str, int], tag: str, stage: str = "04") -> str:
    """Median balanced accuracy on the test part vs training epoch, one line per experiment, best epoch marked.

    Args:
        med: columns experiment, epoch, bal_acc (median over participants).
        best: chosen epoch per experiment.
    """
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (e, g) in enumerate(med.groupby("experiment")):
        ax.plot(g.epoch, 100 * g.bal_acc, color=viz.SERIES[i], label=e)
        b = g[g.epoch == best[e]]
        ax.scatter(b.epoch, 100 * b.bal_acc, color=viz.SERIES[i], s=30, zorder=3)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("balanced accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("Валидация: медиана по участникам, точка = выбранная эпоха")
    path = viz.save(fig, f"{stage}_epochs_{tag}")
    plt.close(fig)
    return str(path)
