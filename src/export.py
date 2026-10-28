"""Stage 8: the network on a device. ONNX export, int8 post-training quantization in onnxruntime, size / speed /
operations / accuracy before and after, and the latency budget of the whole system.

Deployed form. A decision is made every 50 ms from the last 200 ms of signal; the TCN's receptive field is exactly
those 205 samples (1024 Hz), so the exported graph takes one window [1, 12, 205] and returns the 17 logits of its
last step. Its output equals the streaming output of stages 4-6 at the same moment (checked by `check_window`).

Quantization (int8). A tensor x is stored as integers q with a scale s and a zero point z:
    x ~ s * (q - z),  q = clip(round(x / s) + z, 0..255),  s = (max - min) / 255
Weights are fixed, so their range is known (per output channel with `per_channel`). Activations depend on the input:
    dynamic  their range is measured on every call (robust, but extra work at run time)
    static   their range is fixed beforehand from calibration windows (MinMax: extreme values; Percentile: the
             0.001-99.999 % range, outliers clipped); everything then runs in integer arithmetic
`keep_fp32: [input]` leaves the input scaling (Div) and the first 1x1 convolution (stem) in float: the raw signal
has the widest dynamic range, as with an ADC, 8 bits there cost most.

Quantization windows (`calib`): `train` = random windows of the base network's training trials (other people),
`own` = windows of the session's labelled 3x1 calibration (its first second of every gesture).
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import adapt
import tcn
from evaluate import score_rows

METRICS = ["bal_acc", "macro_f1", "agree"]


class LastStep(nn.Module):
    """TCN on one decision window [B, C, RF] -> logits of the last step [B, n_classes]."""

    def __init__(self, model: tcn.TCN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, :, -1]


def export_onnx(model: tcn.TCN, path: Path) -> Path:
    """Export a CPU copy of the model in eval mode (window in, last-step logits out) to ONNX, opset 17."""
    m = LastStep(copy.deepcopy(model).cpu()).eval()
    x = torch.zeros(1, model.stem.in_channels, model.receptive_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(m, (x,), str(path), input_names=["x"], output_names=["logits"], opset_version=17,
                          dynamo=False)
    return path


def windows(X: np.ndarray, ends: np.ndarray, rf: int) -> np.ndarray:
    """Decision windows of every trial at every end -> [N * len(ends), C, rf] (trial-major, as predict_windows)."""
    return np.stack([x[:, e - rf + 1:e + 1] for x in X for e in ends])


def random_windows(X: np.ndarray, rows: np.ndarray, n: int, rf: int, seed: int) -> np.ndarray:
    """n windows from random trials (`rows`) at random positions."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows, n)
    starts = rng.integers(0, X.shape[2] - rf + 1, n)
    return np.stack([X[i, :, s:s + rf] for i, s in zip(pick, starts)])


def input_nodes(path: Path) -> list[str]:
    """Names of the input scaling (Div) and the stem convolution in an exported graph."""
    import onnx

    return [n.name for n in onnx.load(str(path)).graph.node if n.name.endswith("/Div") or "/stem/" in n.name]


def quantize(fp32: Path, out: Path, spec: dict, calib: np.ndarray | None) -> Path:
    """int8 copy of an exported model by a variant spec (see the module docstring).

    Args:
        fp32: exported float model.
        spec: mode (dynamic | static); for static: method (minmax | percentile), optional keep_fp32: [input].
        calib: quantization windows [M, C, RF] for static.
    """
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, arr: np.ndarray):
            self.it = iter(arr)

        def get_next(self):
            a = next(self.it, None)
            return None if a is None else {"x": a[None]}

    # onnxruntime writes intermediate files through ONNX's C++ shape inference, which cannot open non-ASCII paths
    # on Windows (the project path is Cyrillic): quantize in the system temp folder, then move the result
    with tempfile.TemporaryDirectory() as tmp:
        src, pre, dst = Path(tmp) / "fp32.onnx", Path(tmp) / "pre.onnx", Path(tmp) / "int8.onnx"
        shutil.copyfile(fp32, src)
        quant_pre_process(str(src), str(pre))        # shape inference + graph optimisation before quantizing
        if spec["mode"] == "dynamic":
            quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8)
        else:
            method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}[spec["method"]]
            keep = input_nodes(pre) if "input" in spec.get("keep_fp32", []) else []
            quantize_static(str(pre), str(dst), Reader(calib), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                            calibrate_method=method, nodes_to_exclude=keep)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), out)
    return out


def session(path: Path, threads: int = 0):
    """onnxruntime CPU session; threads = 0 lets onnxruntime choose."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def predict_onnx(path: Path, W: np.ndarray) -> np.ndarray:
    """Class (0-based) of every window, one window per call as on the device (a dynamic range is per call)."""
    s = session(path)
    return np.array([s.run(None, {"x": w[None]})[0].argmax() for w in W])


def latency_ms(fn, x, runs: int, warmup: int) -> dict:
    """Wall time of one call fn(x): median, 90th and 99th percentile, ms."""
    for _ in range(warmup):
        fn(x)
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        fn(x)
        t[i] = time.perf_counter() - t0
    return {"latency_ms_median": 1000 * np.median(t), "latency_ms_p90": 1000 * np.quantile(t, 0.9),
            "latency_ms_p99": 1000 * np.quantile(t, 0.99)}


def operations(model: tcn.TCN, step: int) -> dict:
    """Parameters and multiply-accumulate operations (MACs) per decision.

    Every convolution of the TCN keeps the length of its input, so one output position costs
    sum(C_out * C_in * kernel) MACs over all convolutions. The window form recomputes all RF positions on every
    decision; a streaming form that keeps the past activations of every layer would compute only the `step` new
    positions.
    """
    per_pos = sum(m.out_channels * m.in_channels // m.groups * m.kernel_size[0]
                  for m in model.modules() if isinstance(m, nn.Conv1d))
    return {"params": tcn.n_params(model), "macs_per_position": per_pos,
            "macs_window": per_pos * model.receptive_field, "macs_streaming": per_pos * step}


def check_window(model: tcn.TCN, X: np.ndarray, ends: np.ndarray) -> float:
    """Max |logit difference| between the window form and the streaming form on the first trial, on the CPU in
    float32 (on the GPU cuDNN may use reduced-precision TF32 convolutions)."""
    m = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        full = m(torch.from_numpy(X[:1]))[0, :, ends].T.numpy()
        win = LastStep(m)(torch.from_numpy(windows(X[:1], ends, m.receptive_field))).numpy()
    return float(np.abs(full - win).max())


def evaluate_session(base: tcn.TCN, X: np.ndarray, meta: pd.DataFrame, sess: np.ndarray, cfg: dict, ends: np.ndarray,
                     fs: float, device: str, W_train: np.ndarray, work: Path, tags: dict) -> tuple[list[dict], dict]:
    """Adapt the base network to one target session, export it, quantize every variant and score all of them on
    the session's test trials.

    Returns:
        Score rows (one per variant: bal_acc, macro_f1, agree = share of windows with the class of PyTorch fp32)
        and the paths of the exported files by variant.
    """
    y = meta.gesture.to_numpy() - 1
    budget = adapt.parse_budget(cfg["calibration"]["budget"])
    X_cal, y_cal = adapt.calib_signals(X, meta, sess, budget, cfg["preprocess"]["warmup_ms"] / 1000, fs)
    m = adapt.adapted(base, X_cal, y_cal, cfg["adapt"], cfg["crop"], cfg.get("seed", 42), device)
    rf = m.receptive_field
    te = np.flatnonzero(sess & (meta.trial.to_numpy() > cfg["calib_trials"]))
    W = windows(X[te], ends, rf)
    win_meta = meta.iloc[np.repeat(te, len(ends))].reset_index(drop=True)
    y_true = np.repeat(y[te], len(ends))
    ref = tcn.predict_windows(m, X[te], ends, device).ravel()
    step = int(ends[1] - ends[0])
    W_own = np.stack([x[:, s:s + rf] for x in X_cal for s in range(0, X_cal.shape[2] - rf + 1, step)])

    def row(name: str, pred: np.ndarray) -> list[dict]:
        r = score_rows(win_meta, y_true, pred, {**tags, "variant": name})
        r[0]["agree"] = float((pred == ref).mean())
        return r

    rows = row("torch_fp32", ref)
    fp32 = export_onnx(m, work / "fp32.onnx")
    paths = {}
    for name, spec in cfg["variants"].items():
        path = fp32 if spec.get("mode", "fp32") == "fp32" else \
            quantize(fp32, work / f"{name}.onnx", spec, W_own if spec.get("calib") == "own" else W_train)
        paths[name] = path
        rows += row(name, predict_onnx(path, W))
    return rows, {"model": m, "paths": paths}


def benchmark(model: tcn.TCN, paths: dict[str, Path], cfg: dict) -> list[dict]:
    """Size and single-thread CPU latency of one decision (batch 1) for PyTorch fp32 and every ONNX variant."""
    bc = cfg["benchmark"]
    rf, c = model.receptive_field, model.stem.in_channels
    x = np.random.default_rng(0).standard_normal((1, c, rf)).astype(np.float32)
    threads = torch.get_num_threads()
    torch.set_num_threads(bc["threads"])
    m = LastStep(copy.deepcopy(model).cpu()).eval()
    with torch.no_grad():
        rows = [{"variant": "torch_fp32", "size_kb": sum(p.numel() * 4 for p in m.state_dict().values()) / 1024,
                 **latency_ms(lambda a: m(torch.from_numpy(a)), x, bc["runs"], bc["warmup"])}]
    torch.set_num_threads(threads)
    for name, path in paths.items():
        s = session(path, bc["threads"])
        rows.append({"variant": name, "size_kb": path.stat().st_size / 1024,
                     **latency_ms(lambda a: s.run(None, {"x": a}), x, bc["runs"], bc["warmup"])})
    return rows


# ---------------------------------------------------------------- latency budget

def group_delay_ms(sos: np.ndarray, fs: float, freqs: list[float]) -> list[float]:
    """Group delay (how long a narrow-band component is delayed by the filter) at the given frequencies, ms."""
    from scipy import signal

    b, a = signal.sos2tf(sos)
    _, gd = signal.group_delay((b, a), w=freqs, fs=fs)
    return list(1000 * gd / fs)


def latency_budget(sweep: dict, model: str, exp: str, filter_ms: float, step_ms: float, inference_ms: float,
                   total: dict | None = None) -> list[dict]:
    """Decomposition of the stage-6 command latency (median, gesture onset -> correct command) at the frozen
    operating point, plus the inference time that the stream evaluation did not include.

    The dev stream sweep (`sweep`, stream_dev) has the latency of every post-processing setting; the steps between
    settings that differ in one knob give the cost of that knob (smoothing, minimum duration). What is left without
    post-processing is the reaction of the model itself: filter delay, waiting for the next decision on the 50-ms
    grid (half a step on average) and the time the 200-ms window needs to collect enough of the gesture.
    The test stream run has only the frozen setting, so it cannot be decomposed; with `total` (stream_test) its
    measured latency is added as the last row.
    """
    op = next(r for r in sweep["operating_points"] if r["model"] == model and r["experiment"] == exp)
    c = pd.DataFrame(sweep["curves_mean"])
    c = c[(c.model == model) & (c.experiment == exp) & (c.refractory_ms == op["refractory_ms"])
          & np.isclose(c.theta, op["theta"])]

    def lat(tau: float, mn: float) -> float:
        hit = c[(c.tau_ms == tau) & (c.min_ms == mn)]
        if hit.empty:
            raise ValueError(f"no stream latency for tau={tau}, min={mn}: `budget.sweep` needs the dev sweep")
        return float(hit.latency_ms_median.iloc[0])

    raw, smooth, full = lat(0, 0), lat(op["tau_ms"], 0), lat(op["tau_ms"], op["min_ms"])
    rows = [
        ("фильтр 20–450 Гц: групповая задержка на пике спектра", filter_ms, "рассчитано"),
        ("сетка решений: ожидание следующего решения (шаг / 2)", step_ms / 2, "оценка"),
        ("окно 200 мс: накопление жеста до уверенного решения", raw - filter_ms - step_ms / 2, "этап 6 − две строки выше"),
        (f"сглаживание τ = {op['tau_ms']:g} мс", smooth - raw, "этап 6"),
        (f"минимальная длительность {op['min_ms']:g} мс", full - smooth, "этап 6"),
        ("инференс (CPU ноутбука, 1 поток)", inference_ms, "этап 8"),
    ]
    out = [{"component": k, "ms": v, "source": s} for k, v, s in rows]
    out.append({"component": "итого (dev)", "ms": full + inference_ms, "source": f"этап 6, dev: {full:.0f} мс + инференс"})
    if total is not None:
        t = next(r for r in total["operating_points"] if r["model"] == model and r["experiment"] == exp)
        if any(t[k] != op[k] for k in ["tau_ms", "min_ms", "refractory_ms", "theta"]):
            raise ValueError("the test operating point differs from the dev one")
        out.append({"component": "итого (test)", "ms": t["latency_ms_median"] + inference_ms,
                    "source": f"этап 6, test: {t['latency_ms_median']:.0f} мс + инференс"})
    return out


# ---------------------------------------------------------------- summary and figure

def per_participant(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean over a participant's target sessions: one row per (variant, participant)."""
    return scores.groupby(["variant", "participant"], as_index=False, sort=False)[METRICS].mean()


def summary(pp: pd.DataFrame, ref: str = "torch_fp32") -> pd.DataFrame:
    """Per variant: mean / SD / min / max balanced accuracy over participants, the paired difference with `ref`
    (mean, SD, how many participants are better and worse) and the agreement with `ref` (mean, min)."""
    base = pp[pp.variant == ref].set_index("participant").bal_acc
    rows = []
    for v, g in pp.groupby("variant", sort=False):
        acc = g.set_index("participant").bal_acc
        d = acc - base.loc[acc.index]
        rows.append({"variant": v, "mean": acc.mean(), "sd": acc.std(), "min": acc.min(), "max": acc.max(),
                     "macro_f1": g.macro_f1.mean(), "diff_mean": d.mean(), "diff_sd": d.std(),
                     "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n": len(d),
                     "agree_mean": g.agree.mean(), "agree_min": g.agree.min()})
    return pd.DataFrame(rows)


def plot_quant(pp: pd.DataFrame, bench: pd.DataFrame, labels: dict[str, str], tag: str) -> str:
    """Left: change of balanced accuracy against PyTorch fp32 per participant (point) and mean ± SD.
    Right: single-thread latency of one decision (median, bar to the 99th percentile) and model size."""
    import matplotlib.pyplot as plt
    import viz

    viz.setup()
    variants = [v for v in labels if v in set(pp.variant) and v != "torch_fp32"]
    base = pp[pp.variant == "torch_fp32"].set_index("participant").bal_acc
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    c = viz.SERIES[0]
    for i, v in enumerate(variants):
        d = 100 * (pp[pp.variant == v].set_index("participant").bal_acc - base)
        ax[0].scatter(np.full(len(d), i), d, s=12, color=c, alpha=0.35, lw=0)
        ax[0].errorbar([i], [d.mean()], yerr=[d.std()], marker="o", ms=5, capsize=3, color=c)
        ax[0].annotate(f"{d.mean():+.1f}", (i, d.mean()), xytext=(6, 2), textcoords="offset points", fontsize=7,
                       color=c)
    ax[0].axhline(0, color=viz.REFERENCE, lw=1)
    ax[0].set_xticks(range(len(variants)), [labels[v] for v in variants], fontsize=7)
    ax[0].set_ylabel("Δ balanced accuracy к PyTorch fp32, п.п.")
    ax[0].set_title("Точность после экспорта и квантования")
    b = bench.set_index("variant").loc[[v for v in ["torch_fp32"] + variants if v in set(bench.variant)]]
    y = np.arange(len(b))
    ax[1].barh(y, b.latency_ms_median, color=c, alpha=0.8)
    ax[1].errorbar(b.latency_ms_median, y, xerr=[np.zeros(len(b)), b.latency_ms_p99 - b.latency_ms_median],
                   fmt="none", color=viz.INK_2, capsize=3)
    for yi, (v, r) in zip(y, b.iterrows()):
        ax[1].annotate(f"{r.latency_ms_median:.2f} мс · {r.size_kb:.0f} КБ", (r.latency_ms_p99, yi), xytext=(4, -3),
                       textcoords="offset points", fontsize=7, color=viz.INK_2)
    ax[1].set_yticks(y, [labels.get(v, v) for v in b.index], fontsize=7)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1.35 * b.latency_ms_p99.max())
    ax[1].set_xlabel("время одного решения, мс (1 поток CPU; черта — до 99-го перцентиля)")
    ax[1].set_title("Задержка и размер")
    fig.suptitle("TCN, E3 + калибровка 3×1: fp32 и int8", x=0.01, ha="left", fontsize=11, fontweight="bold",
                 color=viz.INK)
    fig.tight_layout()
    path = viz.save(fig, f"08_quant_{tag}")
    plt.close(fig)
    return str(path)
