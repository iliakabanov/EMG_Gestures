"""Causal temporal convolutional network (TCN) on the filtered signal: model, training, window predictions.

Every convolution is padded on the left only, so the output at time t depends on samples <= t, and the network
can run on a stream sample by sample. The receptive field (how far into the past one output looks) is
    RF = 1 + 2 * (kernel - 1) * sum(dilations)     (two convolutions per residual block)
and is set equal to the 200 ms window of the feature models, so both make decisions from the same span of signal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d that pads (kernel - 1) * dilation zeros on the left: no output depends on a future sample."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int = 1):
        super().__init__(c_in, c_out, kernel, dilation=dilation)
        self.left = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left, 0)))


class Block(nn.Module):
    """Two dilated causal convolutions with batch norm, ReLU and channel dropout, plus a residual connection.

    Channel dropout (`Dropout1d`) zeroes whole feature maps during training, so the network cannot lean on one
    narrow feature that happens to work in the training session. It is off at inference.
    """

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, kernel, dilation), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout1d(dropout),
            CausalConv1d(ch, ch, kernel, dilation), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout1d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCN(nn.Module):
    """Signal [B, C, T] -> class logits at every time step [B, n_classes, T].

    `scale` holds the per-channel SD of the training signal; inputs are divided by it inside the model,
    so the normalisation travels with the weights and uses training statistics only.
    """

    def __init__(self, n_channels: int, n_classes: int, channels: int = 64, kernel: int = 3,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 20), dropout: float = 0.0):
        super().__init__()
        self.register_buffer("scale", torch.ones(1, n_channels, 1))
        self.stem = nn.Conv1d(n_channels, channels, 1)
        self.blocks = nn.Sequential(*[Block(channels, kernel, d, dropout) for d in dilations])
        self.head = nn.Conv1d(channels, n_classes, 1)
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x / self.scale)))


def n_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def window_ends(win: int, step: int, n_windows: int) -> np.ndarray:
    """Index of the last sample of each decision window (same moments as the feature models)."""
    return np.arange(n_windows) * step + win - 1


def build(n_channels: int, n_classes: int, model_cfg: dict) -> TCN:
    """TCN from the `model` section of a config."""
    return TCN(n_channels, n_classes, channels=model_cfg["channels"], kernel=model_cfg["kernel"],
               dilations=tuple(model_cfg["dilations"]), dropout=model_cfg.get("dropout", 0.0))


def crop_loss(model: TCN, Xd: torch.Tensor, yd: torch.Tensor, trials: np.ndarray, starts: np.ndarray,
              crop: int) -> torch.Tensor:
    """Mean cross-entropy over the positions of each crop whose receptive field lies fully inside the crop.

    Args:
        Xd, yd: signals [N, C, T] and 0-based classes [N] on the model's device.
        trials, starts: trial index and first sample of every crop in the batch.
        crop: crop length, samples.
    """
    xb = torch.stack([Xd[t, :, s:s + crop] for t, s in zip(trials, starts)])
    logits = model(xb)[:, :, model.receptive_field - 1:]
    target = yd[torch.from_numpy(trials).to(Xd.device)][:, None].expand(-1, logits.shape[2])
    return F.cross_entropy(logits, target)


@torch.no_grad()
def predict_windows(model: TCN, X: np.ndarray, ends: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """Predicted class (0-based) for every trial in X [N, C, T] at every index in `ends` -> [N, len(ends)]."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        logits = model(torch.from_numpy(X[i:i + batch]).to(device))
        out.append(logits[:, :, ends].argmax(1).cpu().numpy())
    return np.concatenate(out)


def fit(X: np.ndarray, y: np.ndarray, model_cfg: dict, train_cfg: dict, epochs: int, seed: int, device: str,
        monitor: Callable[[TCN], dict] | None = None, eval_every: int = 0) -> tuple[TCN, list[dict]]:
    """Train a TCN on whole trials with random crops.

    Args:
        X: training signals [N, C, T].
        y: class of each trial, 0-based [N].
        model_cfg: TCN constructor arguments (channels, kernel, dilations, optional dropout).
        train_cfg: crop (samples), crops_per_trial (crops per trial per epoch), batch, lr, weight_decay.
        epochs: number of epochs.
        seed: seed for weights and crop sampling.
        device: "cuda" or "cpu".
        monitor: called every `eval_every` epochs with the model in eval mode; its dict goes to the history.

    Returns:
        Trained model (eval mode) and history: one dict per monitored epoch.

    The loss is the mean cross-entropy over every position of a crop whose receptive field lies fully inside
    the crop: these are exactly the positions where a decision would be made in a stream.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    model = build(X.shape[1], n_classes, model_cfg).to(device)
    model.scale.copy_(torch.from_numpy(X.std(axis=(0, 2)).reshape(1, -1, 1)))
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    Xd = torch.from_numpy(X).to(device)
    yd = torch.from_numpy(y.astype(np.int64)).to(device)
    crop = train_cfg["crop"]
    T = X.shape[2]
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        trials = rng.permutation(np.repeat(np.arange(len(X)), train_cfg["crops_per_trial"]))
        for i in range(0, len(trials), train_cfg["batch"]):
            ti = trials[i:i + train_cfg["batch"]]
            starts = rng.integers(0, T - crop + 1, size=len(ti))
            loss = crop_loss(model, Xd, yd, ti, starts, crop)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if monitor is not None and eval_every and (epoch % eval_every == 0 or epoch == epochs):
            model.eval()
            history.append({"epoch": epoch, "loss": loss.item(), **monitor(model)})
    model.eval()
    return model, history
