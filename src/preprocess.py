"""Causal preprocessing: IIR filters in SOS form and sliding windows.

Filtering uses scipy.signal.sosfilt with an explicit state, never filtfilt: in real time future samples
do not exist, so the offline pipeline must see exactly what a device would see.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from data import FS


def design_filter(fs: float = FS, band_hz: tuple[float, float] = (20.0, 450.0), order: int = 4,
                  notch_hz: tuple[float, ...] = (), notch_q: float = 30.0) -> np.ndarray:
    """Butterworth band-pass followed by optional notch filters, as one SOS cascade.

    Args:
        fs: sampling rate, Hz.
        band_hz: pass band edges, Hz.
        order: Butterworth order per edge (the band-pass has order 2 * order).
        notch_hz: centre frequencies of notch filters, Hz (e.g. mains harmonics).
        notch_q: notch quality factor; bandwidth = f0 / Q.

    Returns:
        SOS array [n_sections, 6].
    """
    sos = [signal.butter(order, band_hz, btype="bandpass", fs=fs, output="sos")]
    for f0 in notch_hz:
        b, a = signal.iirnotch(f0, notch_q, fs=fs)
        sos.append(signal.tf2sos(b, a))
    return np.vstack(sos)


def design_antialias(cutoff_hz: float, order: int = 8, fs: float = FS) -> np.ndarray:
    """Causal Butterworth low-pass applied before keeping every k-th sample.

    At the new rate fs_new only frequencies below fs_new / 2 (the Nyquist frequency) can be represented; anything
    above folds back onto lower frequencies (aliasing). A device filters in analog form before its ADC; here the
    same is simulated on the 2048 Hz signal. With the cutoff at 0.4 * fs_new, what would fold into the band below
    the cutoff (frequencies above 0.6 * fs_new) is attenuated by >= 28 dB at order 8.
    """
    return signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")


def filter_causal(x: np.ndarray, sos: np.ndarray,
                  zi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Filter x [T, C] causally along time.

    Args:
        x: signal [T, C].
        sos: filter from `design_filter`.
        zi: state carried over from the previous chunk [n_sections, 2, C]. None starts from rest (zero state),
            i.e. assumes the signal was 0 before x[0]. GRABMyo trials are already zero-mean, so this is closer
            to the true past than the steady state for x[0], which turns one random EMG sample into a fake step.
            Starting from x[0] only helps signals with a large DC offset. Either way the start-up transient
            lasts ~150 ms on this filter, so callers drop the beginning of each trial.

    Returns:
        Filtered signal [T, C] and the final state, to pass as `zi` for the next chunk of a stream.
    """
    if zi is None:
        zi = np.zeros((sos.shape[0], 2, x.shape[1]))
    return signal.sosfilt(sos, x, axis=0, zi=zi)


def sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """Windows [N, win, C] of x [T, C]; window k covers samples [k*step, k*step + win).

    Each window uses only samples up to its last one, so a decision for it can be made at that moment.
    Returns a view, no copy.
    """
    return np.lib.stride_tricks.sliding_window_view(x, win, axis=0)[::step].transpose(0, 2, 1)


def ms_to_samples(ms: float, fs: float = FS) -> int:
    """Convert milliseconds to the nearest whole number of samples."""
    return int(round(ms * fs / 1000))
