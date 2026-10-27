"""GRABMyo data access: download, file index, trial loading.

Channel map (verified against grabmyo_convert_wfdb_to_mat.py and .hea signal names):
    record columns 0..15  -> F1..F16  forearm (2 rings x 8)
    record column  16     -> U1       unused
    record columns 17..22 -> W1..W6   wrist ring 1
    record columns 23, 24 -> U2, U3   unused
    record columns 25..30 -> W7..W12  wrist ring 2
    record column  31     -> U4       unused
Gesture 17 is rest. Each trial is 10240 samples = 5 s at 2048 Hz. Physical units: mV.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml

# PhysioNet's AWS mirror: ~15x faster than physionet.org (~19 KB/s per connection), byte-identical files.
BASE_URL = "https://physionet-open.s3.amazonaws.com/grabmyo/1.1.0"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

FS = 2048
N_SESSIONS = 3
N_GESTURES = 17
N_TRIALS = 7
REST_GESTURE = 17

FOREARM_IDX = list(range(0, 16))
WRIST_IDX = list(range(17, 23)) + list(range(25, 31))
UNUSED_IDX = [16, 23, 24, 31]

GESTURE_NAMES = [
    "Lateral Prehension", "Thumb Adduction", "Thumb and Little Finger Opposition",
    "Thumb and Index Finger Opposition", "Thumb and Index Finger Extension",
    "Thumb and Little Finger Extension", "Index and Middle Finger Extension",
    "Little Finger Extension", "Index Finger Extension", "Thumb Finger Extension",
    "Wrist Extension", "Wrist Flexion", "Forearm Supination", "Forearm Pronation",
    "Hand Open", "Hand Close", "Rest",
]
# Short labels in the same (MotionSequence.txt) order, for plots and confusion matrices.
GESTURE_SHORT = ["LP", "TA", "TLFO", "TIFO", "TIFE", "TLFE", "IMFE", "LFE", "IFE", "TE",
                 "WE", "WF", "FS", "FP", "HO", "HC", "Rest"]


def record_relpath(session: int, participant: int, gesture: int, trial: int) -> str:
    """Return the path of a record (without extension) relative to the dataset root."""
    folder = f"Session{session}/session{session}_participant{participant}"
    return f"{folder}/session{session}_participant{participant}_gesture{gesture}_trial{trial}"


def _load_checksums(raw_dir: Path) -> dict[str, str]:
    """Read SHA256SUMS.txt (downloading it if needed) into {relpath: sha256}."""
    path = raw_dir / "SHA256SUMS.txt"
    if not path.exists():
        urllib.request.urlretrieve(f"{BASE_URL}/SHA256SUMS.txt", path)
    sums = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    return sums


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(relpath: str, raw_dir: Path, expected: str | None, retries: int = 5) -> str:
    """Download one file, skipping it if it already exists with the right checksum."""
    dst = raw_dir / relpath
    if dst.exists() and (expected is None or _sha256(dst) == expected):
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{relpath}", tmp)
            if expected is not None and _sha256(tmp) != expected:
                raise IOError(f"checksum mismatch: {relpath}")
            tmp.replace(dst)
            return "ok"
        except Exception:
            if attempt == retries - 1:
                raise
    return "fail"


def download(participants: list[int], sessions: list[int] | None = None,
             raw_dir: Path = RAW_DIR, workers: int = 8) -> None:
    """Download all trials (.dat + .hea) of the given participants and sessions, verified by SHA256."""
    sessions = sessions or list(range(1, N_SESSIONS + 1))
    sums = _load_checksums(raw_dir)
    files = [
        record_relpath(s, p, g, t) + ext
        for s in sessions for p in participants
        for g in range(1, N_GESTURES + 1) for t in range(1, N_TRIALS + 1)
        for ext in (".hea", ".dat")
    ]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, f, raw_dir, sums.get(f)): f for f in files}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 200 == 0 or done == len(files):
                print(f"{done}/{len(files)} files", flush=True)


def build_index(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Index all locally available trials: one row per record with participant, session, gesture, trial, path."""
    pattern = re.compile(r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)\.hea$")
    rows = []
    for hea in raw_dir.glob("Session*/session*_participant*/*.hea"):
        m = pattern.search(hea.name)
        if m and hea.with_suffix(".dat").exists():
            s, p, g, t = map(int, m.groups())
            rows.append({"participant": p, "session": s, "gesture": g, "trial": t,
                         "path": str(hea.with_suffix(""))})
    df = pd.DataFrame(rows, columns=["participant", "session", "gesture", "trial", "path"])
    return df.sort_values(["participant", "session", "gesture", "trial"]).reset_index(drop=True)


def load_trial(session: int, participant: int, gesture: int, trial: int,
               channels: list[int] | None = None, raw_dir: Path = RAW_DIR) -> np.ndarray:
    """Load one trial as a float array [T, C] in mV (default channels: the 12 wrist channels)."""
    rec = wfdb.rdrecord(str(raw_dir / record_relpath(session, participant, gesture, trial)))
    channels = WRIST_IDX if channels is None else channels
    return rec.p_signal[:, channels]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GRABMyo participants.")
    parser.add_argument("--subset", choices=["dev", "test", "all"], default="dev")
    args = parser.parse_args()
    if args.subset == "all":
        participants = list(range(1, 44))
    else:
        splits = yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())
        participants = splits[args.subset]
    print(f"downloading participants {participants}", flush=True)
    download(participants)
