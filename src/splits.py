"""Train / test splits E1-E4. Splits are by trial, session and participant only, never by window.

    E1  within session:   trials 1-5            -> trials 6-7 of the same participant and session
    E2  across days:      session 1             -> sessions 2 and 3 of the same participant
    E3  across people:    other participants, all sessions -> held-out participants, all sessions
    E4  people + days:    other participants, session 1    -> held-out participants, sessions 2 and 3

For E3/E4 the held-out participants are the fixed test set (`holdout`, final runs), each participant in turn
(`lopo`, leave-one-participant-out) or groups of participants (`kfold`: people shuffled with a fixed seed and cut
into `n_folds` groups, e.g. 10 dev people -> 5 folds of 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    experiment: str
    name: str
    train: np.ndarray      # boolean mask over rows of the feature table
    test: np.ndarray


def make_splits(meta: pd.DataFrame, experiment: str, cross_subject: str = "lopo",
                test_participants: list[int] | None = None, n_folds: int = 5, seed: int = 42) -> list[Split]:
    """Build the splits of one experiment.

    Args:
        meta: table with participant, session, trial columns (one row per window).
        experiment: "E1" .. "E4".
        cross_subject: "lopo", "kfold" or "holdout" (E3/E4 only).
        test_participants: held-out participants for "holdout".
        n_folds: number of participant groups for "kfold".
        seed: shuffles the participants before they are cut into folds.

    Returns:
        List of splits; within-person experiments have one split per participant (E2) or participant-session (E1).
    """
    p, s, t = meta.participant.to_numpy(), meta.session.to_numpy(), meta.trial.to_numpy()
    people = sorted(np.unique(p))
    if experiment == "E1":
        return [Split("E1", f"p{q}/s{ss}", (p == q) & (s == ss) & (t <= 5), (p == q) & (s == ss) & (t >= 6))
                for q in people for ss in sorted(np.unique(s[p == q]))]
    if experiment == "E2":
        return [Split("E2", f"p{q}", (p == q) & (s == 1), (p == q) & (s > 1)) for q in people]
    if experiment in ("E3", "E4"):
        if cross_subject == "lopo":
            groups = [[q] for q in people]
        elif cross_subject == "kfold":
            shuffled = np.random.default_rng(seed).permutation(people)
            groups = [sorted(g.tolist()) for g in np.array_split(shuffled, n_folds)]
        elif cross_subject == "holdout":
            groups = [list(test_participants)]
        else:
            raise ValueError(cross_subject)
        out = []
        for i, test_ps in enumerate(groups):
            is_test = np.isin(p, test_ps)
            if cross_subject == "kfold":
                name = f"fold{i + 1}"
            elif len(test_ps) == 1:
                name = f"p{test_ps[0]}"
            else:
                name = f"test {[int(q) for q in test_ps]}"
            if experiment == "E3":
                out.append(Split("E3", name, ~is_test, is_test))
            else:
                out.append(Split("E4", name, ~is_test & (s == 1), is_test & (s > 1)))
        return out
    raise ValueError(experiment)
