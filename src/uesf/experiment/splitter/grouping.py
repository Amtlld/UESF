"""Dimension grouping utility for 5-D EEG arrays.

Data shape: ``[n_subjects, n_sessions, n_recordings, n_channels, n_samples]``.

Splits only along the requested dimension; all lower axes stay intact within
each returned group.
"""

from __future__ import annotations

import numpy as np

from uesf.core.exceptions import SplitError, TypeMismatchError

VALID_DIMENSIONS = {"subject", "session", "recording", "flatten"}


def get_groups(data: np.ndarray, dimension: str) -> list[np.ndarray]:
    """Group sample indices of a 5-D EEG array by ``dimension``.

    Args:
        data: 5-D ndarray with shape
            ``(n_subjects, n_sessions, n_recordings, n_channels, n_samples)``.
        dimension: One of ``"subject" | "session" | "recording" | "flatten"``.

    Returns:
        List of 1-D index arrays into the flatten_3d sample space
        (size ``n_subjects * n_sessions * n_recordings``).

    Raises:
        TypeMismatchError: For non-5-D input or unknown dimension.
        SplitError: Never directly here; downstream splitters raise when
            ``len(groups) == 1``.
    """
    if data.ndim != 5:
        raise TypeMismatchError(
            f"Data must be 5-D [subject, session, recording, channel, sample], "
            f"got {data.ndim}-D with shape {data.shape}",
            hint="Ensure datasets are loaded in their original 5-D shape.",
        )
    if dimension not in VALID_DIMENSIONS:
        raise TypeMismatchError(
            f"Unknown dimension: '{dimension}'",
            hint=f"Use one of: {sorted(VALID_DIMENSIONS)}",
        )

    n_sub, n_sess, n_rec = data.shape[:3]
    total = n_sub * n_sess * n_rec

    if dimension == "subject":
        step = n_sess * n_rec
        return [np.arange(i * step, (i + 1) * step) for i in range(n_sub)]

    if dimension == "session":
        # A "session group" is all recordings of one (subject, session) pair.
        groups: list[np.ndarray] = []
        for s in range(n_sub):
            for sess in range(n_sess):
                start = (s * n_sess + sess) * n_rec
                groups.append(np.arange(start, start + n_rec))
        return groups

    if dimension == "recording":
        return [np.array([i], dtype=int) for i in range(total)]

    # flatten: every (subject, session, recording) triple is an independent unit
    return [np.array([i], dtype=int) for i in range(total)]


def require_nontrivial(groups: list[np.ndarray], context: str) -> None:
    """Raise ``SplitError`` when there's only one group — nothing to split."""
    if len(groups) < 2:
        raise SplitError(
            f"{context}: got {len(groups)} group(s), need at least 2 to split.",
            hint="Check that the dimension has more than one unit "
            "(e.g., more than one subject/session/recording).",
        )
