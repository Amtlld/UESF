"""ICA artifact rejection operator (thin wrapper over mne.preprocessing.ICA)."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA as MNE_ICA

from uesf.core.exceptions import MissingRequiredKeyError, ShapeMismatchError

# Keys consumed by UESF; everything else in params is forwarded to mne.preprocessing.ICA().
# Note: MNE's own `fit_params` (ICA constructor arg for algorithm-specific kwargs like
# {extended: true}) is NOT reserved — it passes through to ICA(). UESF's `fit_call_kwargs`
# targets ica.fit() method kwargs (e.g., decim, reject); distinct name avoids confusion.
_UESF_RESERVED_KEYS = frozenset(
    {
        "eog_ch_names",
        "ecg_ch_names",
        "montage",
        "figures_dir",
        "fit_call_kwargs",
        "find_bads_kwargs",
        "ch_names",
    }
)


def ica(
    data: np.ndarray,
    sampling_rate: float,
    params: dict,
    context: dict | None = None,
) -> tuple[np.ndarray, float]:
    """Apply ICA artifact rejection per (session, recording).

    Thin wrapper over ``mne.preprocessing.ICA``. Unrecognized params keys are
    forwarded to ``ICA()`` — users can configure any MNE ICA constructor option
    (``method``, ``n_components``, ``random_state``, ``max_iter``, ``fit_params``,
    ``noise_cov``, ``n_pca_components``, ``allow_ref_meg``, ``verbose``, ...).

    Args:
        data: EEG data (sessions, recordings, channels, samples).
        sampling_rate: Sampling rate in Hz.
        params: UESF-reserved keys:
            - eog_ch_names: list[str] | None — EOG reference channel names;
              each is passed to ``find_bads_eog(ch_name=...)``.
            - ecg_ch_names: list[str] | None — ECG reference channel names.
            - montage: str | None — MNE standard montage name (e.g.,
              "standard_1020"); required for sensible ``plot_components``
              topomaps.
            - figures_dir: str | None — where to save scores/components PNGs.
              Absolute path used verbatim; relative path resolved against CWD.
              None / omitted skips figure generation.
            - fit_call_kwargs: dict — forwarded to ``ica.fit(raw, **fit_call_kwargs)``
              (e.g., decim, reject). Distinct from MNE's own ``fit_params`` ICA
              constructor arg, which passes through unchanged.
            - find_bads_kwargs: dict — forwarded to both ``find_bads_eog`` and
              ``find_bads_ecg``.
            - ch_names: list[str] | None — overrides ``context.electrode_list``.
            All other keys are passed as kwargs to ``mne.preprocessing.ICA()``.
        context: Per-subject metadata from the preprocessor:
            - electrode_list: list[str] | None
            - subject_id: str (used for figure filename prefix)

    Returns:
        (cleaned_data, sampling_rate)
    """
    mne.set_log_level("WARNING")

    ctx = context or {}
    ch_names = params.get("ch_names") or ctx.get("electrode_list")
    if not ch_names:
        raise MissingRequiredKeyError(
            "ICA operator requires channel names; none found in params or context",
            context={"params_has_ch_names": "ch_names" in params},
            hint=(
                "Provide 'electrode_list' in the raw dataset's raw.yml so the "
                "preprocessor can inject it, or set params.ch_names explicitly."
            ),
        )

    n_channels = data.shape[-2]
    if len(ch_names) != n_channels:
        raise ShapeMismatchError(
            f"ICA: len(ch_names)={len(ch_names)} does not match data channel count={n_channels}",
            context={"ch_names": list(ch_names), "data_shape": list(data.shape)},
        )

    eog_ch_names = list(params.get("eog_ch_names") or [])
    ecg_ch_names = list(params.get("ecg_ch_names") or [])
    montage = params.get("montage")
    figures_dir_raw = params.get("figures_dir")
    fit_call_kwargs = dict(params.get("fit_call_kwargs") or {})
    find_bads_kwargs = dict(params.get("find_bads_kwargs") or {})

    ch_name_set = set(ch_names)
    for ch in eog_ch_names + ecg_ch_names:
        if ch not in ch_name_set:
            raise MissingRequiredKeyError(
                f"ICA: reference channel '{ch}' is not in ch_names",
                context={"ch_names": list(ch_names), "requested": ch},
                hint="Ensure eog_ch_names/ecg_ch_names use names present in electrode_list.",
            )

    mne_ica_kwargs = {k: v for k, v in params.items() if k not in _UESF_RESERVED_KEYS}

    figures_dir: Path | None = None
    if figures_dir_raw:
        figures_dir = Path(figures_dir_raw)
        if not figures_dir.is_absolute():
            figures_dir = figures_dir.resolve()
        figures_dir.mkdir(parents=True, exist_ok=True)

    subject_id = ctx.get("subject_id", "unknown")
    n_sessions, n_recordings = data.shape[0], data.shape[1]
    cleaned = np.empty_like(data)

    for s in range(n_sessions):
        for r in range(n_recordings):
            recording = data[s, r].astype(np.float64)

            info = mne.create_info(
                ch_names=list(ch_names),
                sfreq=sampling_rate,
                ch_types="eeg",
            )
            raw = mne.io.RawArray(recording, info, verbose="WARNING")

            if montage:
                raw.set_montage(mne.channels.make_standard_montage(montage))

            mne_ica = MNE_ICA(**mne_ica_kwargs)
            mne_ica.fit(raw, **fit_call_kwargs)

            exclude_set: set[int] = set()
            eog_scores_per_ch: dict[str, np.ndarray] = {}
            for ch in eog_ch_names:
                inds, scores = mne_ica.find_bads_eog(raw, ch_name=ch, **find_bads_kwargs)
                exclude_set.update(inds)
                eog_scores_per_ch[ch] = scores

            ecg_scores_per_ch: dict[str, np.ndarray] = {}
            for ch in ecg_ch_names:
                inds, scores = mne_ica.find_bads_ecg(raw, ch_name=ch, **find_bads_kwargs)
                exclude_set.update(inds)
                ecg_scores_per_ch[ch] = scores

            mne_ica.exclude = sorted(exclude_set)

            if figures_dir is not None:
                _save_ica_figures(
                    mne_ica=mne_ica,
                    eog_scores=eog_scores_per_ch,
                    ecg_scores=ecg_scores_per_ch,
                    figures_dir=figures_dir,
                    subject_id=subject_id,
                    session_idx=s,
                    recording_idx=r,
                )

            mne_ica.apply(raw)
            cleaned[s, r] = raw.get_data().astype(data.dtype)

    return cleaned, sampling_rate


def _save_ica_figures(
    mne_ica: MNE_ICA,
    eog_scores: dict[str, np.ndarray],
    ecg_scores: dict[str, np.ndarray],
    figures_dir: Path,
    subject_id: str,
    session_idx: int,
    recording_idx: int,
) -> None:
    import matplotlib.pyplot as plt

    prefix = f"{subject_id}_s{session_idx}_r{recording_idx}"

    for ch, scores in eog_scores.items():
        fig = mne_ica.plot_scores(scores, show=False)
        fig.savefig(figures_dir / f"{prefix}_eog_{ch}_scores.png")
        plt.close(fig)

    for ch, scores in ecg_scores.items():
        fig = mne_ica.plot_scores(scores, show=False)
        fig.savefig(figures_dir / f"{prefix}_ecg_{ch}_scores.png")
        plt.close(fig)

    if mne_ica.exclude:
        figs = mne_ica.plot_components(picks=mne_ica.exclude, show=False)
        figs_list = figs if isinstance(figs, list) else [figs]
        for idx, fig in enumerate(figs_list):
            suffix = "" if len(figs_list) == 1 else f"_{idx}"
            fig.savefig(figures_dir / f"{prefix}_excluded_components{suffix}.png")
            plt.close(fig)
