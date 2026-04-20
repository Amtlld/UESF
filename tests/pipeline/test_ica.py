"""Tests for the ICA data-stream operator."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import MissingRequiredKeyError, ShapeMismatchError
from uesf.pipeline.operators.ica import ica

CH_NAMES_8 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]


def _make_eeg(n_sessions=1, n_recordings=1, n_channels=8, n_samples=600, seed=0):
    """Synthesize mixed sinusoidal sources + noise — ICA-friendly."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 200.0  # 200 Hz
    data = np.empty((n_sessions, n_recordings, n_channels, n_samples), dtype=np.float32)
    for s in range(n_sessions):
        for r in range(n_recordings):
            # Build a few independent sources and mix them across channels.
            sources = np.stack(
                [
                    np.sin(2 * np.pi * 10 * t),
                    np.sin(2 * np.pi * 20 * t + 0.5),
                    np.sign(np.sin(2 * np.pi * 5 * t)),
                    rng.standard_normal(n_samples) * 0.3,
                ],
                axis=0,
            )
            mixing = rng.standard_normal((n_channels, sources.shape[0]))
            mixed = mixing @ sources + rng.standard_normal((n_channels, n_samples)) * 0.1
            data[s, r] = mixed.astype(np.float32)
    return data


class TestICABasic:
    def test_shape_and_dtype_preserved(self):
        data = _make_eeg()
        out, sr = ica(
            data,
            200.0,
            {"method": "infomax", "n_components": 4, "random_state": 0, "max_iter": 200},
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        assert out.shape == data.shape
        assert out.dtype == data.dtype
        assert sr == 200.0

    def test_uses_context_electrode_list(self):
        data = _make_eeg()
        out, _ = ica(
            data,
            200.0,
            {"method": "infomax", "n_components": 4, "random_state": 0, "max_iter": 200},
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        assert out.shape == data.shape

    def test_params_ch_names_overrides_context(self):
        data = _make_eeg()
        custom = [f"ch{i}" for i in range(8)]
        out, _ = ica(
            data,
            200.0,
            {
                "method": "infomax",
                "ch_names": custom,
                "n_components": 4,
                "random_state": 0,
                "max_iter": 200,
            },
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        assert out.shape == data.shape


class TestICAErrors:
    def test_missing_ch_names_raises(self):
        data = _make_eeg()
        with pytest.raises(MissingRequiredKeyError):
            ica(data, 200.0, {"n_components": 4}, context=None)

    def test_ch_names_length_mismatch_raises(self):
        data = _make_eeg(n_channels=8)
        with pytest.raises(ShapeMismatchError):
            ica(
                data,
                200.0,
                {"ch_names": ["A", "B", "C"], "n_components": 2},
                context=None,
            )

    def test_eog_channel_not_in_ch_names_raises(self):
        data = _make_eeg()
        with pytest.raises(MissingRequiredKeyError):
            ica(
                data,
                200.0,
                {"eog_ch_names": ["NotAChannel"], "n_components": 4},
                context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
            )


class TestICAFigures:
    def test_no_figures_dir_skips_files(self, tmp_path):
        data = _make_eeg()
        ica(
            data,
            200.0,
            {
                "method": "infomax",
                "n_components": 4,
                "random_state": 0,
                "max_iter": 200,
                "eog_ch_names": ["Fp1"],
            },
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        assert list(tmp_path.iterdir()) == []

    def test_figures_dir_creates_eog_scores(self, tmp_path):
        data = _make_eeg()
        figs_dir = tmp_path / "ica_figs"
        ica(
            data,
            200.0,
            {
                "method": "infomax",
                "n_components": 4,
                "random_state": 0,
                "max_iter": 200,
                "eog_ch_names": ["Fp1", "Fp2"],
                "figures_dir": str(figs_dir),
            },
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        assert figs_dir.exists()
        files = {p.name for p in figs_dir.iterdir()}
        assert "sub01_s0_r0_eog_Fp1_scores.png" in files
        assert "sub01_s0_r0_eog_Fp2_scores.png" in files

    def test_relative_figures_dir_resolved_against_cwd(self, tmp_path, monkeypatch):
        data = _make_eeg()
        monkeypatch.chdir(tmp_path)
        ica(
            data,
            200.0,
            {
                "method": "infomax",
                "n_components": 4,
                "random_state": 0,
                "max_iter": 200,
                "eog_ch_names": ["Fp1"],
                "figures_dir": "ica_figs_rel",
            },
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        resolved = (tmp_path / "ica_figs_rel").resolve()
        assert resolved.exists()
        assert (resolved / "sub01_s0_r0_eog_Fp1_scores.png").exists()


class TestICAMNEPassthrough:
    def test_deterministic_with_random_state(self):
        data = _make_eeg(seed=7)
        params = {
            "method": "infomax",
            "n_components": 4,
            "random_state": 123,
            "max_iter": 200,
        }
        ctx = {"electrode_list": CH_NAMES_8, "subject_id": "sub01"}
        out1, _ = ica(data, 200.0, params, context=ctx)
        out2, _ = ica(data, 200.0, params, context=ctx)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_multi_session_recording_iterates(self):
        data = _make_eeg(n_sessions=2, n_recordings=2)
        out, _ = ica(
            data,
            200.0,
            {"method": "infomax", "n_components": 4, "random_state": 0, "max_iter": 200},
            context={"electrode_list": CH_NAMES_8, "subject_id": "sub01"},
        )
        assert out.shape == (2, 2, 8, 600)
