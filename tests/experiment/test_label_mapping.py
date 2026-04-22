"""Tests for experiment-level per-dataset label_mapping."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import ConfigError
from uesf.experiment.alignment import LabelAligner
from uesf.experiment.label_mapping import apply_label_mapping


def _faced_meta() -> dict:
    """Replicate FACED-style 9-class emotion numeric_to_semantic."""
    return {
        "n_classes": 9,
        "numeric_to_semantic": {
            "0": "anger",
            "1": "disgust",
            "2": "fear",
            "3": "sadness",
            "4": "neutrality",
            "5": "amusement",
            "6": "inspiration",
            "7": "joy",
            "8": "tenderness",
        },
    }


def _thuep_meta() -> dict:
    """Replicate THU-EP-style 9-class emotion numeric_to_semantic (capitalized)."""
    return {
        "n_classes": 9,
        "numeric_to_semantic": {
            "0": "Anger",
            "1": "Disgust",
            "2": "Fear",
            "3": "Sadness",
            "4": "Neutral",
            "5": "Amusement",
            "6": "Inspiration",
            "7": "Joy",
            "8": "Tenderness",
        },
    }


_FACED_MAPPING = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "neutrality": "neutral",
    "amusement": "positive",
    "inspiration": "positive",
    "joy": "positive",
    "tenderness": "positive",
}

_THUEP_MAPPING = {
    "Anger": "negative",
    "Disgust": "negative",
    "Fear": "negative",
    "Sadness": "negative",
    "Neutral": "neutral",
    "Amusement": "positive",
    "Inspiration": "positive",
    "Joy": "positive",
    "Tenderness": "positive",
}


class TestApplyLabelMappingBasic:
    def test_remaps_labels_and_updates_metadata(self):
        labels = np.array([0, 1, 4, 5, 8, 3, 7, 6, 2], dtype=np.int64)
        labels_cache = {"faced": labels.copy()}
        metadata_cache = {"faced": _faced_meta()}
        datasets_config = {
            "faced": {"name": "faced_test", "label_mapping": dict(_FACED_MAPPING)}
        }

        apply_label_mapping(datasets_config, labels_cache, metadata_cache)

        # ASCII-sorted new semantics: negative < neutral < positive
        # anger=0→negative=0, disgust=1→negative=0, neutrality=4→neutral=1,
        # amusement=5→positive=2, tenderness=8→positive=2, sadness=3→negative=0,
        # joy=7→positive=2, inspiration=6→positive=2, fear=2→negative=0
        expected = np.array([0, 0, 1, 2, 2, 0, 2, 2, 0], dtype=np.int64)
        np.testing.assert_array_equal(labels_cache["faced"], expected)

        meta = metadata_cache["faced"]
        assert meta["n_classes"] == 3
        assert meta["numeric_to_semantic"] == {
            "0": "negative",
            "1": "neutral",
            "2": "positive",
        }

    def test_preserves_dtype(self):
        labels_cache = {"a": np.array([0, 1], dtype=np.int32)}
        metadata_cache = {
            "a": {
                "n_classes": 2,
                "numeric_to_semantic": {"0": "left", "1": "right"},
            }
        }
        datasets_config = {
            "a": {"label_mapping": {"left": "foo", "right": "bar"}}
        }
        apply_label_mapping(datasets_config, labels_cache, metadata_cache)
        assert labels_cache["a"].dtype == np.int32

    def test_alias_without_mapping_untouched(self):
        labels_a = np.array([0, 1, 2])
        labels_b = np.array([3, 4, 5])
        labels_cache = {"a": labels_a.copy(), "b": labels_b.copy()}
        metadata_cache = {
            "a": {
                "n_classes": 3,
                "numeric_to_semantic": {"0": "x", "1": "y", "2": "z"},
            },
            "b": {
                "n_classes": 3,
                "numeric_to_semantic": {"3": "x", "4": "y", "5": "z"},
            },
        }
        datasets_config = {
            "a": {"label_mapping": {"x": "u", "y": "v", "z": "v"}},
            "b": {},
        }
        apply_label_mapping(datasets_config, labels_cache, metadata_cache)
        np.testing.assert_array_equal(labels_cache["b"], labels_b)
        assert metadata_cache["b"]["n_classes"] == 3

    def test_empty_mapping_is_noop(self):
        labels_cache = {"a": np.array([0, 1])}
        metadata_cache = {
            "a": {"n_classes": 2, "numeric_to_semantic": {"0": "l", "1": "r"}}
        }
        datasets_config = {"a": {"label_mapping": {}}}
        apply_label_mapping(datasets_config, labels_cache, metadata_cache)
        assert metadata_cache["a"]["n_classes"] == 2


class TestApplyLabelMappingMultiDataset:
    def test_heterogeneous_sources_align_to_shared_target_set(self):
        """FACED ↔ THU-EP: different source semantics, same target semantics."""
        faced_labels = np.arange(9, dtype=np.int64)  # 0..8
        thuep_labels = np.arange(9, dtype=np.int64)
        labels_cache = {
            "faced": faced_labels.copy(),
            "thuep": thuep_labels.copy(),
        }
        metadata_cache = {"faced": _faced_meta(), "thuep": _thuep_meta()}
        datasets_config = {
            "faced": {"label_mapping": dict(_FACED_MAPPING)},
            "thuep": {"label_mapping": dict(_THUEP_MAPPING)},
        }

        apply_label_mapping(datasets_config, labels_cache, metadata_cache)

        # Both datasets end up with identical post-mapping n2s — this is what
        # allows LabelAligner.validate() to succeed.
        assert (
            metadata_cache["faced"]["numeric_to_semantic"]
            == metadata_cache["thuep"]["numeric_to_semantic"]
        )
        assert metadata_cache["faced"]["n_classes"] == 3
        assert metadata_cache["thuep"]["n_classes"] == 3

        # Verify LabelAligner is happy with the post-mapping state.
        LabelAligner().validate(metadata_cache)

        # Sanity: both remapped label arrays must cover the full {0,1,2} space.
        assert set(np.unique(labels_cache["faced"]).tolist()) == {0, 1, 2}
        assert set(np.unique(labels_cache["thuep"]).tolist()) == {0, 1, 2}


class TestApplyLabelMappingErrors:
    def test_missing_numeric_to_semantic_raises(self):
        labels_cache = {"a": np.array([0, 1])}
        metadata_cache = {"a": {"n_classes": 2}}
        datasets_config = {"a": {"label_mapping": {"x": "foo"}}}
        with pytest.raises(ConfigError, match="numeric_to_semantic"):
            apply_label_mapping(datasets_config, labels_cache, metadata_cache)

    def test_missing_key_raises(self):
        """Mapping must cover every semantic in numeric_to_semantic."""
        labels_cache = {"a": np.array([0, 1, 2])}
        metadata_cache = {
            "a": {
                "n_classes": 3,
                "numeric_to_semantic": {"0": "x", "1": "y", "2": "z"},
            }
        }
        datasets_config = {"a": {"label_mapping": {"x": "foo", "y": "bar"}}}
        with pytest.raises(ConfigError, match="Missing.*'z'"):
            apply_label_mapping(datasets_config, labels_cache, metadata_cache)

    def test_extra_key_raises(self):
        labels_cache = {"a": np.array([0, 1])}
        metadata_cache = {
            "a": {
                "n_classes": 2,
                "numeric_to_semantic": {"0": "x", "1": "y"},
            }
        }
        datasets_config = {
            "a": {"label_mapping": {"x": "foo", "y": "bar", "zzz": "baz"}}
        }
        with pytest.raises(ConfigError, match="Extra.*'zzz'"):
            apply_label_mapping(datasets_config, labels_cache, metadata_cache)


class TestApplyLabelMappingMaskedSource:
    def test_double_remap_on_masked_dataset_semantics(self):
        """A dataset that was already remapped once (e.g. via masked_datasets)
        can be remapped a second time — its numeric_to_semantic is just the
        currently-effective semantic space."""
        # Pretend this dataset was already collapsed to {negative, neutral, positive}.
        labels_cache = {"m": np.array([0, 0, 1, 2, 2, 1], dtype=np.int64)}
        metadata_cache = {
            "m": {
                "n_classes": 3,
                "numeric_to_semantic": {
                    "0": "negative",
                    "1": "neutral",
                    "2": "positive",
                },
            }
        }
        # Collapse further to binary (affect vs neutral).
        datasets_config = {
            "m": {
                "label_mapping": {
                    "negative": "affect",
                    "positive": "affect",
                    "neutral": "neutral",
                }
            }
        }
        apply_label_mapping(datasets_config, labels_cache, metadata_cache)

        # ASCII-sorted: affect < neutral → affect=0, neutral=1
        expected = np.array([0, 0, 1, 0, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(labels_cache["m"], expected)
        assert metadata_cache["m"]["n_classes"] == 2
        assert metadata_cache["m"]["numeric_to_semantic"] == {
            "0": "affect",
            "1": "neutral",
        }
