"""Tests for ConfigValidator — R1–R30 normalization and validation rules."""

from __future__ import annotations

import pytest

from uesf.core.exceptions import (
    ConfigError,
    MissingRequiredKeyError,
    TypeMismatchError,
)
from uesf.experiment.config_schema import ConfigValidator


def _regular_holdout(datasets_count=1, **split_overrides):
    datasets = {f"ds{i}": {"name": f"ds{i}_pre"} for i in range(datasets_count)}
    split = {
        "strategy": "holdout",
        "dimension": "subject",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }
    split.update(split_overrides)
    return {
        "experiment_name": "e",
        "datasets": datasets,
        "split": split,
        "model": {"name": "m"},
    }


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_defaults_seed_mode_shuffle(self):
        cfg = ConfigValidator.normalize(_regular_holdout())
        assert cfg["seed"] == 42
        assert cfg["mode"] == "regular"
        assert cfg["split"]["shuffle"] is True  # R27

    def test_legacy_k_folds_renames_to_k(self):
        raw = _regular_holdout()
        raw["split"] = {
            "strategy": "k-fold",
            "dimension": "subject",
            "k-folds": 5,
        }
        cfg = ConfigValidator.normalize(raw)
        assert cfg["split"]["k"] == 5
        assert "k-folds" not in cfg["split"]

    def test_kfold_val_ratio_conversion(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a_pre"}},
            "split": {"strategy": "k-fold", "dimension": "subject", "k": 5, "val_ratio": 0.1},
        }
        cfg = ConfigValidator.normalize(raw)
        # val_ratio_in_train = 0.1 / (1 - 1/5) = 0.125
        assert abs(cfg["split"]["val_ratio_in_train"] - 0.125) < 1e-9

    def test_val_split_shuffle_default(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a_pre"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.8,
                "test_ratio": 0.2,
                "val_split": {"dimension": "session", "val_ratio": 0.1},
            },
        }
        cfg = ConfigValidator.normalize(raw)
        assert cfg["split"]["val_split"]["shuffle"] is True

    def test_uda_domain_shuffle_default(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a_pre"}},
            "mode": "uda",
            "uda": {
                "domain": {
                    "strategy": "holdout",
                    "dimension": "subject",
                    "target_count": 1,
                },
                "adaptation": "transductive",
            },
            "model": {"name": "m"},
        }
        cfg = ConfigValidator.normalize(raw)
        assert cfg["uda"]["domain"]["shuffle"] is True

    def test_normalize_does_not_mutate_input(self):
        raw = _regular_holdout()
        before_split = dict(raw["split"])
        ConfigValidator.normalize(raw)
        assert raw["split"] == before_split


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class TestValidateBasic:
    def test_valid_regular_holdout_passes(self):
        cfg = ConfigValidator.normalize(_regular_holdout())
        ConfigValidator.validate(cfg)  # no exception

    def test_mode_regular_without_split_r1(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "model": {"name": "m"},
        }
        with pytest.raises(MissingRequiredKeyError, match="split"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_mode_regular_with_uda_block_r1(self):
        raw = _regular_holdout()
        raw["uda"] = {}
        with pytest.raises(ConfigError, match="must not contain 'uda'"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_mode_uda_without_uda_block_r2(self):
        raw = {
            "experiment_name": "e",
            "mode": "uda",
            "datasets": {"a": {"name": "a"}},
            "model": {"name": "m"},
        }
        with pytest.raises(MissingRequiredKeyError, match="uda"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))


class TestRegularSplitRules:
    def test_r3_dataset_dim_requires_2_datasets(self):
        raw = _regular_holdout(
            datasets_count=1,
            dimension="dataset",
            assign={"train": ["ds0"], "test": []},
        )
        with pytest.raises(ConfigError, match="at least 2 datasets"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r6_holdout_ratio_sum(self):
        raw = _regular_holdout(train_ratio=0.6, val_ratio=0.1, test_ratio=0.1)
        with pytest.raises(TypeMismatchError, match="≈ 1.0"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r6_holdout_with_val_split_two_way(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.8,
                "test_ratio": 0.2,
                "val_split": {"dimension": "session", "val_ratio": 0.1},
            },
            "model": {"name": "m"},
        }
        ConfigValidator.validate(ConfigValidator.normalize(raw))  # ok

    def test_r6_holdout_with_val_split_bad_sum(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.8,
                "test_ratio": 0.3,
                "val_split": {"dimension": "session", "val_ratio": 0.1},
            },
            "model": {"name": "m"},
        }
        with pytest.raises(TypeMismatchError, match="≈ 1.0"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r7_invalid_k(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "split": {"strategy": "k-fold", "dimension": "subject", "k": 1},
            "model": {"name": "m"},
        }
        with pytest.raises(TypeMismatchError, match="k > 1"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r13_dataset_holdout_missing_assign(self):
        raw = _regular_holdout(datasets_count=2, dimension="dataset")
        # remove ratios since they are irrelevant for dataset holdout
        raw["split"] = {"strategy": "holdout", "dimension": "dataset"}
        with pytest.raises(MissingRequiredKeyError, match="assign"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r13_assign_missing_alias(self):
        raw = _regular_holdout(datasets_count=3, dimension="dataset")
        raw["split"] = {
            "strategy": "holdout",
            "dimension": "dataset",
            "assign": {"train": ["ds0"], "test": ["ds1"]},
        }
        with pytest.raises(ConfigError, match="must cover"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r13_assign_no_val_key(self):
        raw = _regular_holdout(datasets_count=3, dimension="dataset")
        raw["split"] = {
            "strategy": "holdout",
            "dimension": "dataset",
            "assign": {"train": ["ds0"], "val": ["ds1"], "test": ["ds2"]},
        }
        with pytest.raises(ConfigError, match="val"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r14_dataset_kfold_k_must_match_n(self):
        raw = _regular_holdout(datasets_count=3, dimension="dataset")
        raw["split"] = {"strategy": "k-fold", "dimension": "dataset", "k": 2}
        with pytest.raises(TypeMismatchError, match="R14"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r14_dataset_kfold_k_minus_one_ok(self):
        raw = _regular_holdout(datasets_count=3, dimension="dataset")
        raw["split"] = {"strategy": "k-fold", "dimension": "dataset", "k": -1}
        ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r15_val_ratio_out_of_range(self):
        raw = _regular_holdout()
        raw["split"]["val_ratio"] = 1.5
        with pytest.raises(TypeMismatchError, match="R15"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r19_val_split_and_val_ratio_mutual_exclusion(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "val_split": {"dimension": "session", "val_ratio": 0.1},
            },
            "model": {"name": "m"},
        }
        with pytest.raises(ConfigError, match="R19"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r20_dataset_dim_requires_val_split(self):
        raw = _regular_holdout(datasets_count=3, dimension="dataset")
        raw["split"] = {
            "strategy": "holdout",
            "dimension": "dataset",
            "assign": {"train": ["ds0", "ds1"], "test": ["ds2"]},
            "val_ratio": 0.1,
        }
        with pytest.raises(ConfigError, match="R20"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r21_val_split_val_ratio_zero_rejected(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.8,
                "test_ratio": 0.2,
                "val_split": {"dimension": "session", "val_ratio": 0.0},
            },
            "model": {"name": "m"},
        }
        with pytest.raises(TypeMismatchError, match="R21"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r25_multi_dataset_requires_dataset_dim(self):
        raw = _regular_holdout(datasets_count=2)  # dimension=subject by default
        with pytest.raises(ConfigError, match="R25"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r26_flatten_requires_shuffle(self):
        raw = _regular_holdout(dimension="flatten", shuffle=False)
        with pytest.raises(TypeMismatchError, match="R26"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r26_flatten_default_shuffle_passes(self):
        raw = _regular_holdout(dimension="flatten")
        ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r30_flatten_main_requires_flatten_val_split(self):
        raw = {
            "experiment_name": "e",
            "datasets": {"a": {"name": "a"}},
            "split": {
                "strategy": "holdout",
                "dimension": "flatten",
                "train_ratio": 0.8,
                "test_ratio": 0.2,
                "val_split": {"dimension": "session", "val_ratio": 0.1},
            },
            "model": {"name": "m"},
        }
        with pytest.raises(ConfigError, match="R30"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))


class TestUDARules:
    def _base_uda(self, **overrides):
        raw = {
            "experiment_name": "u",
            "datasets": {"a": {"name": "a"}},
            "mode": "uda",
            "uda": {
                "domain": {
                    "strategy": "holdout",
                    "dimension": "subject",
                    "target_count": 1,
                },
                "adaptation": "transductive",
            },
            "model": {"name": "m"},
        }
        raw["uda"].update(overrides)
        return raw

    def test_valid_uda_transductive(self):
        raw = self._base_uda()
        ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r4_inductive_requires_target_split(self):
        raw = self._base_uda(adaptation="inductive")
        with pytest.raises(MissingRequiredKeyError, match="target.split"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r5_transductive_rejects_target_split(self):
        raw = self._base_uda(
            target={
                "split": {
                    "strategy": "holdout",
                    "dimension": "recording",
                    "train_ratio": 0.7,
                    "test_ratio": 0.3,
                }
            },
        )
        with pytest.raises(ConfigError, match="R5"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r8_cross_dataset_holdout_needs_source_target(self):
        raw = {
            "experiment_name": "u",
            "datasets": {"a": {"name": "a"}, "b": {"name": "b"}},
            "mode": "uda",
            "uda": {
                "domain": {
                    "strategy": "holdout",
                    "dimension": "dataset",
                },
                "adaptation": "transductive",
            },
            "model": {"name": "m"},
        }
        with pytest.raises(MissingRequiredKeyError, match="R8"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r9_subject_dimension_multi_dataset_rejected(self):
        raw = {
            "experiment_name": "u",
            "datasets": {"a": {"name": "a"}, "b": {"name": "b"}},
            "mode": "uda",
            "uda": {
                "domain": {
                    "strategy": "holdout",
                    "dimension": "subject",
                    "target_count": 1,
                },
                "adaptation": "transductive",
            },
            "model": {"name": "m"},
        }
        with pytest.raises(ConfigError, match="R25|R9"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r10_source_split_no_strategy(self):
        raw = self._base_uda(
            source={
                "split": {
                    "dimension": "recording",
                    "val_ratio": 0.2,
                    "strategy": "holdout",
                }
            },
        )
        with pytest.raises(ConfigError, match="R10"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r11_domain_dim_equals_source_dim_rejected(self):
        raw = self._base_uda(
            source={
                "split": {"dimension": "subject", "val_ratio": 0.2},
            },
        )
        with pytest.raises(ConfigError, match="R11"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r11_domain_dim_equals_target_dim_rejected(self):
        raw = self._base_uda(
            adaptation="inductive",
            target={
                "split": {
                    "strategy": "holdout",
                    "dimension": "subject",
                    "train_ratio": 0.7,
                    "test_ratio": 0.3,
                }
            },
        )
        with pytest.raises(ConfigError, match="R11"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r12_target_count_and_ratio_mutual_exclusion(self):
        raw = self._base_uda()
        raw["uda"]["domain"]["target_ratio"] = 0.2
        with pytest.raises(ConfigError, match="R12"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r23_uda_transforms_global_rejected(self):
        raw = self._base_uda()
        raw["datasets"]["a"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "global"}
        ]
        with pytest.raises(ConfigError, match="R23"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r24_transductive_rejects_early_stopping(self):
        raw = self._base_uda()
        raw["training"] = {"epochs": 10, "early_stopping": {"metric": "val_acc"}}
        with pytest.raises(ConfigError, match="R24"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r24_transductive_rejects_checkpoint(self):
        raw = self._base_uda()
        raw["training"] = {"epochs": 10, "checkpoint": {"metric": "val_acc"}}
        with pytest.raises(ConfigError, match="R24"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r28_domain_dim_recording_rejected(self):
        raw = self._base_uda()
        raw["uda"]["domain"]["dimension"] = "recording"
        with pytest.raises(TypeMismatchError, match="R28"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))


class TestTransformsRules:
    def test_r22_invalid_scope(self):
        raw = _regular_holdout()
        raw["datasets"]["ds0"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "per_fold"}
        ]
        with pytest.raises(TypeMismatchError, match="R22"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r29_cross_dataset_pipeline_mismatch_length(self):
        raw = _regular_holdout(
            datasets_count=3, dimension="dataset",
        )
        raw["split"] = {
            "strategy": "holdout",
            "dimension": "dataset",
            "assign": {"train": ["ds0", "ds1"], "test": ["ds2"]},
        }
        raw["datasets"]["ds0"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "per_dataset"}
        ]
        raw["datasets"]["ds1"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "per_dataset"},
            {"name": "zscore_normalize", "scope": "per_dataset"},
        ]
        raw["datasets"]["ds2"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "per_dataset"}
        ]
        with pytest.raises(ConfigError, match="R29"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r29_cross_dataset_pipeline_scope_mismatch(self):
        raw = _regular_holdout(datasets_count=2, dimension="dataset")
        raw["split"] = {
            "strategy": "holdout",
            "dimension": "dataset",
            "assign": {"train": ["ds0"], "test": ["ds1"]},
        }
        raw["datasets"]["ds0"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "per_dataset"}
        ]
        raw["datasets"]["ds1"]["transforms"] = [
            {"name": "zscore_normalize", "scope": "global"}
        ]
        with pytest.raises(ConfigError, match="R29"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))


class TestLoggingRules:
    def test_r17_invalid_backend(self):
        raw = _regular_holdout()
        raw["training"] = {"epochs": 1, "logging": {"backend": "wandb"}}
        with pytest.raises(TypeMismatchError, match="R17"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r18_invalid_log_every_n_epochs(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_every_n_epochs": 0},
        }
        with pytest.raises(TypeMismatchError, match="R18"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r18_valid_log_every_n_epochs(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_every_n_epochs": 2},
        }
        ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r18a_invalid_log_every_n_steps(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_every_n_steps": 0},
        }
        with pytest.raises(TypeMismatchError, match="R18a"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r18a_string_rejected(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_every_n_steps": "5"},
        }
        with pytest.raises(TypeMismatchError, match="R18a"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r18a_valid_positive_int(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_every_n_steps": 5},
        }
        ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r17b_log_graph_must_be_bool(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_graph": "yes"},
        }
        with pytest.raises(TypeMismatchError, match="R17b"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r17c_log_lr_must_be_bool(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "log_lr": "on"},
        }
        with pytest.raises(TypeMismatchError, match="R17c"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r35_train_step_scalars_must_be_list(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "train_step_scalars": "loss"},
        }
        with pytest.raises(TypeMismatchError, match="R35"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r35_empty_string_rejected(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "train_step_scalars": ["loss", ""]},
        }
        with pytest.raises(TypeMismatchError, match="R35"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r36_train_metrics_must_be_list(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "train_metrics": "accuracy"},
        }
        with pytest.raises(TypeMismatchError, match="R36"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r37_val_metrics_subset_ok(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "val_metrics": ["accuracy"]},
        }
        raw["evaluation"] = {"metrics": ["accuracy", "f1_score"]}
        ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r37_val_metrics_not_subset_rejected(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "val_metrics": ["accuracy", "auroc"]},
        }
        raw["evaluation"] = {"metrics": ["accuracy", "f1_score"]}
        with pytest.raises(ConfigError, match="R37"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r37_val_metrics_requires_evaluation_metrics(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "val_metrics": ["accuracy"]},
        }
        with pytest.raises(ConfigError, match="R37"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_r38_test_metrics_must_be_list(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {"backend": "tensorboard", "test_metrics": "auroc"},
        }
        with pytest.raises(TypeMismatchError, match="R38"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))

    def test_all_new_fields_valid_passes(self):
        raw = _regular_holdout()
        raw["training"] = {
            "epochs": 1,
            "logging": {
                "backend": "tensorboard",
                "log_every_n_epochs": 1,
                "log_every_n_steps": 5,
                "log_graph": False,
                "log_lr": True,
                "train_step_scalars": ["loss"],
                "train_metrics": ["accuracy"],
                "val_metrics": ["accuracy"],
                "test_metrics": ["accuracy"],
            },
        }
        raw["evaluation"] = {"metrics": ["accuracy", "f1_score"]}
        ConfigValidator.validate(ConfigValidator.normalize(raw))


class TestSingleDatasetAlignmentWarning:
    def test_r16_single_dataset_with_alignment_warns(self, caplog):
        raw = _regular_holdout()
        raw["alignment"] = {"channel": "intersection"}
        with caplog.at_level("WARNING", logger="uesf.experiment.config_schema"):
            ConfigValidator.validate(ConfigValidator.normalize(raw))
        assert any("R16" in rec.message or "alignment" in rec.message for rec in caplog.records)
