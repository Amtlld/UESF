"""Experiment config normalization and validation.

:class:`ConfigValidator` implements the 30 rules (R1–R30) listed in
``docs/version_design/0.1.0.dev6_design/07_config_validation_rules.md``.

Responsibilities are split:
- :meth:`ConfigValidator.normalize` fills defaults and rewrites legacy keys.
- :meth:`ConfigValidator.validate` runs R1–R30 on the normalized config.
"""

from __future__ import annotations

import copy

from uesf.core.exceptions import (
    ConfigError,
    MissingRequiredKeyError,
    TypeMismatchError,
)
from uesf.core.logging import get_logger

logger = get_logger("experiment.config_schema")

_RATIO_TOL = 1e-4
_ALLOWED_SCOPES = {"per_dataset", "global"}
_ALLOWED_DOMAIN_DIMS = {"dataset", "subject", "session"}
_ALLOWED_SPLIT_DIMS = {"subject", "session", "recording", "flatten"}
_ALLOWED_TOP_DIMS = _ALLOWED_SPLIT_DIMS | {"dataset"}


class ConfigValidator:
    """Normalize + validate experiment YAML configs."""

    @staticmethod
    def normalize(raw_config: dict) -> dict:
        cfg = copy.deepcopy(raw_config)

        cfg.setdefault("seed", 42)
        cfg.setdefault("mode", "regular")

        _rename_kfold_key(cfg.get("split"))
        # normalize main split block
        if "split" in cfg and cfg["split"] is not None:
            _normalize_split_block(cfg["split"])

        # UDA normalization
        if cfg.get("mode") == "uda":
            uda = cfg.get("uda")
            if uda is None:
                return cfg  # let validate() raise a clear error
            domain = uda.get("domain")
            if isinstance(domain, dict):
                _rename_kfold_key(domain)
                domain.setdefault("shuffle", True)

            # source.split (ValSplit-like)
            source = uda.get("source")
            if isinstance(source, dict) and isinstance(source.get("split"), dict):
                s = source["split"]
                s.setdefault("shuffle", True)
                # explicit flag — ValSplit never has strategy/train_ratio/test_ratio
                s["_role"] = "val_split"

            # target.split
            target = uda.get("target")
            if isinstance(target, dict) and isinstance(target.get("split"), dict):
                _rename_kfold_key(target["split"])
                _normalize_split_block(target["split"])

        return cfg

    @staticmethod
    def validate(config: dict) -> None:
        mode = config.get("mode", "regular")
        if mode not in ("regular", "uda"):
            raise TypeMismatchError(
                f"mode must be 'regular' or 'uda', got {mode!r}",
            )

        datasets = config.get("datasets", {})
        if not isinstance(datasets, dict) or not datasets:
            raise MissingRequiredKeyError(
                "datasets block is required and must declare at least one alias.",
            )

        # R1 / R2
        has_split = "split" in config and config["split"] is not None
        has_uda = "uda" in config and config["uda"] is not None
        if mode == "regular":
            if not has_split:
                raise MissingRequiredKeyError(
                    "mode='regular' requires top-level 'split'.",
                )
            if has_uda:
                raise ConfigError(
                    "mode='regular' must not contain 'uda' block.",
                )
        else:  # uda
            if not has_uda:
                raise MissingRequiredKeyError(
                    "mode='uda' requires 'uda' block.",
                )
            if has_split:
                raise ConfigError(
                    "mode='uda' must not contain top-level 'split'.",
                )

        # transforms checks (R22/R23/R29)
        _validate_transforms(datasets, mode)

        if mode == "regular":
            _validate_regular_split(config["split"], datasets)
        else:
            _validate_uda(config["uda"], datasets, config)

        # training block validation (R17/R18/R24)
        _validate_training(config, mode)

        # R16 — multi-dataset + alignment warning already in caller; single-dataset + alignment warn here
        if len(datasets) == 1 and config.get("alignment"):
            logger.warning(
                "alignment block ignored for single-dataset experiments (R16)."
            )


# ---------------------------------------------------------------------------
# normalize helpers
# ---------------------------------------------------------------------------


def _rename_kfold_key(block: dict | None) -> None:
    if not isinstance(block, dict):
        return
    for legacy in ("k-folds", "k_folds"):
        if legacy in block and "k" not in block:
            block["k"] = block.pop(legacy)
        elif legacy in block:
            block.pop(legacy)


def _normalize_split_block(split: dict) -> None:
    """Fill shuffle defaults (R27) and convert k-fold val_ratio → val_ratio_in_train."""
    split.setdefault("shuffle", True)
    if "val_split" in split and isinstance(split["val_split"], dict):
        split["val_split"].setdefault("shuffle", True)
    if split.get("strategy") == "k-fold":
        k = split.get("k")
        val_ratio = split.get("val_ratio")
        has_val_split = "val_split" in split
        if val_ratio is not None and not has_val_split and isinstance(k, int) and k > 1:
            # val_ratio / (1 - 1/k)  (user expresses val as fraction of whole)
            split["val_ratio_in_train"] = val_ratio / (1 - 1 / k)


# ---------------------------------------------------------------------------
# transforms (R22 / R23 / R29)
# ---------------------------------------------------------------------------


def _validate_transforms(datasets: dict, mode: str) -> None:
    per_alias_steps: dict[str, list[dict]] = {}
    for alias, dcfg in datasets.items():
        steps = (dcfg or {}).get("transforms") or []
        normalized_steps: list[dict] = []
        for step in steps:
            if "name" not in step:
                raise MissingRequiredKeyError(
                    f"dataset '{alias}': transform entry missing 'name'",
                )
            scope = step.get("scope", "per_dataset")
            if scope not in _ALLOWED_SCOPES:
                raise TypeMismatchError(
                    f"dataset '{alias}': transform scope must be one of {sorted(_ALLOWED_SCOPES)}, "
                    f"got {scope!r} (R22)",
                )
            if mode == "uda" and scope == "global":
                raise ConfigError(
                    f"dataset '{alias}': scope='global' not permitted in UDA mode (R23).",
                    hint="Use scope='per_dataset' for UDA transforms.",
                )
            normalized_steps.append({"name": step["name"], "scope": scope, "params": step.get("params", {})})
        per_alias_steps[alias] = normalized_steps

    # R29 — all aliases must present the same pipeline shape when multiple datasets
    if len(datasets) > 1:
        aliases = list(datasets.keys())
        reference_alias = aliases[0]
        reference = per_alias_steps[reference_alias]
        for alias in aliases[1:]:
            steps = per_alias_steps[alias]
            if len(steps) != len(reference):
                raise ConfigError(
                    "Cross-dataset transforms pipeline length mismatch (R29): "
                    f"'{reference_alias}' has {len(reference)} steps, "
                    f"'{alias}' has {len(steps)}.",
                )
            for i, (a, b) in enumerate(zip(reference, steps)):
                if a["name"] != b["name"] or a["scope"] != b["scope"]:
                    raise ConfigError(
                        "Cross-dataset transforms mismatch at position "
                        f"{i} (R29): '{reference_alias}'={a['name']}/{a['scope']} vs "
                        f"'{alias}'={b['name']}/{b['scope']}.",
                    )


# ---------------------------------------------------------------------------
# regular split validation
# ---------------------------------------------------------------------------


def _validate_regular_split(split: dict, datasets: dict) -> None:
    strategy = split.get("strategy")
    dimension = split.get("dimension")
    if strategy not in ("holdout", "k-fold"):
        raise TypeMismatchError(
            f"split.strategy must be 'holdout' or 'k-fold', got {strategy!r}",
        )
    if dimension not in _ALLOWED_TOP_DIMS:
        raise TypeMismatchError(
            f"split.dimension must be one of {sorted(_ALLOWED_TOP_DIMS)}, got {dimension!r}",
        )

    # R26
    if dimension == "flatten" and split.get("shuffle") is False:
        raise TypeMismatchError(
            "split.dimension='flatten' requires shuffle=true (R26).",
        )
    val_split = split.get("val_split")
    if isinstance(val_split, dict):
        if val_split.get("dimension") not in _ALLOWED_SPLIT_DIMS:
            raise TypeMismatchError(
                f"val_split.dimension must be one of {sorted(_ALLOWED_SPLIT_DIMS)}.",
            )
        if val_split.get("dimension") == "flatten" and val_split.get("shuffle") is False:
            raise TypeMismatchError(
                "val_split.dimension='flatten' requires shuffle=true (R26).",
            )
        # R21
        vr = val_split.get("val_ratio")
        if not isinstance(vr, (int, float)) or not (0 < vr < 1):
            raise TypeMismatchError(
                f"val_split.val_ratio must be in (0, 1), got {vr!r} (R21).",
            )
        # R30
        if dimension == "flatten" and val_split["dimension"] != "flatten":
            raise ConfigError(
                "Main split.dimension='flatten' requires val_split.dimension='flatten' (R30).",
            )

    # R19
    if isinstance(val_split, dict) and "val_ratio" in split:
        raise ConfigError(
            "split.val_split and split.val_ratio are mutually exclusive (R19).",
        )

    # ratio sanity (R15 / R6)
    if "val_ratio" in split:
        vr = split["val_ratio"]
        if not isinstance(vr, (int, float)) or not (0 <= vr < 1):
            raise TypeMismatchError(
                f"split.val_ratio must be in [0, 1), got {vr!r} (R15).",
            )

    # R25 multi-dataset requires dimension=dataset
    if len(datasets) > 1 and dimension != "dataset":
        raise ConfigError(
            "Multi-dataset experiments require split.dimension='dataset' (R25).",
            hint="Different datasets have incompatible subject/session namespaces.",
        )

    # R3 — dataset dim requires >=2 datasets
    if dimension == "dataset" and len(datasets) < 2:
        raise ConfigError(
            "split.dimension='dataset' requires at least 2 datasets (R3).",
        )

    if strategy == "holdout":
        if dimension == "dataset":
            # R13 — assign required, covers all datasets
            assign = split.get("assign")
            if not isinstance(assign, dict):
                raise MissingRequiredKeyError(
                    "split.strategy='holdout' + dimension='dataset' requires 'assign' "
                    "with 'train' and 'test' (R13).",
                )
            train = assign.get("train", [])
            test = assign.get("test", [])
            if "val" in assign:
                raise ConfigError(
                    "split.assign must not include 'val' for dimension='dataset' (R13).",
                )
            declared = set(datasets.keys())
            specified = set(train) | set(test)
            if specified - declared:
                raise ConfigError(
                    f"split.assign references unknown aliases: {sorted(specified - declared)}",
                )
            if declared - specified:
                raise ConfigError(
                    f"split.assign must cover every declared dataset (R13); missing: "
                    f"{sorted(declared - specified)}",
                )
            # R20 — no val_ratio with dimension=dataset
            if "val_ratio" in split:
                raise ConfigError(
                    "split.val_ratio not allowed with dimension='dataset' (R20); use val_split.",
                )
        else:
            # Numeric ratio validation (R6)
            if isinstance(val_split, dict):
                # Two-way split: train + test = 1
                train = split.get("train_ratio")
                test = split.get("test_ratio")
                if train is None or test is None:
                    raise MissingRequiredKeyError(
                        "holdout + val_split requires train_ratio and test_ratio.",
                    )
                if abs(train + test - 1.0) > _RATIO_TOL:
                    raise TypeMismatchError(
                        f"train_ratio + test_ratio must be ≈ 1.0 (R6), got {train + test}.",
                    )
            else:
                train = split.get("train_ratio")
                val = split.get("val_ratio", 0.0)
                test = split.get("test_ratio")
                if train is None or test is None:
                    raise MissingRequiredKeyError(
                        "holdout requires train_ratio and test_ratio.",
                    )
                if abs(train + val + test - 1.0) > _RATIO_TOL:
                    raise TypeMismatchError(
                        f"train_ratio + val_ratio + test_ratio must be ≈ 1.0 (R6), "
                        f"got {train + val + test}.",
                    )
    else:  # k-fold
        k = split.get("k")
        if dimension == "dataset":
            # R14
            n_datasets = len(datasets)
            if k != -1 and k != n_datasets:
                raise TypeMismatchError(
                    f"split.strategy='k-fold' + dimension='dataset' requires k == -1 "
                    f"or k == len(datasets) ({n_datasets}), got k={k!r} (R14).",
                )
            if "val_ratio" in split:
                raise ConfigError(
                    "split.val_ratio not allowed with dimension='dataset' (R20); use val_split.",
                )
        else:
            if k is None or (k != -1 and (not isinstance(k, int) or k < 2)):
                raise TypeMismatchError(
                    f"split.strategy='k-fold' requires k > 1 or k=-1, got {k!r} (R7).",
                )
            # R15 — val_ratio_in_train already set by normalize when applicable
            v_in_train = split.get("val_ratio_in_train")
            if v_in_train is not None and not (0 <= v_in_train < 1):
                raise TypeMismatchError(
                    f"k-fold val_ratio_in_train must be in [0, 1), got {v_in_train} (R15).",
                )


# ---------------------------------------------------------------------------
# UDA validation
# ---------------------------------------------------------------------------


def _validate_uda(uda: dict, datasets: dict, full_cfg: dict) -> None:
    domain = uda.get("domain")
    if not isinstance(domain, dict):
        raise MissingRequiredKeyError(
            "uda block requires 'domain' sub-block.",
        )
    adaptation = uda.get("adaptation")
    if adaptation not in ("transductive", "inductive"):
        raise TypeMismatchError(
            f"uda.adaptation must be 'transductive' or 'inductive', got {adaptation!r}",
        )

    dim = domain.get("dimension")
    if dim not in _ALLOWED_DOMAIN_DIMS:
        raise TypeMismatchError(
            f"uda.domain.dimension must be one of {sorted(_ALLOWED_DOMAIN_DIMS)}, got {dim!r} (R28).",
        )

    strategy = domain.get("strategy")
    if strategy not in ("holdout", "k-fold"):
        raise TypeMismatchError(
            f"uda.domain.strategy must be 'holdout' or 'k-fold', got {strategy!r}",
        )

    # R25 inverse for UDA
    if len(datasets) > 1 and dim != "dataset":
        raise ConfigError(
            "Multi-dataset UDA requires uda.domain.dimension='dataset' (R25).",
        )

    # R9 — subject/session requires single dataset
    if dim in ("subject", "session") and len(datasets) != 1:
        raise ConfigError(
            "uda.domain.dimension=subject|session requires exactly one dataset (R9).",
        )

    # R8 — cross-dataset holdout requires explicit source/target
    if dim == "dataset" and strategy == "holdout":
        if not domain.get("source") or not domain.get("target"):
            raise MissingRequiredKeyError(
                "uda.domain (dataset+holdout) requires 'source' (list) and 'target' (alias) (R8).",
            )
        specified = set(domain["source"]) | {domain["target"]}
        unknown = specified - set(datasets.keys())
        if unknown:
            raise ConfigError(
                f"uda.domain references unknown aliases: {sorted(unknown)}",
            )

    # R12 — target_count vs target_ratio
    if dim in ("subject", "session") and strategy == "holdout":
        if ("target_count" in domain) and ("target_ratio" in domain):
            raise ConfigError(
                "uda.domain.target_count and target_ratio are mutually exclusive (R12).",
            )
        if ("target_count" not in domain) and ("target_ratio" not in domain):
            raise MissingRequiredKeyError(
                "uda.domain (holdout + subject/session) requires target_count or target_ratio.",
            )

    if strategy == "k-fold":
        k = domain.get("k")
        if k is None or (k != -1 and (not isinstance(k, int) or k < 2)):
            raise TypeMismatchError(
                f"uda.domain.k must be >1 or -1, got {k!r} (R7).",
            )

    # source (R10) + R11 + R21 (val_ratio)
    source = uda.get("source")
    if source is not None:
        s = source.get("split") if isinstance(source, dict) else None
        if isinstance(s, dict):
            forbidden = {"strategy", "train_ratio", "test_ratio"}.intersection(s)
            if forbidden:
                raise ConfigError(
                    f"uda.source.split must not contain {sorted(forbidden)} (R10).",
                )
            if "dimension" not in s:
                raise MissingRequiredKeyError(
                    "uda.source.split requires 'dimension'.",
                )
            if s["dimension"] not in _ALLOWED_SPLIT_DIMS:
                raise TypeMismatchError(
                    f"uda.source.split.dimension must be one of {sorted(_ALLOWED_SPLIT_DIMS)}.",
                )
            if "val_ratio" not in s:
                raise MissingRequiredKeyError(
                    "uda.source.split requires 'val_ratio'.",
                )
            vr = s["val_ratio"]
            if not isinstance(vr, (int, float)) or not (0 <= vr < 1):
                raise TypeMismatchError(
                    f"uda.source.split.val_ratio must be in [0, 1), got {vr!r} (R15).",
                )
            if s["dimension"] == "flatten" and s.get("shuffle") is False:
                raise TypeMismatchError(
                    "uda.source.split.dimension='flatten' requires shuffle=true (R26).",
                )
            # R11
            if s["dimension"] == dim:
                raise ConfigError(
                    "uda.domain.dimension must differ from uda.source.split.dimension (R11).",
                )

    # target
    target_block = uda.get("target") or {}
    t_split = target_block.get("split") if isinstance(target_block, dict) else None
    if adaptation == "transductive":
        # R5
        if t_split is not None:
            raise ConfigError(
                "uda.target.split must be absent when adaptation='transductive' (R5).",
            )
    else:  # inductive
        # R4
        if not isinstance(t_split, dict):
            raise MissingRequiredKeyError(
                "uda.target.split is required when adaptation='inductive' (R4).",
            )
        if t_split.get("strategy") not in ("holdout", "k-fold"):
            raise TypeMismatchError(
                f"uda.target.split.strategy must be holdout|k-fold, got {t_split.get('strategy')!r}",
            )
        if t_split.get("dimension") not in _ALLOWED_SPLIT_DIMS:
            raise TypeMismatchError(
                f"uda.target.split.dimension must be one of {sorted(_ALLOWED_SPLIT_DIMS)}.",
            )
        if t_split["dimension"] == dim:
            raise ConfigError(
                "uda.domain.dimension must differ from uda.target.split.dimension (R11).",
            )
        if t_split["dimension"] == "flatten" and t_split.get("shuffle") is False:
            raise TypeMismatchError(
                "uda.target.split.dimension='flatten' requires shuffle=true (R26).",
            )
        # Required test_ratio / k (R4)
        if t_split["strategy"] == "holdout":
            if "test_ratio" not in t_split:
                raise MissingRequiredKeyError(
                    "uda.target.split (inductive holdout) requires 'test_ratio' (R4).",
                )
        else:
            if "k" not in t_split:
                raise MissingRequiredKeyError(
                    "uda.target.split (inductive k-fold) requires 'k' (R4).",
                )
        # Re-use regular-style ratio validation
        _validate_target_split_ratios(t_split)


def _validate_target_split_ratios(split: dict) -> None:
    """Apply R6/R15/R19/R21/R30 to an inductive target.split block."""
    val_split = split.get("val_split")
    if isinstance(val_split, dict):
        if val_split.get("dimension") not in _ALLOWED_SPLIT_DIMS:
            raise TypeMismatchError(
                f"target.split.val_split.dimension invalid: {val_split.get('dimension')!r}.",
            )
        if val_split.get("dimension") == "flatten" and val_split.get("shuffle") is False:
            raise TypeMismatchError(
                "target.split.val_split.dimension='flatten' requires shuffle=true (R26).",
            )
        vr = val_split.get("val_ratio")
        if not isinstance(vr, (int, float)) or not (0 < vr < 1):
            raise TypeMismatchError(
                f"target.split.val_split.val_ratio must be in (0, 1) (R21), got {vr!r}.",
            )
        if split.get("dimension") == "flatten" and val_split["dimension"] != "flatten":
            raise ConfigError(
                "target.split.dimension='flatten' requires val_split.dimension='flatten' (R30).",
            )

    if isinstance(val_split, dict) and "val_ratio" in split:
        raise ConfigError(
            "target.split.val_split and target.split.val_ratio are mutually exclusive (R19).",
        )

    if split["strategy"] == "holdout":
        if isinstance(val_split, dict):
            if abs(split["train_ratio"] + split["test_ratio"] - 1.0) > _RATIO_TOL:
                raise TypeMismatchError(
                    "target.split train_ratio + test_ratio must be ≈ 1.0 (R6).",
                )
        else:
            train = split.get("train_ratio")
            val = split.get("val_ratio", 0.0)
            test = split.get("test_ratio")
            if train is None or test is None:
                raise MissingRequiredKeyError(
                    "target.split holdout requires train_ratio and test_ratio.",
                )
            if abs(train + val + test - 1.0) > _RATIO_TOL:
                raise TypeMismatchError(
                    "target.split ratios must sum to 1.0 (R6).",
                )
            if not (0 <= val < 1):
                raise TypeMismatchError(
                    f"target.split.val_ratio must be in [0, 1) (R15), got {val}.",
                )


# ---------------------------------------------------------------------------
# training block (R17/R18/R24)
# ---------------------------------------------------------------------------


def _validate_training(config: dict, mode: str) -> None:
    training = config.get("training")
    if not isinstance(training, dict):
        return  # training is optional at the schema level; downstream catches missing fields

    logging_block = training.get("logging")
    if isinstance(logging_block, dict):
        backend = logging_block.get("backend")
        if backend is not None and backend != "tensorboard":
            raise TypeMismatchError(
                f"training.logging.backend must be 'tensorboard', got {backend!r} (R17).",
            )
        freq = logging_block.get("log_every_n_epochs")
        if freq is not None:
            if not isinstance(freq, int) or freq <= 0:
                raise TypeMismatchError(
                    f"training.logging.log_every_n_epochs must be a positive int, got {freq!r} (R18).",
                )

    # R24
    if mode == "uda":
        adaptation = config["uda"].get("adaptation")
        if adaptation == "transductive":
            if "early_stopping" in training:
                raise ConfigError(
                    "training.early_stopping not allowed for transductive UDA (R24).",
                )
            if "checkpoint" in training:
                raise ConfigError(
                    "training.checkpoint not allowed for transductive UDA (R24).",
                )
