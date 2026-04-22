"""Experiment-level per-dataset label remapping.

Declared in experiment YAML as ``datasets[alias].label_mapping``:

.. code-block:: yaml

    datasets:
      faced_test:
        label_mapping:
          anger: negative
          neutrality: neutral
          amusement: positive
          ...

Runs after :meth:`ExperimentManager._load_all_datasets` and before
:meth:`ExperimentManager._apply_alignment`. For each alias declaring
a mapping, labels in ``labels_cache[alias]`` are remapped to new numeric
IDs assigned by ASCII-sorted new semantic values (matching
:func:`uesf.managers.data_manager.DataManager.create_masked`), and
``metadata_cache[alias]`` is updated so downstream consumers
(:class:`LabelAligner`, model instantiation via ``meta["n_classes"]``) see
the post-mapping state.
"""

from __future__ import annotations

import numpy as np

from uesf.core.exceptions import ConfigError
from uesf.core.logging import get_logger

logger = get_logger("experiment.label_mapping")


def apply_label_mapping(
    datasets_config: dict,
    labels_cache: dict[str, np.ndarray],
    metadata_cache: dict[str, dict],
) -> None:
    """Apply per-dataset label remapping in place.

    Args:
        datasets_config: The ``datasets`` block of a normalized experiment
            config — a dict keyed by alias. Only aliases whose entry carries
            a non-empty ``label_mapping`` are touched.
        labels_cache: Alias → flattened 1-D label array. Remapped in place.
        metadata_cache: Alias → meta dict with ``numeric_to_semantic`` and
            ``n_classes``. Both fields are replaced with the post-mapping
            values for any remapped alias.

    Raises:
        ConfigError: If the alias has no ``numeric_to_semantic`` metadata,
            or the mapping keys do not exactly cover that alias's semantic
            value set.
    """
    for alias, ds_cfg in datasets_config.items():
        mapping = (ds_cfg or {}).get("label_mapping")
        if not mapping:
            continue

        meta = metadata_cache[alias]
        n2s = meta.get("numeric_to_semantic")
        if not n2s:
            raise ConfigError(
                f"Dataset '{alias}': label_mapping requires the dataset to "
                "have a 'numeric_to_semantic' mapping in its metadata.",
                hint="Re-register the raw dataset with a numeric_to_semantic "
                "declaration, or remove label_mapping from the config.",
            )

        old_semantic_set = set(n2s.values())
        mapping_keys = set(mapping.keys())
        missing = old_semantic_set - mapping_keys
        extra = mapping_keys - old_semantic_set
        if missing or extra:
            raise ConfigError(
                f"Dataset '{alias}': label_mapping keys must exactly cover "
                f"the dataset's semantic label set. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}.",
                context={
                    "dataset_semantics": sorted(old_semantic_set),
                    "mapping_keys": sorted(mapping_keys),
                },
                hint="Add entries for all missing semantic labels, or remove "
                "entries whose keys are not present in this dataset.",
            )

        new_semantics = sorted(set(mapping.values()))
        semantic_to_new_num = {s: i for i, s in enumerate(new_semantics)}

        old_num_to_new_num: dict[int, int] = {}
        for old_num_key, old_semantic in n2s.items():
            new_num = semantic_to_new_num[mapping[old_semantic]]
            old_num_to_new_num[int(old_num_key)] = new_num

        old_labels = labels_cache[alias]
        new_labels = np.empty_like(old_labels)
        for old_val, new_val in old_num_to_new_num.items():
            new_labels[old_labels == old_val] = new_val
        labels_cache[alias] = new_labels

        meta["numeric_to_semantic"] = {
            str(i): s for i, s in enumerate(new_semantics)
        }
        meta["n_classes"] = len(new_semantics)

        logger.info(
            "Applied label_mapping on '%s': %d → %d classes (new n2s=%s)",
            alias,
            len(old_semantic_set),
            len(new_semantics),
            meta["numeric_to_semantic"],
        )
