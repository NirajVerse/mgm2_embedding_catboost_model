from __future__ import annotations

from pathlib import Path
import unittest

from mgm2_embedding_model.model_config import (
    apply_runtime_overrides,
    load_model_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelConfigTests(unittest.TestCase):
    def test_protocol_is_frozen_to_expected_grid_and_counts(self) -> None:
        config = load_model_config(PROJECT_ROOT / "configs" / "modeling.json")

        self.assertEqual(len(config.candidates), 6)
        self.assertEqual(config.settings.maximum_iterations, 4000)
        self.assertEqual(config.settings.early_stopping_rounds, 200)
        self.assertEqual(config.settings.random_seed, 42)
        self.assertEqual(config.settings.task_type, "CPU")
        self.assertEqual(config.expected["development_samples"], 946)
        self.assertEqual(config.expected["test_samples"], 245)
        self.assertEqual(config.expected["features"], 224)
        self.assertIn("prepared_manifest", config.expected_sha256)

    def test_runtime_overrides_do_not_change_scientific_protocol(self) -> None:
        config = load_model_config(PROJECT_ROOT / "configs" / "modeling.json")
        changed = apply_runtime_overrides(
            config, task_type="gpu", devices="0", thread_count=4, verbose=0
        )

        self.assertEqual(changed.settings.task_type, "GPU")
        self.assertEqual(changed.settings.devices, "0")
        self.assertEqual(changed.settings.thread_count, 4)
        self.assertEqual(changed.settings.verbose, 0)
        self.assertEqual(changed.candidates, config.candidates)
        self.assertEqual(changed.expected, config.expected)

    def test_invalid_thread_count_is_rejected(self) -> None:
        config = load_model_config(PROJECT_ROOT / "configs" / "modeling.json")
        with self.assertRaisesRegex(ValueError, "thread_count"):
            apply_runtime_overrides(config, thread_count=0)


if __name__ == "__main__":
    unittest.main()
