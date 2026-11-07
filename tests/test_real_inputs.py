from __future__ import annotations

import unittest
from pathlib import Path

from mgm2_embedding_model.config import load_config
from mgm2_embedding_model.prepare import load_and_validate_sources


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "prepare.json"


@unittest.skipUnless(CONFIG_PATH.is_file(), "real preparation config is unavailable")
class RealInputContractTests(unittest.TestCase):
    def test_frozen_real_inputs_pass_raw_validation(self) -> None:
        config = load_config(CONFIG_PATH)
        source_paths = [
            config.original_embeddings,
            config.mapseq_embeddings,
            config.original_metadata,
            config.mapseq_metadata,
            config.frozen_splits,
            config.processed_material_ids,
        ]
        if not all(path.is_file() for path in source_paths):
            self.skipTest("one or more external source files are unavailable")
        sources, recorder = load_and_validate_sources(config)
        self.assertEqual(recorder.status, "PASS")
        self.assertEqual(sources.original.embeddings.shape, (3896, 224))
        self.assertEqual(sources.mapseq.embeddings.shape, (3896, 224))
        self.assertEqual(len(sources.metadata_rows), 3691)
        self.assertEqual(len(sources.split_rows), 1191)


if __name__ == "__main__":
    unittest.main()
