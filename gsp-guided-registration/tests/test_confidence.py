import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gsp_reg.confidence import validate_predator


class ConfidencePreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def create_source(self, dataset):
        for name in (
            "models/architectures.py", "models/blocks.py", "models/gcn.py",
            "configs/models.py", "configs/test/%s.yaml" % dataset,
            "lib/utils.py", "kernels/kernel_points.py",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_missing_source_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "docs/CONFIDENCE.md"):
            validate_predator(self.root, self.root / "indoor.pth", "indoor")

    def test_missing_and_empty_weights_are_rejected(self):
        self.create_source("indoor")
        weights = self.root / "indoor.pth"
        for create in (False, True):
            if create:
                weights.touch()
            with self.assertRaisesRegex(FileNotFoundError, "Missing or empty"):
                validate_predator(self.root, weights, "indoor")

    def test_valid_assets_for_both_datasets(self):
        with patch("gsp_reg.confidence.find_spec", return_value=object()):
            for dataset in ("indoor", "kitti"):
                self.create_source(dataset)
                weights = self.root / (dataset + ".pth")
                weights.write_bytes(b"fixture")
                validate_predator(self.root, weights, dataset)

    def test_missing_python_dependencies_are_rejected(self):
        self.create_source("indoor")
        weights = self.root / "indoor.pth"
        weights.write_bytes(b"fixture")
        with patch("gsp_reg.confidence.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "requirements.txt"):
                validate_predator(self.root, weights, "indoor")


if __name__ == "__main__":
    unittest.main()
