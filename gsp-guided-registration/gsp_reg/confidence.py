from importlib.util import find_spec
from pathlib import Path


def validate_predator(root, checkpoint, dataset):
    if dataset not in ("indoor", "kitti"):
        raise ValueError("Expected indoor or kitti PREDATOR configuration")
    root = Path(root)
    required = (
        "models/architectures.py", "models/blocks.py", "models/gcn.py",
        "configs/models.py", "configs/test/%s.yaml" % dataset,
        "lib/utils.py", "kernels/kernel_points.py",
    )
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    hint = (
        "See docs/CONFIDENCE.md to install the official source and weights. "
        "Alternatively choose --sample_mode random "
        "(SAMPLE_MODE=random for scripts/evaluate.sh). "
        "The sampling mode will not be changed automatically."
    )
    if missing:
        raise FileNotFoundError("Missing PREDATOR source files: %s. %s" % (", ".join(missing), hint))
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError("Missing or empty PREDATOR checkpoint: %s. %s" % (checkpoint, hint))
    missing_dependencies = [name for name in ("yaml", "sklearn") if find_spec(name) is None]
    if missing_dependencies:
        raise ImportError(
            "Missing PREDATOR dependencies: %s. Install requirements.txt in the active "
            "Python environment. %s" % (", ".join(missing_dependencies), hint)
        )
