import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description="Verify and install descriptor checkpoints.")
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "checkpoints"
    manifest = json.loads((target / "manifest.json").read_text())
    for item in manifest["checkpoints"]:
        source = args.source_dir / item["filename"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Checksum mismatch: %s" % source.name)
        destination = target / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        print("Verified and installed:", destination.name)


if __name__ == "__main__":
    main()
