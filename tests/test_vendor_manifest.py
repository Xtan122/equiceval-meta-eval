import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_files_match_lock():
    lock = json.loads((ROOT / "VENDOR.lock").read_text())
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        assert path.exists(), f"missing vendored file: {relative}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"vendored file changed: {relative}"
