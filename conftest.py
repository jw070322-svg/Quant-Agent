import sys
from pathlib import Path

# Make src-layout package importable without requiring `pip install -e .`
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# W0 evidence isolation: snapshot protected files at session start and verify
# at session end that the test suite left them byte-identical. Tests must use
# temporary roots for anything that writes (see tests/test_w0_boundary.py).
import hashlib  # noqa: E402

_PROTECTED_PATTERNS = [
    "datasets/manifest_*.json",
    "datasets/normalized/**/*.csv",
    "datasets/updates/**/*.csv",
    "datasets/updates/**/*.json",
]


def _protected_hashes():
    root = Path(__file__).resolve().parent
    out = {}
    for pat in _PROTECTED_PATTERNS:
        for p in sorted(root.glob(pat)):
            out[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def pytest_configure(config):
    config._w0_protected_snapshot = _protected_hashes()


def pytest_collection_modifyitems(session, config, items):
    """Run the W0 boundary check last so it observes every suite side effect."""
    boundary = [i for i in items if "w0_boundary" in i.nodeid]
    for item in boundary:
        items.remove(item)
    items.extend(boundary)
