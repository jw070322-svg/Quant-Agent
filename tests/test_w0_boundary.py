"""W0 evidence-isolation boundary check (runs LAST in every suite).

Verifies that the test session left all protected project files byte-identical:
ingestion manifests, normalized data, provider snapshots/update summaries.
The snapshot is taken by conftest.py at pytest_configure."""
import pytest


def test_w0_protected_files_unchanged(request):
    snapshot = getattr(request.config, "_w0_protected_snapshot", None)
    if snapshot is None:
        pytest.skip("no protected-file snapshot recorded")
    from conftest import _protected_hashes  # rootdir conftest
    current = _protected_hashes()
    changed = {key for key in set(snapshot) | set(current)
               if snapshot.get(key) != current.get(key)}
    assert not changed, (
        "test suite modified protected project files: "
        f"{sorted(changed)} — tests must use temporary roots "
        "(W0 evidence isolation, comprehensive plan 2026-11-17 §6)")
