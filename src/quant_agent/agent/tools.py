"""Compatibility import for the original run-artifact tools."""

import sys

from ..storage import run_history

# Keep legacy imports and assignments to RUNS_DIR on the same module object.
sys.modules[__name__] = run_history
