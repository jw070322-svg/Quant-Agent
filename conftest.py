import sys
from pathlib import Path

# Make src-layout package importable without requiring `pip install -e .`
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
