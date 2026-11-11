"""Run repository tests against the checked-out packages, never stale installations."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)
