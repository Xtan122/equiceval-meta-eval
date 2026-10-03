"""Make the vendored method code importable as ``src.*``.

``vendor/src`` merges the EquiCEval engine (from ``../EquiCEval``) with the
EquivaFormulation adapter and the independent relabel oracle (from the
monorepo). ``vendor/EquivaMap`` is the vendored third-party method clone.
"""
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
