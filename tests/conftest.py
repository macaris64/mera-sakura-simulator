"""
Global test configuration.

sys.modules injection happens at collection time — before any test module
imports sakura_simulator — so `import mera` inside engine.py resolves to the
mock, not the real (absent) SDK.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock


# --- MERA mock ---------------------------------------------------------------

_mera = ModuleType("mera")


class _MockTarget:
    """Minimal stand-in for mera.Target."""

    SAKURA_II = "SAKURA_II"

    def __init__(self, target_type: str):
        self.target_type = target_type


_mera.Target = _MockTarget
sys.modules["mera"] = _mera


# --- Streamlit mock ----------------------------------------------------------
# cache_resource must pass functions through unchanged; a plain MagicMock used
# as a decorator would replace the decorated function with another MagicMock.

_st_mock = MagicMock()
_st_mock.cache_resource.side_effect = lambda fn: fn
sys.modules["streamlit"] = _st_mock
