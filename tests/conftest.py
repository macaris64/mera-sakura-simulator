"""
Global test configuration.

Streamlit is mocked via sys.modules injection at collection time so tests can
import and call app.py functions without a running Streamlit server.
cache_resource must pass functions through unchanged; a plain MagicMock used as
a decorator would replace the decorated function with another MagicMock.

TyperOption flag_value fix: Typer 0.12.5 passes flag_value=None to TyperOption,
but Click 8.3.3 changed its UNSET sentinel from None to Sentinel.UNSET. When
flag_value is not Sentinel.UNSET, Click auto-marks the option as is_flag=True,
breaking int/str options. This patch applies the correct UNSET sentinel before
sakura_simulator.cli is imported so all TyperOptions are created correctly.
"""

import sys
from unittest.mock import MagicMock

# --- Typer + Click 8.3.3 compatibility patch -----------------------------------
import click.core as _click_core
import typer.core as _typer_core

_orig_typer_option_init = _typer_core.TyperOption.__init__


def _fixed_typer_option_init(self, *args, **kwargs):
    if "flag_value" in kwargs and kwargs["flag_value"] is None:
        kwargs["flag_value"] = _click_core.UNSET
    _orig_typer_option_init(self, *args, **kwargs)


_typer_core.TyperOption.__init__ = _fixed_typer_option_init
# ------------------------------------------------------------------------------

_st_mock = MagicMock()
_st_mock.cache_resource.side_effect = lambda fn: fn
sys.modules["streamlit"] = _st_mock
