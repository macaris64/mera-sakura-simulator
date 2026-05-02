"""
Global test configuration.

TyperOption flag_value fix: Typer 0.12.5 passes flag_value=None to TyperOption,
but Click 8.3.3 changed its UNSET sentinel from None to Sentinel.UNSET. When
flag_value is not Sentinel.UNSET, Click auto-marks the option as is_flag=True,
breaking int/str options. This patch applies the correct UNSET sentinel before
sakura_simulator.cli is imported so all TyperOptions are created correctly.
"""

import enum
import sys
import types
from unittest.mock import MagicMock

# The real `mera` package pulls in seaborn → pandas → pyarrow (C-headers only, broken).
# Inject a hand-crafted mera stub before any test file is imported so that
# `import mera` at module level in test files gets the mock, not the real package.
# Tests that need to swap out mera for their own mock save/restore sys.modules["mera"].


class _Target(enum.Enum):
    Simulator = "Simulator"
    InterpreterHw = "InterpreterHw"


class _Platform(enum.Enum):
    SAKURA_2C = "SAKURA_2C"
    SAKURA_1 = "SAKURA_1"


_mera_mock = types.ModuleType("mera")
_mera_mock.Target = _Target
_mera_mock.Platform = _Platform
_mera_mock.TVMDeployer = MagicMock
_mera_mock.ModelLoader = MagicMock
_mera_mock.__version__ = "1.6.0"

_mera_deployment_mock = types.ModuleType("mera.mera_deployment")
_mera_deployment_mock.load_mera_deployment = MagicMock()
_mera_deployment_mock.MeraTvmDeployment = MagicMock
_mera_deployment_mock.MeraTvmPrjDeployment = MagicMock
_mera_deployment_mock.MeraTvmModelRunner = MagicMock
_mera_mock.mera_deployment = _mera_deployment_mock

sys.modules["mera"] = _mera_mock
sys.modules["mera.mera_deployment"] = _mera_deployment_mock

# --- Typer + Click 8.3.3 compatibility patch -----------------------------------
import click.core as _click_core  # noqa: E402
import typer.core as _typer_core  # noqa: E402

_orig_typer_option_init = _typer_core.TyperOption.__init__


def _fixed_typer_option_init(self, *args, **kwargs):
    if "flag_value" in kwargs and kwargs["flag_value"] is None:
        kwargs["flag_value"] = _click_core.UNSET
    _orig_typer_option_init(self, *args, **kwargs)


_typer_core.TyperOption.__init__ = _fixed_typer_option_init
# ------------------------------------------------------------------------------
