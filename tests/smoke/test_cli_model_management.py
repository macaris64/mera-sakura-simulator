"""CLI smoke tests — resnet50 model management end-to-end.

Exercises sakura models list/inspect/download/remove via Typer's CliRunner
without any mocking of httpx or the registry. Network access is required for
the download step (fetches from Hugging Face).

Run standalone:  poetry run pytest tests/smoke/ --no-cov -v
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sakura_simulator.cli import app

runner = CliRunner()
MODEL_PATH = Path("models/resnet50.onnx")


@pytest.fixture(autouse=True)
def clean_model_file():
    MODEL_PATH.unlink(missing_ok=True)
    yield
    MODEL_PATH.unlink(missing_ok=True)


class TestResnet50List:
    def test_list_includes_resnet50(self):
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0, result.output
        assert "resnet50" in result.output

    def test_list_shows_not_space_ready_before_download(self):
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0, result.output
        resnet_line = next(line for line in result.output.splitlines() if "resnet50" in line)
        assert "NO" in resnet_line


class TestResnet50Inspect:
    def test_inspect_resnet50_exits_successfully(self):
        result = runner.invoke(app, ["models", "inspect", "resnet50"])
        assert result.exit_code == 0, result.output

    def test_inspect_resnet50_shows_max_power_watts(self):
        result = runner.invoke(app, ["models", "inspect", "resnet50"])
        assert "12.5" in result.output

    def test_inspect_resnet50_shows_required_memory_mb(self):
        result = runner.invoke(app, ["models", "inspect", "resnet50"])
        assert "512" in result.output

    def test_inspect_resnet50_shows_space_ready_no(self):
        result = runner.invoke(app, ["models", "inspect", "resnet50"])
        assert "NO" in result.output


class TestResnet50Lifecycle:
    def test_download_inspect_remove_full_lifecycle(self):
        # --- download ---
        dl = runner.invoke(app, ["models", "download", "resnet50"])
        assert dl.exit_code == 0, dl.output
        assert "Downloaded" in dl.output
        assert MODEL_PATH.exists()

        # --- list shows Space-Ready after download ---
        list_after_dl = runner.invoke(app, ["models", "list"])
        assert list_after_dl.exit_code == 0
        resnet_line = next(line for line in list_after_dl.output.splitlines() if "resnet50" in line)
        assert "YES" in resnet_line

        # --- inspect shows Space-Ready YES ---
        insp = runner.invoke(app, ["models", "inspect", "resnet50"])
        assert insp.exit_code == 0
        assert "YES" in insp.output

        # --- remove ---
        rm = runner.invoke(app, ["models", "remove", "resnet50"])
        assert rm.exit_code == 0, rm.output
        assert "Removed" in rm.output
        assert not MODEL_PATH.exists()

        # --- list shows Space-Ready NO after remove ---
        list_after_rm = runner.invoke(app, ["models", "list"])
        resnet_line_after = next(
            line for line in list_after_rm.output.splitlines() if "resnet50" in line
        )
        assert "NO" in resnet_line_after
