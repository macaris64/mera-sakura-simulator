"""BDD tests for the Typer CLI."""

import sys
from unittest.mock import MagicMock

from typer.testing import CliRunner

from sakura_simulator.cli import app

runner = CliRunner()


class TestHelloCommand:
    def test_given_hello_command_when_invoked_then_exits_successfully(self):
        # Given: the CLI app
        # When: `sakura hello` is invoked
        result = runner.invoke(app, ["hello"])
        # Then: exit code is 0
        assert result.exit_code == 0

    def test_given_hello_command_when_invoked_then_prints_full_greeting(self):
        # Given: the CLI app
        # When: `sakura hello` is invoked
        result = runner.invoke(app, ["hello"])
        # Then: the exact greeting appears in stdout
        assert "Hello from Sakura-II: Titan Biosignature Engine Active" in result.output


class TestModelsListCommand:
    def setup_method(self):
        mock_module = MagicMock()
        self.mock_registry = MagicMock()
        mock_module.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_module

    def test_given_valid_manifest_when_models_list_invoked_then_shows_model_names(self):
        # Given: registry returns two models
        m1 = MagicMock()
        m1.name = "resnet50"
        m1.version = "1.0.0"
        m2 = MagicMock()
        m2.name = "mobilenet_v2"
        m2.version = "2.0.0"
        self.mock_registry.list_models.return_value = [m1, m2]
        self.mock_registry.is_space_ready.return_value = True
        # When: models list is invoked
        result = runner.invoke(app, ["models", "list"])
        # Then: exits successfully and both model names appear in output
        assert result.exit_code == 0
        assert "resnet50" in result.output
        assert "mobilenet_v2" in result.output

    def test_given_missing_manifest_when_models_list_invoked_then_exits_with_error(self):
        # Given: registry constructor raises FileNotFoundError
        sys.modules["sakura_simulator.registry"].ModelRegistry.side_effect = FileNotFoundError(
            "manifest missing"
        )
        # When: models list is invoked
        result = runner.invoke(app, ["models", "list"])
        # Then: exits with non-zero code
        assert result.exit_code == 1


class TestModelsInspectCommand:
    def setup_method(self):
        mock_module = MagicMock()
        self.mock_registry = MagicMock()
        mock_module.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_module

    def test_given_known_model_when_inspect_invoked_then_shows_npu_constraints(self):
        # Given: registry contains the requested model with full metadata
        entry = MagicMock()
        entry.name = "resnet50"
        entry.version = "1.0.0"
        entry.path = "models/resnet50.mera"
        entry.npu_constraints.max_power_watts = 12.5
        entry.npu_constraints.required_memory_mb = 512
        self.mock_registry.get_model.return_value = entry
        self.mock_registry.is_space_ready.return_value = True
        # When: models inspect resnet50 is invoked
        result = runner.invoke(app, ["models", "inspect", "resnet50"])
        # Then: exits successfully and shows key details
        assert result.exit_code == 0
        assert "resnet50" in result.output
        assert "12.5" in result.output

    def test_given_unknown_model_when_inspect_invoked_then_exits_with_code_1(self):
        # Given: registry returns None for the requested name
        self.mock_registry.get_model.return_value = None
        # When: models inspect ghost_model is invoked
        result = runner.invoke(app, ["models", "inspect", "ghost_model"])
        # Then: exits with code 1
        assert result.exit_code == 1


class TestModelsDownloadCommand:
    def setup_method(self):
        from pathlib import Path

        mock_module = MagicMock()
        self.mock_registry = MagicMock()
        mock_module.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_module
        self.fake_path = Path("/tmp/models/resnet50.mera")

    def test_given_valid_model_when_download_invoked_then_shows_path_and_exits_successfully(self):
        # Given: registry download succeeds and returns the model path
        self.mock_registry.download.return_value = self.fake_path
        # When: models download resnet50 is invoked
        result = runner.invoke(app, ["models", "download", "resnet50"])
        # Then: exits successfully and confirms path in output
        assert result.exit_code == 0
        assert "Downloaded" in result.output

    def test_given_download_error_when_invoked_then_shows_error_and_exits_with_code_1(self):
        # Given: registry download raises ValueError (unknown name or checksum mismatch)
        self.mock_registry.download.side_effect = ValueError("Checksum mismatch for 'resnet50'")
        # When: models download resnet50 is invoked
        result = runner.invoke(app, ["models", "download", "resnet50"])
        # Then: exits with code 1 and error appears in output
        assert result.exit_code == 1


class TestModelsRemoveCommand:
    def setup_method(self):
        from pathlib import Path

        mock_module = MagicMock()
        self.mock_registry = MagicMock()
        mock_module.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_module
        self.fake_path = Path("/tmp/models/resnet50.mera")

    def test_given_valid_model_when_remove_invoked_then_shows_path_and_exits_successfully(self):
        # Given: registry remove succeeds and returns the model path
        self.mock_registry.remove.return_value = self.fake_path
        # When: models remove resnet50 is invoked
        result = runner.invoke(app, ["models", "remove", "resnet50"])
        # Then: exits successfully and confirms path in output
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_given_remove_error_when_invoked_then_shows_error_and_exits_with_code_1(self):
        # Given: registry remove raises FileNotFoundError (model file not on disk)
        self.mock_registry.remove.side_effect = FileNotFoundError("resnet50.mera not found")
        # When: models remove resnet50 is invoked
        result = runner.invoke(app, ["models", "remove", "resnet50"])
        # Then: exits with code 1
        assert result.exit_code == 1
