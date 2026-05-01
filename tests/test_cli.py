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


class TestModelsCompileCommand:
    def setup_method(self):
        mock_reg_mod = MagicMock()
        self.mock_registry = MagicMock()
        mock_reg_mod.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_reg_mod

        mock_compiler_mod = MagicMock()
        self.mock_compiler = MagicMock()
        mock_compiler_mod.MeraCompiler.return_value = self.mock_compiler
        sys.modules["sakura_simulator.compiler"] = mock_compiler_mod

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.registry", None)
        sys.modules.pop("sakura_simulator.compiler", None)

    def test_given_manifest_not_found_when_compile_invoked_then_exits_with_code_1(self):
        # Given: registry constructor raises FileNotFoundError
        sys.modules["sakura_simulator.registry"].ModelRegistry.side_effect = FileNotFoundError(
            "manifest missing"
        )
        # When: models compile is invoked
        result = runner.invoke(app, ["models", "compile", "resnet50"])
        # Then: exits with code 1 and error in output
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_given_model_not_found_when_compile_invoked_then_exits_with_code_1(self):
        # Given: registry returns None for the requested model
        self.mock_registry.get_model.return_value = None
        # When: models compile ghost_model is invoked
        result = runner.invoke(app, ["models", "compile", "ghost_model"])
        # Then: exits with code 1 and error in output
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_given_compile_raises_value_error_when_invoked_then_exits_with_code_1(self):
        # Given: compiler raises ValueError (unsupported format, missing file, etc.)
        self.mock_compiler.compile.side_effect = ValueError("Unsupported format 'tflite'")
        # When: models compile resnet50 is invoked
        result = runner.invoke(app, ["models", "compile", "resnet50"])
        # Then: exits with code 1 and error in output
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_given_valid_model_when_compile_invoked_then_shows_artifact_path_and_exits_0(self):
        # Given: compiler succeeds and returns the artifact path
        from pathlib import Path

        self.mock_compiler.compile.return_value = Path("/tmp/artifacts/resnet50")
        # When: models compile resnet50 is invoked
        result = runner.invoke(app, ["models", "compile", "resnet50"])
        # Then: exits successfully and shows the compiled path
        assert result.exit_code == 0
        assert "Compiled:" in result.output


class TestModelsInferCommand:
    def setup_method(self):
        mock_reg_mod = MagicMock()
        self.mock_registry = MagicMock()
        mock_reg_mod.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_reg_mod

        mock_runtime_mod = MagicMock()
        self.mock_runtime = MagicMock()
        mock_runtime_mod.MeraRuntime.return_value = self.mock_runtime
        sys.modules["sakura_simulator.runtime"] = mock_runtime_mod

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.registry", None)
        sys.modules.pop("sakura_simulator.runtime", None)

    def _make_infer_result(self, text="The biosignature is confirmed.", latency=55.0, n_tokens=7):
        r = MagicMock()
        r.text = text
        r.latency_ms = latency
        r.token_ids = list(range(n_tokens))
        return r

    def test_given_manifest_not_found_when_infer_invoked_then_exits_with_code_1(self):
        # Given: registry constructor raises FileNotFoundError
        sys.modules["sakura_simulator.registry"].ModelRegistry.side_effect = FileNotFoundError(
            "manifest missing"
        )
        # When: models infer is invoked
        result = runner.invoke(app, ["models", "infer", "tinyllama", "--prompt", "hi"])
        # Then: exits with code 1
        assert result.exit_code == 1

    def test_given_model_not_found_when_infer_invoked_then_exits_with_code_1(self):
        # Given: registry returns None for the model
        self.mock_registry.get_model.return_value = None
        # When: models infer is invoked
        result = runner.invoke(app, ["models", "infer", "ghost", "--prompt", "hi"])
        # Then: exits with code 1 and error in output
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_given_model_not_compiled_when_infer_invoked_then_exits_with_code_1(self):
        # Given: model exists but is not compiled
        self.mock_registry.is_compiled.return_value = False
        # When: models infer is invoked
        result = runner.invoke(app, ["models", "infer", "tinyllama", "--prompt", "hi"])
        # Then: exits with code 1 and compile hint in output
        assert result.exit_code == 1
        assert "not compiled" in result.output

    def test_given_runtime_raises_value_error_when_infer_invoked_then_exits_with_code_1(self):
        # Given: model is compiled but MeraRuntime.infer() raises ValueError
        self.mock_registry.is_compiled.return_value = True
        self.mock_runtime.infer.side_effect = ValueError("not an LLM")
        # When: models infer is invoked
        result = runner.invoke(app, ["models", "infer", "tinyllama", "--prompt", "hi"])
        # Then: exits with code 1 and error message
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_given_compiled_llm_when_infer_invoked_then_prints_text_and_stats(self):
        # Given: model is compiled and inference succeeds
        self.mock_registry.is_compiled.return_value = True
        self.mock_runtime.infer.return_value = self._make_infer_result()
        # When: models infer is invoked
        result = runner.invoke(
            app, ["models", "infer", "tinyllama", "--prompt", "What is a biosignature?"]
        )
        # Then: exits successfully and output contains generated text and stats
        assert result.exit_code == 0
        assert "The biosignature is confirmed." in result.output
        assert "ms" in result.output

    def test_given_custom_options_when_infer_invoked_then_runtime_called_with_correct_args(self):
        # Given: model is compiled
        self.mock_registry.is_compiled.return_value = True
        self.mock_runtime.infer.return_value = self._make_infer_result()
        # When: models infer is invoked with custom --max-new-tokens and --temperature
        runner.invoke(
            app,
            [
                "models",
                "infer",
                "tinyllama",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "64",
                "--temperature",
                "0.7",
            ],
        )
        # Then: MeraRuntime.infer received the custom options
        call_kwargs = self.mock_runtime.infer.call_args[1]
        assert call_kwargs["max_new_tokens"] == 64
        assert abs(call_kwargs["temperature"] - 0.7) < 1e-6


class TestModelsRunCommand:
    def setup_method(self):
        mock_reg_mod = MagicMock()
        self.mock_registry = MagicMock()
        mock_reg_mod.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_reg_mod

        mock_runtime_mod = MagicMock()
        self.mock_runtime = MagicMock()
        mock_runtime_mod.MeraRuntime.return_value = self.mock_runtime
        sys.modules["sakura_simulator.runtime"] = mock_runtime_mod

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.registry", None)
        sys.modules.pop("sakura_simulator.runtime", None)

    def _configure_success_result(self):
        mock_result = MagicMock()
        mock_result.avg_latency_ms = 5.0
        mock_result.min_latency_ms = 4.0
        mock_result.p95_latency_ms = 5.5
        mock_result.outputs = [{"name": "output0", "shape": [1, 1000], "dtype": "float32"}]
        self.mock_runtime.run.return_value = mock_result

    def test_given_manifest_not_found_when_run_invoked_then_exits_with_code_1(self):
        # Given: registry constructor raises FileNotFoundError
        sys.modules["sakura_simulator.registry"].ModelRegistry.side_effect = FileNotFoundError(
            "manifest missing"
        )
        # When: models run is invoked
        result = runner.invoke(app, ["models", "run", "resnet50"])
        # Then: exits with code 1
        assert result.exit_code == 1

    def test_given_model_not_found_when_run_invoked_then_exits_with_code_1(self):
        # Given: registry returns None for the requested model
        self.mock_registry.get_model.return_value = None
        # When: models run ghost_model is invoked
        result = runner.invoke(app, ["models", "run", "ghost_model"])
        # Then: exits with code 1 and error in output
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_given_model_not_compiled_when_run_invoked_then_shows_hint_and_exits_1(self):
        # Given: model exists but is not compiled
        self.mock_registry.is_compiled.return_value = False
        # When: models run resnet50 is invoked
        result = runner.invoke(app, ["models", "run", "resnet50"])
        # Then: exits with code 1 and hint about compile command
        assert result.exit_code == 1
        assert "not compiled" in result.output

    def test_given_run_raises_value_error_when_invoked_then_exits_with_code_1(self):
        # Given: model is compiled but runtime raises ValueError
        self.mock_registry.is_compiled.return_value = True
        self.mock_runtime.run.side_effect = ValueError("no inputs configured")
        # When: models run resnet50 is invoked
        result = runner.invoke(app, ["models", "run", "resnet50"])
        # Then: exits with code 1
        assert result.exit_code == 1

    def test_given_valid_compiled_model_when_run_invoked_then_shows_latency_and_exits_0(self):
        # Given: model is compiled and runtime returns a successful result
        self.mock_registry.is_compiled.return_value = True
        self._configure_success_result()
        # When: models run resnet50 is invoked
        result = runner.invoke(app, ["models", "run", "resnet50"])
        # Then: exits successfully and shows latency + output info
        assert result.exit_code == 0
        assert "Avg latency" in result.output
        assert "Output:" in result.output

    def test_given_valid_model_when_run_invoked_with_iters_5_then_runtime_called_with_iters_5(self):
        # Given: model is compiled and runtime returns a successful result
        self.mock_registry.is_compiled.return_value = True
        self._configure_success_result()
        # When: models run resnet50 --iters 5 is invoked
        result = runner.invoke(app, ["models", "run", "resnet50", "--iters", "5"])
        # Then: exits successfully and runtime was called with iters=5
        assert result.exit_code == 0
        self.mock_runtime.run.assert_called_once()
        call_kwargs = self.mock_runtime.run.call_args[1]
        assert call_kwargs["iters"] == 5
