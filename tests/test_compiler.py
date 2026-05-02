"""BDD tests for MeraCompiler — compile pipeline for SAKURA-II."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestHostArch:
    def test_given_x86_64_machine_when_host_arch_then_returns_x86(self):
        # Given: platform.machine() reports x86_64 (standard Linux x86)
        from sakura_simulator.compiler import _host_arch

        with patch("sakura_simulator.compiler.platform.machine", return_value="x86_64"):
            # When / Then
            assert _host_arch() == "x86"

    def test_given_unknown_machine_when_host_arch_then_returns_x86_fallback(self):
        # Given: platform.machine() returns '' (empty — common in containers)
        from sakura_simulator.compiler import _host_arch

        with patch("sakura_simulator.compiler.platform.machine", return_value=""):
            # When / Then: falls back to 'x86' default
            assert _host_arch() == "x86"


class TestMeraCompilerInit:
    def setup_method(self):
        import mera as _real_mera  # noqa: F401 — ensure real module for teardown restore

        self._saved_mera = sys.modules["mera"]
        # Remove any mock previously injected by CLI tests
        sys.modules.pop("sakura_simulator.compiler", None)
        self.mock_mera = MagicMock()
        sys.modules["mera"] = self.mock_mera

    def teardown_method(self):
        sys.modules["mera"] = self._saved_mera

    def test_given_no_args_when_init_then_uses_simulator_target(self):
        # Given: no arguments
        from sakura_simulator.compiler import MeraCompiler

        # When: MeraCompiler is constructed with defaults
        compiler = MeraCompiler()
        # Then: target is mera.Target.Simulator
        assert compiler._target is self.mock_mera.Target.Simulator

    def test_given_no_args_when_init_then_uses_sakura_2c_platform(self):
        # Given: no arguments
        from sakura_simulator.compiler import MeraCompiler

        # When: MeraCompiler is constructed with defaults
        compiler = MeraCompiler()
        # Then: platform is mera.Platform.SAKURA_2C
        assert compiler._platform is self.mock_mera.Platform.SAKURA_2C

    def test_given_custom_target_when_init_then_stores_custom_target(self):
        # Given: a specific target value
        from sakura_simulator.compiler import MeraCompiler

        # When: MeraCompiler is constructed with a custom target
        compiler = MeraCompiler(target="my_target")
        # Then: the provided target is stored
        assert compiler._target == "my_target"

    def test_given_custom_platform_when_init_then_stores_custom_platform(self):
        # Given: a specific platform value
        from sakura_simulator.compiler import MeraCompiler

        # When: MeraCompiler is constructed with a custom platform
        compiler = MeraCompiler(platform="my_platform")
        # Then: the provided platform is stored
        assert compiler._platform == "my_platform"


class TestMeraCompilerCompile:
    def setup_method(self):
        import mera as _real_mera  # noqa: F401 — ensure real module for teardown restore

        self._saved_mera = sys.modules["mera"]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        # Remove any mock previously injected by CLI tests
        sys.modules.pop("sakura_simulator.compiler", None)
        self.mock_mera = MagicMock()
        sys.modules["mera"] = self.mock_mera

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules["mera"] = self._saved_mera

    def _make_entry(self, **overrides):
        entry = MagicMock()
        entry.name = "resnet50"
        entry.format = "onnx"
        entry.path = str(self.tmpdir / "resnet50.onnx")
        entry.artifact_dir = str(self.tmpdir / "artifacts" / "resnet50")
        for k, v in overrides.items():
            setattr(entry, k, v)
        return entry

    def test_given_format_not_onnx_when_compile_then_raises_value_error(self):
        # Given: a model entry with unsupported format
        from sakura_simulator.compiler import MeraCompiler

        compiler = MeraCompiler()
        entry = self._make_entry(format="tflite")
        # When / Then: ValueError mentioning the unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            compiler.compile(entry)

    def test_given_artifact_dir_none_when_compile_then_raises_value_error(self):
        # Given: a model entry with no artifact_dir
        from sakura_simulator.compiler import MeraCompiler

        compiler = MeraCompiler()
        entry = self._make_entry(artifact_dir=None)
        # When / Then: ValueError mentioning no artifact_dir
        with pytest.raises(ValueError, match="no artifact_dir"):
            compiler.compile(entry)

    def test_given_source_model_missing_when_compile_then_raises_value_error(self):
        # Given: a model entry whose source file does not exist on disk
        from sakura_simulator.compiler import MeraCompiler

        compiler = MeraCompiler()
        entry = self._make_entry(path=str(self.tmpdir / "nonexistent.onnx"))
        # When / Then: ValueError mentioning source model not found
        with pytest.raises(ValueError, match="Source model not found"):
            compiler.compile(entry)

    def test_given_valid_entry_when_compile_then_calls_tvm_deployer(self):
        # Given: a valid entry with source file on disk
        from sakura_simulator.compiler import MeraCompiler

        source_file = self.tmpdir / "resnet50.onnx"
        source_file.write_bytes(b"fake onnx")
        compiler = MeraCompiler()
        entry = self._make_entry(path=str(source_file))
        # When: compile is called
        result = compiler.compile(entry)
        # Then: TVMDeployer was instantiated with output dir only (MERA 1.6 API)
        self.mock_mera.TVMDeployer.assert_called_once_with(str(Path(entry.artifact_dir)))
        assert result == Path(entry.artifact_dir)

    def test_given_valid_entry_when_compile_then_model_loader_and_deploy_invoked(self):
        # Given: a valid entry with source file on disk
        from sakura_simulator.compiler import MeraCompiler

        source_file = self.tmpdir / "resnet50.onnx"
        source_file.write_bytes(b"fake onnx")
        compiler = MeraCompiler()
        entry = self._make_entry(path=str(source_file))
        # When: compile is called
        compiler.compile(entry)
        # Then: ModelLoader(deployer), from_onnx, deploy(model, mera_platform, target)
        mock_deployer = self.mock_mera.TVMDeployer.return_value
        self.mock_mera.ModelLoader.assert_called_once_with(mock_deployer)
        mock_loader = self.mock_mera.ModelLoader.return_value
        mock_loader.from_onnx.assert_called_once_with(str(source_file), model_name=entry.name)
        mock_model = mock_loader.from_onnx.return_value
        mock_deployer.deploy.assert_called_once_with(
            mock_model,
            mera_platform=compiler._platform,
            target=compiler._target,
            host_arch="x86",
        )
