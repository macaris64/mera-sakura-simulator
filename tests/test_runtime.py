"""BDD tests for MeraRuntime and RunResult — inference pipeline for SAKURA-II."""

import builtins
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mera.mera_deployment as mera_dep
import pytest

from sakura_simulator.runtime import (
    RunResult,
    _build_simulator_runner,
    _load_mera_deployment,
    _make_runner,
    _resolve_deployment_dir,
    _SimulatorGraphRunner,
)


class TestLoadMeraDeployment:
    def test_given_path_and_target_when_load_mera_deployment_then_forwards_to_mera(self):
        with patch.object(mera_dep, "load_mera_deployment") as mock_load:
            mock_load.return_value = MagicMock()
            # When
            out = _load_mera_deployment("/tmp/artifacts/model", "sim_target")
            # Then
            mock_load.assert_called_once_with("/tmp/artifacts/model", target="sim_target")
            assert out is mock_load.return_value


class TestResolveDeploymentDir:
    def test_given_deploy_so_in_result_when_resolve_then_returns_result_dir(self, tmp_path: Path):
        artifact_dir = tmp_path / "artifacts"
        result_dir = artifact_dir / "build" / "Simulator" / "result"
        result_dir.mkdir(parents=True)
        (result_dir / "deploy.so").write_bytes(b"x")

        class _T:
            str_val = "Simulator"

        resolved = _resolve_deployment_dir(artifact_dir, target=_T())
        assert resolved == result_dir

    def test_given_no_result_dir_when_resolve_then_returns_artifact_dir(self, tmp_path: Path):
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        class _T:
            str_val = "Simulator"

        resolved = _resolve_deployment_dir(artifact_dir, target=_T())
        assert resolved == artifact_dir

    def test_given_target_without_strval_when_resolve_then_uses_str(
        self, tmp_path: Path,
    ):
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        class _NoStrVal:
            def __str__(self) -> str:
                return "CustomTarget"

        resolved = _resolve_deployment_dir(artifact_dir, target=_NoStrVal())
        assert resolved == artifact_dir

    def test_given_no_strval_and_result_has_deploy_when_resolve_then_returns_result(
        self, tmp_path: Path,
    ):
        artifact_dir = tmp_path / "artifacts"
        result_dir = artifact_dir / "build" / "CustomTarget" / "result"
        result_dir.mkdir(parents=True)
        (result_dir / "deploy.so").write_bytes(b"x")

        class _NoStrVal:
            def __str__(self) -> str:
                return "CustomTarget"

        resolved = _resolve_deployment_dir(artifact_dir, target=_NoStrVal())
        assert resolved == result_dir


class TestSimulatorGraphRunner:
    def test_given_non_dict_input_when_set_input_then_raises_type_error(self):
        rt = MagicMock()
        runner = _SimulatorGraphRunner(rt)
        with pytest.raises(TypeError, match="dict"):
            runner.set_input([])  # type: ignore[arg-type]

    def test_given_dict_when_set_input_run_get_outputs_then_delegates_to_rt_mod(self):
        rt = MagicMock()
        rt.get_num_outputs.return_value = 2
        out0 = MagicMock()
        out0.asnumpy.return_value = MagicMock(shape=(1,), dtype="float32")
        out1 = SimpleNamespace(shape=(2, 3), dtype="int32")
        rt.get_output.side_effect = [out0, out1]

        runner = _SimulatorGraphRunner(rt)
        ret = runner.set_input({"a": 1})
        assert ret is runner
        assert runner.run() is runner
        rt.set_input.assert_called_once_with("a", 1)
        rt.run.assert_called_once()
        got = runner.get_outputs()
        assert len(got) == 2
        assert got[0] is out0.asnumpy.return_value
        assert got[1] is out1


class TestBuildSimulatorRunner:
    def test_given_missing_deploy_so_when_build_then_raises_value_error(self, tmp_path: Path):
        d = tmp_path / "result"
        d.mkdir()
        (d / "deploy.json").write_text("{}")
        (d / "deploy.params").write_bytes(b"x")
        with pytest.raises(ValueError, match="Missing deploy.so"):
            _build_simulator_runner(d)

    def test_given_tvm_import_blocked_when_build_then_value_error(self, tmp_path: Path):
        d = tmp_path / "r"
        d.mkdir()
        (d / "deploy.so").write_bytes(b"x")
        (d / "deploy.json").write_text("{}")
        (d / "deploy.params").write_bytes(b"x")

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tvm" or name.startswith("tvm."):
                raise ImportError("blocked")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(ValueError, match="TVM is required"):
                _build_simulator_runner(d)

    def test_given_load_module_fails_when_build_then_value_error(self, tmp_path: Path):
        pytest.importorskip("tvm")
        import tvm.runtime as tvm_rt

        d = tmp_path / "r"
        d.mkdir()
        (d / "deploy.so").write_bytes(b"x")
        (d / "deploy.json").write_text("{}")
        (d / "deploy.params").write_bytes(b"x")

        with patch.object(tvm_rt, "load_module", side_effect=OSError("bad so")):
            with pytest.raises(ValueError, match="Failed to load TVM deployment"):
                _build_simulator_runner(d)

    def test_given_tvm_mocks_when_build_then_returns_simulator_runner(self, tmp_path: Path):
        pytest.importorskip("tvm")
        d = tmp_path / "r"
        d.mkdir()
        (d / "deploy.so").write_bytes(b"x")
        (d / "deploy.json").write_text('{"graph":""}')
        params = b"\x01\x02"
        (d / "deploy.params").write_bytes(params)

        mock_rt = MagicMock()
        fake_lib = object()
        with (
            patch("tvm.runtime.load_module", return_value=fake_lib) as mock_lm,
            patch("tvm.contrib.graph_executor.create", return_value=mock_rt) as mock_create,
        ):
            runner = _build_simulator_runner(d)

        mock_lm.assert_called_once_with(str(d / "deploy.so"))
        mock_create.assert_called_once()
        call_json, call_lib, _cpu = mock_create.call_args[0]
        assert call_json == '{"graph":""}'
        assert call_lib is fake_lib
        mock_rt.load_params.assert_called_once_with(params)
        assert isinstance(runner, _SimulatorGraphRunner)
        assert runner._rt is mock_rt


class TestMakeRunnerBranching:
    def test_given_simulator_target_when_make_runner_then_builds_graph_runner_not_load_mera(
        self, tmp_path: Path,
    ):
        d = tmp_path / "r"
        d.mkdir()
        mock_runner = MagicMock()
        mock_runner.get_outputs.return_value = []

        p_build = patch(
            "sakura_simulator.runtime._build_simulator_runner",
            return_value=mock_runner,
        )
        with (
            patch("sakura_simulator.runtime._is_simulator_target", return_value=True),
            p_build as mock_build,
            patch("sakura_simulator.runtime._load_mera_deployment") as mock_load,
        ):
            out = _make_runner(d, target="sim")
        mock_build.assert_called_once_with(d)
        mock_load.assert_not_called()
        assert out is mock_runner

    def test_given_non_simulator_when_make_runner_then_loads_mera(self, tmp_path: Path):
        d = tmp_path / "r"
        d.mkdir()
        mock_dep = MagicMock()
        mock_dep.get_runner.return_value = MagicMock()

        p_load = patch(
            "sakura_simulator.runtime._load_mera_deployment",
            return_value=mock_dep,
        )
        with (
            patch("sakura_simulator.runtime._is_simulator_target", return_value=False),
            p_load as mock_load,
            patch("sakura_simulator.runtime._build_simulator_runner") as mock_build,
        ):
            r = _make_runner(d, target="ip")
        mock_load.assert_called_once_with(str(d), "ip")
        mock_build.assert_not_called()
        assert r is mock_dep.get_runner.return_value


class TestIsSimulatorTarget:
    def test_given_real_mera_simulator_enum_when_checked_then_true(self):
        import mera

        from sakura_simulator.runtime import _is_simulator_target

        assert _is_simulator_target(mera.Target.Simulator) is True

    def test_given_ip_strval_when_checked_then_false(self):
        from sakura_simulator.runtime import _is_simulator_target

        t = MagicMock()
        t.str_val = "IP"
        assert _is_simulator_target(t) is False


class TestRunResult:
    def test_given_empty_latency_when_avg_then_returns_zero(self):
        # Given: RunResult with no latency measurements
        # When: avg_latency_ms is accessed
        result = RunResult(outputs=[], latency_ms=[])
        # Then: returns 0.0
        assert result.avg_latency_ms == 0.0

    def test_given_latency_list_when_avg_then_returns_mean(self):
        # Given: RunResult with three measurements
        # When: avg_latency_ms is accessed
        result = RunResult(outputs=[], latency_ms=[10.0, 20.0, 30.0])
        # Then: returns the arithmetic mean
        assert result.avg_latency_ms == 20.0

    def test_given_empty_latency_when_min_then_returns_zero(self):
        # Given: RunResult with no latency measurements
        # When: min_latency_ms is accessed
        result = RunResult(outputs=[], latency_ms=[])
        # Then: returns 0.0
        assert result.min_latency_ms == 0.0

    def test_given_latency_list_when_min_then_returns_minimum(self):
        # Given: RunResult with three measurements
        # When: min_latency_ms is accessed
        result = RunResult(outputs=[], latency_ms=[10.0, 5.0, 15.0])
        # Then: returns the smallest value
        assert result.min_latency_ms == 5.0

    def test_given_empty_latency_when_p95_then_returns_zero(self):
        # Given: RunResult with no latency measurements
        # When: p95_latency_ms is accessed
        result = RunResult(outputs=[], latency_ms=[])
        # Then: returns 0.0
        assert result.p95_latency_ms == 0.0

    def test_given_single_latency_when_p95_then_returns_that_value(self):
        # Given: RunResult with exactly one measurement
        # When: p95_latency_ms is accessed
        # Then: int(1 * 0.95) - 1 = -1 → max(0, -1) = 0 → sorted[0]
        result = RunResult(outputs=[], latency_ms=[42.0])
        assert result.p95_latency_ms == 42.0

    def test_given_twenty_latencies_when_p95_then_returns_correct_percentile(self):
        # Given: RunResult with 20 evenly-spaced measurements (1.0 … 20.0)
        # When: p95_latency_ms is accessed
        # Then: idx = max(0, int(20 * 0.95) - 1) = max(0, 18) = 18 → sorted[18] = 19.0
        result = RunResult(outputs=[], latency_ms=list(range(1, 21)))
        assert result.p95_latency_ms == 19.0


class TestMeraRuntimeRun:
    def setup_method(self):
        import mera as _real_mera  # noqa: F401 — ensure real module for teardown restore

        self._saved_mera = sys.modules["mera"]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        # Remove any mock previously injected by CLI/engine tests
        sys.modules.pop("sakura_simulator.runtime", None)
        self.mock_mera = MagicMock()
        sim_target = MagicMock()
        sim_target.str_val = "Simulator"
        self.mock_mera.Target.Simulator = sim_target
        self._saved_numpy = sys.modules.get("numpy")
        self.mock_numpy = MagicMock()
        sys.modules["mera"] = self.mock_mera
        sys.modules["numpy"] = self.mock_numpy

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules["mera"] = self._saved_mera
        saved_np = getattr(self, "_saved_numpy", None)
        if saved_np is not None:
            sys.modules["numpy"] = saved_np
        else:
            sys.modules.pop("numpy", None)

    def _make_entry(self, inputs=None):
        entry = MagicMock()
        entry.name = "resnet50"
        entry.inputs = inputs
        return entry

    def _deployment_and_runner(self):
        mock_deployment = MagicMock()
        mock_runner = MagicMock()
        mock_deployment.get_runner.return_value = mock_runner
        mock_array = MagicMock()
        mock_array.shape = [1, 1000]
        mock_array.dtype = "float32"
        mock_runner.get_outputs.return_value = [mock_array]
        return mock_deployment, mock_runner

    def _ip_target(self):
        t = MagicMock()
        t.str_val = "IP"
        return t

    def test_given_artifact_dir_missing_when_run_then_raises_value_error(self):
        # Given: artifact_dir does not exist on disk
        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime(target=self._ip_target())
        entry = self._make_entry(inputs=[MagicMock()])
        # When / Then: ValueError mentioning artifact directory
        with pytest.raises(ValueError, match="Artifact directory not found"):
            runtime.run(entry, self.tmpdir / "nonexistent")

    def test_given_entry_inputs_none_when_run_then_raises_value_error(self):
        # Given: artifact_dir exists but entry.inputs is None
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime(target=self._ip_target())
        entry = self._make_entry(inputs=None)
        # When / Then: ValueError mentioning no inputs
        with pytest.raises(ValueError, match="no inputs configured"):
            runtime.run(entry, artifact_path)

    def test_given_entry_inputs_empty_list_when_run_then_raises_value_error(self):
        # Given: artifact_dir exists but entry.inputs is an empty list
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime(target=self._ip_target())
        entry = self._make_entry(inputs=[])
        # When / Then: ValueError mentioning no inputs
        with pytest.raises(ValueError, match="no inputs configured"):
            runtime.run(entry, artifact_path)

    def test_given_valid_entry_when_run_one_iter_then_latency_has_one_element(self):
        # Given: valid artifact_dir and entry with one input spec
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        mock_deployment, mock_runner = self._deployment_and_runner()
        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1, 3, 224, 224]
        inp.dtype = "float32"
        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime(target=self._ip_target())
        entry = self._make_entry(inputs=[inp])
        with patch("sakura_simulator.runtime._load_mera_deployment", return_value=mock_deployment):
            # When: run is called with iters=1
            result = runtime.run(entry, artifact_path, iters=1)
        # Then: exactly one latency measurement
        assert len(result.latency_ms) == 1
        assert len(result.outputs) == 1
        assert result.outputs[0]["name"] == "output_0"

    def test_given_valid_entry_when_run_five_iters_then_latency_has_five_elements(self):
        # Given: valid artifact_dir and entry with one input spec
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        mock_deployment, mock_runner = self._deployment_and_runner()
        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1, 3, 224, 224]
        inp.dtype = "float32"
        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime(target=self._ip_target())
        entry = self._make_entry(inputs=[inp])
        with patch("sakura_simulator.runtime._load_mera_deployment", return_value=mock_deployment):
            # When: run is called with iters=5
            result = runtime.run(entry, artifact_path, iters=5)
        # Then: five latency measurements
        assert len(result.latency_ms) == 5
        assert mock_runner.set_input.call_count == 5
        assert mock_runner.run.call_count == 5

    def test_given_input_with_no_name_when_run_then_uses_index_fallback_key(self):
        # Given: an input spec with name=None
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        mock_deployment, mock_runner = self._deployment_and_runner()
        inp = MagicMock()
        inp.name = None
        inp.shape = [1, 3, 224, 224]
        inp.dtype = "float32"
        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime(target=self._ip_target())
        entry = self._make_entry(inputs=[inp])
        with patch("sakura_simulator.runtime._load_mera_deployment", return_value=mock_deployment):
            runtime.run(entry, artifact_path, iters=1)
        # Then: set_input was called with a dict containing "input_0"
        call_args = mock_runner.set_input.call_args[0][0]
        assert "input_0" in call_args

    def test_given_simulator_when_run_then_uses_build_simulator_runner_path(self):
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        result_dir = artifact_path / "build" / "Simulator" / "result"
        result_dir.mkdir(parents=True)
        (result_dir / "deploy.so").write_bytes(b"x")

        mock_runner = MagicMock()
        mock_arr = MagicMock()
        mock_arr.shape = [1, 2]
        mock_arr.dtype = "float32"
        mock_runner.get_outputs.return_value = [mock_arr]
        self.mock_numpy.zeros.return_value = MagicMock()

        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1, 3, 224, 224]
        inp.dtype = "float32"
        entry = self._make_entry(inputs=[inp])

        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime()
        p_build = patch(
            "sakura_simulator.runtime._build_simulator_runner",
            return_value=mock_runner,
        )
        with p_build as mock_build:
            result = runtime.run(entry, artifact_path, iters=1)
        mock_build.assert_called_once()
        assert len(result.latency_ms) == 1
        mock_runner.set_input.assert_called_once()

    def test_given_make_runner_raises_runtime_error_when_run_then_wraps_value_error(self):
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1]
        inp.dtype = "float32"
        entry = self._make_entry(inputs=[inp])

        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime()
        with patch("sakura_simulator.runtime._make_runner", side_effect=RuntimeError("boom")):
            with pytest.raises(ValueError, match="Run failed"):
                runtime.run(entry, artifact_path, iters=1)

    def test_given_value_error_from_make_runner_when_run_then_reraises_same(self):
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1]
        inp.dtype = "float32"
        entry = self._make_entry(inputs=[inp])

        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime()
        with patch("sakura_simulator.runtime._make_runner", side_effect=ValueError("precise")):
            with pytest.raises(ValueError, match="precise"):
                runtime.run(entry, artifact_path, iters=1)

    def test_given_keyboard_interrupt_during_run_when_run_then_propagates(self):
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1]
        inp.dtype = "float32"
        entry = self._make_entry(inputs=[inp])

        mock_runner = MagicMock()
        mock_runner.run.side_effect = KeyboardInterrupt

        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime()
        with patch("sakura_simulator.runtime._make_runner", return_value=mock_runner):
            with pytest.raises(KeyboardInterrupt):
                runtime.run(entry, artifact_path, iters=1)

    def test_given_system_exit_during_run_when_run_then_propagates(self):
        artifact_path = self.tmpdir / "artifacts"
        artifact_path.mkdir()
        inp = MagicMock()
        inp.name = "data"
        inp.shape = [1]
        inp.dtype = "float32"
        entry = self._make_entry(inputs=[inp])

        mock_runner = MagicMock()
        mock_runner.run.side_effect = SystemExit(2)

        from sakura_simulator.runtime import MeraRuntime

        runtime = MeraRuntime()
        with patch("sakura_simulator.runtime._make_runner", return_value=mock_runner):
            with pytest.raises(SystemExit) as exc_info:
                runtime.run(entry, artifact_path, iters=1)
        assert exc_info.value.code == 2
