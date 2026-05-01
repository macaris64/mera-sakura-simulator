"""MeraRuntime: load compiled artifacts and run inference on SAKURA-II."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from sakura_simulator.registry import ModelEntry


def _resolve_deployment_dir(artifact_dir: Path, *, target) -> Path:
    """Resolve the directory containing deploy.so/deploy.json/deploy.params.

    MERA's TVMDeployer writes a project-style directory layout like:
      artifact_dir/
        build/<Target>/result/deploy.so

    `mera.mera_deployment.load_mera_deployment()` can load either:
    - a MERA project directory (if it recognizes it), or
    - a direct "result" directory containing deploy.* files.

    To be robust across versions/layout detection, we prefer the result directory
    when it exists for the requested target.
    """
    target_str = getattr(target, "str_val", None) or str(target)
    candidate = artifact_dir / "build" / target_str / "result"
    if (candidate / "deploy.so").exists():
        return candidate
    return artifact_dir


def _load_mera_deployment(path: str, target):
    """Load a built MERA TVM deployment from disk (hook for tests)."""
    import mera.mera_deployment as _md

    return _md.load_mera_deployment(path, target=target)


def _is_simulator_target(target) -> bool:
    """Return True if MERA target is the software Simulator."""
    if getattr(target, "str_val", None) == "Simulator":
        return True
    import mera

    return target is mera.Target.Simulator


class _SimulatorGraphRunner:
    """TVM graph executor wrapper matching the subset of MeraTvmModelRunner used here."""

    def __init__(self, rt_mod):
        self._rt = rt_mod

    def set_input(self, data):
        if not isinstance(data, dict):
            raise TypeError("inputs must be a dict[str, ndarray]")
        for k, v in data.items():
            self._rt.set_input(k, v)
        return self

    def run(self):
        self._rt.run()
        return self

    def get_outputs(self):
        n = self._rt.get_num_outputs()
        out = []
        for i in range(n):
            arr = self._rt.get_output(i)
            out.append(arr.asnumpy() if hasattr(arr, "asnumpy") else arr)
        return out


def _build_simulator_runner(deployment_dir: Path):
    """Load deploy.{so,json,params} via TVM graph executor (no mera_runtime_init_device)."""
    paths = {
        "deploy.so": deployment_dir / "deploy.so",
        "deploy.json": deployment_dir / "deploy.json",
        "deploy.params": deployment_dir / "deploy.params",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} under {deployment_dir}")

    try:
        import tvm
        from tvm.contrib.graph_executor import create as graph_create
        from tvm.runtime import load_module
    except (ImportError, OSError) as exc:
        raise ValueError(f"TVM is required for inference: {exc}") from exc

    try:
        lib = load_module(str(paths["deploy.so"]))
        graph_json = paths["deploy.json"].read_text(encoding="utf-8")
        rt_mod = graph_create(graph_json, lib, tvm.cpu(0))
        rt_mod.load_params(paths["deploy.params"].read_bytes())
    except Exception as exc:
        raise ValueError(f"Failed to load TVM deployment from {deployment_dir}: {exc}") from exc

    return _SimulatorGraphRunner(rt_mod)


def _make_runner(deployment_dir: Path, target):
    """Create a runner: Simulator uses TVM graph executor; other targets use MERA get_runner()."""
    if _is_simulator_target(target):
        return _build_simulator_runner(deployment_dir)
    deployment = _load_mera_deployment(str(deployment_dir), target)
    return deployment.get_runner()


@dataclass
class RunResult:
    """Holds inference outputs and per-iteration latency measurements."""

    outputs: list[dict] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def min_latency_ms(self) -> float:
        return min(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        s = sorted(self.latency_ms)
        return s[max(0, int(len(s) * 0.95) - 1)]


class MeraRuntime:
    """Loads compiled MERA artifacts and executes inference using dummy inputs."""

    def __init__(self, target=None):
        import mera  # lazy — mocked in tests

        self._target = target if target is not None else mera.Target.Simulator

    def run(
        self,
        entry: ModelEntry,
        artifact_dir: Path | str,
        *,
        iters: int = 1,
    ) -> RunResult:
        """Run inference on compiled artifacts, generating dummy inputs from entry.inputs.

        Raises ValueError if artifact_dir does not exist or entry.inputs is not configured.
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            raise ValueError(f"Artifact directory not found: {artifact_path}")
        if not entry.inputs:
            raise ValueError(f"Model '{entry.name}' has no inputs configured")

        import numpy as np  # lazy — available as transitive dep

        deployment_dir = _resolve_deployment_dir(artifact_path, target=self._target)

        try:
            runner = _make_runner(deployment_dir, self._target)

            inputs = {
                (inp.name or f"input_{i}"): np.zeros(inp.shape, dtype=inp.dtype)
                for i, inp in enumerate(entry.inputs)
            }

            latencies: list[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                runner.set_input(inputs)
                runner.run()
                latencies.append((time.perf_counter() - t0) * 1000)

            raw_outputs = runner.get_outputs()
            outputs_summary = [
                {"name": f"output_{i}", "shape": list(arr.shape), "dtype": str(arr.dtype)}
                for i, arr in enumerate(raw_outputs)
            ]

            return RunResult(outputs=outputs_summary, latency_ms=latencies)
        except ValueError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise ValueError(f"Run failed: {exc}") from exc
