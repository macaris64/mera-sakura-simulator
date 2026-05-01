"""SakuraEngine: wraps the MERA NPU target and platform for SAKURA-II."""

from pathlib import Path

import mera

GREETING = "Hello from Sakura-II: Titan Biosignature Engine Active"


class SakuraEngine:
    """Binds a MERA Target + Platform and exposes the NPU greeting."""

    def __init__(
        self,
        target: mera.Target = mera.Target.Simulator,
        platform: mera.Platform = mera.Platform.SAKURA_2C,
    ):
        self._target = target
        self._platform = platform

    @property
    def target(self) -> mera.Target:
        return self._target

    @property
    def platform(self) -> mera.Platform:
        return self._platform

    def greeting(self) -> str:
        return GREETING

    def load_model(self, entry) -> str:
        """Load a model onto the NPU, or stage it for activation if the SDK is not ready."""
        try:
            self._target.load(entry.path)
            return f"[NPU] Model {entry.name} loaded on {self._platform}"
        except AttributeError:
            return f"[SIMULATOR] Model {entry.name} staged for NPU activation"

    def compile_model(self, entry) -> Path:
        """Compile a source model to deployment artifacts via MeraCompiler."""
        from sakura_simulator.compiler import MeraCompiler

        return MeraCompiler(self._target, self._platform).compile(entry)

    def run_model(self, entry, *, iters: int = 1):
        """Run inference on compiled artifacts via MeraRuntime."""
        from sakura_simulator.runtime import MeraRuntime

        return MeraRuntime(target=self._target).run(entry, entry.artifact_dir, iters=iters)

    def run_model_infer(
        self,
        entry,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ):
        """Run LLM inference on a compiled model via MeraRuntime.infer()."""
        from sakura_simulator.runtime import MeraRuntime

        return MeraRuntime(target=self._target).infer(
            entry,
            entry.artifact_dir,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
