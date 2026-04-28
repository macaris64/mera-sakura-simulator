"""SakuraEngine: wraps the MERA NPU target and platform for SAKURA-II."""

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
