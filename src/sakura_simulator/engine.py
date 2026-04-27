"""SakuraEngine: hardware-agnostic wrapper around the MERA NPU target."""

try:
    import mera
except ImportError:  # pragma: no cover
    # Simulation fallback — real MERA SDK not installed.
    from types import ModuleType as _ModuleType

    mera = _ModuleType("mera")

    class _SimTarget:
        SAKURA_II = "SAKURA_II"

        def __init__(self, target_type: str):
            self.target_type = target_type

    mera.Target = _SimTarget  # type: ignore[attr-defined]

GREETING = "Hello from Sakura-II: Titan Biosignature Engine Active"


class SakuraEngine:
    """Initializes a MERA Target and exposes the NPU greeting."""

    def __init__(self, target=None):
        if target is None:
            target = mera.Target(mera.Target.SAKURA_II)
        self._target = target

    @property
    def target(self):
        return self._target

    def greeting(self) -> str:
        return GREETING
