"""MeraCompiler: compile a source model into SAKURA-II deployment artifacts."""

from pathlib import Path

from sakura_simulator.registry import ModelEntry


class MeraCompiler:
    """Wraps MERA TVMDeployer to compile ONNX models for a given target/platform."""

    def __init__(self, target=None, platform=None):
        import mera  # lazy — mocked in tests

        self._target = target if target is not None else mera.Target.Simulator
        self._platform = platform if platform is not None else mera.Platform.SAKURA_2C

    def compile(self, entry: ModelEntry) -> Path:
        """Compile a source ONNX model to artifact_dir and return the artifact path.

        Raises ValueError if the format is not 'onnx', artifact_dir is not configured,
        or the source model file does not exist on disk.
        """
        if entry.format != "onnx":
            raise ValueError(f"Unsupported format '{entry.format}': only 'onnx' is supported")
        if entry.artifact_dir is None:
            raise ValueError(f"Model '{entry.name}' has no artifact_dir configured")
        source_path = Path(entry.path)
        if not source_path.exists():
            raise ValueError(f"Source model not found: {source_path}")

        artifact_path = Path(entry.artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        import mera  # lazy — mocked in tests

        deployer = mera.TVMDeployer(str(artifact_path))
        loader = mera.ModelLoader(deployer)
        model = loader.from_onnx(str(source_path), model_name=entry.name)
        deployer.deploy(model, mera_platform=self._platform, target=self._target)
        return artifact_path
