"""Model registry: load, validate, and inspect YAML-defined NPU model entries."""

import hashlib
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel


class NPUConstraints(BaseModel):
    max_power_watts: float
    required_memory_mb: int


class ModelInput(BaseModel):
    name: str | None = None
    dtype: str
    shape: list[int]


class ModelEntry(BaseModel):
    name: str
    version: str
    path: str
    checksum: str
    source_url: str | None = None
    format: str | None = None
    artifact_dir: str | None = None
    inputs: list[ModelInput] | None = None
    npu_constraints: NPUConstraints
    model_type: str = "vision"
    tokenizer_path: str | None = None
    context_length: int | None = None
    generation_config: dict | None = None


class ModelManifest(BaseModel):
    models: list[ModelEntry]


class ModelRegistry:
    """Load a YAML manifest and provide model lookup and SHA-256 integrity checking."""

    def __init__(self, manifest_path: str | Path = "configs/models.yaml") -> None:
        self._path = Path(manifest_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Model manifest not found: {self._path}")
        with self._path.open() as fh:
            data = yaml.safe_load(fh)
        self._manifest = ModelManifest.model_validate(data)

    def list_models(self) -> list[ModelEntry]:
        """Return all registered model entries."""
        return self._manifest.models

    def get_model(self, name: str) -> ModelEntry | None:
        """Return the ModelEntry for the given name, or None if not found."""
        for entry in self._manifest.models:
            if entry.name == name:
                return entry
        return None

    def is_space_ready(self, entry: ModelEntry) -> bool:
        """Return True iff the model file exists and its SHA-256 matches the manifest."""
        model_path = Path(entry.path)
        if not model_path.exists():
            return False
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return actual == entry.checksum

    def is_compiled(self, entry: ModelEntry) -> bool:
        """Return True iff artifact_dir is configured and exists on disk."""
        if entry.artifact_dir is None:
            return False
        return Path(entry.artifact_dir).exists()

    def download(self, name: str) -> Path:
        """Download a model file and verify its SHA-256 checksum immediately after write.

        Raises ValueError for unknown model name, missing source_url, or checksum mismatch.
        Raises httpx.HTTPStatusError for non-2xx HTTP responses.
        """
        entry = self.get_model(name)
        if entry is None:
            raise ValueError(f"Model '{name}' not in registry")
        if entry.source_url is None:
            raise ValueError(f"Model '{name}' has no source_url configured")

        model_path = Path(entry.path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        response = httpx.get(entry.source_url, follow_redirects=True, timeout=300.0)
        response.raise_for_status()
        model_path.write_bytes(response.content)

        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != entry.checksum:
            model_path.unlink()
            raise ValueError(
                f"Checksum mismatch for '{name}': expected {entry.checksum}, got {actual}"
            )

        return model_path

    def remove(self, name: str) -> Path:
        """Delete a downloaded model file from disk.

        Raises ValueError if the model is not in the registry.
        Raises FileNotFoundError if the file does not exist on disk.
        """
        entry = self.get_model(name)
        if entry is None:
            raise ValueError(f"Model '{name}' not in registry")
        model_path = Path(entry.path)
        model_path.unlink()
        return model_path
