"""Model registry: load, validate, and inspect YAML-defined NPU model entries."""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel


class NPUConstraints(BaseModel):
    max_power_watts: float
    required_memory_mb: int


class ModelEntry(BaseModel):
    name: str
    version: str
    path: str
    checksum: str
    npu_constraints: NPUConstraints


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
