"""Smoke tests — integration workflows for get, download, and remove.

These tests use real ModelRegistry objects with real temp files.
httpx is mocked only to avoid live network calls; all file I/O and SHA-256
verification run against actual data on disk.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from sakura_simulator.registry import ModelRegistry


def _make_registry(tmpdir: Path, content: bytes, source_url: str = "https://example.com/m.mera"):
    """Helper: write a real manifest and return a loaded ModelRegistry."""
    checksum = hashlib.sha256(content).hexdigest()
    model_path = tmpdir / "models" / "sakura-slm.mera"
    manifest_path = tmpdir / "models.yaml"
    manifest_path.write_text(
        yaml.dump(
            {
                "models": [
                    {
                        "name": "sakura-slm-v1",
                        "version": "1.0.0",
                        "path": str(model_path),
                        "checksum": checksum,
                        "source_url": source_url,
                        "npu_constraints": {"max_power_watts": 10.0, "required_memory_mb": 256},
                    }
                ]
            }
        )
    )
    return ModelRegistry(manifest_path), model_path, checksum


class TestSmokeGetModel:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_smoke_get_model_returns_entry_with_all_fields(self):
        # Given: a real manifest loaded into a real registry
        registry, _, _ = _make_registry(self.tmpdir, b"model payload")
        # When: get_model is called by name
        entry = registry.get_model("sakura-slm-v1")
        # Then: all manifest fields are accessible on the entry
        assert entry is not None
        assert entry.name == "sakura-slm-v1"
        assert entry.version == "1.0.0"
        assert entry.source_url == "https://example.com/m.mera"
        assert entry.npu_constraints.max_power_watts == 10.0
        assert entry.npu_constraints.required_memory_mb == 256

    def test_smoke_get_unknown_model_returns_none(self):
        # Given: a real registry
        registry, _, _ = _make_registry(self.tmpdir, b"model payload")
        # When: get_model is called with a name not in the manifest
        result = registry.get_model("nonexistent-model")
        # Then: None is returned
        assert result is None


class TestSmokeDownloadModel:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.content = b"binary model payload for SAKURA-II"

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_smoke_download_writes_file_and_passes_integrity_check(self):
        # Given: registry with a downloadable model; HTTP returns exact content
        registry, model_path, checksum = _make_registry(self.tmpdir, self.content)
        mock_response = MagicMock()
        mock_response.content = self.content
        # When: download is called
        with patch("sakura_simulator.registry.httpx.get", return_value=mock_response):
            returned_path = registry.download("sakura-slm-v1")
        # Then: file written to the declared path with matching bytes
        assert returned_path == model_path
        assert model_path.exists()
        assert model_path.read_bytes() == self.content
        # And: SHA-256 of the written file matches the manifest checksum
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == checksum

    def test_smoke_download_then_is_space_ready_returns_true(self):
        # Given: registry + mocked HTTP
        registry, _, _ = _make_registry(self.tmpdir, self.content)
        mock_response = MagicMock()
        mock_response.content = self.content
        # When: model is downloaded
        with patch("sakura_simulator.registry.httpx.get", return_value=mock_response):
            registry.download("sakura-slm-v1")
        # Then: is_space_ready confirms the file is mission-ready
        entry = registry.get_model("sakura-slm-v1")
        assert registry.is_space_ready(entry) is True


class TestSmokeRemoveModel:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.content = b"model binary data"

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_smoke_remove_deletes_file_from_disk(self):
        # Given: a model file that exists on disk
        registry, model_path, _ = _make_registry(self.tmpdir, self.content)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(self.content)
        assert model_path.exists()
        # When: remove is called
        returned_path = registry.remove("sakura-slm-v1")
        # Then: file is deleted and the path is returned
        assert returned_path == model_path
        assert not model_path.exists()

    def test_smoke_remove_then_is_space_ready_returns_false(self):
        # Given: model file on disk, confirmed space-ready
        registry, model_path, checksum = _make_registry(self.tmpdir, self.content)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(self.content)
        entry = registry.get_model("sakura-slm-v1")
        assert registry.is_space_ready(entry) is True
        # When: the model file is removed
        registry.remove("sakura-slm-v1")
        # Then: is_space_ready reflects the file is gone
        assert registry.is_space_ready(entry) is False

    def test_smoke_full_lifecycle_download_verify_remove(self):
        # Given: registry with a downloadable model
        registry, model_path, _ = _make_registry(self.tmpdir, self.content)
        entry = registry.get_model("sakura-slm-v1")
        mock_response = MagicMock()
        mock_response.content = self.content
        # When: full lifecycle — download, verify, remove
        assert registry.is_space_ready(entry) is False  # not yet on disk
        with patch("sakura_simulator.registry.httpx.get", return_value=mock_response):
            registry.download("sakura-slm-v1")
        assert registry.is_space_ready(entry) is True  # mission-ready
        registry.remove("sakura-slm-v1")
        assert registry.is_space_ready(entry) is False  # decommissioned
