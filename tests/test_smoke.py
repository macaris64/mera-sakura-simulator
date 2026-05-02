"""Smoke tests — integration workflows for get, download, remove, and infer.

These tests use real ModelRegistry objects with real temp files.
httpx is mocked only to avoid live network calls; all file I/O and SHA-256
verification run against actual data on disk.
"""

import hashlib
import sys
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


def _make_llm_manifest(tmpdir: Path) -> Path:
    """Helper: write a real LLM manifest entry and return the path."""
    manifest_path = tmpdir / "models.yaml"
    artifact_dir = tmpdir / "artifacts"
    manifest_path.write_text(
        yaml.dump(
            {
                "models": [
                    {
                        "name": "tinyllama-smoke",
                        "version": "1.0.0",
                        "path": str(tmpdir / "tinyllama.onnx"),
                        "checksum": "abc",
                        "model_type": "llm",
                        "tokenizer_path": "tokenizers/tinyllama",
                        "context_length": 512,
                        "artifact_dir": str(artifact_dir),
                        "npu_constraints": {"max_power_watts": 15.0, "required_memory_mb": 2048},
                    }
                ]
            }
        )
    )
    return manifest_path


class TestSmokeInferCommand:
    """Smoke tests for models infer CLI path — no live model/tokenizer/mera required."""

    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.manifest_path = _make_llm_manifest(self.tmpdir)
        # Inject a default runtime mock so models_infer's lazy import doesn't
        # trigger a fresh module import (which would update the package attribute
        # and break patch() in TestMakeRunnerBranching later in the run).
        self._default_rt_mod = MagicMock()
        sys.modules["sakura_simulator.runtime"] = self._default_rt_mod

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules.pop("sakura_simulator.runtime", None)

    def test_smoke_infer_model_not_found_exits_1(self):
        # Given: manifest exists but model name is not registered
        from typer.testing import CliRunner as _CR

        from sakura_simulator.cli import app as _app

        # When: infer is called with an unknown model name
        result = _CR().invoke(
            _app,
            [
                "models",
                "infer",
                "nonexistent",
                "--prompt",
                "hi",
                "--manifest",
                str(self.manifest_path),
            ],
        )
        # Then: exits with code 1
        assert result.exit_code == 1

    def test_smoke_infer_not_compiled_exits_1_with_hint(self):
        # Given: model exists but artifact_dir does not exist (not compiled)
        from typer.testing import CliRunner as _CR

        from sakura_simulator.cli import app as _app

        # When: infer is called
        result = _CR().invoke(
            _app,
            [
                "models",
                "infer",
                "tinyllama-smoke",
                "--prompt",
                "What is life?",
                "--manifest",
                str(self.manifest_path),
            ],
        )
        # Then: exits with code 1 and compile hint in output
        assert result.exit_code == 1
        assert "not compiled" in result.output

    def test_smoke_infer_compiled_model_calls_runtime_and_prints_text(self):
        # Given: artifact_dir exists (model is compiled); MeraRuntime.infer is mocked
        (self.tmpdir / "artifacts").mkdir()

        mock_rt_mod = MagicMock()
        mock_rt = MagicMock()
        mock_rt_mod.MeraRuntime.return_value = mock_rt
        infer_result = MagicMock()
        infer_result.text = "Biosignature detected."
        infer_result.latency_ms = 200.0
        infer_result.token_ids = [1, 2, 3]
        mock_rt.infer.return_value = infer_result
        sys.modules["sakura_simulator.runtime"] = mock_rt_mod

        from typer.testing import CliRunner as _CR

        from sakura_simulator.cli import app as _app

        # When: infer is called
        result = _CR().invoke(
            _app,
            [
                "models",
                "infer",
                "tinyllama-smoke",
                "--prompt",
                "What is life?",
                "--manifest",
                str(self.manifest_path),
            ],
        )
        # Then: exits successfully and generated text is in output
        assert result.exit_code == 0
        assert "Biosignature detected." in result.output
