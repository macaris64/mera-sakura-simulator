"""BDD tests for ModelRegistry — schema validation, integrity checking, and lookup."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from sakura_simulator.registry import (
    ModelEntry,
    ModelInput,
    ModelManifest,
    ModelRegistry,
    NPUConstraints,
)


class TestModelInputModel:
    def test_given_all_fields_when_constructed_then_stores_values(self):
        # Given: all fields for a model input
        # When: ModelInput is created
        inp = ModelInput(name="data", dtype="float32", shape=[1, 3, 224, 224])
        # Then: all fields are stored
        assert inp.name == "data"
        assert inp.dtype == "float32"
        assert inp.shape == [1, 3, 224, 224]

    def test_given_no_name_when_constructed_then_name_is_none(self):
        # Given: name is omitted
        # When: ModelInput is created without a name
        inp = ModelInput(dtype="float32", shape=[1])
        # Then: name defaults to None
        assert inp.name is None

    def test_given_missing_dtype_when_constructed_then_raises_validation_error(self):
        # Given: dtype is omitted (required field)
        # When / Then: ValidationError is raised
        with pytest.raises(ValidationError):
            ModelInput(shape=[1, 3, 224, 224])  # type: ignore[call-arg]

    def test_given_missing_shape_when_constructed_then_raises_validation_error(self):
        # Given: shape is omitted (required field)
        # When / Then: ValidationError is raised
        with pytest.raises(ValidationError):
            ModelInput(dtype="float32")  # type: ignore[call-arg]


class TestModelEntryExtended:
    def _make_constraints(self) -> NPUConstraints:
        return NPUConstraints(max_power_watts=10.0, required_memory_mb=256)

    def test_given_no_optional_fields_when_constructed_then_defaults_are_none(self):
        # Given: only required fields (backward-compat check)
        # When: ModelEntry is created without new optional fields
        entry = ModelEntry(
            name="m1",
            version="1.0",
            path="models/m1.onnx",
            checksum="abc",
            npu_constraints=self._make_constraints(),
        )
        # Then: new optional fields default to None
        assert entry.format is None
        assert entry.artifact_dir is None
        assert entry.inputs is None

    def test_given_format_and_artifact_dir_when_constructed_then_fields_stored(self):
        # Given: format and artifact_dir provided
        # When: ModelEntry is created with those fields
        entry = ModelEntry(
            name="m1",
            version="1.0",
            path="models/m1.onnx",
            checksum="abc",
            format="onnx",
            artifact_dir="artifacts/m1/1.0",
            npu_constraints=self._make_constraints(),
        )
        # Then: both fields are stored
        assert entry.format == "onnx"
        assert entry.artifact_dir == "artifacts/m1/1.0"

    def test_given_inputs_list_when_constructed_then_inputs_parsed(self):
        # Given: inputs list with one tensor spec
        # When: ModelEntry is created with inputs
        entry = ModelEntry(
            name="m1",
            version="1.0",
            path="models/m1.onnx",
            checksum="abc",
            inputs=[{"name": "data", "dtype": "float32", "shape": [1, 3, 224, 224]}],
            npu_constraints=self._make_constraints(),
        )
        # Then: inputs are parsed into ModelInput objects
        assert entry.inputs is not None
        assert len(entry.inputs) == 1
        assert entry.inputs[0].name == "data"
        assert entry.inputs[0].dtype == "float32"
        assert entry.inputs[0].shape == [1, 3, 224, 224]

    def test_given_no_llm_fields_when_constructed_then_model_type_defaults_to_vision(self):
        # Given: no LLM-specific fields provided
        # When: ModelEntry is created with only required fields
        entry = ModelEntry(
            name="m1",
            version="1.0",
            path="models/m1.onnx",
            checksum="abc",
            npu_constraints=self._make_constraints(),
        )
        # Then: model_type defaults to "vision", LLM fields default to None
        assert entry.model_type == "vision"
        assert entry.tokenizer_path is None
        assert entry.context_length is None
        assert entry.generation_config is None

    def test_given_all_llm_fields_when_constructed_then_all_fields_stored(self):
        # Given: all four LLM optional fields provided
        # When: ModelEntry is created with LLM metadata
        entry = ModelEntry(
            name="tinyllama",
            version="1.0.0",
            path="models/tinyllama.onnx",
            checksum="abc",
            npu_constraints=self._make_constraints(),
            model_type="llm",
            tokenizer_path="tokenizers/tinyllama",
            context_length=2048,
            generation_config={"max_new_tokens": 128, "temperature": 1.0},
        )
        # Then: all LLM fields are accessible
        assert entry.model_type == "llm"
        assert entry.tokenizer_path == "tokenizers/tinyllama"
        assert entry.context_length == 2048
        assert entry.generation_config == {"max_new_tokens": 128, "temperature": 1.0}

    def test_given_model_type_llm_only_when_constructed_then_optional_llm_fields_none(self):
        # Given: model_type set to "llm" but optional LLM fields not provided
        # When: ModelEntry is created
        entry = ModelEntry(
            name="m1",
            version="1.0",
            path="p",
            checksum="c",
            npu_constraints=self._make_constraints(),
            model_type="llm",
        )
        # Then: optional fields default to None
        assert entry.tokenizer_path is None
        assert entry.context_length is None
        assert entry.generation_config is None


class TestModelRegistryIsCompiled:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        manifest_path = self.tmpdir / "models.yaml"
        manifest_path.write_text(
            yaml.dump(
                {
                    "models": [
                        {
                            "name": "m1",
                            "version": "1.0",
                            "path": str(self.tmpdir / "m1.onnx"),
                            "checksum": "abc",
                            "npu_constraints": {"max_power_watts": 5.0, "required_memory_mb": 128},
                        }
                    ]
                }
            )
        )
        self.registry = ModelRegistry(manifest_path)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_given_artifact_dir_none_when_is_compiled_then_returns_false(self):
        # Given: entry with no artifact_dir configured
        entry = self.registry.get_model("m1")
        assert entry is not None
        # When: is_compiled is called
        result = self.registry.is_compiled(entry)
        # Then: False (nothing to compile to)
        assert result is False

    def test_given_artifact_dir_not_on_disk_when_is_compiled_then_returns_false(self):
        # Given: entry with artifact_dir pointing to a non-existent directory
        entry = self.registry.get_model("m1")
        assert entry is not None
        entry = entry.model_copy(update={"artifact_dir": str(self.tmpdir / "nonexistent")})
        # When: is_compiled is called
        result = self.registry.is_compiled(entry)
        # Then: False (directory does not exist)
        assert result is False

    def test_given_artifact_dir_exists_on_disk_when_is_compiled_then_returns_true(self):
        # Given: entry with artifact_dir pointing to an existing directory
        artifact_path = self.tmpdir / "artifacts" / "m1"
        artifact_path.mkdir(parents=True)
        entry = self.registry.get_model("m1")
        assert entry is not None
        entry = entry.model_copy(update={"artifact_dir": str(artifact_path)})
        # When: is_compiled is called
        result = self.registry.is_compiled(entry)
        # Then: True (compiled artifacts are present)
        assert result is True


class TestNPUConstraintsModel:
    def test_given_valid_data_when_constructed_then_stores_fields(self):
        # Given: valid NPU constraint values
        # When: NPUConstraints is created
        constraints = NPUConstraints(max_power_watts=12.5, required_memory_mb=512)
        # Then: fields are stored correctly
        assert constraints.max_power_watts == 12.5
        assert constraints.required_memory_mb == 512

    def test_given_missing_field_when_constructed_then_raises_validation_error(self):
        # Given: missing required_memory_mb field
        # When / Then: ValidationError is raised
        with pytest.raises(ValidationError):
            NPUConstraints(max_power_watts=10.0)  # type: ignore[call-arg]


class TestModelEntryModel:
    def test_given_complete_data_when_constructed_then_stores_all_fields(self):
        # Given: all required fields for a model entry
        constraints = NPUConstraints(max_power_watts=10.0, required_memory_mb=256)
        # When: ModelEntry is created
        entry = ModelEntry(
            name="resnet50",
            version="1.0.0",
            path="models/resnet50.mera",
            checksum="abc123",
            npu_constraints=constraints,
        )
        # Then: all fields are accessible
        assert entry.name == "resnet50"
        assert entry.version == "1.0.0"
        assert entry.checksum == "abc123"
        assert entry.npu_constraints.max_power_watts == 10.0

    def test_given_missing_name_when_constructed_then_raises_validation_error(self):
        # Given: name field is absent
        constraints = NPUConstraints(max_power_watts=10.0, required_memory_mb=256)
        # When / Then: ValidationError is raised
        with pytest.raises(ValidationError):
            ModelEntry(  # type: ignore[call-arg]
                version="1.0.0",
                path="models/r.mera",
                checksum="abc",
                npu_constraints=constraints,
            )


class TestModelManifestModel:
    def test_given_list_of_entries_when_constructed_then_stores_models(self):
        # Given: one valid model entry
        constraints = NPUConstraints(max_power_watts=5.0, required_memory_mb=128)
        entry = ModelEntry(
            name="m1", version="1.0", path="m1.mera", checksum="x", npu_constraints=constraints
        )
        # When: ModelManifest is created
        manifest = ModelManifest(models=[entry])
        # Then: models list is populated
        assert len(manifest.models) == 1
        assert manifest.models[0].name == "m1"


class TestModelRegistryLoad:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def _write_manifest(self, data: dict) -> Path:
        path = self.tmpdir / "models.yaml"
        path.write_text(yaml.dump(data))
        return path

    def test_given_valid_manifest_when_registry_created_then_loads_all_models(self):
        # Given: a well-formed manifest YAML with one model
        manifest_path = self._write_manifest(
            {
                "models": [
                    {
                        "name": "resnet50",
                        "version": "1.0.0",
                        "path": str(self.tmpdir / "r.mera"),
                        "checksum": "abc123",
                        "npu_constraints": {"max_power_watts": 10.0, "required_memory_mb": 256},
                    }
                ]
            }
        )
        # When: registry is created
        registry = ModelRegistry(manifest_path)
        # Then: model list is populated
        assert len(registry.list_models()) == 1
        assert registry.list_models()[0].name == "resnet50"

    def test_given_missing_manifest_when_registry_created_then_raises_file_not_found(self):
        # Given: a path that does not exist on disk
        missing_path = self.tmpdir / "nonexistent.yaml"
        # When / Then: FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            ModelRegistry(missing_path)

    def test_given_invalid_yaml_schema_when_registry_created_then_raises_validation_error(self):
        # Given: YAML missing required fields (no checksum, no npu_constraints)
        manifest_path = self._write_manifest(
            {"models": [{"name": "bad_model", "version": "1.0.0"}]}
        )
        # When / Then: Pydantic ValidationError is raised
        with pytest.raises(ValidationError):
            ModelRegistry(manifest_path)


class TestModelRegistryGetModel:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        manifest_path = self.tmpdir / "models.yaml"
        manifest_path.write_text(
            yaml.dump(
                {
                    "models": [
                        {
                            "name": "resnet50",
                            "version": "1.0.0",
                            "path": str(self.tmpdir / "r.mera"),
                            "checksum": "abc",
                            "npu_constraints": {"max_power_watts": 10.0, "required_memory_mb": 256},
                        }
                    ]
                }
            )
        )
        self.registry = ModelRegistry(manifest_path)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_given_known_name_when_get_model_called_then_returns_entry(self):
        # Given: registry loaded with resnet50
        # When: get_model is called with the known name
        entry = self.registry.get_model("resnet50")
        # Then: the matching entry is returned
        assert entry is not None
        assert entry.name == "resnet50"

    def test_given_unknown_name_when_get_model_called_then_returns_none(self):
        # Given: registry loaded with resnet50 only
        # When: get_model is called with an unregistered name
        result = self.registry.get_model("ghost_model")
        # Then: None is returned
        assert result is None


class TestModelRegistryIsSpaceReady:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def _make_registry(self, checksum: str, model_path: str) -> ModelRegistry:
        manifest_path = self.tmpdir / "models.yaml"
        manifest_path.write_text(
            yaml.dump(
                {
                    "models": [
                        {
                            "name": "m1",
                            "version": "1.0",
                            "path": model_path,
                            "checksum": checksum,
                            "npu_constraints": {"max_power_watts": 5.0, "required_memory_mb": 128},
                        }
                    ]
                }
            )
        )
        return ModelRegistry(manifest_path)

    def test_given_file_exists_and_checksum_correct_when_is_space_ready_then_returns_true(self):
        # Given: model file exists with content matching the manifest checksum
        model_file = self.tmpdir / "m1.mera"
        content = b"model binary data"
        model_file.write_bytes(content)
        correct_checksum = hashlib.sha256(content).hexdigest()
        registry = self._make_registry(correct_checksum, str(model_file))
        entry = registry.get_model("m1")
        # When: is_space_ready is called
        result = registry.is_space_ready(entry)
        # Then: True (integrity verified)
        assert result is True

    def test_given_file_exists_and_checksum_wrong_when_is_space_ready_then_returns_false(self):
        # Given: model file exists but its hash differs from the manifest checksum
        model_file = self.tmpdir / "m1.mera"
        model_file.write_bytes(b"model binary data")
        registry = self._make_registry("000000wrong_checksum", str(model_file))
        entry = registry.get_model("m1")
        # When: is_space_ready is called
        result = registry.is_space_ready(entry)
        # Then: False (bit-flip / tamper detected)
        assert result is False

    def test_given_file_missing_when_is_space_ready_then_returns_false(self):
        # Given: model file path referenced in manifest does not exist on disk
        missing_path = str(self.tmpdir / "nonexistent.mera")
        registry = self._make_registry("any_checksum", missing_path)
        entry = registry.get_model("m1")
        # When: is_space_ready is called
        result = registry.is_space_ready(entry)
        # Then: False (file absent — mission abort)
        assert result is False


class TestModelRegistryDownload:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.content = b"fake model binary payload"
        self.checksum = hashlib.sha256(self.content).hexdigest()
        manifest_path = self.tmpdir / "models.yaml"
        manifest_path.write_text(
            yaml.dump(
                {
                    "models": [
                        {
                            "name": "resnet50",
                            "version": "1.0.0",
                            "path": str(self.tmpdir / "models" / "resnet50.mera"),
                            "checksum": self.checksum,
                            "source_url": "https://example.com/resnet50.mera",
                            "npu_constraints": {"max_power_watts": 10.0, "required_memory_mb": 256},
                        }
                    ]
                }
            )
        )
        self.registry = ModelRegistry(manifest_path)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_given_unknown_model_when_download_called_then_raises_value_error(self):
        # Given: registry does not contain "ghost_model"
        # When / Then: ValueError with "not in registry"
        with pytest.raises(ValueError, match="not in registry"):
            self.registry.download("ghost_model")

    def test_given_model_without_source_url_when_download_called_then_raises_value_error(self):
        # Given: manifest entry with no source_url
        no_url_path = self.tmpdir / "no_url.yaml"
        no_url_path.write_text(
            yaml.dump(
                {
                    "models": [
                        {
                            "name": "local_only",
                            "version": "1.0",
                            "path": str(self.tmpdir / "local.mera"),
                            "checksum": "abc",
                            "npu_constraints": {"max_power_watts": 5.0, "required_memory_mb": 128},
                        }
                    ]
                }
            )
        )
        registry = ModelRegistry(no_url_path)
        # When / Then: ValueError with "no source_url"
        with pytest.raises(ValueError, match="no source_url"):
            registry.download("local_only")

    def test_given_valid_model_when_download_called_then_saves_file_and_returns_path(self):
        # Given: HTTP response returns content matching the manifest checksum
        mock_response = MagicMock()
        mock_response.content = self.content
        # When: download is called
        with patch("sakura_simulator.registry.httpx.get", return_value=mock_response):
            path = self.registry.download("resnet50")
        # Then: file written to disk and path returned
        assert path.exists()
        assert path.read_bytes() == self.content
        mock_response.raise_for_status.assert_called_once()

    def test_given_checksum_mismatch_when_download_called_then_raises_and_removes_file(self):
        # Given: HTTP response returns corrupted/tampered content
        mock_response = MagicMock()
        mock_response.content = b"corrupted payload bit-flip detected"
        model_path = self.tmpdir / "models" / "resnet50.mera"
        # When / Then: ValueError with "Checksum mismatch"
        with patch("sakura_simulator.registry.httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                self.registry.download("resnet50")
        # And: the corrupted file is removed from disk
        assert not model_path.exists()


class TestModelRegistryRemove:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.model_file = self.tmpdir / "models" / "resnet50.mera"
        manifest_path = self.tmpdir / "models.yaml"
        manifest_path.write_text(
            yaml.dump(
                {
                    "models": [
                        {
                            "name": "resnet50",
                            "version": "1.0.0",
                            "path": str(self.model_file),
                            "checksum": "abc",
                            "npu_constraints": {"max_power_watts": 10.0, "required_memory_mb": 256},
                        }
                    ]
                }
            )
        )
        self.registry = ModelRegistry(manifest_path)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_given_unknown_model_when_remove_called_then_raises_value_error(self):
        # Given: registry does not contain "ghost_model"
        # When / Then: ValueError with "not in registry"
        with pytest.raises(ValueError, match="not in registry"):
            self.registry.remove("ghost_model")

    def test_given_downloaded_model_when_remove_called_then_deletes_file_and_returns_path(self):
        # Given: model file exists on disk
        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_file.write_bytes(b"model data")
        # When: remove is called
        returned_path = self.registry.remove("resnet50")
        # Then: file is gone and the path is returned
        assert returned_path == self.model_file
        assert not self.model_file.exists()
