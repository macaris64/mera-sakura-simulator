"""BDD tests for ModelRegistry — schema validation, integrity checking, and lookup."""

import hashlib
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sakura_simulator.registry import ModelEntry, ModelManifest, ModelRegistry, NPUConstraints


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
