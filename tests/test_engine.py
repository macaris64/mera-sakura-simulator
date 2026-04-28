"""BDD tests for SakuraEngine."""

from unittest.mock import MagicMock

import mera

from sakura_simulator import SakuraEngine as SakuraEngineFromPackage
from sakura_simulator.engine import GREETING, SakuraEngine


class TestSakuraEngineInitialization:
    def test_given_no_args_when_init_then_uses_simulator_target(self):
        # Given: no arguments
        # When: engine is constructed with defaults
        engine = SakuraEngine()
        # Then: target is the Simulator enum member
        assert engine.target is mera.Target.Simulator

    def test_given_no_args_when_init_then_uses_sakura_2c_platform(self):
        # Given: no arguments
        # When: engine is constructed with defaults
        engine = SakuraEngine()
        # Then: platform is the SAKURA_2C enum member
        assert engine.platform is mera.Platform.SAKURA_2C

    def test_given_custom_target_when_init_then_stores_provided_target(self):
        # Given: a specific non-default target
        # When: engine constructed with that target
        engine = SakuraEngine(target=mera.Target.InterpreterHw)
        # Then: the provided target is stored
        assert engine.target is mera.Target.InterpreterHw

    def test_given_custom_platform_when_init_then_stores_provided_platform(self):
        # Given: a specific non-default platform
        # When: engine constructed with that platform
        engine = SakuraEngine(platform=mera.Platform.SAKURA_1)
        # Then: the provided platform is stored
        assert engine.platform is mera.Platform.SAKURA_1


class TestSakuraEngineGreeting:
    def test_given_initialized_engine_when_greeting_called_then_returns_exact_string(self):
        # Given: a running engine
        engine = SakuraEngine()
        # When: greeting() is called
        result = engine.greeting()
        # Then: returns the canonical greeting
        assert result == "Hello from Sakura-II: Titan Biosignature Engine Active"

    def test_given_greeting_constant_when_imported_then_matches_method_return(self):
        # Given: the module-level GREETING constant and the package re-export
        engine = SakuraEngineFromPackage()
        # When: compared to engine.greeting()
        # Then: they are identical
        assert engine.greeting() == GREETING


class TestSakuraEngineLoadModel:
    def test_given_npu_capable_target_when_load_model_called_then_returns_npu_message(self):
        # Given: a target that exposes a load() method (NPU SDK ready)
        mock_target = MagicMock()
        entry = MagicMock()
        entry.name = "resnet50"
        entry.path = "/models/resnet50.mera"
        engine = SakuraEngine(target=mock_target)
        # When: load_model is called
        result = engine.load_model(entry)
        # Then: NPU activation message returned and load() was invoked with the model path
        assert "[NPU]" in result
        assert "resnet50" in result
        mock_target.load.assert_called_once_with(entry.path)

    def test_given_simulator_target_when_load_model_called_then_returns_staged_message(self):
        # Given: default mera.Target.Simulator (enum member — no load() method)
        engine = SakuraEngine()
        entry = MagicMock()
        entry.name = "mobilenet_v2"
        entry.path = "/models/mobilenet_v2.mera"
        # When: load_model is called
        result = engine.load_model(entry)
        # Then: simulator staging message returned
        assert "[SIMULATOR]" in result
        assert "mobilenet_v2" in result
        assert "staged for NPU activation" in result
