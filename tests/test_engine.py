"""BDD tests for SakuraEngine."""

from unittest.mock import MagicMock

from sakura_simulator.engine import GREETING, SakuraEngine
from sakura_simulator import SakuraEngine as SakuraEngineFromPackage


class TestSakuraEngineInitialization:

    def test_given_no_target_when_init_then_creates_sakura_ii_target(self):
        # Given: no target argument
        # When: engine is constructed
        engine = SakuraEngine()
        # Then: internal target is a MockTarget with SAKURA_II type
        assert engine.target is not None
        assert engine.target.target_type == "SAKURA_II"

    def test_given_custom_target_when_init_then_stores_provided_target(self):
        # Given: a pre-built mock target
        custom_target = MagicMock()
        # When: engine constructed with that target
        engine = SakuraEngine(target=custom_target)
        # Then: the provided target is stored
        assert engine.target is custom_target


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
