"""BDD tests for the Streamlit UI page."""

import sys
from unittest.mock import MagicMock

import sakura_simulator.app as app_module


class TestStreamlitPage:
    def setup_method(self):
        """Reset mock state, re-apply cache_resource pass-through, and stub registry."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn
        # Stub registry to raise FileNotFoundError so _render_model_control_center
        # always hits the warning branch in these tests — order-independent.
        mock_module = MagicMock()
        mock_module.ModelRegistry.side_effect = FileNotFoundError("no manifest")
        sys.modules["sakura_simulator.registry"] = mock_module

    def test_given_app_module_when_main_called_then_sets_page_config(self):
        # Given: streamlit mocked, button not clicked
        self.st.button.return_value = False
        # When: main() renders the page
        app_module.main()
        # Then: set_page_config was called once
        self.st.set_page_config.assert_called_once()

    def test_given_app_module_when_main_called_then_sets_title(self):
        # Given: button not clicked
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then
        self.st.title.assert_called_once_with("SAKURA-II NPU Simulator")

    def test_given_button_not_clicked_when_main_called_then_success_not_shown(self):
        # Given: button returns False
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: st.success is NOT called
        self.st.success.assert_not_called()

    def test_given_button_clicked_when_main_called_then_displays_greeting(self):
        # Given: button IS clicked
        self.st.button.return_value = True
        # When
        app_module.main()
        # Then: success shows the exact greeting
        self.st.success.assert_called_once_with(
            "Hello from Sakura-II: Titan Biosignature Engine Active"
        )


class TestModelControlCenter:
    def setup_method(self):
        """Reset streamlit mock and inject a successful registry stub with two models."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn

        # Provide a proper 2-tuple for st.sidebar.columns so col1, col2 unpacks cleanly
        self.mock_col1 = MagicMock()
        self.mock_col2 = MagicMock()
        self.mock_col1.button.return_value = False
        self.mock_col2.button.return_value = False
        self.st.sidebar.columns.return_value = (self.mock_col1, self.mock_col2)

        self.mock_registry_cls = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_registry_cls.return_value = self.mock_registry

        m1 = MagicMock()
        m1.name = "resnet50"
        m2 = MagicMock()
        m2.name = "mobilenet_v2"
        self.mock_registry.list_models.return_value = [m1, m2]
        self.mock_registry.is_space_ready.return_value = True
        self.mock_registry.is_compiled.return_value = True

        mock_module = MagicMock()
        mock_module.ModelRegistry = self.mock_registry_cls
        sys.modules["sakura_simulator.registry"] = mock_module

    def test_given_manifest_loads_when_main_called_then_sidebar_header_shown(self):
        # Given: registry loads successfully
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: Model Control Center header appears in sidebar
        self.st.sidebar.header.assert_called_once_with("Model Control Center")

    def test_given_manifest_loads_when_main_called_then_selectbox_contains_model_names(self):
        # Given: registry returns two models
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: selectbox is populated with both model names
        call_args = self.st.sidebar.selectbox.call_args
        assert "resnet50" in call_args[0][1]
        assert "mobilenet_v2" in call_args[0][1]

    def test_given_model_selected_when_main_called_then_stored_in_session_state(self):
        # Given: selectbox returns a selected model name
        self.st.sidebar.selectbox.return_value = "resnet50"
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: session_state is updated with the active model
        self.st.session_state.__setitem__.assert_called_with("active_model", "resnet50")

    def test_given_manifest_missing_when_main_called_then_sidebar_warning_shown(self):
        # Given: registry constructor raises FileNotFoundError
        self.mock_registry_cls.side_effect = FileNotFoundError("no manifest")
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: sidebar warning is shown and selectbox is never rendered
        self.st.sidebar.warning.assert_called_once()
        self.st.sidebar.selectbox.assert_not_called()

    def test_given_model_integrity_fails_when_main_called_then_red_indicator_shown(self):
        # Given: is_space_ready returns False (checksum mismatch / tampered file)
        self.mock_registry.is_space_ready.return_value = False
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: at least one red indicator markdown was rendered
        markdown_calls = [str(c) for c in self.st.sidebar.markdown.call_args_list]
        assert any(":red_circle:" in c for c in markdown_calls)

    def test_given_empty_registry_when_main_called_then_no_indicators_shown(self):
        # Given: registry contains no models
        self.mock_registry.list_models.return_value = []
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: selectbox is called with empty list, no indicator markdown rendered
        self.st.sidebar.selectbox.assert_called_once_with("Active Model", [])
        self.st.sidebar.markdown.assert_not_called()


class TestModelControlCenterCompileRun:
    def setup_method(self):
        """Reset mocks and inject registry, compiler, and runtime stubs."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn
        self.st.button.return_value = False

        self.mock_col1 = MagicMock()
        self.mock_col2 = MagicMock()
        self.mock_col1.button.return_value = False
        self.mock_col2.button.return_value = False
        self.st.sidebar.columns.return_value = (self.mock_col1, self.mock_col2)

        # Registry stub with one model
        self.mock_registry = MagicMock()
        self.entry = MagicMock()
        self.entry.name = "resnet50"
        self.entry.version = "2.7.0"
        self.entry.artifact_dir = "artifacts/resnet50/2.7.0"
        self.mock_registry.list_models.return_value = [self.entry]
        self.mock_registry.is_space_ready.return_value = True
        self.mock_registry.is_compiled.return_value = True

        mock_reg_mod = MagicMock()
        mock_reg_mod.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_reg_mod

        # Compiler stub
        self.mock_compiler = MagicMock()
        mock_compiler_mod = MagicMock()
        mock_compiler_mod.MeraCompiler.return_value = self.mock_compiler
        sys.modules["sakura_simulator.compiler"] = mock_compiler_mod

        # Runtime stub
        self.mock_runtime = MagicMock()
        mock_runtime_mod = MagicMock()
        mock_runtime_mod.MeraRuntime.return_value = self.mock_runtime
        sys.modules["sakura_simulator.runtime"] = mock_runtime_mod

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.registry", None)
        sys.modules.pop("sakura_simulator.compiler", None)
        sys.modules.pop("sakura_simulator.runtime", None)

    def test_given_compiled_model_when_rendered_then_shows_green_compiled_indicator(self):
        # Given: is_compiled returns True
        self.mock_registry.is_compiled.return_value = True
        # When: main() renders the page
        app_module.main()
        # Then: at least one green compiled indicator appears in sidebar markdown
        markdown_calls = [str(c) for c in self.st.sidebar.markdown.call_args_list]
        assert any(":green_circle:" in c and "compiled" in c for c in markdown_calls)

    def test_given_not_compiled_model_when_rendered_then_shows_red_compiled_indicator(self):
        # Given: is_compiled returns False
        self.mock_registry.is_compiled.return_value = False
        # When: main() renders the page
        app_module.main()
        # Then: at least one red compiled indicator appears in sidebar markdown
        markdown_calls = [str(c) for c in self.st.sidebar.markdown.call_args_list]
        assert any(":red_circle:" in c and "compiled" in c for c in markdown_calls)

    def test_given_compile_button_clicked_when_compile_succeeds_then_shows_success(self):
        # Given: compile button is clicked and compiler succeeds
        from pathlib import Path

        self.mock_col1.button.return_value = True
        self.mock_compiler.compile.return_value = Path("/tmp/artifacts/resnet50")
        # When: main() renders the page
        app_module.main()
        # Then: success message is shown in sidebar
        self.st.sidebar.success.assert_called_once()

    def test_given_compile_button_clicked_when_compile_fails_then_shows_error(self):
        # Given: compile button is clicked and compiler raises ValueError
        self.mock_col1.button.return_value = True
        self.mock_compiler.compile.side_effect = ValueError("Source model not found")
        # When: main() renders the page
        app_module.main()
        # Then: error message is shown in sidebar
        self.st.sidebar.error.assert_called()

    def test_given_run_button_clicked_when_run_succeeds_then_shows_latency_info(self):
        # Given: run button is clicked and runtime returns a successful result
        self.mock_col2.button.return_value = True
        mock_result = MagicMock()
        mock_result.avg_latency_ms = 3.5
        self.mock_runtime.run.return_value = mock_result
        # When: main() renders the page
        app_module.main()
        # Then: info message with latency is shown in sidebar
        self.st.sidebar.info.assert_called_once()

    def test_given_run_button_clicked_when_run_fails_then_shows_error(self):
        # Given: run button is clicked and runtime raises ValueError
        self.mock_col2.button.return_value = True
        self.mock_runtime.run.side_effect = ValueError("Artifact directory not found")
        # When: main() renders the page
        app_module.main()
        # Then: error message is shown in sidebar
        self.st.sidebar.error.assert_called()


class TestLLMInferPanel:
    def setup_method(self):
        """Reset mocks and inject registry + runtime stubs for an LLM model."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn
        self.st.button.return_value = False
        self.st.button.side_effect = None  # clear any iterator left by previous test

        self.mock_col1 = MagicMock()
        self.mock_col2 = MagicMock()
        self.mock_col1.button.return_value = False
        self.mock_col2.button.return_value = False
        self.st.sidebar.columns.return_value = (self.mock_col1, self.mock_col2)

        # LLM model entry
        self.entry = MagicMock()
        self.entry.name = "tinyllama"
        self.entry.version = "1.0.0"
        self.entry.artifact_dir = "artifacts/tinyllama/1.0.0"
        self.entry.model_type = "llm"

        self.mock_registry = MagicMock()
        self.mock_registry.list_models.return_value = [self.entry]
        self.mock_registry.is_space_ready.return_value = True
        self.mock_registry.is_compiled.return_value = True

        mock_reg_mod = MagicMock()
        mock_reg_mod.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_reg_mod

        # Runtime stub
        self.mock_runtime = MagicMock()
        mock_runtime_mod = MagicMock()
        mock_runtime_mod.MeraRuntime.return_value = self.mock_runtime
        sys.modules["sakura_simulator.runtime"] = mock_runtime_mod

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.registry", None)
        sys.modules.pop("sakura_simulator.runtime", None)

    def _make_infer_result(self):
        r = MagicMock()
        r.text = "I am a language model."
        r.latency_ms = 120.0
        r.token_ids = list(range(8))
        return r

    def test_given_llm_entry_when_rendered_then_llm_subheader_is_shown(self):
        # Given: entry.model_type == "llm"
        # When: main() renders the page
        app_module.main()
        # Then: st.subheader is called (LLM panel rendered)
        calls = [str(c) for c in self.st.subheader.call_args_list]
        assert any("LLM" in c or "tinyllama" in c for c in calls)

    def test_given_vision_entry_when_rendered_then_no_llm_subheader(self):
        # Given: entry.model_type == "vision" (LLM panel skipped)
        self.entry.model_type = "vision"
        # When: main() renders the page
        app_module.main()
        # Then: st.subheader is never called
        self.st.subheader.assert_not_called()

    def test_given_generate_button_clicked_when_infer_succeeds_then_code_and_caption_shown(self):
        # Given: Generate button is clicked and infer() succeeds
        self.mock_runtime.infer.return_value = self._make_infer_result()
        # Generate button is first st.button call; Activate Engine is second
        self.st.button.side_effect = [True, False]
        # When: main() renders the page
        app_module.main()
        # Then: st.code shows the generated text; st.caption shows stats
        self.st.code.assert_called_once_with("I am a language model.", language=None)
        self.st.caption.assert_called_once()

    def test_given_generate_button_clicked_when_infer_raises_then_error_shown(self):
        # Given: Generate button is clicked but infer() raises ValueError
        self.mock_runtime.infer.side_effect = ValueError("artifact not found")
        self.st.button.side_effect = [True, False]
        # When: main() renders the page
        app_module.main()
        # Then: st.error is called with the failure message
        self.st.error.assert_called_once()
        assert "Inference failed" in str(self.st.error.call_args)

    def test_given_generate_button_not_clicked_when_rendered_then_no_code_shown(self):
        # Given: Generate button is not clicked
        self.st.button.return_value = False
        # When: main() renders the page
        app_module.main()
        # Then: st.code is never called
        self.st.code.assert_not_called()
