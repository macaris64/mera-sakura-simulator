"""BDD tests for the Streamlit UI page."""

import sys
from unittest.mock import MagicMock, call, patch

import sakura_simulator.app as app_module


class TestStreamlitPage:
    def setup_method(self):
        """Reset mock state, re-apply cache_resource pass-through, and stub registry."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn
        self.st.session_state = MagicMock()
        self.st.session_state.get.return_value = None
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

    def test_given_chat_model_set_when_main_called_then_chat_panel_rendered(self):
        # Given: session_state contains a chat_model
        self.st.session_state = {
            "chat_model": "tinyllama",
            "chat_history_tinyllama": [],
        }
        mock_reg_mod = MagicMock()
        mock_reg = MagicMock()
        mock_reg.get_model.return_value = None
        mock_reg_mod.ModelRegistry.side_effect = [
            FileNotFoundError("no manifest"),  # first call: _render_model_control_center
            mock_reg,                           # second call: _render_chat_panel
        ]
        sys.modules["sakura_simulator.registry"] = mock_reg_mod
        self.st.button.return_value = False
        self.st.chat_input.return_value = None
        # When
        app_module.main()
        # Then: chat panel subheader appears
        self.st.subheader.assert_called_once_with("Chat — tinyllama")

    def test_given_no_chat_model_when_main_called_then_chat_panel_not_rendered(self):
        # Given: no chat_model in session_state
        self.st.session_state.get.return_value = None
        self.st.button.return_value = False
        # When
        app_module.main()
        # Then: subheader (chat panel) is never rendered
        self.st.subheader.assert_not_called()


class TestModelControlCenter:
    def setup_method(self):
        """Reset streamlit mock and inject a successful registry stub with two models."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn
        self.st.session_state = MagicMock()
        self.st.session_state.get.return_value = None

        self.mock_col1 = MagicMock()
        self.mock_col2 = MagicMock()
        self.mock_col3 = MagicMock()
        self.mock_col4 = MagicMock()
        self.mock_col5 = MagicMock()
        for col in [
            self.mock_col1,
            self.mock_col2,
            self.mock_col3,
            self.mock_col4,
            self.mock_col5,
        ]:
            col.button.return_value = False
        # Two models → 4 sidebar.columns calls: [3-col, 2-col] × 2
        self.st.sidebar.columns.side_effect = [
            (self.mock_col1, self.mock_col2, self.mock_col3),
            (self.mock_col4, self.mock_col5),
            (self.mock_col1, self.mock_col2, self.mock_col3),
            (self.mock_col4, self.mock_col5),
        ]

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
        self.st.sidebar.columns.side_effect = None
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
        self.st.sidebar.columns.side_effect = None
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
        self.st.session_state = MagicMock()
        self.st.session_state.get.return_value = None

        self.mock_col1 = MagicMock()  # Compile
        self.mock_col2 = MagicMock()  # Run
        self.mock_col3 = MagicMock()  # →
        self.mock_col4 = MagicMock()  # Download
        self.mock_col5 = MagicMock()  # Remove
        for col in [
            self.mock_col1,
            self.mock_col2,
            self.mock_col3,
            self.mock_col4,
            self.mock_col5,
        ]:
            col.button.return_value = False
        self.st.sidebar.columns.side_effect = [
            (self.mock_col1, self.mock_col2, self.mock_col3),
            (self.mock_col4, self.mock_col5),
        ]

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

    def _reset_columns(self):
        """Re-arm sidebar.columns side_effect (consumed once per main() call)."""
        self.st.sidebar.columns.side_effect = [
            (self.mock_col1, self.mock_col2, self.mock_col3),
            (self.mock_col4, self.mock_col5),
        ]

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

    def test_given_arrow_button_clicked_when_rendered_then_chat_model_set_in_session_state(self):
        # Given: → button is clicked
        self.mock_col3.button.return_value = True
        # When
        app_module.main()
        # Then: session_state["chat_model"] is set to the entry name
        self.st.session_state.__setitem__.assert_any_call("chat_model", "resnet50")

    def test_given_download_button_clicked_when_download_succeeds_then_sidebar_success_shown(self):
        # Given: download button clicked, registry.download returns a path
        self.mock_col4.button.return_value = True
        mock_path = MagicMock()
        mock_path.name = "resnet50.onnx"
        self.mock_registry.download.return_value = mock_path
        # When
        app_module.main()
        # Then: success shown with filename
        self.st.sidebar.success.assert_called_once()
        assert "Downloaded" in str(self.st.sidebar.success.call_args)
        assert "resnet50.onnx" in str(self.st.sidebar.success.call_args)

    def test_given_download_button_clicked_when_download_raises_value_error_then_sidebar_error_shown(
        self,
    ):
        # Given: download fails with ValueError
        self.mock_col4.button.return_value = True
        self.mock_registry.download.side_effect = ValueError("bad url")
        # When
        app_module.main()
        # Then
        self.st.sidebar.error.assert_called()
        assert "Download failed" in str(self.st.sidebar.error.call_args)

    def test_given_download_button_clicked_when_download_raises_file_not_found_then_sidebar_error_shown(
        self,
    ):
        # Given: download fails with FileNotFoundError
        self.mock_col4.button.return_value = True
        self.mock_registry.download.side_effect = FileNotFoundError("no manifest")
        # When
        app_module.main()
        # Then
        self.st.sidebar.error.assert_called()

    def test_given_remove_button_clicked_when_remove_succeeds_then_sidebar_success_shown(self):
        # Given: remove button clicked
        self.mock_col5.button.return_value = True
        # When
        app_module.main()
        # Then: success message contains model name
        self.st.sidebar.success.assert_called_once()
        assert "Removed" in str(self.st.sidebar.success.call_args)
        assert "resnet50" in str(self.st.sidebar.success.call_args)

    def test_given_remove_button_clicked_when_remove_raises_value_error_then_sidebar_error_shown(
        self,
    ):
        # Given: remove fails with ValueError
        self.mock_col5.button.return_value = True
        self.mock_registry.remove.side_effect = ValueError("not downloaded")
        # When
        app_module.main()
        # Then
        self.st.sidebar.error.assert_called()
        assert "Remove failed" in str(self.st.sidebar.error.call_args)

    def test_given_remove_button_clicked_when_remove_raises_file_not_found_then_sidebar_error_shown(
        self,
    ):
        # Given: remove fails with FileNotFoundError
        self.mock_col5.button.return_value = True
        self.mock_registry.remove.side_effect = FileNotFoundError("no file")
        # When
        app_module.main()
        # Then
        self.st.sidebar.error.assert_called()


class TestChatPanel:
    def setup_method(self):
        """Reset mocks and set up registry + runtime stubs for chat panel tests."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn
        self.st.button.return_value = False
        self.st.chat_input.return_value = None
        # Real dict so chat panel can use `in`, `[]`, `.append()`
        self.st.session_state = {
            "chat_model": "tinyllama",
            "chat_history_tinyllama": [],
        }

        self.entry = MagicMock()
        self.entry.name = "tinyllama"
        self.entry.artifact_dir = "artifacts/tinyllama/1.0.0"
        self.entry.model_type = "llm"

        self.mock_registry = MagicMock()
        self.mock_registry.get_model.return_value = self.entry

        mock_reg_mod = MagicMock()
        mock_reg_mod.ModelRegistry.return_value = self.mock_registry
        sys.modules["sakura_simulator.registry"] = mock_reg_mod

        self.mock_runtime = MagicMock()
        mock_runtime_mod = MagicMock()
        mock_runtime_mod.MeraRuntime.return_value = self.mock_runtime
        sys.modules["sakura_simulator.runtime"] = mock_runtime_mod

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.registry", None)
        sys.modules.pop("sakura_simulator.runtime", None)

    def _make_infer_result(self, text="MERA is a compiler."):
        r = MagicMock()
        r.text = text
        r.latency_ms = 80.0
        r.token_ids = list(range(5))
        return r

    def test_given_render_called_then_subheader_shows_model_name(self):
        # Given / When
        app_module._render_chat_panel("tinyllama")
        # Then
        self.st.subheader.assert_called_once_with("Chat — tinyllama")

    def test_given_close_button_clicked_then_chat_model_cleared_and_returns_early(self):
        # Given: × Close is clicked
        self.st.button.return_value = True
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: chat_model cleared, no further rendering
        assert self.st.session_state["chat_model"] is None
        self.st.chat_input.assert_not_called()

    def test_given_close_button_not_clicked_then_chat_input_rendered(self):
        # Given: close button not clicked
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: chat_input is rendered
        self.st.chat_input.assert_called_once_with("Type a message...")

    def test_given_no_chat_history_then_history_initialized_as_empty_list(self):
        # Given: history key absent
        del self.st.session_state["chat_history_tinyllama"]
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: key created and empty
        assert self.st.session_state["chat_history_tinyllama"] == []

    def test_given_existing_chat_history_then_history_not_reset(self):
        # Given: history already populated
        existing = [{"role": "user", "content": "hi", "time": "09:00"}]
        self.st.session_state["chat_history_tinyllama"] = existing
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: same list object (not replaced)
        assert self.st.session_state["chat_history_tinyllama"] is existing

    def test_given_messages_in_history_then_chat_messages_displayed(self):
        # Given: two messages in history
        self.st.session_state["chat_history_tinyllama"] = [
            {"role": "user", "content": "Hello", "time": "09:00"},
            {"role": "assistant", "content": "Hi!", "time": "09:01"},
        ]
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: chat_message called twice (once per message)
        assert self.st.chat_message.call_count == 2

    def test_given_no_user_input_then_infer_not_called(self):
        # Given: chat_input returns None (nothing submitted)
        self.st.chat_input.return_value = None
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: no inference, no rerun
        self.mock_runtime.infer.assert_not_called()
        self.st.rerun.assert_not_called()

    def test_given_user_input_when_infer_succeeds_then_messages_appended_and_rerun_called(self):
        # Given: user sends a message
        self.st.chat_input.return_value = "What is MERA?"
        self.st.button.return_value = False
        self.mock_runtime.infer.return_value = self._make_infer_result("MERA is a compiler.")
        with patch("sakura_simulator.app.datetime") as mock_dt:
            mock_dt.datetime.now.return_value.strftime.return_value = "12:00"
            # When
            app_module._render_chat_panel("tinyllama")
        # Then: user + assistant messages appended
        history = self.st.session_state["chat_history_tinyllama"]
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is MERA?", "time": "12:00"}
        assert history[1] == {
            "role": "assistant",
            "content": "MERA is a compiler.",
            "time": "12:00",
        }
        self.st.rerun.assert_called_once()

    def test_given_model_not_in_registry_when_user_sends_message_then_error_appended(self):
        # Given: registry returns None for the model
        self.mock_registry.get_model.return_value = None
        self.st.chat_input.return_value = "hello"
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then: error message appended as assistant
        history = self.st.session_state["chat_history_tinyllama"]
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert "Error" in history[1]["content"]
        self.st.rerun.assert_called_once()

    def test_given_infer_raises_value_error_then_error_message_appended(self):
        # Given: infer raises ValueError
        self.st.chat_input.return_value = "generate text"
        self.st.button.return_value = False
        self.mock_runtime.infer.side_effect = ValueError("artifact not found")
        # When
        app_module._render_chat_panel("tinyllama")
        # Then
        history = self.st.session_state["chat_history_tinyllama"]
        assert len(history) == 2
        assert "Error" in history[1]["content"]
        assert "artifact not found" in history[1]["content"]
        self.st.rerun.assert_called_once()

    def test_given_infer_raises_file_not_found_then_error_message_appended(self):
        # Given: infer raises FileNotFoundError
        self.st.chat_input.return_value = "generate text"
        self.st.button.return_value = False
        self.mock_runtime.infer.side_effect = FileNotFoundError("no artifact dir")
        # When
        app_module._render_chat_panel("tinyllama")
        # Then
        history = self.st.session_state["chat_history_tinyllama"]
        assert len(history) == 2
        assert "Error" in history[1]["content"]
        self.st.rerun.assert_called_once()

    def test_given_registry_raises_file_not_found_then_error_appended(self):
        # Given: ModelRegistry constructor raises FileNotFoundError when chat tries to load it
        mock_reg_mod = MagicMock()
        mock_reg_mod.ModelRegistry.side_effect = FileNotFoundError("no manifest")
        sys.modules["sakura_simulator.registry"] = mock_reg_mod
        self.st.chat_input.return_value = "hello"
        self.st.button.return_value = False
        # When
        app_module._render_chat_panel("tinyllama")
        # Then
        history = self.st.session_state["chat_history_tinyllama"]
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert "Error" in history[1]["content"]
        self.st.rerun.assert_called_once()
