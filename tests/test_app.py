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

        self.mock_registry_cls = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_registry_cls.return_value = self.mock_registry

        m1 = MagicMock()
        m1.name = "resnet50"
        m2 = MagicMock()
        m2.name = "mobilenet_v2"
        self.mock_registry.list_models.return_value = [m1, m2]
        self.mock_registry.is_space_ready.return_value = True

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
