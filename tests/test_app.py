"""BDD tests for the Streamlit UI page."""

import sys

import sakura_simulator.app as app_module


class TestStreamlitPage:
    def setup_method(self):
        """Reset mock state and re-apply cache_resource pass-through before each test."""
        self.st = sys.modules["streamlit"]
        self.st.reset_mock()
        self.st.cache_resource.side_effect = lambda fn: fn

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
