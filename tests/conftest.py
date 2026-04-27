"""
Global test configuration.

Streamlit is mocked via sys.modules injection at collection time so tests can
import and call app.py functions without a running Streamlit server.
cache_resource must pass functions through unchanged; a plain MagicMock used as
a decorator would replace the decorated function with another MagicMock.
"""

import sys
from unittest.mock import MagicMock

_st_mock = MagicMock()
_st_mock.cache_resource.side_effect = lambda fn: fn
sys.modules["streamlit"] = _st_mock
