"""BDD tests for SakuraTokenizer — encode/decode adapter over transformers.AutoTokenizer."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestSakuraTokenizerInit:
    def setup_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)
        self.mock_auto_tok_cls = MagicMock()
        self.mock_auto_tok_cls.from_pretrained.return_value = MagicMock()
        mock_transformers = ModuleType("transformers")
        mock_transformers.AutoTokenizer = self.mock_auto_tok_cls
        sys.modules["transformers"] = mock_transformers

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)

    def test_given_tokenizer_path_when_init_then_calls_from_pretrained(self):
        # Given: transformers is mocked; a tokenizer path is provided
        # When: SakuraTokenizer is constructed
        from sakura_simulator.tokenizer import SakuraTokenizer

        SakuraTokenizer("tokenizers/tinyllama")
        # Then: AutoTokenizer.from_pretrained is called with the path
        self.mock_auto_tok_cls.from_pretrained.assert_called_once_with("tokenizers/tinyllama")

    def test_given_transformers_not_installed_when_init_then_raises_import_error(self):
        # Given: transformers is blocked in sys.modules
        sys.modules["transformers"] = None  # type: ignore[assignment]
        from sakura_simulator.tokenizer import SakuraTokenizer

        # When / Then: ImportError (or SystemError) is raised on construction
        with pytest.raises((ImportError, SystemError)):
            SakuraTokenizer("bad/path")


class TestSakuraTokenizerEncode:
    def setup_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)
        self.mock_inner_tok = MagicMock()
        mock_transformers = ModuleType("transformers")
        mock_auto_tok_cls = MagicMock()
        mock_auto_tok_cls.from_pretrained.return_value = self.mock_inner_tok
        mock_transformers.AutoTokenizer = mock_auto_tok_cls
        sys.modules["transformers"] = mock_transformers
        from sakura_simulator.tokenizer import SakuraTokenizer

        self.tok = SakuraTokenizer("tokenizers/tinyllama")

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)

    def _make_encoding(self, seq_len: int = 3):
        enc = MagicMock()
        enc.__getitem__ = lambda _self, k: {
            "input_ids": np.array([[1] * seq_len]),
            "attention_mask": np.array([[1] * seq_len]),
        }[k]
        self.mock_inner_tok.return_value = enc
        return enc

    def test_given_text_no_max_length_when_encode_then_returns_arrays_with_int64_dtype(self):
        # Given: max_length is None (branch: no truncation kwargs)
        self._make_encoding(3)
        # When: encode is called without max_length
        result = self.tok.encode("hello world")
        # Then: both keys present and dtype is int64
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].dtype == np.int64
        assert result["attention_mask"].dtype == np.int64

    def test_given_text_no_max_length_when_encode_then_truncation_not_in_kwargs(self):
        # Given: max_length is None
        self._make_encoding()
        # When: encode is called without max_length
        self.tok.encode("hi")
        # Then: inner tokenizer was NOT called with truncation kwarg
        call_kwargs = self.mock_inner_tok.call_args[1]
        assert "truncation" not in call_kwargs
        assert "max_length" not in call_kwargs

    def test_given_text_with_max_length_when_encode_then_truncation_kwargs_added(self):
        # Given: max_length is provided (branch: truncation kwargs injected)
        self._make_encoding(2)
        # When: encode is called with max_length=512
        self.tok.encode("some long text", max_length=512)
        # Then: inner tokenizer receives max_length and truncation=True
        call_kwargs = self.mock_inner_tok.call_args[1]
        assert call_kwargs.get("max_length") == 512
        assert call_kwargs.get("truncation") is True


class TestSakuraTokenizerDecode:
    def setup_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)
        self.mock_inner_tok = MagicMock()
        mock_transformers = ModuleType("transformers")
        mock_auto_tok_cls = MagicMock()
        mock_auto_tok_cls.from_pretrained.return_value = self.mock_inner_tok
        mock_transformers.AutoTokenizer = mock_auto_tok_cls
        sys.modules["transformers"] = mock_transformers
        from sakura_simulator.tokenizer import SakuraTokenizer

        self.tok = SakuraTokenizer("tokenizers/tinyllama")

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)

    def test_given_1d_array_when_decode_then_forwards_ids_with_skip_special_tokens(self):
        # Given: a 1-D token-ID array
        self.mock_inner_tok.decode.return_value = "hello"
        # When: decode is called
        result = self.tok.decode(np.array([1, 2, 3], dtype=np.int64))
        # Then: inner tok.decode receives the flat list and skip_special_tokens=True
        self.mock_inner_tok.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)
        assert result == "hello"

    def test_given_2d_array_when_decode_then_flattens_before_forwarding(self):
        # Given: a 2-D token-ID array (shape [1, seq_len])
        self.mock_inner_tok.decode.return_value = "world"
        # When: decode is called
        result = self.tok.decode(np.array([[4, 5, 6]], dtype=np.int64))
        # Then: inner tok.decode receives flattened list
        self.mock_inner_tok.decode.assert_called_once_with([4, 5, 6], skip_special_tokens=True)
        assert result == "world"


class TestSakuraTokenizerEosTokenId:
    def setup_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)
        self.mock_inner_tok = MagicMock()
        mock_transformers = ModuleType("transformers")
        mock_auto_tok_cls = MagicMock()
        mock_auto_tok_cls.from_pretrained.return_value = self.mock_inner_tok
        mock_transformers.AutoTokenizer = mock_auto_tok_cls
        sys.modules["transformers"] = mock_transformers
        from sakura_simulator.tokenizer import SakuraTokenizer

        self.tok = SakuraTokenizer("tokenizers/tinyllama")

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)

    def test_given_eos_token_id_set_when_accessed_then_returns_integer(self):
        # Given: inner tokenizer has eos_token_id = 2 (branch: int return)
        self.mock_inner_tok.eos_token_id = 2
        # When / Then
        assert self.tok.eos_token_id == 2

    def test_given_no_eos_token_when_accessed_then_returns_none(self):
        # Given: inner tokenizer has eos_token_id = None (branch: None return)
        self.mock_inner_tok.eos_token_id = None
        # When / Then
        assert self.tok.eos_token_id is None


class TestSakuraTokenizerPadTokenId:
    def setup_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)
        self.mock_inner_tok = MagicMock()
        mock_transformers = ModuleType("transformers")
        mock_auto_tok_cls = MagicMock()
        mock_auto_tok_cls.from_pretrained.return_value = self.mock_inner_tok
        mock_transformers.AutoTokenizer = mock_auto_tok_cls
        sys.modules["transformers"] = mock_transformers
        from sakura_simulator.tokenizer import SakuraTokenizer

        self.tok = SakuraTokenizer("tokenizers/distilgpt2")

    def teardown_method(self):
        sys.modules.pop("sakura_simulator.tokenizer", None)
        sys.modules.pop("transformers", None)

    def test_given_pad_token_id_set_when_accessed_then_returns_integer(self):
        # Given: inner tokenizer has pad_token_id = 50256 (branch: int return)
        self.mock_inner_tok.pad_token_id = 50256
        # When / Then
        assert self.tok.pad_token_id == 50256

    def test_given_no_pad_token_when_accessed_then_returns_none(self):
        # Given: inner tokenizer has pad_token_id = None (branch: None return)
        self.mock_inner_tok.pad_token_id = None
        # When / Then
        assert self.tok.pad_token_id is None
