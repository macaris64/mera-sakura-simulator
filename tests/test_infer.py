"""BDD tests for MeraRuntime.infer(), _sample_next_token, and InferResult."""

import dataclasses
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sakura_simulator.runtime import InferResult, _sample_next_token

# ---------------------------------------------------------------------------
# _sample_next_token — module-level helper
# ---------------------------------------------------------------------------


class TestSampleNextToken:
    def test_given_temperature_zero_when_sample_then_returns_argmax(self):
        # Given: logits with a clear maximum at index 1 and temperature == 0.0
        # When: _sample_next_token is called with greedy mode
        logits = np.array([0.1, 5.0, 0.3], dtype=np.float32)
        result = _sample_next_token(logits, 0.0)
        # Then: the index with the highest logit is returned
        assert result == 1

    def test_given_temperature_nonzero_when_sample_then_returns_valid_index(self):
        # Given: logits and temperature > 0 (multinomial branch)
        # When: _sample_next_token is called
        logits = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        result = _sample_next_token(logits, 1.0)
        # Then: the result is a valid vocab index
        assert 0 <= result < 3

    def test_given_large_logits_temperature_nonzero_when_sample_then_no_nan(self):
        # Given: large logits values that could cause numerical issues
        # When: _sample_next_token is called repeatedly
        logits = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        for _ in range(10):
            result = _sample_next_token(logits, 0.5)
            assert 0 <= result < 3


# ---------------------------------------------------------------------------
# InferResult dataclass
# ---------------------------------------------------------------------------


class TestInferResult:
    def test_given_all_fields_when_constructed_then_stored_correctly(self):
        # Given / When: InferResult with all fields
        result = InferResult(text="hello world", token_ids=[1, 2, 3], latency_ms=42.5)
        # Then: fields are accessible
        assert result.text == "hello world"
        assert result.token_ids == [1, 2, 3]
        assert result.latency_ms == 42.5

    def test_given_infer_result_when_inspected_then_fields_are_typed_primitives(self):
        # Given: InferResult dataclass
        # When: fields are inspected
        field_names = {f.name for f in dataclasses.fields(InferResult)}
        # Then: exactly the three proto-ready primitive fields exist
        assert field_names == {"text", "token_ids", "latency_ms"}


# ---------------------------------------------------------------------------
# MeraRuntime.infer() — guard-clause branches
# ---------------------------------------------------------------------------


class TestMeraRuntimeInferValidation:
    def setup_method(self):
        self._saved_mera = sys.modules["mera"]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        sys.modules.pop("sakura_simulator.runtime", None)
        mock_mera = MagicMock()
        sim_target = MagicMock()
        sim_target.str_val = "Simulator"
        mock_mera.Target.Simulator = sim_target
        sys.modules["mera"] = mock_mera

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules["mera"] = self._saved_mera
        sys.modules.pop("sakura_simulator.runtime", None)
        sys.modules.pop("sakura_simulator.tokenizer", None)

    def _vision_entry(self):
        e = MagicMock()
        e.name = "resnet50"
        e.model_type = "vision"
        return e

    def _llm_entry(self, tokenizer_path="tokenizers/tinyllama"):
        e = MagicMock()
        e.name = "tinyllama"
        e.model_type = "llm"
        e.tokenizer_path = tokenizer_path
        e.context_length = None
        e.use_kv_cache = False
        return e

    def test_given_vision_entry_when_infer_then_raises_value_error(self):
        # Given: a vision model entry (model_type != "llm")
        from sakura_simulator.runtime import MeraRuntime

        # When / Then: ValueError with "not an LLM"
        with pytest.raises(ValueError, match="not an LLM"):
            MeraRuntime().infer(self._vision_entry(), self.tmpdir, "hello")

    def test_given_llm_entry_no_tokenizer_path_when_infer_then_raises_value_error(self):
        # Given: LLM entry but tokenizer_path is None
        from sakura_simulator.runtime import MeraRuntime

        # When / Then: ValueError about missing tokenizer_path
        with pytest.raises(ValueError, match="no tokenizer_path"):
            MeraRuntime().infer(self._llm_entry(tokenizer_path=None), self.tmpdir, "hello")

    def test_given_artifact_dir_missing_when_infer_then_raises_value_error(self):
        # Given: LLM entry with valid tokenizer_path but artifact_dir does not exist
        from sakura_simulator.runtime import MeraRuntime

        # When / Then: ValueError about missing artifact directory
        with pytest.raises(ValueError, match="Artifact directory not found"):
            MeraRuntime().infer(self._llm_entry(), self.tmpdir / "nonexistent", "hello")


# ---------------------------------------------------------------------------
# MeraRuntime.infer() — decode loop branches
# ---------------------------------------------------------------------------


class TestMeraRuntimeInferDecodeLoop:
    def setup_method(self):
        self._saved_mera = sys.modules["mera"]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        sys.modules.pop("sakura_simulator.runtime", None)
        mock_mera = MagicMock()
        sim_target = MagicMock()
        sim_target.str_val = "Simulator"
        mock_mera.Target.Simulator = sim_target
        sys.modules["mera"] = mock_mera

        self.artifact_path = self.tmpdir / "artifacts"
        self.artifact_path.mkdir()

        # Inject mock tokenizer module
        self.mock_tokenizer_cls = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_tok_mod = MagicMock()
        mock_tok_mod.SakuraTokenizer = self.mock_tokenizer_cls
        sys.modules["sakura_simulator.tokenizer"] = mock_tok_mod

        # Trigger a fresh import now so the package attribute is updated to this
        # module instance before any patch() context is entered in test bodies.
        # This ensures patch("sakura_simulator.runtime.*") and infer()'s __globals__
        # reference the same module dict.
        import importlib

        importlib.import_module("sakura_simulator.runtime")

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules["mera"] = self._saved_mera
        sys.modules.pop("sakura_simulator.runtime", None)
        sys.modules.pop("sakura_simulator.tokenizer", None)

    def _make_entry(self, context_length=None):
        e = MagicMock()
        e.name = "tinyllama"
        e.model_type = "llm"
        e.tokenizer_path = "tokenizers/tinyllama"
        e.context_length = context_length
        e.use_kv_cache = False
        return e

    def _configure_tokenizer(self, prompt_len: int = 3, eos_id: int | None = 2, pad_id: int = 0):
        input_ids = np.ones((1, prompt_len), dtype=np.int64)
        attn_mask = np.ones((1, prompt_len), dtype=np.int64)
        self.mock_tokenizer.encode.return_value = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        self.mock_tokenizer.eos_token_id = eos_id
        self.mock_tokenizer.pad_token_id = pad_id
        self.mock_tokenizer.decode.return_value = "generated text"

    def _make_mock_runner(self, context_length: int | None = None):
        mock_runner = MagicMock()
        seq_len = context_length if context_length is not None else 1
        mock_runner.get_outputs.return_value = [np.zeros((1, seq_len, 10), dtype=np.float32)]
        return mock_runner

    def test_given_eos_emitted_when_infer_then_stops_and_excludes_eos_from_token_ids(self):
        # Given: tokenizer emits EOS after 1 real token; EOS is excluded from generated list
        self._configure_tokenizer(prompt_len=2, eos_id=5)
        mock_runner = self._make_mock_runner()
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=10
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=10, temperature=0.0
            )
        # Then: EOS token is not in token_ids; real token 3 is included
        assert 5 not in result.token_ids
        assert result.token_ids == [3]
        assert result.text == "generated text"

    def test_given_max_tokens_exhausted_when_infer_then_returns_all_generated_tokens(self):
        # Given: EOS never emitted; loop runs exactly max_new_tokens times
        self._configure_tokenizer(prompt_len=2, eos_id=99)
        mock_runner = self._make_mock_runner()
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=7),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=3
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=3, temperature=0.0
            )
        # Then: exactly 3 tokens generated
        assert len(result.token_ids) == 3
        assert result.text == "generated text"

    def test_given_eos_id_none_when_infer_then_runs_full_max_new_tokens(self):
        # Given: tokenizer has no EOS token (eos_id is None — EOS check short-circuits False)
        self._configure_tokenizer(prompt_len=2, eos_id=None)
        mock_runner = self._make_mock_runner()
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=3),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=4
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hi", max_new_tokens=4
            )
        # Then: loop runs to exhaustion — 4 tokens generated
        assert len(result.token_ids) == 4

    def test_given_context_length_set_when_infer_then_forwarded_to_tokenizer_encode(self):
        # Given: entry has context_length=512
        self._configure_tokenizer(eos_id=5)
        mock_runner = self._make_mock_runner(context_length=512)
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called
            MeraRuntime().infer(self._make_entry(context_length=512), self.artifact_path, "test")
        # Then: encode was called with max_length=512
        self.mock_tokenizer.encode.assert_called_once_with("test", max_length=512)

    def test_given_context_length_none_when_infer_then_encode_called_with_max_length_none(self):
        # Given: entry has context_length=None
        self._configure_tokenizer(eos_id=5)
        mock_runner = self._make_mock_runner()
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called
            MeraRuntime().infer(self._make_entry(context_length=None), self.artifact_path, "test")
        # Then: encode was called with max_length=None
        self.mock_tokenizer.encode.assert_called_once_with("test", max_length=None)

    def test_given_valid_infer_when_complete_then_returns_infer_result_with_positive_latency(self):
        # Given: happy-path setup
        self._configure_tokenizer(prompt_len=2, eos_id=5)
        mock_runner = self._make_mock_runner()
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 5]),
        ):
            from sakura_simulator.runtime import InferResult as _IR
            from sakura_simulator.runtime import MeraRuntime

            # When: infer completes successfully
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=10
            )
        # Then: result is an InferResult with non-negative latency
        assert isinstance(result, _IR)
        assert isinstance(result.text, str)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# MeraRuntime.infer() — static-padded branch (context_length is not None)
# ---------------------------------------------------------------------------


class TestMeraRuntimeInferStaticPadded:
    def setup_method(self):
        self._saved_mera = sys.modules["mera"]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        sys.modules.pop("sakura_simulator.runtime", None)
        mock_mera = MagicMock()
        sim_target = MagicMock()
        sim_target.str_val = "Simulator"
        mock_mera.Target.Simulator = sim_target
        sys.modules["mera"] = mock_mera

        self.artifact_path = self.tmpdir / "artifacts"
        self.artifact_path.mkdir()

        self.mock_tokenizer_cls = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_tok_mod = MagicMock()
        mock_tok_mod.SakuraTokenizer = self.mock_tokenizer_cls
        sys.modules["sakura_simulator.tokenizer"] = mock_tok_mod

        import importlib

        importlib.import_module("sakura_simulator.runtime")

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules["mera"] = self._saved_mera
        sys.modules.pop("sakura_simulator.runtime", None)
        sys.modules.pop("sakura_simulator.tokenizer", None)

    def _make_entry(self, context_length: int = 512):
        e = MagicMock()
        e.name = "distilgpt2"
        e.model_type = "llm"
        e.tokenizer_path = "tokenizers/distilgpt2"
        e.context_length = context_length
        e.use_kv_cache = False
        return e

    def _configure_tokenizer(
        self, prompt_len: int = 3, eos_id: int | None = 5, pad_id: int | None = 0
    ):
        input_ids = np.ones((1, prompt_len), dtype=np.int64)
        attn_mask = np.ones((1, prompt_len), dtype=np.int64)
        self.mock_tokenizer.encode.return_value = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        self.mock_tokenizer.eos_token_id = eos_id
        self.mock_tokenizer.pad_token_id = pad_id
        self.mock_tokenizer.decode.return_value = "generated text"

    def _make_mock_runner(self, context_length: int = 512):
        mock_runner = MagicMock()
        mock_runner.get_outputs.return_value = [np.zeros((1, context_length, 10), dtype=np.float32)]
        return mock_runner

    def test_given_context_length_when_infer_then_runner_receives_padded_input_shape(self):
        # Given: entry has context_length=512 and prompt is 3 tokens
        self._configure_tokenizer(prompt_len=3, eos_id=5, pad_id=0)
        mock_runner = self._make_mock_runner(context_length=512)
        captured_inputs = {}

        def capture(data):
            if not captured_inputs:
                captured_inputs.update({k: v.copy() for k, v in data.items()})

        mock_runner.set_input.side_effect = capture
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with context_length=512
            MeraRuntime().infer(self._make_entry(context_length=512), self.artifact_path, "hi")
        # Then: runner received input_ids with shape [1, 512]
        assert captured_inputs["input_ids"].shape == (1, 512)
        assert captured_inputs["attention_mask"].shape == (1, 512)

    def test_given_context_length_when_infer_then_padding_positions_have_zero_attention(self):
        # Given: entry has context_length=512 and prompt is 3 tokens
        self._configure_tokenizer(prompt_len=3, eos_id=5, pad_id=0)
        mock_runner = self._make_mock_runner(context_length=512)
        captured_mask = {}

        def capture(data):
            if not captured_mask:
                # Copy the arrays so later in-place mutations don't affect the snapshot
                captured_mask.update({k: v.copy() for k, v in data.items()})

        mock_runner.set_input.side_effect = capture
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            MeraRuntime().infer(self._make_entry(context_length=512), self.artifact_path, "hi")
        # Then: positions beyond prompt_len=3 start as zero in the first call
        assert captured_mask["attention_mask"][0, 3] == 0
        assert captured_mask["attention_mask"][0, 511] == 0

    def test_given_context_length_and_eos_emitted_when_infer_then_stops_early(self):
        # Given: context_length=512, EOS token is 5; sampler returns real token then EOS
        self._configure_tokenizer(prompt_len=2, eos_id=5, pad_id=50256)
        mock_runner = self._make_mock_runner(context_length=512)
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[7, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=100
            result = MeraRuntime().infer(
                self._make_entry(context_length=512),
                self.artifact_path,
                "hello",
                max_new_tokens=100,
            )
        # Then: only the real token (7) is in token_ids; EOS (5) is excluded
        assert result.token_ids == [7]
        assert 5 not in result.token_ids

    def test_given_context_length_and_pad_token_id_none_when_infer_then_falls_back_to_eos_id(self):
        # Given: pad_token_id is None — fallback branch: use eos_id as pad
        self._configure_tokenizer(prompt_len=2, eos_id=50256, pad_id=None)
        mock_runner = self._make_mock_runner(context_length=512)
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 50256]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer runs (should not crash despite pad_token_id=None)
            result = MeraRuntime().infer(
                self._make_entry(context_length=512), self.artifact_path, "hello", max_new_tokens=10
            )
        # Then: inference completes; eos_id used for padding; only token 3 generated
        assert result.token_ids == [3]

    def test_given_context_length_and_no_eos_emitted_when_infer_then_generates_max_new_tokens(self):
        # Given: context_length=512; EOS never emitted — loop exhausts max_new_tokens
        self._configure_tokenizer(prompt_len=2, eos_id=99, pad_id=0)
        mock_runner = self._make_mock_runner(context_length=512)
        with (
            patch("sakura_simulator.runtime._make_runner", return_value=mock_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=7),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=3
            result = MeraRuntime().infer(
                self._make_entry(context_length=512), self.artifact_path, "hello", max_new_tokens=3
            )
        # Then: exactly 3 tokens generated (loop ran to exhaustion, no break)
        assert len(result.token_ids) == 3


# ---------------------------------------------------------------------------
# MeraRuntime._infer_with_kv_cache() — KV cache split-model decode path
# ---------------------------------------------------------------------------


class TestMeraRuntimeInferKVCache:
    def setup_method(self):
        self._saved_mera = sys.modules["mera"]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        sys.modules.pop("sakura_simulator.runtime", None)
        mock_mera = MagicMock()
        sim_target = MagicMock()
        sim_target.str_val = "Simulator"
        mock_mera.Target.Simulator = sim_target
        sys.modules["mera"] = mock_mera

        self.artifact_path = self.tmpdir / "artifacts"
        self.artifact_path.mkdir()

        self.mock_tokenizer_cls = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_tok_mod = MagicMock()
        mock_tok_mod.SakuraTokenizer = self.mock_tokenizer_cls
        sys.modules["sakura_simulator.tokenizer"] = mock_tok_mod

        import importlib

        importlib.import_module("sakura_simulator.runtime")

    def teardown_method(self):
        self._tmpdir.cleanup()
        sys.modules["mera"] = self._saved_mera
        sys.modules.pop("sakura_simulator.runtime", None)
        sys.modules.pop("sakura_simulator.tokenizer", None)

    def _make_entry(self, kv_decode_artifact_dir=None, context_length: int = 512):
        e = MagicMock()
        e.name = "distilgpt2-kvcache"
        e.model_type = "llm"
        e.tokenizer_path = "tokenizers/distilgpt2"
        e.context_length = context_length
        e.use_kv_cache = True
        e.kv_decode_artifact_dir = (
            kv_decode_artifact_dir
            if kv_decode_artifact_dir is not None
            else str(self.artifact_path)
        )
        return e

    def _configure_tokenizer(
        self, prompt_len: int = 3, eos_id: int | None = 99, pad_id: int | None = 0
    ):
        input_ids = np.ones((1, prompt_len), dtype=np.int64)
        attn_mask = np.ones((1, prompt_len), dtype=np.int64)
        self.mock_tokenizer.encode.return_value = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        self.mock_tokenizer.eos_token_id = eos_id
        self.mock_tokenizer.pad_token_id = pad_id
        self.mock_tokenizer.decode.return_value = "generated text"

    def _make_prefill_runner(self, prompt_len: int = 3, ctx: int = 512):
        runner = MagicMock()
        logits = np.zeros((1, ctx, 10), dtype=np.float32)
        kv = np.zeros((6, 2, 1, 12, ctx - 1, 64), dtype=np.float32)
        runner.get_outputs.return_value = [logits, kv]
        return runner

    def _make_decode_runner(self, ctx: int = 512):
        runner = MagicMock()
        logits = np.zeros((1, 1, 10), dtype=np.float32)
        kv = np.zeros((6, 2, 1, 12, ctx - 1, 64), dtype=np.float32)
        runner.get_outputs.return_value = [logits, kv]
        return runner

    def test_given_kv_cache_entry_missing_decode_dir_when_infer_then_raises_value_error(self):
        # Given: use_kv_cache=True but kv_decode_artifact_dir is None
        entry = self._make_entry(kv_decode_artifact_dir=None)
        entry.kv_decode_artifact_dir = None
        from sakura_simulator.runtime import MeraRuntime

        # When / Then: ValueError about missing kv_decode_artifact_dir
        with pytest.raises(ValueError, match="kv_decode_artifact_dir"):
            MeraRuntime().infer(entry, self.artifact_path, "hello")

    def test_given_kv_cache_entry_no_context_length_when_infer_then_raises_value_error(self):
        # Given: use_kv_cache=True but context_length is None (static shapes require it)
        entry = self._make_entry(context_length=None)
        entry.context_length = None
        from sakura_simulator.runtime import MeraRuntime

        # When / Then: ValueError about missing context_length
        with pytest.raises(ValueError, match="context_length"):
            MeraRuntime().infer(entry, self.artifact_path, "hello")

    def test_given_kv_cache_entry_and_pad_token_id_none_with_eos_when_infer_then_uses_eos_as_pad(
        self,
    ):  # noqa: E501
        # Given: pad_token_id is None, eos_id is 50256 — fallback: use eos_id for padding
        self._configure_tokenizer(prompt_len=2, eos_id=50256, pad_id=None)
        prefill_runner = self._make_prefill_runner(prompt_len=2)
        mock_make_runner = MagicMock(return_value=prefill_runner)
        with (
            patch("sakura_simulator.runtime._make_runner", mock_make_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=50256),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer runs; EOS sampled immediately after prefill
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=10
            )
        # Then: no tokens generated; decode runner never created
        assert result.token_ids == []
        mock_make_runner.assert_called_once()

    def test_given_kv_cache_entry_and_pad_token_id_none_no_eos_when_infer_then_uses_zero_as_pad(
        self,
    ):  # noqa: E501
        # Given: both pad_token_id and eos_id are None — fallback: use 0 as pad
        self._configure_tokenizer(prompt_len=2, eos_id=None, pad_id=None)
        prefill_runner = self._make_prefill_runner(prompt_len=2)
        mock_make_runner = MagicMock(return_value=prefill_runner)
        with (
            patch("sakura_simulator.runtime._make_runner", mock_make_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=7),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer runs with max_new_tokens=1 (only prefill step)
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=1
            )
        # Then: one token generated; no crash (0 used as pad fallback)
        assert result.token_ids == [7]

    def test_given_kv_cache_entry_when_infer_then_prefill_runner_called_with_full_prompt_shape(
        self,
    ):  # noqa: E501
        # Given: KV cache entry with context_length=512, prompt of length 5
        self._configure_tokenizer(prompt_len=5, eos_id=99, pad_id=0)
        prefill_runner = self._make_prefill_runner(prompt_len=5)
        decode_runner = self._make_decode_runner()
        captured = {}

        def capture(data):
            if not captured:
                captured.update({k: v.copy() if hasattr(v, "copy") else v for k, v in data.items()})

        prefill_runner.set_input.side_effect = capture
        with (
            patch(
                "sakura_simulator.runtime._make_runner",
                side_effect=[prefill_runner, decode_runner],
            ),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 99]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called
            MeraRuntime().infer(self._make_entry(), self.artifact_path, "test", max_new_tokens=10)
        # Then: prefill runner received inputs padded to static context_length=512
        assert captured["input_ids"].shape == (1, 512)
        assert captured["attention_mask"].shape == (1, 512)

    def test_given_kv_cache_entry_when_infer_then_decode_runner_receives_kv_tensors(self):
        # Given: KV cache entry; prefill emits KV state that decode runner should receive
        self._configure_tokenizer(prompt_len=3, eos_id=99, pad_id=0)
        prefill_runner = self._make_prefill_runner(prompt_len=3)
        decode_runner = self._make_decode_runner()
        captured_inputs = {}

        def capture(data):
            if not captured_inputs:
                captured_inputs.update(data)

        decode_runner.set_input.side_effect = capture
        with (
            patch(
                "sakura_simulator.runtime._make_runner",
                side_effect=[prefill_runner, decode_runner],
            ),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[3, 99]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens allowing one decode step
            MeraRuntime().infer(self._make_entry(), self.artifact_path, "test", max_new_tokens=10)
        # Then: decode runner received input_ids [1,1], static attention_mask [1,512], and past_kv
        assert "past_kv" in captured_inputs
        assert "input_ids" in captured_inputs
        assert captured_inputs["input_ids"].shape == (1, 1)
        assert captured_inputs["attention_mask"].shape == (1, 512)

    def test_given_kv_cache_entry_and_eos_after_prefill_when_infer_then_returns_no_decode_tokens(
        self,
    ):  # noqa: E501
        # Given: _sample_next_token returns EOS immediately after prefill
        self._configure_tokenizer(prompt_len=3, eos_id=5)
        prefill_runner = self._make_prefill_runner(prompt_len=3)
        mock_make_runner = MagicMock(return_value=prefill_runner)
        with (
            patch("sakura_simulator.runtime._make_runner", mock_make_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=5),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=10
            )
        # Then: no tokens generated; decode runner never created
        assert result.token_ids == []
        mock_make_runner.assert_called_once()

    def test_given_kv_cache_entry_and_eos_in_decode_loop_when_infer_then_stops_early(self):
        # Given: prefill emits real token; first decode step emits EOS
        self._configure_tokenizer(prompt_len=3, eos_id=5)
        prefill_runner = self._make_prefill_runner(prompt_len=3)
        decode_runner = self._make_decode_runner()
        with (
            patch(
                "sakura_simulator.runtime._make_runner",
                side_effect=[prefill_runner, decode_runner],
            ),
            patch("sakura_simulator.runtime._sample_next_token", side_effect=[7, 5]),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=10
            )
        # Then: only the prefill token (7) is included; EOS (5) is excluded
        assert result.token_ids == [7]
        assert 5 not in result.token_ids

    def test_given_kv_cache_entry_when_max_tokens_exhausted_then_all_tokens_returned(self):
        # Given: EOS never emitted; loop runs to max_new_tokens
        self._configure_tokenizer(prompt_len=3, eos_id=99)
        prefill_runner = self._make_prefill_runner(prompt_len=3)
        decode_runner = self._make_decode_runner()
        with (
            patch(
                "sakura_simulator.runtime._make_runner",
                side_effect=[prefill_runner, decode_runner],
            ),
            patch("sakura_simulator.runtime._sample_next_token", return_value=7),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=3
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=3
            )
        # Then: exactly 3 tokens (1 from prefill + 2 from decode loop)
        assert len(result.token_ids) == 3

    def test_given_kv_cache_entry_and_max_new_tokens_one_when_infer_then_only_prefill_runner_used(
        self,
    ):  # noqa: E501
        # Given: max_new_tokens=1 — only the prefill sample is needed, no decode runner
        self._configure_tokenizer(prompt_len=3, eos_id=99)
        prefill_runner = self._make_prefill_runner(prompt_len=3)
        mock_make_runner = MagicMock(return_value=prefill_runner)
        with (
            patch("sakura_simulator.runtime._make_runner", mock_make_runner),
            patch("sakura_simulator.runtime._sample_next_token", return_value=7),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer is called with max_new_tokens=1
            result = MeraRuntime().infer(
                self._make_entry(), self.artifact_path, "hello", max_new_tokens=1
            )
        # Then: exactly 1 token; decode runner never created
        assert result.token_ids == [7]
        mock_make_runner.assert_called_once()

    def test_given_kv_cache_entry_when_infer_then_kv_state_updated_each_decode_step(self):
        # Given: max_new_tokens=3 → decode loop runs max_new_tokens-1=2 times
        self._configure_tokenizer(prompt_len=3, eos_id=99)
        prefill_runner = self._make_prefill_runner(prompt_len=3)
        decode_runner = self._make_decode_runner()
        with (
            patch(
                "sakura_simulator.runtime._make_runner",
                side_effect=[prefill_runner, decode_runner],
            ),
            patch("sakura_simulator.runtime._sample_next_token", return_value=7),
        ):
            from sakura_simulator.runtime import MeraRuntime

            # When: infer runs to exhaustion
            MeraRuntime().infer(self._make_entry(), self.artifact_path, "hello", max_new_tokens=3)
        # Then: decode runner set_input called once per decode step (2 times)
        assert decode_runner.set_input.call_count == 2
