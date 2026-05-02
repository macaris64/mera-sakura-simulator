"""SakuraTokenizer: thin adapter over transformers.AutoTokenizer for LLM inference."""

from __future__ import annotations

import os

import numpy as np

# TensorFlow 2.9 in this environment uses a protobuf version incompatible with the
# installed protobuf (>=4). Transformers' to_py_obj() tries to detect TF during decode
# and crashes when TF imports its proto descriptors. Set USE_TF=0 before the first
# lazy transformers import so is_tf_available() returns False and the broken path is
# never entered. This must be set before any transformers import.
os.environ.setdefault("USE_TF", "0")


class SakuraTokenizer:
    """Encode text to token-ID arrays; decode token-ID arrays to text.

    Wraps transformers.AutoTokenizer as a lazy import so tests can mock
    sakura_simulator.tokenizer without requiring the transformers package.
    """

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer  # lazy — mocked in tests

        self._tok = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(self, text: str, *, max_length: int | None = None) -> dict[str, np.ndarray]:
        """Return {"input_ids": int64 [1, seq_len], "attention_mask": int64 [1, seq_len]}.

        If max_length is given, truncates to that length with truncation=True.
        """
        kwargs: dict = {"return_tensors": "np", "return_attention_mask": True}
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True
        encoding = self._tok(text, **kwargs)
        return {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }

    def decode(self, token_ids: np.ndarray) -> str:
        """Convert a 1-D or 2-D int64 token-ID array back to a string."""
        ids = token_ids.flatten().tolist()
        return self._tok.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID, or None if the tokenizer has none."""
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID, or None if the tokenizer has none."""
        return self._tok.pad_token_id
