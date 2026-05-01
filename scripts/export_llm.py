#!/usr/bin/env python3
"""Export DistilGPT-2 to static-shape ONNX and update configs/models.yaml.

Usage:
    poetry install --extras llm
    python scripts/export_llm.py

Produces:
    models/distilgpt2.onnx          — static shape [1, 512], opset 14
    tokenizers/distilgpt2/          — HuggingFace tokenizer files
    configs/models.yaml             — updated with real SHA-256 checksum
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import yaml

MODEL_ID = "distilgpt2"
MAX_LEN = 512
ONNX_PATH = pathlib.Path("models/distilgpt2.onnx")
TOK_PATH = pathlib.Path("tokenizers/distilgpt2")
MANIFEST = pathlib.Path("configs/models.yaml")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    print(f"ERROR: {exc}\nRun: poetry install --extras llm", file=sys.stderr)
    sys.exit(1)

# ── Download and save tokenizer ───────────────────────────────────────────────
print(f"Downloading tokenizer for {MODEL_ID} …")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  # GPT-2 has no dedicated pad token
TOK_PATH.mkdir(parents=True, exist_ok=True)
tok.save_pretrained(str(TOK_PATH))
print(f"  Tokenizer saved → {TOK_PATH}")
print(f"  pad_token_id = {tok.pad_token_id}  eos_token_id = {tok.eos_token_id}")

# ── Export model to ONNX (static shape [1, 512]) ──────────────────────────────
print(f"\nDownloading model {MODEL_ID} and exporting to ONNX …")
ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.config.use_cache = False  # disable KV-cache so tracing uses the simple single-pass path
model.eval()

dummy_ids = torch.zeros(1, MAX_LEN, dtype=torch.long)
dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long)


class _NoCache(torch.nn.Module):
    """Wrapper that calls the model with use_cache=False so torch.onnx.export can trace it."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits


with torch.no_grad():
    torch.onnx.export(
        _NoCache(model),
        args=(dummy_ids, dummy_mask),
        f=str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=14,
    )
print(f"  ONNX model saved → {ONNX_PATH}  ({ONNX_PATH.stat().st_size // 1_000_000} MB)")

# ── Compute real SHA-256 ──────────────────────────────────────────────────────
sha256 = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
print(f"  SHA-256: {sha256}")

# ── Patch manifest ────────────────────────────────────────────────────────────
print("\nUpdating configs/models.yaml …")
manifest = yaml.safe_load(MANIFEST.read_text())
new_entry = {
    "name": MODEL_ID,
    "version": "1.0.0",
    "model_type": "llm",
    "path": str(ONNX_PATH),
    "checksum": sha256,
    "format": "onnx",
    "artifact_dir": f"artifacts/{MODEL_ID}/1.0.0",
    "tokenizer_path": str(TOK_PATH),
    "context_length": MAX_LEN,
    "inputs": [
        {"name": "input_ids", "dtype": "int64", "shape": [1, MAX_LEN]},
        {"name": "attention_mask", "dtype": "int64", "shape": [1, MAX_LEN]},
    ],
    "npu_constraints": {"max_power_watts": 15.0, "required_memory_mb": 512},
}
manifest["models"] = [m for m in manifest["models"] if m["name"] != MODEL_ID]
manifest["models"].append(new_entry)
MANIFEST.write_text(yaml.dump(manifest, default_flow_style=False, allow_unicode=True))
print(f"  Manifest updated → {MANIFEST}")

print("\nDone. Next steps:")
print("  poetry run sakura models compile distilgpt2")
print('  poetry run sakura models infer distilgpt2 --prompt "The meaning of life is"')
