#!/usr/bin/env python3
"""Export any AutoModelForCausalLM as a split prefill + decode ONNX pair with KV cache I/O.

Usage:
    poetry install --extras llm
    python scripts/export_llm_kv.py distilgpt2
    python scripts/export_llm_kv.py HuggingFaceTB/SmolLM2-135M-Instruct --name smollm2-135m-instruct-kvcache

Produces (example for distilgpt2):
    models/distilgpt2-kvcache-prefill.onnx
    models/distilgpt2-kvcache-decode.onnx
    configs/models.yaml  (entry updated with real SHA-256 + kv_decode_path)

KV cache tensor convention:
    past_kv / present_kv shape: [num_layers, 2, batch, num_heads, seq_len, head_dim]
    The "2" axis encodes key (index 0) and value (index 1) per layer.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import yaml

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    print(f"ERROR: {exc}\nRun: poetry install --extras llm", file=sys.stderr)
    sys.exit(1)

MAX_LEN = 512
MANIFEST = pathlib.Path("configs/models.yaml")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LLM as KV-cache prefill+decode ONNX pair")
    p.add_argument("model_id", help="HuggingFace model ID (e.g. distilgpt2)")
    p.add_argument("--name", help="Registry name for the KV model (default: <model_id>-kvcache)")
    p.add_argument("--max-len", type=int, default=MAX_LEN, help="Max sequence length (default: 512)")
    p.add_argument("--tok-path", help="Local tokenizer dir (skip download if present)")
    return p.parse_args()


class _PrefillWrapper(torch.nn.Module):
    """(input_ids [1,L], attention_mask [1,L]) → (logits [1,L,V], past_kv [N,2,B,H,L-1,D]).

    Truncates KV to L-1 positions so the decode stage always receives a fixed-size buffer.
    """

    def __init__(self, m, max_len: int):
        super().__init__()
        self.m = m
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        present = torch.stack(
            [torch.stack([k, v], dim=0) for k, v in out.past_key_values], dim=0
        )
        return out.logits, present[:, :, :, :, : self.max_len - 1, :]


class _DecodeWrapper(torch.nn.Module):
    """(input_ids [1,1], attention_mask [1,L], past_kv [N,2,B,H,L-1,D])
    → (logits [1,1,V], present_kv [N,2,B,H,L-1,D]).

    Sliding window: appends current-token KV then drops the oldest position so the
    output buffer stays at a fixed L-1 length.
    """

    def __init__(self, m, n_layers: int):
        super().__init__()
        self.m = m
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask, past_kv):
        pkv = tuple((past_kv[i, 0], past_kv[i, 1]) for i in range(self.n_layers))
        out = self.m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=pkv,
            use_cache=True,
        )
        present = torch.stack(
            [torch.stack([k, v], dim=0) for k, v in out.past_key_values], dim=0
        )
        # Drop oldest position so output shape matches input shape [N,2,B,H,L-1,D]
        return out.logits, present[:, :, :, :, 1:, :]


def main() -> None:
    args = _parse_args()
    model_id = args.model_id
    short_name = args.name or (model_id.split("/")[-1] + "-kvcache")
    max_len = args.max_len

    prefill_onnx = pathlib.Path(f"models/{short_name}-prefill.onnx")
    decode_onnx = pathlib.Path(f"models/{short_name}-decode.onnx")
    tok_path = pathlib.Path(args.tok_path) if args.tok_path else pathlib.Path(f"tokenizers/{short_name.replace('-kvcache', '')}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    if not tok_path.exists():
        print(f"Downloading tokenizer for {model_id} …")
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok_path.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(tok_path))
        print(f"  Saved → {tok_path}")
    else:
        print(f"Tokenizer already at {tok_path}, skipping.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    # Detect architecture dimensions from a dummy forward pass (model-agnostic).
    dummy_ids = torch.zeros(1, 4, dtype=torch.long)
    dummy_mask = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        probe = model(input_ids=dummy_ids, attention_mask=dummy_mask, use_cache=True)
    num_layers = len(probe.past_key_values)
    num_heads = probe.past_key_values[0][0].shape[1]
    head_dim = probe.past_key_values[0][0].shape[3]
    print(f"  layers={num_layers}  heads={num_heads}  head_dim={head_dim}")

    dummy_ids_full = torch.zeros(1, max_len, dtype=torch.long)
    dummy_mask_full = torch.ones(1, max_len, dtype=torch.long)

    # ── Prefill export ────────────────────────────────────────────────────────
    print(f"\nExporting prefill model → {prefill_onnx} …")
    prefill_onnx.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            _PrefillWrapper(model, max_len),
            args=(dummy_ids_full, dummy_mask_full),
            f=str(prefill_onnx),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "past_kv"],
            opset_version=14,
        )
    print(f"  Saved ({prefill_onnx.stat().st_size // 1_000_000} MB)")

    # ── Decode export ─────────────────────────────────────────────────────────
    print(f"\nExporting decode model → {decode_onnx} …")
    decode_onnx.parent.mkdir(parents=True, exist_ok=True)
    decode_ids = torch.zeros(1, 1, dtype=torch.long)
    # attention_mask covers all max_len positions (past_kv L-1 + current token)
    decode_mask = torch.ones(1, max_len, dtype=torch.long)
    dummy_past_kv = torch.zeros(num_layers, 2, 1, num_heads, max_len - 1, head_dim)
    with torch.no_grad():
        torch.onnx.export(
            _DecodeWrapper(model, num_layers),
            args=(decode_ids, decode_mask, dummy_past_kv),
            f=str(decode_onnx),
            input_names=["input_ids", "attention_mask", "past_kv"],
            output_names=["logits", "present_kv"],
            opset_version=14,
        )
    print(f"  Saved ({decode_onnx.stat().st_size // 1_000_000} MB)")

    # ── Checksum ──────────────────────────────────────────────────────────────
    sha256 = hashlib.sha256(prefill_onnx.read_bytes()).hexdigest()
    print(f"\nSHA-256 (prefill): {sha256}")

    # ── Patch manifest ────────────────────────────────────────────────────────
    print("Updating configs/models.yaml …")
    manifest = yaml.safe_load(MANIFEST.read_text())
    new_entry = {
        "name": short_name,
        "version": "1.0.0",
        "model_type": "llm",
        "use_kv_cache": True,
        "path": str(prefill_onnx),
        "checksum": sha256,
        "format": "onnx",
        "artifact_dir": f"artifacts/{short_name}/prefill",
        "kv_decode_path": str(decode_onnx),
        "kv_decode_artifact_dir": f"artifacts/{short_name}/decode",
        "tokenizer_path": str(tok_path),
        "context_length": max_len,
        "inputs": [
            {"name": "input_ids", "dtype": "int64", "shape": [1, max_len]},
            {"name": "attention_mask", "dtype": "int64", "shape": [1, max_len]},
        ],
        "npu_constraints": {"max_power_watts": 15.0, "required_memory_mb": 768},
    }
    manifest["models"] = [m for m in manifest["models"] if m["name"] != short_name]
    manifest["models"].append(new_entry)
    MANIFEST.write_text(yaml.dump(manifest, default_flow_style=False, allow_unicode=True))
    print(f"  Manifest updated → {MANIFEST}")

    print(f"\nDone. Next steps:")
    print(f"  poetry run sakura models compile {short_name}")
    print(f'  poetry run sakura models infer {short_name} --prompt "Hello"')


if __name__ == "__main__":
    main()
