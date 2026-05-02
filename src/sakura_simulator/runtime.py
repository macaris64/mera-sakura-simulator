"""MeraRuntime: load compiled artifacts and run inference on SAKURA-II."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from sakura_simulator.registry import ModelEntry


def _resolve_deployment_dir(artifact_dir: Path, *, target) -> Path:
    """Resolve the directory containing deploy.so/deploy.json/deploy.params.

    MERA's TVMDeployer writes a project-style directory layout like:
      artifact_dir/
        build/<Target>/result/deploy.so

    `mera.mera_deployment.load_mera_deployment()` can load either:
    - a MERA project directory (if it recognizes it), or
    - a direct "result" directory containing deploy.* files.

    To be robust across versions/layout detection, we prefer the result directory
    when it exists for the requested target.
    """
    target_str = getattr(target, "str_val", None) or str(target)
    candidate = artifact_dir / "build" / target_str / "result"
    if (candidate / "deploy.so").exists():
        return candidate
    return artifact_dir


def _load_mera_deployment(path: str, target):
    """Load a built MERA TVM deployment from disk (hook for tests)."""
    import mera.mera_deployment as _md

    return _md.load_mera_deployment(path, target=target)


def _is_simulator_target(target) -> bool:
    """Return True if MERA target is the software Simulator."""
    if getattr(target, "str_val", None) == "Simulator":
        return True
    import mera

    return target is mera.Target.Simulator


class _SimulatorGraphRunner:
    """TVM graph executor wrapper matching the subset of MeraTvmModelRunner used here."""

    def __init__(self, rt_mod):
        self._rt = rt_mod

    def set_input(self, data):
        if not isinstance(data, dict):
            raise TypeError("inputs must be a dict[str, ndarray]")
        for k, v in data.items():
            self._rt.set_input(k, v)
        return self

    def run(self):
        self._rt.run()
        return self

    def get_outputs(self):
        n = self._rt.get_num_outputs()
        out = []
        for i in range(n):
            arr = self._rt.get_output(i)
            out.append(arr.asnumpy() if hasattr(arr, "asnumpy") else arr)
        return out


def _build_simulator_runner(deployment_dir: Path):
    """Load deploy.{so,json,params} via TVM graph executor (no mera_runtime_init_device)."""
    paths = {
        "deploy.so": deployment_dir / "deploy.so",
        "deploy.json": deployment_dir / "deploy.json",
        "deploy.params": deployment_dir / "deploy.params",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} under {deployment_dir}")

    try:
        import tvm
        from tvm.contrib.graph_executor import create as graph_create
        from tvm.runtime import load_module
    except (ImportError, OSError) as exc:
        raise ValueError(f"TVM is required for inference: {exc}") from exc

    try:
        lib = load_module(str(paths["deploy.so"]))
        graph_json = paths["deploy.json"].read_text(encoding="utf-8")
        rt_mod = graph_create(graph_json, lib, tvm.cpu(0))
        rt_mod.load_params(paths["deploy.params"].read_bytes())
    except Exception as exc:
        raise ValueError(f"Failed to load TVM deployment from {deployment_dir}: {exc}") from exc

    return _SimulatorGraphRunner(rt_mod)


def _make_runner(deployment_dir: Path, target):
    """Create a runner: Simulator uses TVM graph executor; other targets use MERA get_runner()."""
    if _is_simulator_target(target):
        return _build_simulator_runner(deployment_dir)
    deployment = _load_mera_deployment(str(deployment_dir), target)
    return deployment.get_runner()


@dataclass
class InferResult:
    """Holds the generated text and token-level metadata from LLM inference.

    All fields are typed primitives — no internal dicts — so a protobuf
    serialization layer can be added later by mapping each field to a proto
    field directly without touching the business logic that produces InferResult.
    """

    text: str
    token_ids: list[int]
    latency_ms: float


@dataclass
class RunResult:
    """Holds inference outputs and per-iteration latency measurements."""

    outputs: list[dict] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def min_latency_ms(self) -> float:
        return min(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        s = sorted(self.latency_ms)
        return s[max(0, int(len(s) * 0.95) - 1)]


def _sample_next_token(logits_1d, temperature: float) -> int:
    """Greedy decoding if temperature == 0.0; multinomial sampling otherwise."""
    import numpy as np

    if temperature == 0.0:
        return int(np.argmax(logits_1d))
    logits_1d = logits_1d / temperature
    probs = np.exp(logits_1d - logits_1d.max())
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


class MeraRuntime:
    """Loads compiled MERA artifacts and executes inference using dummy inputs."""

    def __init__(self, target=None):
        import mera  # lazy — mocked in tests

        self._target = target if target is not None else mera.Target.Simulator

    def run(
        self,
        entry: ModelEntry,
        artifact_dir: Path | str,
        *,
        iters: int = 1,
    ) -> RunResult:
        """Run inference on compiled artifacts, generating dummy inputs from entry.inputs.

        Raises ValueError if artifact_dir does not exist or entry.inputs is not configured.
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            raise ValueError(f"Artifact directory not found: {artifact_path}")
        if not entry.inputs:
            raise ValueError(f"Model '{entry.name}' has no inputs configured")

        import numpy as np  # lazy — available as transitive dep

        deployment_dir = _resolve_deployment_dir(artifact_path, target=self._target)

        try:
            runner = _make_runner(deployment_dir, self._target)

            inputs = {
                (inp.name or f"input_{i}"): np.zeros(inp.shape, dtype=inp.dtype)
                for i, inp in enumerate(entry.inputs)
            }

            latencies: list[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                runner.set_input(inputs)
                runner.run()
                latencies.append((time.perf_counter() - t0) * 1000)

            raw_outputs = runner.get_outputs()
            outputs_summary = [
                {"name": f"output_{i}", "shape": list(arr.shape), "dtype": str(arr.dtype)}
                for i, arr in enumerate(raw_outputs)
            ]

            return RunResult(outputs=outputs_summary, latency_ms=latencies)
        except ValueError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise ValueError(f"Run failed: {exc}") from exc

    def infer(
        self,
        entry: ModelEntry,
        artifact_dir: Path | str,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> InferResult:
        """Run autoregressive LLM inference on a prompt and return generated text.

        Raises ValueError if entry.model_type != 'llm', tokenizer_path is not set,
        or artifact_dir does not exist.
        """
        if entry.model_type != "llm":
            raise ValueError(
                f"Model '{entry.name}' is not an LLM (model_type={entry.model_type!r}). "
                "Use MeraRuntime.run() for vision models."
            )
        if entry.tokenizer_path is None:
            raise ValueError(f"Model '{entry.name}' has no tokenizer_path configured.")
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            raise ValueError(f"Artifact directory not found: {artifact_path}")
        if entry.use_kv_cache and entry.kv_decode_artifact_dir is None:
            raise ValueError(
                f"Model '{entry.name}' has use_kv_cache=True but no kv_decode_artifact_dir."
            )
        if entry.use_kv_cache:
            return self._infer_with_kv_cache(
                entry,
                artifact_path,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        import numpy as np

        from sakura_simulator.tokenizer import SakuraTokenizer  # lazy — mocked in tests

        tokenizer = SakuraTokenizer(entry.tokenizer_path)
        encoded = tokenizer.encode(prompt, max_length=entry.context_length)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_len = input_ids.shape[1]
        eos_id = tokenizer.eos_token_id

        deployment_dir = _resolve_deployment_dir(artifact_path, target=self._target)
        runner = _make_runner(deployment_dir, self._target)

        generated: list[int] = []
        t0 = time.perf_counter()
        if entry.context_length is not None:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = eos_id if eos_id is not None else 0
            ctx = entry.context_length
            input_ids_buf = np.full((1, ctx), pad_id, dtype=np.int64)
            input_ids_buf[0, :prompt_len] = input_ids[0]
            attn_buf = np.zeros((1, ctx), dtype=np.int64)
            attn_buf[0, :prompt_len] = attention_mask[0]
            input_ids, attention_mask = input_ids_buf, attn_buf
            current_pos = prompt_len
            for _ in range(max_new_tokens):
                runner.set_input({"input_ids": input_ids, "attention_mask": attention_mask})
                runner.run()
                logits = runner.get_outputs()[0]
                logits_last = logits[0, current_pos - 1, :]
                next_token = _sample_next_token(logits_last, temperature)
                if eos_id is not None and next_token == eos_id:
                    break
                generated.append(next_token)
                input_ids[0, current_pos] = next_token
                attention_mask[0, current_pos] = 1
                current_pos += 1
        else:
            for _ in range(max_new_tokens):
                runner.set_input({"input_ids": input_ids, "attention_mask": attention_mask})
                runner.run()
                logits = runner.get_outputs()[0]
                logits_last = logits[0, -1, :]
                next_token = _sample_next_token(logits_last, temperature)
                if eos_id is not None and next_token == eos_id:
                    break
                generated.append(next_token)
                new_tok = np.array([[next_token]], dtype=np.int64)
                input_ids = np.concatenate([input_ids, new_tok], axis=1)
                attention_mask = np.concatenate(
                    [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
                )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = np.array(generated, dtype=np.int64).reshape(1, -1)
        text = tokenizer.decode(generated_ids)
        return InferResult(text=text, token_ids=generated, latency_ms=latency_ms)

    def _infer_with_kv_cache(
        self,
        entry: ModelEntry,
        artifact_path: Path,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> InferResult:
        """KV-cache autoregressive decode: one prefill pass then single-token decode steps.

        All shapes are fully static (required by the MERA/TVM compiler):
          Prefill inputs:  {input_ids [1, ctx], attention_mask [1, ctx]}
          Prefill outputs: [logits [1, ctx, V], past_kv [N, 2, 1, H, ctx-1, D]]
          Decode inputs:   {input_ids [1,1], attention_mask [1, ctx], past_kv [N,2,1,H,ctx-1,D]}
          Decode outputs:  [logits [1, 1, V], present_kv [N, 2, 1, H, ctx-1, D]]
        """
        import numpy as np

        from sakura_simulator.tokenizer import SakuraTokenizer

        if entry.context_length is None:
            raise ValueError(
                f"Model '{entry.name}' has use_kv_cache=True but context_length is not configured."
            )

        tokenizer = SakuraTokenizer(entry.tokenizer_path)
        encoded = tokenizer.encode(prompt, max_length=entry.context_length)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_len = input_ids.shape[1]
        eos_id = tokenizer.eos_token_id

        ctx = entry.context_length
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        # Pad prefill inputs to static shape [1, ctx]
        input_ids_buf = np.full((1, ctx), pad_id, dtype=np.int64)
        input_ids_buf[0, :prompt_len] = input_ids[0]
        attn_buf = np.zeros((1, ctx), dtype=np.int64)
        attn_buf[0, :prompt_len] = attention_mask[0]

        prefill_dir = _resolve_deployment_dir(artifact_path, target=self._target)
        prefill_runner = _make_runner(prefill_dir, self._target)

        t0 = time.perf_counter()
        prefill_runner.set_input({"input_ids": input_ids_buf, "attention_mask": attn_buf})
        prefill_runner.run()
        prefill_outputs = prefill_runner.get_outputs()
        kv_cache = prefill_outputs[1]
        # Sample from the last real token position (not the pad positions)
        next_token = _sample_next_token(prefill_outputs[0][0, prompt_len - 1, :], temperature)

        # Static decode attention mask [1, ctx]: covers the full KV buffer each step
        attention_mask_decode = np.ones((1, ctx), dtype=np.int64)

        generated: list[int] = []
        if eos_id is None or next_token != eos_id:
            generated.append(next_token)
            if max_new_tokens > 1:
                decode_dir = _resolve_deployment_dir(
                    Path(entry.kv_decode_artifact_dir), target=self._target
                )
                decode_runner = _make_runner(decode_dir, self._target)
                for _ in range(max_new_tokens - 1):
                    decode_inputs = {
                        "input_ids": np.array([[next_token]], dtype=np.int64),
                        "attention_mask": attention_mask_decode,
                        "past_kv": kv_cache,
                    }
                    decode_runner.set_input(decode_inputs)
                    decode_runner.run()
                    decode_outputs = decode_runner.get_outputs()
                    kv_cache = decode_outputs[1]
                    next_token = _sample_next_token(decode_outputs[0][0, 0, :], temperature)
                    if eos_id is not None and next_token == eos_id:
                        break
                    generated.append(next_token)
                    # Slide mask left: drop oldest slot, add new active slot
                    attention_mask_decode = np.concatenate(
                        [attention_mask_decode[:, 1:], np.ones((1, 1), dtype=np.int64)], axis=1
                    )

        latency_ms = (time.perf_counter() - t0) * 1000
        generated_ids = np.array(generated, dtype=np.int64).reshape(1, -1)
        text = tokenizer.decode(generated_ids)
        return InferResult(text=text, token_ids=generated, latency_ms=latency_ms)
