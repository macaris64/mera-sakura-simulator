# mera-sakura-simulator

A **Hello World simulator** for the [EdgeCortix SAKURA-II NPU](https://www.edgecortix.com/en/products/sakura) using the [MERA framework](https://www.edgecortix.com/en/products/mera) — built with strict **Test-Driven Development (TDD)** and **Behavior-Driven Development (BDD)** practices.

> Runs against the real [MERA SDK](https://github.com/Edgecortix-Inc/mera) (`mera.Target.Simulator` + `mera.Platform.SAKURA_2C`). No physical NPU required.

---

## Features

- `SakuraEngine` — wraps [`mera.Target`](https://github.com/Edgecortix-Inc/mera) and `mera.Platform` for SAKURA-II NPU initialization
- **CLI** via [Typer](https://typer.tiangolo.com/): `sakura hello`, `sakura models list/inspect/download/remove`
- **Model Registry**: YAML-driven manifest with Pydantic schema validation, SHA-256 integrity checking, HTTP download via httpx, and disk-level remove
- **Web UI** via [Streamlit](https://streamlit.io/): activate the engine and manage models from a browser
- **100% branch coverage** enforced by pytest-cov on every commit
- **BDD-style tests** with Given-When-Then structure, plus smoke tests for end-to-end workflows
- **Pre-commit hook** that blocks commits if tests fail
- **Claude Code integration**: custom slash commands, agent instructions, and NPU telemetry plugin

---

## Quick Start

```bash
# 1. Install Poetry (once per machine)
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# 2. Configure Poetry to create .venv inside the project
poetry config virtualenvs.in-project true

# 3. Install dependencies
poetry install

# 4. Wire up the pre-commit hook
git config core.hooksPath .githooks

# 5. Run the test suite
poetry run pytest
```

---

## Usage

### CLI

```bash
# Engine greeting
poetry run sakura hello
# Hello from Sakura-II: Titan Biosignature Engine Active

# List all registered models with Space-Ready integrity status
poetry run sakura models list

# Inspect NPU constraints for a specific model
poetry run sakura models inspect resnet50

# Download a model file and verify its SHA-256 checksum
poetry run sakura models download resnet50

# Remove a downloaded model file from disk
poetry run sakura models remove resnet50
```

### Web UI

```bash
poetry run streamlit run src/sakura_simulator/app.py
# Open http://localhost:8501
```

---

## Test Coverage

54 tests across 5 test files. **100% branch coverage** enforced on every run.

| Module | Statements | Branches | Cover |
|---|---|---|---|
| `engine.py` | 20 | 0 | **100%** |
| `cli.py` | 60 | 4 | **100%** |
| `registry.py` | 61 | 16 | **100%** |
| `app.py` | 28 | 4 | **100%** |
| `__init__.py` | 3 | 0 | **100%** |
| **Total** | **172** | **24** | **100%** |

Run `poetry run pytest` to reproduce. HTML report: `open htmlcov/index.html`.

---

## Project Structure

```
src/sakura_simulator/
    engine.py        SakuraEngine — wraps mera.Target, owns the greeting constant
    cli.py           Typer CLI (hello, models list/inspect/download/remove)
    app.py           Streamlit UI page + Model Control Center sidebar
    registry.py      ModelRegistry — YAML loader, Pydantic validation, SHA-256, download, remove
    __init__.py      Package init, re-exports SakuraEngine

configs/
    models.yaml      Model manifest — name, version, path, checksum, source_url, npu_constraints

tests/
    conftest.py      sys.modules mock injection (streamlit + mera)
    test_engine.py   BDD tests — SakuraEngine (8 tests)
    test_cli.py      BDD tests — CLI (10 tests)
    test_app.py      BDD tests — Streamlit page (10 tests)
    test_registry.py BDD tests — ModelRegistry (19 tests)
    test_smoke.py    Smoke tests — get / download / remove workflow (7 tests)

.claude/             Claude Code settings, agent instructions, slash commands
.claude_plugin/      NPU telemetry simulation plugin for Claude Code
.githooks/           pre-commit hook (runs pytest before every commit)
CLAUDE.md            Developer source of truth — all commands documented
```

---

## Development

See [CLAUDE.md](CLAUDE.md) for the full command reference and TDD workflow.

### TDD Cycle

```bash
# Red — write a failing test
poetry run pytest --no-cov -x

# Green — implement minimum code to pass
poetry run pytest --no-cov -x

# Refactor + verify coverage
poetry run pytest          # must show 100%
```

### All commands at a glance

| Command | Description |
|---------|-------------|
| `poetry run pytest` | Full suite + 100% branch coverage |
| `poetry run pytest --no-cov -x` | Fast TDD iteration |
| `poetry run sakura hello` | Engine greeting |
| `poetry run sakura models list` | List models with Space-Ready status |
| `poetry run sakura models inspect <name>` | Show NPU constraints for a model |
| `poetry run sakura models download <name>` | Download model + verify SHA-256 |
| `poetry run sakura models remove <name>` | Remove downloaded model from disk |
| `poetry run streamlit run src/sakura_simulator/app.py` | Launch UI |

---

## MERA SDK

This project depends on the real [EdgeCortix MERA SDK](https://github.com/Edgecortix-Inc/mera) (`mera>=1.6.0` from PyPI). `SakuraEngine` uses:

- **[`mera.Target.Simulator`](https://github.com/Edgecortix-Inc/mera)** — software simulation target (no physical hardware needed)
- **`mera.Platform.SAKURA_2C`** — the [SAKURA-II](https://www.edgecortix.com/en/products/sakura) chip platform (`DNAA600L0003`)

The [MERA framework](https://www.edgecortix.com/en/products/mera) supports PyTorch, TensorFlow, and ONNX models with INT8 quantization and multiple deployment backends.

Only `streamlit` is mocked in tests (to avoid requiring a running Streamlit server); `mera` is imported directly from the installed package.

---

## License

MIT — see [LICENSE](LICENSE).
