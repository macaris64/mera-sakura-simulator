# mera-sakura-simulator

A **Hello World simulator** for the [EdgeCortix SAKURA-II NPU](https://www.edgecortix.com/sakura-ii) using the MERA framework — built with strict **Test-Driven Development (TDD)** and **Behavior-Driven Development (BDD)** practices.

> Hardware-agnostic: runs everywhere via a MERA mock layer. No physical NPU or SDK required.

---

## Features

- `SakuraEngine` — wraps `mera.Target` for SAKURA-II NPU initialization
- **CLI** via [Typer](https://typer.tiangolo.com/): `sakura hello`
- **Web UI** via [Streamlit](https://streamlit.io/): activate the engine from a browser
- **100% branch coverage** enforced by pytest-cov on every commit
- **BDD-style tests** with Given-When-Then structure
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
poetry run sakura hello
# Hello from Sakura-II: Titan Biosignature Engine Active
```

### Web UI

```bash
poetry run streamlit run src/sakura_simulator/app.py
# Open http://localhost:8501
```

---

## Project Structure

```
src/sakura_simulator/
    engine.py       SakuraEngine — wraps mera.Target, owns the greeting constant
    cli.py          Typer CLI (`sakura hello`)
    app.py          Streamlit UI page
    __init__.py     Package init, re-exports SakuraEngine

tests/
    conftest.py     sys.modules mock injection (mera + streamlit)
    test_engine.py  BDD tests — SakuraEngine (4 tests)
    test_cli.py     BDD tests — CLI (2 tests)
    test_app.py     BDD tests — Streamlit page (4 tests)

.claude/            Claude Code settings, agent instructions, slash commands
.claude_plugin/     NPU telemetry simulation plugin for Claude Code
.githooks/          pre-commit hook (runs pytest before every commit)
CLAUDE.md           Developer source of truth — all commands documented
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
| `poetry run sakura hello` | Run CLI |
| `poetry run streamlit run src/sakura_simulator/app.py` | Launch UI |

---

## Mock Architecture

`mera` is not a standard pip package. The project uses two layers:

1. **Tests**: `tests/conftest.py` injects a minimal `ModuleType("mera")` into `sys.modules` before any test file is collected — so `import mera` inside `engine.py` always resolves to the mock.
2. **Runtime**: `engine.py` has a `try/except ImportError` fallback that builds an inline simulation target, so `sakura hello` works without the real SDK.

---

## License

MIT — see [LICENSE](LICENSE).
