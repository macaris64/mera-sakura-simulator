# CLAUDE.md — SAKURA-II NPU Simulator

Source of truth for all developer commands, architecture decisions, and TDD workflow.

---

## Quick Start

```bash
# 1. Install Poetry (once per machine)
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# 2. Configure Poetry (once per machine)
poetry config virtualenvs.in-project true

# 3. Bootstrap project
poetry install          # creates .venv/, installs all deps + editable src install

# 4. Activate git hooks
git config core.hooksPath .githooks

# 5. Verify everything works
poetry run pytest       # must show 100% coverage, all 10 tests PASSED
```

---

## All Developer Commands

| Command | Description |
|---------|-------------|
| `poetry run pytest` | Full test suite with 100% branch coverage enforcement |
| `poetry run pytest --no-cov -x` | Fast iteration — stop at first failure, no coverage |
| `poetry run pytest --no-cov tests/test_engine.py` | Run a single test file |
| `poetry run sakura hello` | Execute CLI greeting command |
| `poetry run sakura --help` | Show CLI help |
| `poetry run streamlit run src/sakura_simulator/app.py` | Launch web UI (http://localhost:8501) |
| `poetry add <pkg>` | Add a runtime dependency |
| `poetry add --group dev <pkg>` | Add a dev-only dependency |
| `poetry shell` | Activate the .venv in current shell |
| `open htmlcov/index.html` | View HTML coverage report after `poetry run pytest` |

---

## Architecture

### Package Layout

```
src/sakura_simulator/       Main package (Poetry src layout)
    __init__.py             Exports SakuraEngine, __version__
    engine.py               SakuraEngine — wraps mera.Target, owns GREETING constant
    cli.py                  Typer CLI — callback() + hello() subcommand
    app.py                  Streamlit UI — main() + _get_engine() cached factory
tests/
    conftest.py             sys.modules mock injection (mera + streamlit)
    test_engine.py          BDD tests for SakuraEngine (4 tests)
    test_cli.py             BDD tests for CLI (2 tests)
    test_app.py             BDD tests for Streamlit page (4 tests)
.githooks/
    pre-commit              Blocks git commit if pytest fails
.claude/
    settings.json           Permissions + PreToolUse hook for commit gating
    AGENTS.md               Agent instructions for TDD workflow
    commands/               Custom slash commands (/test, /run-cli, /run-ui)
.claude_plugin/
    plugin.json             Manifest: get_npu_status + get_telemetry tools
    telemetry.py            Simulated NPU telemetry data model
    npu_monitor.py          Tool handlers + dispatch() router
```

### Greeting String (IMMUTABLE)

```
Hello from Sakura-II: Titan Biosignature Engine Active
```

Defined as `GREETING` constant in `src/sakura_simulator/engine.py:6`.
Asserted directly in tests — never change without updating all test assertions.

### Mock Architecture

`mera` is NOT installed as a pip package. `tests/conftest.py` injects a
hand-crafted `ModuleType("mera")` with a `_MockTarget` class into `sys.modules`
before pytest collects any test modules. This allows `import mera` in `engine.py`
to succeed without the real EdgeCortix SDK.

`streamlit` is mocked similarly via `MagicMock`. Critical detail:
`st.cache_resource.side_effect = lambda fn: fn` makes it a pass-through decorator.
Without this, `@st.cache_resource` would replace `_get_engine` with a `MagicMock`,
breaking the button-click branch test. After any `st.reset_mock()` call in tests,
you must re-apply this `side_effect`.

### CLI Design

The Typer app has an explicit `@app.callback()` to force a Click Group structure.
This makes `hello` a true named subcommand, so `sakura hello` works both from the
command line and via `CliRunner.invoke(app, ["hello"])` in tests.

---

## TDD Workflow

### Red → Green → Refactor cycle

```bash
# 1. Red — write a failing test
#    Edit tests/test_<module>.py and add a new Given-When-Then test.
poetry run pytest --no-cov -x           # confirm it fails

# 2. Green — write minimum implementation
#    Edit src/sakura_simulator/<module>.py with minimum code to pass.
poetry run pytest --no-cov -x           # confirm it passes

# 3. Refactor + coverage check
poetry run pytest                        # must still show 100% coverage
```

### BDD Test Structure

Every test method follows the Given-When-Then pattern:

```python
def test_given_<state>_when_<action>_then_<outcome>(self):
    # Given: set up preconditions
    # When: execute the action
    # Then: assert the outcome
```

---

## Claude Code Plugin

`.claude_plugin/` provides two simulation tools to Claude Code agents:

- **`get_npu_status`** — returns simulated SAKURA-II hardware status JSON
  (online, temperature_c, utilization_pct, target, timestamp)
- **`get_telemetry`** — returns simulated inference performance metrics JSON
  (inference_latency_ms, throughput_inferences_per_sec, power_draw_watts, window_ms)

Plugin manifest: `.claude_plugin/plugin.json`
Tool handlers: `.claude_plugin/npu_monitor.py`
Data models: `.claude_plugin/telemetry.py`

---

## Coverage Configuration

Coverage is configured entirely in `pyproject.toml`:

- `--cov-branch` — branch coverage required, not just line coverage
- `--cov-fail-under=100` — CI fails if total drops below 100%
- `[tool.coverage.paths]` — maps `.venv/site-packages/sakura_simulator` back to
  `src/sakura_simulator` so the editable install doesn't cause 0% false negatives
- `exclude_lines` — excludes `if __name__ == "__main__":` guards
