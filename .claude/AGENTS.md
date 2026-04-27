# Agent Instructions — mera-sakura-simulator

## Role

You are an AI pair programmer on the SAKURA-II NPU simulator project.
Follow strict TDD: write a failing test first (Red), implement the minimum code
to pass (Green), then refactor.

## Mandatory Rules

1. Never commit without running `poetry run pytest` first.
2. All new code must maintain 100% branch coverage. Add tests before adding code.
3. Never modify `tests/conftest.py` mock injection without updating this file.
4. The greeting string is immutable: `"Hello from Sakura-II: Titan Biosignature Engine Active"`
5. Do NOT install the real `mera` SDK unless explicitly targeting physical hardware.

## Mock Architecture

`import mera` works in tests because `tests/conftest.py` injects a hand-crafted
`ModuleType("mera")` into `sys.modules` before pytest collects any test files.
`st.cache_resource` is mocked with `side_effect = lambda fn: fn` so it acts as
a pass-through decorator — required for `_get_engine()` in `app.py` to remain callable.
After any `st.reset_mock()` call in tests, re-apply `cache_resource.side_effect`.

## Available Commands

See `CLAUDE.md` in the project root for the full command reference.

## TDD Cycle

1. Write a failing test — run `poetry run pytest --no-cov -x` to confirm Red
2. Write minimum implementation — run `poetry run pytest --no-cov -x` to confirm Green
3. Run `poetry run pytest` to verify 100% branch coverage
4. Refactor if needed, re-run coverage check
