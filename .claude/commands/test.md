---
description: Run the full test suite with 100% branch coverage enforcement
---

Run all tests with coverage:

```bash
poetry run pytest
```

To run without coverage for faster TDD iteration:

```bash
poetry run pytest --no-cov -x
```

To view HTML coverage report after a run:

```bash
open htmlcov/index.html
```
