# Contributing

Thank you for your interest in UMOSLS-DL.

## Before opening a change

- Keep raw research data and model checkpoints out of Git.
- Use UTF-8 for source code and documentation.
- Preserve the scientific meaning of features, units, filters, and metrics.
- Document any change to preprocessing or model architecture.

## Local checks

Create a virtual environment and install the development dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

For model changes, include a small synthetic-data test. For experiment changes,
record the random seed, package versions, data revision, and configuration.

## Pull requests

Use a focused branch and explain:

1. What changed.
2. Why it changed.
3. How it was tested.
4. Whether results or data assumptions changed.
