# Reproducibility status

## Available

- Original experiment notebook and presentation.
- RePN architecture extracted into an importable Python module.
- Dataset schema and filtering rules.
- Default model and training hyperparameters.
- Deterministic seed helper and checkpoint-based training loop.
- Synthetic unit tests for preprocessing and model shape behaviour.

## Not available

- The raw TLS PLY dataset.
- The code that generated the `Group_<scan_id>.npy` neighbourhood arrays.
- A version-locked environment from the original 2023 experiment.
- Original fold checkpoints and predictions.

Because the group-generation step and measurements are absent, the reported
metrics cannot currently be independently reproduced from a fresh clone.

## Recommended path to full reproduction

1. Confirm whether the research dataset can be published.
2. Recover or reimplement neighbourhood generation with documented KD-tree,
   radius, and farthest-point-sampling settings.
3. Add a command-line training entry point that consumes a configuration file.
4. Pin a tested environment and record hardware/CUDA details.
5. Store data and model artifacts in a release or data repository rather than
   the Git history.
6. Run all folds from a clean checkout and publish an experiment manifest with
   metrics and checksums.

The original notebook remains in `notebooks/legacy/` as historical evidence.
New experiments should use the package modules and record their configuration.
