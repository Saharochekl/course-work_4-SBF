# GO-3055 regression baseline

This directory freezes the scientific outputs that existed at git commit
`5ea77b9` (`Reg baseline`). It was built from the already-computed files in
`code/sbf2_batch_outputs`; creating it did not rerun the SBF pipeline.

The baseline is deliberately compact. Full-frame FITS products are not useful
as byte-for-byte regression fixtures because headers, paths, and serialization
details may change without changing the science result.

## Files

- `baseline_targets.csv`: target-level measurements and quality decisions for
  all 14 GO-3055 galaxies.
- `baseline_calibration.json`: sample-level calibration and model-comparison
  values, plus the accepted numerical tolerances.
- `baseline_environment.txt`: environment and source hashes at the baseline
  commit. These are provenance, not pass/fail requirements.

## Check the existing outputs

From the project root:

```bash
./astro_env/bin/python code/check_regression_baseline.py
```

This check reads existing CSV outputs only and normally finishes in less than a
second. It does not execute `sbf-2.ipynb` and does not open the large FITS files.

## Intended test cadence

1. Run this cheap check after interface and reporting changes.
2. After the filter-input refactor, rerun only NGC 1380 in an isolated output
   directory and compare it with this baseline.
3. Rerun NGC 4621 as the warning-path smoke test.
4. Rerun all 14 targets once before declaring the refactor complete.

The target magnitude tolerance is 0.005 mag. A larger difference is a
scientific change and must be explained, not silently accepted by widening the
tolerance.

