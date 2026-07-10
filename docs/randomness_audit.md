# Randomness audit

The canonical seed is 42. Dataset splitting is deterministic temporal leave-one-out. TruncatedSVD, index construction/training, calibration query sampling, evaluation sampling, bootstrap resampling, embedding sensitivity, PQ diagnostics, exposure analysis, and scale stress receive an explicit seed through their canonical configuration or CLI.

CPU thread settings are captured in `results/_meta/hardware/`; canonical index builds default to one OpenMP thread. GPU nondeterminism is inapplicable because no canonical GPU execution path exists.
