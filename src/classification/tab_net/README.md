# TabNet Classification Pipeline

This folder contains the recent tabular-model comparison pipeline for predicting
`injury_present`.

The pipeline is intentionally separate from the Decision Tree baseline because
TabNet uses different preprocessing:

- numeric features remain numeric after training-only imputation;
- binary features remain 0/1 after training-only imputation;
- categorical features are integer encoded and passed to TabNet with
  `cat_idxs` and `cat_dims`.

The comparison should reuse the same target, leakage exclusions, split seed,
validation-threshold selection rule, and final metrics as the Decision Tree
baseline.
