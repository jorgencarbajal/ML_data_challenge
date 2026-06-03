File Summaries

common.py

This is the shared helper file for the whole decision tree baseline. It defines constants like DATA_PATH, TARGET, split sizes, random seed, and tree hyperparameters. It also contains reusable functions for loading the dataset, excluding risky columns, choosing predictors, grouping features into numeric/categorical/binary, splitting the data, building the preprocessing pipeline, building the full decision tree model, selecting the best F1 threshold, and summarizing class balance.

01_inspect_data.py

This is the first exploratory audit script. It loads data/v1.csv, checks that injury_present exists and is a valid binary target, prints dataset size and memory usage, audits every column’s dtype/missingness/unique values, checks identifier duplication, and scans column names for possible leakage candidates.

02_select_features.py

This script formalizes which columns are included or excluded. It loads the data, prints the target distribution, lists excluded columns by reason, lists included predictor columns by feature type, and validates that every column is accounted for exactly once: either included as a predictor or explicitly excluded.

03_split_data.py

This script verifies the train/validation/test split strategy. It loads predictors and target, groups features, prints the effective split proportions, creates stratified splits, prints class balance for each split, and checks that split row counts match and indices do not overlap.

04_preprocess_data.py

This script tests the preprocessing pipeline before modeling. It creates the train/validation/test split, builds the ColumnTransformer, fits preprocessing only on training data, transforms validation/test data, prints output matrix shapes and encoded feature counts, and validates that preprocessing preserves row counts and creates consistent feature dimensions across splits.

05_train_validate_tree.py

This script trains the baseline decision tree and evaluates it on the validation set. It builds the full sklearn Pipeline, fits the model on training data, gets validation probabilities, compares the default 0.50 threshold against a threshold chosen to maximize validation F1, prints validation metrics, prints a confusion matrix, and summarizes the learned tree depth/leaves.

Small note for later: this file uses type annotations np.ndarray and Pipeline, but those names are not imported in the file. That may cause a runtime error depending on your Python version.

06_evaluate_test_tree.py

This script performs the official held-out test evaluation. It trains on the training split, selects the threshold using validation probabilities only, freezes that threshold, then evaluates both validation and test performance using PR-AUC, F1, precision, recall, predicted injury count, and a test confusion matrix. Its main purpose is to keep the test set clean.

07_save_baseline_results.py

This final script saves the baseline experiment results to JSON at data/decision_tree_baseline_results.json. It trains the model, selects the validation threshold, computes validation/test metrics, records model settings, split sizes, feature counts, tree parameters, threshold details, and writes everything into a reproducible results file.