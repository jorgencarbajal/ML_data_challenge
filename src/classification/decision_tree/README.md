# Overview

This file is a numbered, script-based baseline pipeline for a binary classification task: predicting `injury present` using scikit-learn `DecisionTreeClassifier`.

The workflow order:

1. inspect the raw dataset
2. choose columns that fit the method
3. split the data
4. preprocess the features
5. train and validate decision tree
6. test of the held-out set
7. save the baseline results

There is shared logic inside `common.py`. The ordering is meant to test specific files as they are number to ensure execution is working as intended. We ensure the exclusion of target columns to prevent data leakage.