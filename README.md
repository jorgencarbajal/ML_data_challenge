# ML Data Challenge

This repository contains a self-contained Jupyter notebook for the FRA highway-rail grade crossing incident data challenge.

## How to Run

Place the raw CSV file in the `data` folder with this filename:

```text
data/Highway-Rail_Grade_Crossing_Incident_Data_(Form_57)_20260603.csv
```

Then open and run:

```text
data_challenge.ipynb
```

The notebook is self-contained. It loads the raw CSV, rebuilds the cleaned working dataset, trains/evaluates the classification methods, runs the clustering methods, and creates the report figures.

Expected runtime:

- Decision Tree is quick.
- TabNet trains on CPU and may take a few minutes.
- Clustering builds a 2,000 by 2,000 distance matrix and may also take a few minutes.

For the cleanest rerun, restart the notebook kernel and run all cells from the top.

## Project Structure

```text
data_challenge.ipynb        Self-contained final notebook
data/                       Raw data and generated/intermediate data files
src/                        Script-based development pipelines
```

The `src` folder contains the script-based pipelines used during development. The notebook is the main reproducible deliverable and does not require loading saved model-result artifacts.

# Report V1 Working Source

This file is a detailed working source document for the final data challenge report and notebook. It is not meant to be pasted directly as the final 4-page ACM SIGKDD-style report. The final report should be a compressed version of this document, written in the student's own final wording, with verified references and only the strongest tables/figures.

Academic integrity note: because this is coursework, treat this file as a structured project record and study guide. Verify every claim, understand each method, and rewrite the final report in your own voice. Do not invent experiments, citations, or results beyond the artifacts actually produced in this repository.

## 1. Rubric Coverage Checklist

### Documentation About Choices

Status: strong, if final report summarizes this clearly.

Choices that should be explicitly documented:

- Why the raw dataset was reduced before modeling.
- Why missingness and reporting-era effects led to removing `Crossing Warning Explanation`, `Roadway Condition`, `User Age`, and `Signaled Crossing Warning`.
- Why Form 55A injury/fatality totals were used to define binary outcomes.
- Why outcome totals, identifiers, exact dates, and post-incident consequence fields were excluded from predictors.
- Why the project selected classification and clustering as the two task families.
- Why K-medoids/PAM was selected as the classical clustering method.
- Why HDBSCAN was selected as the density-based clustering comparison.
- Why Decision Tree was selected as the classical classification method.
- Why TabNet was selected as the recent tabular classification method.
- Why clustering used a 2,000-row sample while classification used the full cleaned dataset.
- Why validation-threshold selection was used for classification.

### Compare With Baselines

Status: strong.

Baseline comparisons now exist for both selected task families:

- Clustering: K-medoids/PAM baseline vs HDBSCAN comparison.
- Classification: Decision Tree baseline vs TabNet comparison.

The final report should include both a numeric comparison table and a short interpretation for each task.

### Reasoning Samples

Status: mostly strong, but final report must be explicit.

Only the clustering task used a sample. The reason was computational: PAM and HDBSCAN used a precomputed pairwise distance matrix, and a full 250,806 by 250,806 matrix would be too expensive in memory and runtime.

Sampling justification to include:

- A simple random sample of 2,000 incidents was taken from the full cleaned dataset with seed 42.
- The same sampled rows were used for K-medoids and HDBSCAN.
- The sample was not stratified by injury/fatality because clustering was unsupervised and outcomes were not allowed to influence cluster formation.
- The sample outcome rates were close to the full dataset outcome rates:
  - Full `fatality_present`: 8.26%; sample `fatality_present`: 7.90%.
  - Full `injury_present`: 27.36%; sample `injury_present`: 25.35%.
- This supports the sample as a reasonable pilot representation of broad incident outcomes, although it does not prove perfect representativeness for every feature or year.

Limitation to include:

- A stronger final experiment would repeat clustering over multiple random samples and average the cluster-quality summaries. The current project uses a fixed representative sample because of time and computational constraints.

### Cross Validation

Status: this needs careful handling.

The current notebook and executable Decision Tree/TabNet pipelines use train/validation/test splitting, not full k-fold cross-validation for the final reported test metrics. The older `notes/report_draft.md` includes a 3-fold stratified cross-validation comparison for Decision Tree tuning, but the current final notebook reports the newer train/validation/test threshold pipeline.

Do not falsely claim that every final model was evaluated by k-fold cross-validation unless that code is actually run and saved.

Options for satisfying the rubric:

1. Add a small cross-validation section before final submission.
   - For classification: run 3-fold stratified CV on the training/modeling data for Decision Tree and possibly a smaller TabNet or fewer epochs if time permits.
   - For clustering: repeated random samples can function as a stability check, but it is not standard supervised cross-validation.

2. If no new CV is run, be transparent:
   - "Final model selection used a stratified train/validation/test design. Earlier Decision Tree sensitivity used 3-fold stratified cross-validation, but the final artifact is reported with a validation-selected threshold and a held-out test set."

Because the rubric explicitly says "Needs to use cross validation for evaluation," the safest path is to add at least a small classification CV or repeated-split validation table before final submission.

### Completeness

Status: strong after notebook assembly.

The code now covers:

- Data cleaning and feature construction.
- Clustering baseline and comparison.
- Classification baseline and comparison.
- Self-contained notebook execution from the raw data file.
- Current status notes.

Remaining work:

- Read and polish the final notebook.
- Use selected notebook plots/tables in the report.
- Read/comment through code.
- Condense this file into a final 4-page report.

## 2. Project Overview

The project uses the FRA highway-rail grade crossing incident dataset from the Form 57 data downloads page:

- Dataset page: https://data.transportation.gov/stories/s/Form-57-Data-Downloads/i5dw-jvsi/

The project requires selecting two major data-mining task families from classification, clustering, association rule mining, and anomaly detection. The final selected task families are:

1. Classification.
2. Clustering.

The cleaned dataset used for modeling is:

- `data/v1.csv`

The raw dataset contained approximately:

- 250,806 incidents.
- 154 original fields.

The cleaned Version 1 dataset contains:

- 250,806 incidents.
- 49 columns.

The main goal is not to maximize performance at all costs. The project is graded on reasoning, documentation, comparison, experimental design, and the ability to explain the pipeline. Therefore, the final report should emphasize the decisions made, why they were made, and what the results do and do not prove.

## 3. Final Research Questions

### Classification Question

Main question:

Can recorded highway-rail crossing incident characteristics be used to classify whether a recorded incident involved at least one reported injury?

Target:

```text
injury_present = 1 if Total Injured Form 55A > 0
injury_present = 0 otherwise
```

Important boundary:

This is not a model of whether an incident will occur at a crossing in the future. It is a model of injury occurrence among already-recorded incidents using incident-context variables.

Subquestions:

- Does a classical Decision Tree learn useful injury-related signal beyond the no-skill baseline?
- Does TabNet, a more recent deep tabular method, improve ranking and classification metrics over the Decision Tree?
- Are the most useful predictors interpretable incident-context variables rather than outcome leakage?
- How does the model behave under class imbalance, where no-injury incidents are the majority?

### Clustering Question

Main question:

Do highway-rail crossing incidents form stable and interpretable groups based on incident characteristics alone?

Important boundary:

Clustering is descriptive. Outcome variables such as `fatality_present` and `injury_present` are not used to form clusters. They are inspected only after clustering for interpretation.

Subquestions:

- What broad incident profiles emerge when K-medoids forces every sampled incident into a fixed number of clusters?
- Does HDBSCAN find similar dense incident profiles, or does it label much of the data as ambiguous/noise?
- How do the cluster-size patterns and interpretability differ between a partitioning method and a density-based method?

## 4. Task Selection and Project Pivots

The project originally considered clustering and association rule mining. Association rule mining was considered because it could identify combinations of incident characteristics associated with injury or fatality outcomes. However, this path was deprioritized because it risked producing many weak, repetitive, or difficult-to-interpret rules. Classification created a cleaner supervised question and a clearer evaluation framework.

The final decision was:

- Classification: Decision Tree vs TabNet.
- Clustering: K-medoids/PAM vs HDBSCAN.

This combination satisfies the requirement to perform two task families and compare at least two substantially different methods for each task.

## 5. Data Cleaning and Version 1 Dataset

### Purpose of Cleaning

The first major phase reduced the raw FRA dataset into a smaller working dataset before task-specific modeling. The preprocessing goal was to preserve useful incident-context information, create usable injury/fatality outcomes, remove or transform problematic fields, and clearly separate modeling features from audit-only or outcome-related fields.

### Initial Feature Reduction

The raw dataset was reduced from 154 original fields to 51 candidate source fields. These fields were chosen because they were potentially useful for:

- incident-context analysis;
- modeling;
- outcome construction;
- auditing and validation.

After missingness and reporting-era audits, four fields were removed:

| Removed feature | Reason |
|---|---|
| `Crossing Warning Explanation` | About 99.62% missing |
| `Roadway Condition` | About 87.28% missing and mostly unavailable before 2012 |
| `User Age` | About 72.04% missing and unavailable in many earlier years |
| `Signaled Crossing Warning` | About 50.73% missing |

After these removals, Version 1 retained 47 original/source fields before derived variables were added.

### Duplicate Report Key Audit

`Report Key` was retained temporarily as an audit identifier. The duplicate audit found:

| Duplicate audit result | Count |
|---|---:|
| Rows associated with duplicated `Report Key` values | 47 |
| Distinct duplicated-key groups | 23 |
| Groups identical across retained fields | 14 |
| Groups with conflicting retained values | 9 |

Because some duplicated keys had conflicting retained values, they could not all be treated as exact duplicates. The Version 1 simplifying decision was to keep the first occurrence of each duplicated `Report Key` and remove later occurrences. This affected very few incidents relative to the full dataset.

Final modeling rule:

- `Report Key` is audit-only.
- `Report Key` is never used as a predictor.

### Outcome Construction

The project compared Form 55A and Form 57 injury/fatality totals before choosing the outcome source.

Agreement:

| Comparison | Agreement |
|---|---:|
| Form 55A killed total vs Form 57 killed total | 99.59% |
| Form 55A injured total vs Form 57 injured total | 98.25% |

Because agreement was high, Form 55A totals were used to create binary outcomes:

```text
fatality_present = 1 if Total Killed Form 55A > 0, otherwise 0
injury_present   = 1 if Total Injured Form 55A > 0, otherwise 0
```

Outcome frequencies:

| Outcome | Positive incidents | Percent |
|---|---:|---:|
| `fatality_present` | 20,711 | 8.26% |
| `injury_present` | 68,617 | 27.36% |

The original injury/fatality totals were retained for validation and interpretation, but excluded from model predictors to prevent outcome leakage.

### Warning-Device Transformation

The original dataset contained twelve warning-device slots:

```text
Crossing Warning Expanded 1
...
Crossing Warning Expanded 12
```

These fields represented multiple warning devices that could appear in any slot. Treating each slot as a separate categorical variable would be misleading because the slot position does not define a separate concept. The fields were replaced with binary indicators:

```text
has_gate
has_cantilever_fls
has_standard_fls
has_wig_wags
has_highway_traffic_signals
has_audible
has_crossbucks
has_stop_signs
has_watchman
has_flagged_by_crew
has_other_warning
has_no_warning_device
```

Each indicator equals 1 when that warning device appears anywhere among the twelve original warning-device fields for an incident.

### Time Transformation

The raw time fields included:

```text
Date
Month
Hour
Time
```

The project created:

| Derived feature | Purpose |
|---|---|
| `year` | Audit reporting-era effects only |
| `season` | Broad seasonal/environmental context |
| `time_of_day` | Interpretable timing category |

`Date` and `year` are excluded from modeling because they can encode reporting-era artifacts rather than incident characteristics.

### Numeric Cleaning

Numeric audit fields:

```text
Train Speed
Estimated Vehicle Speed
Number Vehicle Occupants
Number of Cars
Temperature
Vehicle Damage Cost
```

`Vehicle Damage Cost` was cleaned by removing commas and dollar signs, but excluded from modeling because it is post-incident and not inflation-adjusted.

Clearly suspicious numeric measurements were set to missing:

| Feature | Cleaning rule | Values replaced |
|---|---|---:|
| `Train Speed` | Greater than 110 mph | 3 |
| `Estimated Vehicle Speed` | Greater than 120 mph | 7 |
| `Number of Cars` | Greater than 300 | 9 |
| `Temperature` | Below -80 F or above 130 F | 34 |

In total, 53 suspicious numeric values were replaced with missing values.

### Final Feature-Use Rules

Audit-only:

```text
Report Key
Date
year
```

Outcome or outcome-source fields excluded from predictors:

```text
Crossing Users Killed
Crossing Users Injured
Employees Killed
Employees Injured
Passengers Killed
Passengers Injured
Total Killed Form 55A
Total Injured Form 55A
Total Killed Form 57
Total Injured Form 57
fatality_present
injury_present
```

Post-incident consequence excluded:

```text
Vehicle Damage Cost
```

Context features retained:

- highway-user characteristics;
- rail-equipment and track characteristics;
- visibility, weather, and obstruction fields;
- highway-user action and driver-context fields;
- crossing illumination and signal connection;
- `season` and `time_of_day`;
- binary warning-device indicators;
- cleaned numeric context variables.

## 6. Final Modeling Feature Spaces

### Classification Feature Space

The final classification target is:

```text
injury_present
```

The final raw predictor count is 32:

| Predictor group | Count |
|---|---:|
| Numeric | 5 |
| Categorical | 16 |
| Binary warning-device indicators | 11 |
| Total | 32 |

`has_no_warning_device` was removed because it was constant:

```text
has_no_warning_device: {0: 250806}
```

Final classification split:

| Split | Rows | Injury cases | No-injury cases | Injury percent |
|---|---:|---:|---:|---:|
| Full | 250,806 | 68,617 | 182,189 | 27.36% |
| Train | 160,515 | 43,914 | 116,601 | 27.36% |
| Validation | 40,129 | 10,979 | 29,150 | 27.36% |
| Test | 50,162 | 13,724 | 36,438 | 27.36% |

Split settings:

```text
RANDOM_SEED = 13
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20 of the post-test modeling set
```

The effective split is:

- 64% train.
- 16% validation.
- 20% test.

### Clustering Feature Space

The original clustering feature set had 33 features:

- 5 numeric.
- 16 categorical.
- 12 binary warning-device indicators.

Sensitivity testing showed that `time_of_day` overlapped with or added noise relative to `Visibility`. Removing `time_of_day` improved the K-medoids silhouette score. Therefore, the final fitted clustering feature count was 32:

- all 5 numeric features;
- all 12 binary warning-device indicators;
- all categorical features except `time_of_day`.

`time_of_day` remained available only for post-cluster profiling.

Outcome variables were never used for clustering.

## 7. Evaluation Design

### Classification Metrics

Accuracy was not the main metric because no-injury incidents are the majority class. A model predicting no injury for every incident would already achieve 72.64% accuracy.

Final classification metrics:

- Average precision / PR-AUC: evaluates probability ranking under class imbalance.
- F1: balances precision and recall at the selected threshold.
- Precision: fraction of predicted injury cases that were actually injury cases.
- Recall: fraction of actual injury cases identified.
- Confusion matrix: shows true negatives, false positives, false negatives, and true positives.
- Predicted injury cases: helps interpret how aggressive the threshold is.

Threshold rule:

- The threshold was selected using validation F1.
- The selected threshold was frozen.
- The held-out test set was used only for final evaluation.

This avoids selecting a threshold based on the test set.

### Clustering Metrics

Final clustering metrics:

- Silhouette score.
- Total PAM cost for K-medoids.
- Cluster count.
- Cluster sizes.
- Noise percent for HDBSCAN.
- Clustered rows for HDBSCAN.
- Post-hoc outcome rates for interpretation only.
- Qualitative profile interpretability using overrepresented categorical values and warning-device rates.

Important comparison note:

- K-medoids silhouette is calculated over all sampled incidents because every row is assigned to a cluster.
- HDBSCAN clustered-point silhouette is calculated only over non-noise clustered points.
- Therefore, HDBSCAN's silhouette score should not be treated as directly identical to K-medoids' silhouette score.

### Sampling Design for Clustering

PAM/K-medoids and precomputed-distance HDBSCAN both require a pairwise distance matrix. A full distance matrix for 250,806 incidents would be computationally unrealistic for this project.

Clustering used:

```text
Sample size = 2,000
Random seed = 42
```

The same sample and same Gower-style distance matrix were used for K-medoids and HDBSCAN. This was important for a fair comparison.

The sample was not outcome-stratified because clustering is unsupervised. Injury and fatality outcomes were intentionally prevented from shaping cluster formation.

## 8. Method 1: K-medoids / PAM Clustering Baseline

### Why K-medoids

K-medoids is a classical partitioning clustering method. It is similar in spirit to K-means, but cluster centers are actual observations called medoids. This is useful here because:

- the dataset has mixed numeric, categorical, and binary features;
- the method can use a precomputed dissimilarity matrix;
- medoids are actual recorded incidents, which can be easier to interpret than artificial means;
- it avoids assuming Euclidean geometry over all raw features.

Implementation:

- Python package: `kmedoids`.
- Algorithm: PAM / Partitioning Around Medoids.
- Distance: custom Gower-style mixed-data dissimilarity.

### Gower-Style Distance

Numeric features:

```text
absolute difference / full-dataset feature range
```

Categorical features:

```text
same category = 0
different category = 1
```

Binary warning-device features:

- Treated asymmetrically.
- Shared presence of a device counts as similarity.
- One present and one absent counts as difference.
- Shared absence is not treated as strong evidence of similarity.

Distance matrix summary:

| Quantity | Value |
|---|---:|
| Matrix shape | 2,000 by 2,000 |
| Memory | 15.26 MB |
| Minimum pairwise distance | 0.0017 |
| Median pairwise distance | 0.4168 |
| Maximum pairwise distance | 0.8440 |

### K Selection

PAM was evaluated for `k=2` through `k=8` after removing `time_of_day` from fitted clustering features.

| k | Silhouette | Total cost | Smallest cluster | Largest cluster |
|---:|---:|---:|---:|---:|
| 2 | 0.1657 | 547.3750 | 437 | 1,563 |
| 3 | 0.1444 | 518.6757 | 386 | 1,182 |
| 4 | 0.1007 | 496.0873 | 356 | 856 |
| 5 | 0.0925 | 479.2149 | 290 | 682 |
| 6 | 0.0956 | 465.6280 | 187 | 680 |
| 7 | 0.0724 | 454.5081 | 182 | 431 |
| 8 | 0.0750 | 445.3983 | 131 | 412 |

Selection rule:

- Highest silhouette score.
- Smaller `k` breaks ties.

Selected:

```text
k = 2
silhouette = 0.1657
total cost = 547.3750
```

Interpretation boundary:

The silhouette score is not high. The result should be interpreted as a weak-to-moderate two-cluster partition with substantial overlap, not as evidence of two cleanly separated natural groups.

### K-medoids Final Profiles

Cluster sizes:

| Cluster | Rows | Percent |
|---:|---:|---:|
| 0 | 1,563 | 78.15% |
| 1 | 437 | 21.85% |

Numeric medians:

| Feature | Cluster 0 | Cluster 1 |
|---|---:|---:|
| `Train Speed` | 20.0 | 30.0 |
| `Estimated Vehicle Speed` | 8.0 | 0.0 |
| `Number Vehicle Occupants` | 1.0 | 1.0 |
| `Number of Cars` | 19.0 | 55.0 |
| `Temperature` | 57.0 | 61.0 |

Cluster 0:

- 67.95% `Did not stop`, compared with 53.55% overall.
- 86.24% `Moving over crossing`, compared with 72.80% overall.
- 95.52% `Driver In Vehicle = Yes`, compared with 83.90% overall.
- Lower rates of gates, standard flashing lights, and audible devices.
- Higher crossbuck rate than overall.

Working label:

```text
moving-over-crossing / fewer-active-warning profile
```

Cluster 1:

- 63.84% `Stopped on crossing`, compared with 25.55% overall.
- 49.66% `Driver In Vehicle = No`, compared with 13.65% overall.
- 41.65% `Stopped on crossing` position and 29.98% `Stalled or stuck on crossing`.
- 92.22% rail equipment struck highway user.
- Strongly higher rates of gates, standard flashing lights, and audible devices.

Working label:

```text
stopped-or-stalled / active-warning profile
```

Post-hoc outcomes:

| Cluster | Fatality present | Injury present |
|---:|---:|---:|
| 0 | 7.74% | 28.34% |
| 1 | 8.47% | 14.65% |

These outcome rates are descriptive only and should not be interpreted causally.

## 9. Method 2: HDBSCAN Clustering Comparison

### Why HDBSCAN

HDBSCAN is a density-based clustering method that can identify dense groups while labeling weakly clusterable points as noise. This makes it a useful comparison to K-medoids:

- K-medoids forces every point into one of `k` clusters.
- HDBSCAN can decide that many points are ambiguous/noise.
- HDBSCAN can test whether the broad K-medoids clusters are supported by dense local structure.

Implementation:

- `sklearn.cluster.HDBSCAN`.
- Metric: precomputed Gower-style distance matrix.
- Same 2,000 sampled incidents as K-medoids.
- Same distance matrix as K-medoids.

This makes the comparison fair because both methods see the same incidents and same pairwise dissimilarities.

### HDBSCAN Hyperparameter Search

Grid:

- `cluster_selection_method`: `eom`, `leaf`.
- `min_cluster_size`: 5, 10, 15, 20, 25, 30, 50.
- `min_samples`: None, 1, 3, 5, 10, 20.

Total configurations:

```text
84
```

Selection rule:

- At least 2 clusters.
- Noise percent at or below 90%.
- Smallest cluster size at least 20.
- Highest clustered-point silhouette.

Selected configuration:

| Parameter | Value |
|---|---|
| `min_cluster_size` | 25 |
| `min_samples` | 3 |
| `cluster_selection_method` | `eom` |

### HDBSCAN Final Result

| Quantity | Value |
|---|---:|
| Clusters excluding noise | 2 |
| Noise rows | 1,780 |
| Noise percent | 89.00% |
| Clustered rows | 220 |
| Clustered percent | 11.00% |
| Clustered-point silhouette | 0.3082 |

Group sizes:

| Group | Rows |
|---|---:|
| `noise` | 1,780 |
| `cluster_0` | 190 |
| `cluster_1` | 30 |

Interpretation:

HDBSCAN found only small dense groups. Most sampled incidents were labeled as noise/ambiguous. This is a meaningful contrast with K-medoids, which assigns every incident to a broad cluster.

### HDBSCAN Profiles

Cluster 0:

- 95.79% `Did not stop`.
- 100.00% `Moving over crossing`.
- 99.47% `has_crossbucks`.
- 0.00% gates, standard flashing lights, and audible devices.
- 86.32% `Crossing Illuminated = No`.
- 76.84% `Visibility = Day`.

Working label:

```text
dense moving-over-crossing / passive-crossbuck profile
```

Cluster 1:

- 96.67% `Stopped on crossing` position.
- 90.00% `Stopped on crossing` action.
- 100.00% `has_crossbucks`.
- 0.00% gates, standard flashing lights, and audible devices.
- 96.67% `Highway User = Auto`.
- 93.33% `Visibility = Day`.

Working label:

```text
dense stopped-on-crossing / passive-crossbuck auto profile
```

Noise group:

- 1,780 rows.
- Contains more of the active-warning structure:
  - 24.10% gates.
  - 38.82% standard flashing lights.
  - 22.47% audible devices.
- This suggests that many active-warning incidents do not form a compact density cluster under the selected distance representation.

Post-hoc outcomes:

| Group | Fatality present | Injury present |
|---|---:|---:|
| `cluster_0` | 11.05% | 37.37% |
| `cluster_1` | 13.33% | 33.33% |
| `noise` | 7.47% | 23.93% |

Again, outcomes were not used to form the clusters. These rates are descriptive only.

### Clustering Comparison Summary

| Method | Sample | Distance | Assigned groups | Noise percent | Main score |
|---|---:|---|---|---:|---:|
| K-medoids/PAM | 2,000 | Gower-style | 2 clusters | 0.00% | silhouette 0.1657 on all points |
| HDBSCAN | 2,000 | Same Gower-style matrix | 2 clusters + noise | 89.00% | clustered-point silhouette 0.3082 |

Main conclusion:

K-medoids found broad forced partitions, while HDBSCAN found only small dense profiles and treated most records as ambiguous/noise. This suggests that the incident data contains some local dense patterns but weak global cluster structure.

## 10. Method 3: Decision Tree Classification Baseline

### Why Decision Tree

Decision Trees are classical supervised learning methods. They are appropriate as a baseline because:

- they are interpretable;
- they can model nonlinear rules;
- they can use mixed data after preprocessing;
- feature importance and decision rules are easy to explain;
- they provide a reasonable classical comparator for a newer tabular neural model.

Implementation:

- `sklearn.tree.DecisionTreeClassifier`.

Current final staged Decision Tree settings:

| Setting | Value |
|---|---|
| `max_depth` | 8 |
| `min_samples_leaf` | 100 |
| `class_weight` | None |
| random seed | 13 |

Important note:

The older draft includes a previous class-weighted Decision Tree run with different seed and different results. The current final notebook uses the train/validation/test threshold pipeline with an unweighted depth-8 Decision Tree. Do not mix the old weighted-tree results with the current final comparison table unless clearly labeling them as historical sensitivity experiments.

### Decision Tree Preprocessing

Numeric:

- median imputation.

Categorical:

- most-frequent imputation;
- one-hot encoding;
- `handle_unknown="ignore"`.

Binary:

- most-frequent imputation.

All preprocessing is fitted on the training split only.

### Decision Tree Threshold Selection

The Decision Tree was trained on the training split. The classification threshold was selected on the validation set by maximizing F1.

Selected threshold:

```text
0.2977
```

The selected threshold was then frozen and applied to the held-out test set.

### Decision Tree Results

| Split | AP / PR-AUC | F1 | Precision | Recall | Predicted injury cases | Actual injury cases |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.4631 | 0.5268 | 0.3994 | 0.7737 | 21,271 | 10,979 |
| Test | 0.4663 | 0.5269 | 0.4001 | 0.7715 | 26,466 | 13,724 |

Interpretation:

- The Decision Tree captures meaningful injury-related signal.
- The threshold is recall-heavy, identifying about 77% of actual injury incidents in the test set.
- Precision is around 40%, meaning many predicted injury cases are false positives.
- This tradeoff is expected because the positive class is less common and F1 thresholding favors identifying injury cases.

## 11. Method 4: TabNet Classification Comparison

### Why TabNet

TabNet is a recent deep learning architecture for tabular data. It was selected as the modern classification comparison because:

- it is designed specifically for tabular inputs;
- it can use categorical embeddings;
- it is substantially different from a classical Decision Tree;
- it can learn nonlinear feature interactions through a neural architecture;
- it provides feature importances, although these should be interpreted carefully.

Implementation:

- `pytorch_tabnet.tab_model.TabNetClassifier`.
- CPU PyTorch was used.

### TabNet Preprocessing

TabNet does not use the same one-hot preprocessing as the Decision Tree.

Numeric:

- converted to numeric;
- median-imputed from training data;
- passed as numeric.

Binary:

- checked as 0/1;
- mode-imputed from training data;
- passed as integer 0/1.

Categorical:

- converted to strings;
- missing values filled with `__MISSING__`;
- categories mapped to integer codes using training data only;
- unknown validation/test categories reserved for an unknown code;
- categorical feature indices and cardinalities passed to TabNet as `cat_idxs` and `cat_dims`.

Prepared feature counts:

| Feature group | Count |
|---|---:|
| Numeric | 5 |
| Categorical embeddings | 16 |
| Binary | 11 |
| Total | 32 |

No validation or test categories fell outside the training mappings.

### TabNet Training Setup

Model parameters:

| Parameter | Value |
|---|---:|
| `n_d` | 8 |
| `n_a` | 8 |
| `n_steps` | 3 |
| `gamma` | 1.3 |
| `lambda_sparse` | 0.001 |
| `cat_emb_dim` | 1 |
| seed | 13 |
| device | CPU |

Fit parameters:

| Parameter | Value |
|---|---:|
| `max_epochs` | 30 |
| `patience` | 5 |
| `batch_size` | 8192 |
| `virtual_batch_size` | 512 |
| `eval_metric` | AUC |

Training summary:

| Quantity | Value |
|---|---:|
| Training time | 121.86 seconds |
| Best validation AUC epoch | 27 |
| Best validation AUC | 0.7332 |
| Validation-selected threshold | 0.2696 |

### TabNet Results

| Split | AP / PR-AUC | F1 | Precision | Recall | Predicted injury cases | Actual injury cases |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.4772 | 0.5311 | 0.4023 | 0.7812 | 21,320 | 10,979 |
| Test | 0.4837 | 0.5333 | 0.4041 | 0.7840 | 26,626 | 13,724 |

Test confusion matrix:

|  | Predicted no injury | Predicted injury |
|---|---:|---:|
| Actual no injury | 20,571 | 15,867 |
| Actual injury | 2,965 | 10,759 |

Top TabNet feature importances:

| Feature | Importance |
|---|---:|
| `Number Vehicle Occupants` | 0.3357 |
| `Train Speed` | 0.1436 |
| `Driver In Vehicle` | 0.1348 |
| `Estimated Vehicle Speed` | 0.1209 |
| `Highway User Position` | 0.1092 |
| `Equipment Type` | 0.0520 |

Interpretation:

- TabNet slightly improves the classification metrics compared with the Decision Tree.
- The gain is modest, not dramatic.
- TabNet identified similar broad signal types: occupants, speed, driver status, vehicle speed, and highway-user position.
- The increased complexity may be justified if the goal is small performance improvement, but the Decision Tree remains easier to explain.

## 12. Classification Comparison

Main test comparison:

| Model | AP / PR-AUC | F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.4663 | 0.5269 | 0.4001 | 0.7715 | 0.2977 |
| TabNet | 0.4837 | 0.5333 | 0.4041 | 0.7840 | 0.2696 |

Absolute differences, TabNet minus Decision Tree:

| Metric | Difference |
|---|---:|
| AP / PR-AUC | +0.0174 |
| F1 | +0.0064 |
| Precision | +0.0040 |
| Recall | +0.0125 |

Conclusion:

TabNet outperformed the Decision Tree on all recorded test metrics, but the improvement was small. The final report should not overstate this. A careful conclusion is:

```text
TabNet provided a modest empirical improvement over the Decision Tree under the same split and validation-threshold rule, but the simpler tree remained competitive and easier to interpret.
```

## 13. Recommended Tables and Figures for Final Report

### Classification Table

Include:

- Model.
- AP / PR-AUC.
- F1.
- Precision.
- Recall.
- Threshold.

Optional if space:

- TP, FP, FN, TN.

### Classification Figure

Recommended:

- Grouped bar chart comparing Decision Tree and TabNet on AP, F1, precision, and recall.

Optional:

- Confusion matrix heatmaps for both models.

### Clustering Table

Include:

- Method.
- Sample size.
- Distance.
- Cluster count.
- Noise percent.
- Cluster sizes.
- Main score.

Important footnote:

- K-medoids silhouette is over all sampled rows.
- HDBSCAN silhouette is over non-noise clustered rows only.

### Clustering Figure

Recommended:

- Bar chart of group sizes:
  - K-medoids cluster 0 and cluster 1.
  - HDBSCAN cluster 0, cluster 1, and noise.

Optional:

- MDS 2D projection of the precomputed distance matrix colored by cluster assignment.

If using MDS, include this caveat:

```text
The 2D projection is only a visualization of the precomputed distance matrix. Clustering was performed on the original distance matrix, not on the 2D coordinates.
```

## 14. Limitations and Interpretation Boundaries

### Dataset and Outcome Limitations

- The classification target only indicates whether at least one injury occurred.
- It does not model number of injuries or injury severity.
- The model classifies outcomes among already-recorded incidents.
- It does not predict whether a crossing will experience an incident.
- Some features are recorded after or during the incident context, so the classification model should not be described as a pure prevention tool.

### Leakage Boundaries

Excluded fields:

- identifiers;
- exact dates and reporting year;
- injury and fatality totals;
- binary fatality outcome;
- vehicle damage cost.

This prevents direct or indirect outcome leakage.

### Sampling Limitations

- Clustering used a 2,000-row random sample for computational reasons.
- The sample outcome rates were similar to the full dataset, but this does not guarantee representativeness across all features, years, or rare categories.
- A stronger study would repeat clustering over multiple random samples.

### Clustering Limitations

- K-medoids had a low silhouette score, suggesting weak global cluster separation.
- HDBSCAN labeled 89% of points as noise, suggesting that only a small portion of the data formed dense profiles under this distance representation.
- HDBSCAN and K-medoids silhouettes are not directly identical because they are calculated over different point sets.
- Post-hoc injury/fatality rates by cluster are descriptive, not causal.

### Classification Limitations

- TabNet only modestly improved over the Decision Tree.
- Training was CPU-only and used a modest hyperparameter setup because of time constraints.
- Current final pipelines use train/validation/test splitting; the cross-validation rubric requirement should be addressed before final submission if possible.

## 15. Decision Log

| Decision | Final choice | Reason |
|---|---|---|
| Use full cleaned dataset for classification | Yes | Classification can train/test on full 250,806 rows without pairwise matrix limits |
| Use sample for clustering | Yes, 2,000 rows | Pairwise distance matrix is expensive for full dataset |
| Stratify clustering sample by outcome | No | Clustering is unsupervised; outcomes should not influence cluster formation |
| Use Form 55A outcome totals | Yes | High agreement with Form 57 and selected as outcome source |
| Use `injury_present` as classification target | Yes | Direct binary supervised task with 27.36% positive class |
| Exclude fatality/injury totals from predictors | Yes | Prevent outcome leakage |
| Exclude `Vehicle Damage Cost` | Yes | Post-incident consequence and not inflation-adjusted |
| Use K-medoids for classical clustering | Yes | Works with precomputed mixed-data distances and medoids are real incidents |
| Use HDBSCAN for clustering comparison | Yes | Density-based method can label noise instead of forcing all incidents into clusters |
| Use Decision Tree for classical classification | Yes | Interpretable and classical baseline |
| Use TabNet for recent classification | Yes | Recent deep tabular method and substantially different from Decision Tree |
| Select classification threshold on validation F1 | Yes | Avoid using test set for threshold selection |
| Use accuracy as main classification metric | No | Class imbalance makes accuracy misleading |

## 16. Report Writing Plan

Because the final report is limited to 4 pages excluding references, use this file as source material and compress heavily.

Suggested final report structure:

1. Introduction and task definitions.
   - One paragraph for dataset and goals.
   - One paragraph for classification and clustering questions.

2. Data preparation.
   - Briefly describe feature reduction, outcome creation, warning-device transformation, leakage exclusions.
   - Include one compact table of feature groups.

3. Methods.
   - Classification: Decision Tree and TabNet.
   - Clustering: K-medoids and HDBSCAN.
   - Mention fair comparison design.

4. Experimental design.
   - Classification split, threshold selection, metrics.
   - Clustering sample, distance matrix, metrics.
   - Include sampling justification.
   - Include cross-validation or clearly state validation design plus any CV/sensitivity performed.

5. Results.
   - Classification comparison table.
   - Clustering comparison table.
   - One or two plots.

6. Discussion and limitations.
   - TabNet modestly improves over Decision Tree.
   - HDBSCAN finds small dense profiles while K-medoids forces broad clusters.
   - Avoid causal claims.
   - Mention sample and CV limitations.

## 17. Citation Anchors to Verify

Use these as starting points for the reference section. Verify formatting before final submission.

Dataset:

- FRA Form 57 data downloads page: https://data.transportation.gov/stories/s/Form-57-Data-Downloads/i5dw-jvsi/

Decision Tree:

- scikit-learn Decision Tree documentation: https://scikit-learn.org/stable/modules/tree.html

K-medoids / PAM:

- Kaufman, L. and Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis.
- PAM documentation/source anchor: https://stat.ethz.ch/R-manual/R-patched/library/cluster/html/pam.html

HDBSCAN:

- Campello, R. J. G. B., Moulavi, D., and Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD.
- scikit-learn HDBSCAN documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

TabNet:

- Arik, S. O. and Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.
- AAAI paper page: https://ojs.aaai.org/index.php/AAAI/article/view/16826
- PyTorch TabNet usage documentation: https://tabnet.readthedocs.io/en/v4.4.2/guides/basic_usage.html

## 18. Final Cautions Before Submission

- Do not mix old weighted Decision Tree results from the draft with the current staged Decision Tree artifact unless clearly labeled.
- Do not claim HDBSCAN's silhouette is directly comparable to K-medoids' silhouette without explaining that HDBSCAN's value is calculated only on clustered non-noise points.
- Do not claim clustering results prove injury risk differences. Outcomes were post-hoc descriptions only.
- Do not call the classification model a future incident predictor. It classifies injury occurrence among recorded incidents.
- Address the rubric's cross-validation requirement before final submission if possible.
- Keep the final report concise. This file is intentionally detailed; the report should be selective.
