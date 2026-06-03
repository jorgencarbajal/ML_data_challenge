# Report: What are we answering?

Clustering question to consider: 

- Does the incident dataset contain stable and interpretable natural groupings, and does a density-based method confirm or challenge the groups produced by a partitioning method?
- **Main question:** Do highway–rail crossing incidents form stable and interpretable groups based on incident characteristics alone?
- **K-medoids:** When every incident is forced into one of a fixed number of clusters, what incident profiles emerge? (specific question answered by kmedoids)
- Are the **K-medoids** profiles stable across repeated representative samples?
- **HDBSCAN:** If incidents are not forced into clusters, does HDBSCAN discover similar dense incident profiles?
- How much of the dataset does **HDBSCAN** consider clusterable versus ambiguous or noisy? (specific to the value of HDBSCAN)
- Comparison Question: Do **K-medoids** and **HDBSCAN** identify the same broad incident profiles?
- Does **HDBSCAN** provide a better description of weak cluster structure than **K-medoids**?
- Evaluation: Which method produces more stable and interpretable clustering under repeated sampling?

Classification questions to consider:

- Does the incident dataset contain enough predictive information to identify injury-related incidents using recorded incident characteristics, while excluding direct outcome leakage?
- **Main question:** Can highway–rail crossing incident characteristics be used to classify whether at least one reported injury occurred?
- **Target variable:** `injury_present`
- **Leakage control:** Can injury occurrence be predicted using only incident-context predictors, without including injury totals, fatality outcomes, identifiers, or other post-outcome severity information?
- **Decision Tree:** What interpretable decision patterns emerge when a classical tree-based classifier is used to predict `injury_present`?
- Does the **Decision Tree** perform meaningfully better than the majority-class / no-skill baseline?
- Does the **Decision Tree** remain interpretable without becoming overly complex or overfitting the training data?
- **TabNet:** Does a recent attention-based tabular neural-network model improve injury classification performance over the Decision Tree baseline?
- Does **TabNet** identify useful feature relationships that a simpler Decision Tree may fail to capture?
- Comparison Question: Do **Decision Tree** and **TabNet** identify similar incident characteristics as important for distinguishing injury-present from non-injury incidents?
- Does the increased complexity of **TabNet** provide enough predictive improvement to justify reduced interpretability compared with the **Decision Tree**?
- Class Imbalance Question: Since injury cases are less common than non-injury cases, does accuracy hide weaknesses in identifying the injury-positive class?
- Evaluation: Which method produces better generalization on unseen incidents using precision, recall, F1 score, PR-AUC / average precision, and the confusion matrix?
- Evaluation: How do training and testing results compare for each method, and does either model show signs of overfitting?
- Interpretation Question: Which incident characteristics appear most influential in injury classification, and are those findings consistent across both methods?

## Setup

I started out by setting up the project folder, the only help in this was the syntax for adding a vcs, `git remote add origin http...`. Other than that setup and download was primarily on my own.

Another part of the setup involved setting up the Latex template for the report. It would be very time consuming to sit and attempt to understand what every file in the template folder from the provided website was doing. Instead I had chat summarize what each was for and what was necessary to keep and discard.

## Structure/approach

After setting the environment we now had to plan out the structure of the project. This involved setting up a roadmap that will correctly hit all the targets I need. I initially missed the idea of setting up the questions that would lead to the goal of the project. What is the purpose. I am new to the world of machine learning so didn't really have that thinking cap on. I will continue to attempt to develop that critical thinking skill. 

Aside from the questions I had some reasoning backwards in thinking I should pick my task before sifting threw the features and building an understanding of the features that I was working with. It obviously now makes more sense to have an understanding of the data I am working with before deciding what types of tasks work best.

## Task selection / feature reduction

I did have chat heavily help me in looking through the features and pinpointing which would work best for the clustering and association ideas I had in mind. Again... deciding before understanding the data.

I next decided it was best to look through the filtered 68 features and pick a bucket that works for clustering and another that works for association. I thought to pick and move one but was again dumbfounded in missing the idea of an *exploratory check*. The idea here is that we only want to include features that actually bring value. If the feature is entirely empty with no data, it makes sense to remove it use.

Chat helped me pick the features that best work for each of the two tasks we chose. I initially missed a few and added others. The best approach to have done this completely alone was to go through all 154, refine, then again further refine into cluster/association buckets... ain't nobody got time for that.

I submitted my notes and so far the features.md file to have an ultimate best idea for what tasks I should focus on. After initially thinking that KMeans was going to be it, chat recommended KMedoids. Apparently KMeans works better for continuos data and since this data is not continuous chat recommended medoids. I wouldnt have noticed that but now I know. I will have to ensure I further explore and explain why Kmedoids is better.

Ok I initially thought that I neede to first find the more recent methods after settling on KMedoids and Apriori. But I was again deviated in my thought process that it was better to do that initial pass in the EDA process to see what the true structure of the data is.

Pass1 data refining: reduce to 51 columns, remove duplicates, ensure that the deaths in the data set will be meaningful enough to produce reasonable results. After running the audit function inside the pass1 file, we have determined that the values from 55A and 57 agree over 98% of the time, no need for additional changes. Also 8.3% of the data has a more than 0 fatalities and 23.4% of the data has injuries, this shows that clustering and association is possible, right?

Alot of reducing later... looks like we are at the end of the initial pass. Chat was doing the heavy lifting here. So many things and syntax that wouldve took weeks without chat. The goal is to practice alot of this over the summer.

In the end we have cleaned up missing values, dealt with outliers, categorized categories, etc. This initial `pass_1.csv` is a baseline that we can then specifically refine for clustering and association modeling. I plan on running some initial basic clustering and association mining to see what type of useful patterns emerge from what is in the nice clean set before further refining for task specific methods.

### Chat Report

#### Initial Data Refinement and Version 1 Cleaned Dataset

##### Purpose

The purpose of this stage was to reduce the original FRA Highway-Rail Grade Crossing Incident dataset into a smaller, cleaner working dataset before creating model-specific inputs.

The raw dataset contained approximately **250,806 incidents** and **154 features**. The preprocessing goal was to preserve useful incident-context information, create usable injury/fatality outcomes, remove or transform problematic fields, and clearly separate modeling features from audit-only or outcome-related fields.

---

##### 1. Initial Feature Reduction

The original dataset was first reduced from **154 features** to **51 candidate source fields**. The retained fields were selected because they were potentially useful for:

* incident-context analysis;
* later modeling tasks;
* outcome construction;
* auditing and validation.

The candidate fields included timing information, highway-user and rail-equipment characteristics, warning-device fields, environmental conditions, driver/action fields, injury and fatality totals, and vehicle damage cost.

After missingness and reporting-era audits, four fields were removed from the Version 1 source-field list:

| Removed Feature                | Reason                                                             |
| ------------------------------ | ------------------------------------------------------------------ |
| `Crossing Warning Explanation` | Approximately 99.62% missing                                       |
| `Roadway Condition`            | Approximately 87.28% missing and mostly unavailable before 2012    |
| `User Age`                     | Approximately 72.04% missing and unavailable in many earlier years |
| `Signaled Crossing Warning`    | Approximately 50.73% missing                                       |

After these removals, Version 1 retained **47 original/source fields** before derived variables were added.

---

##### 2. Duplicate `Report Key` Audit

`Report Key` was retained temporarily as an audit identifier to check for duplicated incident records.

The audit found:

| Result                                              | Count |
| --------------------------------------------------- | ----: |
| Rows associated with duplicated `Report Key` values |    47 |
| Distinct duplicated-key groups                      |    23 |
| Groups identical across retained fields             |    14 |
| Groups with conflicting retained values             |     9 |

Because some duplicated keys contained conflicting incident information, the duplicates could not be treated as exact copies in every case.

For Version 1, the simplifying decision was to keep the first occurrence of each duplicated `Report Key` and remove later occurrences. This affected very few incidents relative to the full dataset and was treated as a practical cleaning choice rather than proof that every duplicate represented the same record.

`Report Key` remains audit-only and should not be used as a modeling feature.

---

##### 3. Outcome Audit and Binary Outcome Creation

Before creating outcome variables, the total injury and fatality fields from Form 55A and Form 57 were compared.

The primary candidate outcome-source fields were:

```text
Total Killed Form 55A
Total Injured Form 55A
```

The comparison fields were:

```text
Total Killed Form 57
Total Injured Form 57
```

All four fields were fully populated. Agreement between the two form sources was also very high:

| Comparison                                       | Agreement |
| ------------------------------------------------ | --------: |
| Form 55A killed total vs. Form 57 killed total   |    99.59% |
| Form 55A injured total vs. Form 57 injured total |    98.25% |

Based on this audit, the Form 55A totals were selected as the Version 1 outcome sources.

Two binary outcome variables were created:

```text
fatality_present = 1 if Total Killed Form 55A > 0, otherwise 0
injury_present   = 1 if Total Injured Form 55A > 0, otherwise 0
```

Outcome frequencies were:

| Outcome            | Positive Incidents | Percent |
| ------------------ | -----------------: | ------: |
| `fatality_present` |             20,711 |   8.26% |
| `injury_present`   |             68,617 |  27.36% |

A combined severity variable was considered but not created. Injuries and fatalities were kept separate because they represent different outcomes and are easier to interpret independently.

The original injury and fatality totals were retained in the cleaned working dataset for validation and interpretation, but they must be excluded from model predictors to prevent outcome leakage.

---

##### 4. Missingness and Reporting-Era Decisions

Missingness was evaluated both overall and by year. This was important because some features were not simply missing at random; they were unavailable or inconsistently recorded during earlier reporting periods.

The main decision was to build a consistent all-years Version 1 dataset rather than restrict the project to recent years just to keep a few incomplete variables.

Fields such as `Roadway Condition` and `User Age` may still be useful in a later modern-era supporting analysis, but they were excluded from the broad Version 1 dataset.

Several retained fields contain meaningful `Unknown` or missing categories, including:

| Feature                       | Notable Uncertainty                            |
| ----------------------------- | ---------------------------------------------- |
| `Crossing Illuminated`        | Approximately 18.05% `Unknown`, 3.61% missing  |
| `Warning Connected To Signal` | Approximately 14.28% `Unknown`, 10.59% missing |
| `Driver Passed Vehicle`       | Approximately 7.85% `Unknown`, 2.28% missing   |

These fields were retained because they still describe useful incident context. Unknown or missing responses were preserved rather than being forced into valid categories such as `Yes` or `No`.

Rare categorical values were also retained during Version 1 preprocessing rather than being grouped prematurely.

---

##### 5. Warning-Device Transformation

The dataset originally contained twelve fields:

```text
Crossing Warning Expanded 1
...
Crossing Warning Expanded 12
```

These fields represented multiple warning devices that could be selected for the same incident. Their column positions did not represent separate ordered meanings, so treating them as twelve ordinary categorical features would have been misleading.

They were replaced with twelve binary indicator variables:

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

Each indicator equals `1` when that warning-device type appears anywhere among the original twelve warning fields for an incident, and `0` otherwise.

After this transformation, the original twelve expanded warning fields were removed. This preserved the warning-device information in a clearer and more usable format.

---

##### 6. Time Variable Transformation

The original dataset included detailed time fields:

```text
Date
Month
Hour
Time
```

These were transformed into broader, more interpretable variables:

| Derived Feature | Purpose                                       |
| --------------- | --------------------------------------------- |
| `year`          | Retained for auditing reporting-era effects   |
| `season`        | Provides a broader environmental/time context |
| `time_of_day`   | Provides interpretable timing categories      |

The derived categories were:

```text
season: Winter, Spring, Summer, Fall
time_of_day: Night, Morning, Afternoon, Evening
```

After transformation:

* `Month`, `Hour`, and `Time` were removed;
* `Date` was retained for auditing only;
* `year` was retained for reporting-era checks only;
* `season` and `time_of_day` were retained as usable context features.

`Date` and `year` should be excluded from model inputs because they could cause models to learn reporting periods rather than incident characteristics.

---

##### 7. Numeric Feature Audit and Cleaning

The numeric audit focused on:

```text
Train Speed
Estimated Vehicle Speed
Number Vehicle Occupants
Number of Cars
Temperature
Vehicle Damage Cost
```

The audit examined missingness, formatting problems, minimum and maximum values, extreme-value frequencies, and possible invalid measurements.

###### Vehicle Damage Cost

`Vehicle Damage Cost` initially appeared to contain many non-numeric values because numerous entries included commas or dollar-sign formatting.

The field was cleaned by removing formatting characters and converting the values to numeric form. Approximately **139,307 values** required formatting cleanup.

Although successfully cleaned, `Vehicle Damage Cost` was excluded from Version 1 modeling because:

* it is partly an outcome of an incident rather than a pre-incident condition;
* dollar amounts across many decades are not directly comparable without inflation adjustment.

It was retained only for descriptive or future supporting analysis.

###### Implausible Numeric Measurements

Only clearly suspicious measurements were replaced with missing values:

| Feature                   | Cleaning Rule              | Values Replaced |
| ------------------------- | -------------------------- | --------------: |
| `Train Speed`             | Greater than 110 mph       |               3 |
| `Estimated Vehicle Speed` | Greater than 120 mph       |               7 |
| `Number of Cars`          | Greater than 300           |               9 |
| `Temperature`             | Below -80°F or above 130°F |              34 |

In total, **53 suspicious numeric values** were replaced with missing values.

No incident rows were removed during numeric cleaning. This preserved the usable context information from records that contained only one questionable numeric measurement.

`Number Vehicle Occupants` was retained unchanged because unusually large values may still be valid for buses or other multi-passenger highway users.

---

##### 8. Final Feature-Use Decisions

The cleaned dataset contains several categories of fields with different intended uses.

###### Audit-Only Fields

These fields remain in the working dataset but should not be used for modeling:

```text
Report Key
Date
year
```

###### Context Features Retained for Later Modeling

The usable context feature set includes:

* highway-user characteristics;
* rail-equipment and track characteristics;
* visibility, weather, and obstruction fields;
* highway-user action and driver-context fields;
* crossing illumination and signal-connection fields;
* `season`;
* `time_of_day`;
* binary warning-device indicators;
* cleaned numeric context fields such as train speed, estimated vehicle speed, number of occupants, number of cars, and temperature.

###### Outcome-Related Fields

The original injury and fatality totals and component fields were retained only for outcome construction, validation, or interpretation. They must not be included as predictors.

The binary fields:

```text
fatality_present
injury_present
```

can be used as outcome variables where appropriate, but they should not be included as ordinary incident-context inputs.

###### Excluded Modeling Field

```text
Vehicle Damage Cost
```

was retained in the cleaned dataset but excluded from Version 1 modeling because it is outcome-like and not adjusted for inflation.

---

##### 9. Resulting Version 1 Dataset

The preprocessing pipeline produced `v1.csv`, a cleaned working analysis dataset rather than a final model-ready dataset.

Its structure is:

| Stage                                                                                | Columns |
| ------------------------------------------------------------------------------------ | ------: |
| Original raw dataset                                                                 |     154 |
| Initial retained source fields                                                       |      51 |
| Retained source fields after missingness removals                                    |      47 |
| After adding `fatality_present` and `injury_present`                                 |      49 |
| After replacing twelve warning-device fields with twelve indicators                  |      49 |
| After replacing `Month`, `Hour`, and `Time` with `year`, `season`, and `time_of_day` |      49 |

Therefore, Version 1 contains **49 columns** and approximately **250,000 incident records**, with the final row count depending on application of the duplicate-removal decision.

---

##### Final Preprocessing Conclusion

This stage created a defensible cleaned working dataset from the original FRA incident data. The main preprocessing decisions were:

* reduce the raw feature space to useful context, audit, and outcome fields;
* remove variables with severe missingness or strong reporting-era limitations;
* audit and simplify duplicated incident identifiers;
* verify Form 55A outcome totals and derive binary injury/fatality indicators;
* convert warning-device slots into interpretable binary indicators;
* summarize detailed time fields into broader categories;
* clean formatting problems and replace only clearly suspicious numeric measurements;
* preserve audit and outcome fields in the working dataset while explicitly excluding them from modeling inputs.

The resulting `v1.csv` is ready to be used as the foundation for constructing task-specific model inputs.


## Baseline methods and testing

The goal here is to run a quick cluster and association rule mining on the v1 dataset to see how viable it is, lets try not to waste to much time.

### K-medoids

Partitioning Around Medoids (PAM) operates onf nxn dissimilarity matrix. Expensive and may require sampling based medoid approach. Sampling based Medoid approach CLARA??

Final setup:
- Fitting features: $32$ features; `time_of_day` excluded from distance calculation.
- Algorithm: PAM / K-medoids with Gower-style mixed-date distance.
- Sample: $2000$ incidents, seed `42`.
- Selected result: `k=2`, silhouette score $0.1657$, total cost $547.375$.

Interpretation:
- Cluster 0: 1563 incidents - more often moving over the crossing, did not stop, driver in vehicle, fewer gates/flashers/audible warnings.
- Cluster 1: 437 incidents - more often stopped or stalled on the crossing, driver not in vehicle, train struck the highway user, and crossings with gates/flashers/audible warnings.
- The outcome differences are post-hoc only: injury presence was lower in Cluster 1, while fatality rates were similar. Do not interpret the clusters as injury/fatality risk groups.

#### K-Medoids / PAM Version 1 Baseline Pipeline Notes

##### 1. Purpose of this stage

The purpose of this stage was to build and diagnose a first clustering baseline using **K-medoids**, specifically the **Partitioning Around Medoids (PAM)** algorithm.

The clustering goal was not to predict injuries or fatalities. Instead, the goal was to group highway–rail crossing incidents according to their **incident characteristics**, such as:

* Speed-related variables.
* Equipment and highway-user characteristics.
* Track, weather, visibility, and crossing conditions.
* Warning-device indicators.

For this reason, outcome variables such as `fatality_present` and `injury_present` were intentionally excluded from the clustering feature space. They were retained only for post-cluster interpretation.

---

##### 2. Why K-medoids was used

K-medoids is a partitioning clustering algorithm that is similar to K-means, but its cluster centers are actual observations from the dataset, called **medoids**.

This is useful for this project because:

* A medoid corresponds to a real recorded incident, which makes cluster representatives easier to interpret.
* K-medoids can use a precomputed dissimilarity matrix.
* The dataset contains mixed feature types: numeric, categorical, and binary features.
* K-medoids does not require all features to behave naturally under Euclidean distance, unlike ordinary K-means.

The implemented PAM step eventually used the `kmedoids` Python package rather than relying on a hand-written implementation. A manual PAM version was first tested briefly and produced the same result as the library version for `k=3`, which confirmed that the library call was behaving as expected.

Important course requirement note: K-medoids is a classical clustering method, but its use as the official classical baseline still depends on whether the instructor accepts it under the wording requiring a method seen in class or in the textbook. If necessary, K-means may later need to become the official classical baseline.

---

##### 3. Original clustering feature space

The original clustering input contained **33 incident-characteristic features** divided into three types.

###### Numeric features

These remained numeric because their numerical distances are meaningful for clustering:

* `Train Speed`
* `Estimated Vehicle Speed`
* `Number Vehicle Occupants`
* `Number of Cars`
* `Temperature`

###### Categorical features

These describe incident context and conditions:

* `season`
* `time_of_day`
* `Highway User`
* `Highway User Position`
* `Equipment Involved`
* `Equipment Struck`
* `Equipment Type`
* `Track Type`
* `Warning Connected To Signal`
* `Crossing Illuminated`
* `Visibility`
* `Weather Condition`
* `View Obstruction`
* `Highway User Action`
* `Driver Passed Vehicle`
* `Driver In Vehicle`

###### Binary warning-device features

These were previously derived from the raw warning-device fields and stored as `0/1` indicators:

* `has_gate`
* `has_cantilever_fls`
* `has_standard_fls`
* `has_wig_wags`
* `has_highway_traffic_signals`
* `has_audible`
* `has_crossbucks`
* `has_stop_signs`
* `has_watchman`
* `has_flagged_by_crew`
* `has_other_warning`
* `has_no_warning_device`

###### Profile-only variables

The following columns were kept available for tracking and post-cluster interpretation, but were not used to form the clusters:

* `Report Key`
* `year`
* `fatality_present`
* `injury_present`

This separation prevented fatality or injury information from shaping the clusters.

---

##### 4. Data loading and validation

The first function, `load_and_validate_data()`, loaded the refined dataset from `data/v1.csv`.

It performed the following checks:

1. Confirmed that every expected clustering feature and profile-only column existed.
2. Checked for duplicate `Report Key` values.
3. Stopped the pipeline if duplicate incidents remained.
4. Reduced the dataset to only the relevant clustering and profiling columns.

The loaded refined dataset contained:

* **250,806 incidents**
* **33 originally prepared clustering features**
* Four profile-only columns

The absence of duplicate-report errors confirmed that the working dataset contained one row per retained incident.

---

##### 5. K-medoids data preparation

The function `prepare_kmedoids_data()` created two aligned data tables:

* `model_df`: features that may be used in clustering.
* `profile_df`: identifiers and outcome variables used only after clustering.

It also created an audit table recording how missing values were handled.

###### Numeric feature preparation

Numeric features were converted with:

```python
pd.to_numeric(errors="coerce")
```

This ensures that malformed or non-numeric values become missing values rather than silently interfering with distance calculations.

Missing numeric values were then filled with the median of the feature. Median imputation was chosen as a simple Version 1 baseline because it is less sensitive to extreme values than mean imputation.

Missingness observed in numeric features:

| Feature                    | Missing Count | Missing Percent |
| -------------------------- | ------------: | --------------: |
| `Train Speed`              |         2,564 |           1.02% |
| `Estimated Vehicle Speed`  |        28,072 |          11.19% |
| `Number Vehicle Occupants` |           225 |           0.09% |
| `Number of Cars`           |            68 |           0.03% |
| `Temperature`              |            35 |           0.01% |

The major numeric missingness issue is `Estimated Vehicle Speed`, which has about 11.2% missing values. For the Version 1 baseline, these values were median-imputed, but this should remain a limitation to document or later test.

###### Categorical feature preparation

Categorical missing values were replaced with the explicit category:

```python
"Missing"
```

This prevented missing values from breaking the distance calculation while preserving the fact that the value was unavailable.

Most categorical missingness was small, but a few variables require attention:

| Feature                       | Missing Count | Missing Percent |
| ----------------------------- | ------------: | --------------: |
| `Warning Connected To Signal` |        26,570 |          10.59% |
| `Crossing Illuminated`        |         9,046 |           3.61% |
| `Driver In Vehicle`           |         6,099 |           2.43% |
| `Driver Passed Vehicle`       |         5,728 |           2.28% |
| `Highway User Action`         |         3,115 |           1.24% |

Representing missingness as a category is acceptable for an initial diagnostic run, but there is a risk that clusters could partially reflect missing-data patterns rather than true incident patterns.

###### Binary warning-device preparation

The binary warning-device indicators were checked to ensure that they contained only valid binary values:

```python
0 or 1
```

The pipeline was designed to raise an error if a warning-device feature contained missing values or invalid values. This was important because replacing a missing binary indicator with zero would incorrectly assume that the warning device was absent.

No binary-feature validation error occurred, so the derived warning indicators were usable for this baseline.

---

##### 6. Why a pilot sample was necessary

PAM requires pairwise dissimilarities among incidents. A full pairwise distance matrix for approximately 250,000 incidents would be computationally and memory intensive.

Therefore, a reproducible pilot sample was created using:

* Sample size: **2,000 incidents**
* Random seed: **42**

The same sampled row indices were applied to both:

* The clustering feature table.
* The profile-only table.

This maintained correct alignment between cluster assignments and incident identifiers/outcomes.

The sample was not stratified using injury or fatality variables because clustering is unsupervised. Allowing outcome variables to influence sampling would weaken the claim that clusters were formed only from incident characteristics.

The sampled records had the following outcome rates, retained only for later description:

| Outcome            | Sample Rate |
| ------------------ | ----------: |
| `fatality_present` |       7.90% |
| `injury_present`   |      25.35% |

---

##### 7. Mixed-data distance calculation

Because the clustering features contain numeric, categorical, and binary variables, ordinary Euclidean distance was not appropriate for this first baseline.

A custom **Gower-style dissimilarity matrix** was calculated. This allowed each type of feature to contribute meaningfully to the incident-to-incident distance.

###### Numeric feature distance

For numeric features, the distance between two incidents was calculated as the absolute difference divided by the feature range:

```text
absolute difference / full-dataset feature range
```

This places numeric-feature differences on a scale from 0 to 1 so that speed or temperature cannot dominate solely because it is measured on a larger raw scale.

The numeric ranges were taken from the full prepared dataset rather than only the 2,000-record sample. This makes the scaling more consistent if a different sample is later tested.

###### Categorical feature distance

For categorical features:

* Same category = distance contribution of `0`
* Different category = distance contribution of `1`

For example, two incidents both occurring in `Day` visibility contribute no visibility-related dissimilarity, while an incident occurring in `Day` and another occurring in `Dark` contribute a visibility dissimilarity of 1.

###### Binary warning-device distance

The warning-device features were treated as **asymmetric binary features**.

This means:

* Two records both having a gate count as similarity.
* One record having a gate while the other does not counts as a difference.
* Two records both lacking a gate do not automatically become more similar solely because both are zeros.

This was selected because shared absence of a specific warning device is less informative than shared presence of that device.

###### Distance matrix output

For the final revised feature setup, the 2,000 by 2,000 matrix required approximately **15.26 MB** of memory.

Final revised distance summary:

| Quantity                  |  Value |
| ------------------------- | -----: |
| Minimum pairwise distance | 0.0017 |
| Median pairwise distance  | 0.4168 |
| Maximum pairwise distance | 0.8440 |

The resulting distances varied across a usable range and passed validation checks for symmetry and expected values between 0 and 1.

---

##### 8. Initial PAM baseline using all 33 features

The initial PAM experiment used all 33 prepared clustering features, including `time_of_day`.

PAM was evaluated for values of (k) from 2 through 8. Two main evaluation outputs were recorded:

* **Silhouette score:** measures how clearly incidents belong to their assigned cluster rather than another cluster. Higher is better.
* **Total PAM cost:** sum of each incident’s distance to its medoid. Lower is better, but this naturally decreases as more clusters are added.

Initial all-feature results:

| (k) | Silhouette Score | Total Cost |
| --: | ---------------: | ---------: |
|   2 |           0.0964 |   574.1521 |
|   3 |           0.0953 |   543.5626 |
|   4 |           0.0741 |   522.0764 |
|   5 |           0.0694 |   505.6869 |
|   6 |           0.0728 |   490.6385 |
|   7 |           0.0716 |   480.2846 |
|   8 |           0.0630 |   470.5020 |

The silhouette scores showed that (k=2) and (k=3) were the only reasonable candidates under the original feature design, but both scores were low. This indicated weak cluster separation and suggested that the input representation needed investigation.

The total-cost curve decreased steadily without a strong elbow. Since cost naturally decreases when more clusters are added, it was not used alone to select (k).

---

##### 9. Interpretation of initial cluster profiles

###### Initial (k=2) profile

The two-cluster result was primarily separated by light/time conditions:

* One cluster was heavily associated with `Day`, `Afternoon`, and `Morning`.
* The other cluster was heavily associated with `Dark`, `Evening`, and `Night`.

This indicated that `Visibility` and `time_of_day` may be jointly expressing similar information and disproportionately shaping the clustering.

###### Initial (k=3) profile

The three-cluster solution revealed a more specific third group:

* More incidents where the highway user was stopped or stalled on the crossing.
* More incidents with no driver in the vehicle.
* More gates, audible devices, and standard flashing lights.
* More cases where rail equipment struck the highway user.

This third cluster was more substantively interesting, but it also appeared strongly influenced by the warning-device variables. This justified running sensitivity tests before selecting a final Version 1 feature setup.

---

##### 10. Sensitivity testing

Sensitivity testing was performed to determine whether the clustering result was being overly influenced by potentially redundant or dominant features.

The test fixed (k=3) and compared four feature configurations:

1. Baseline with all features.
2. Removal of all warning-device indicator variables.
3. Removal of `time_of_day`.
4. Removal of `Visibility`.

Sensitivity results:

| Configuration                | Silhouette Score | Cluster Sizes   |
| ---------------------------- | ---------------: | --------------- |
| Baseline: all features       |           0.0953 | 952, 712, 336   |
| No warning-device indicators |           0.1130 | 835, 726, 439   |
| Remove `time_of_day`         |       **0.1444** | 1,182, 432, 386 |
| Remove `Visibility`          |           0.0545 | 921, 375, 704   |

###### Interpretation of sensitivity results

Removing `time_of_day` produced the largest improvement in silhouette score. This suggested that `time_of_day` added overlapping or noisy information when `Visibility` was already present.

Removing `Visibility` sharply worsened the cluster separation. This suggested that visibility contains important information about incident structure and should remain in the clustering feature space.

Removing warning-device indicators improved silhouette compared with the initial baseline, so these features may add some complexity or noise. However, profiling showed that keeping them preserved a meaningful cluster pattern involving incidents where highway users went around gates at crossings with active warning devices. Therefore, the warning-device indicators were retained for the Version 1 baseline.

An important measurement note is that total PAM cost was not used to compare different sensitivity configurations because removing features changes the distance space itself. Silhouette score and interpretation were more suitable for this diagnostic comparison.

---

##### 11. Revised feature decision for K-medoids

Based on the sensitivity testing, the final Version 1 PAM distance calculation used:

* All five numeric features.
* All twelve binary warning-device indicators.
* The categorical features except `time_of_day`.

Therefore:

* Original prepared feature list: **33 features**
* Final features used to calculate PAM distance: **32 features**

`time_of_day` remained available for post-cluster description, but it was excluded from the actual distance calculation and therefore did not form the final clusters.

The final feature decision was:

```python
KMEDOIDS_CATEGORICAL_FEATURES = [
    feature for feature in CATEGORICAL_FEATURES
    if feature != "time_of_day"
]
```

---

##### 12. Re-evaluation of (k) after removing `time_of_day`

After revising the feature space, PAM was again evaluated for (k=2) through (k=8).

Revised results:

| (k) | Silhouette Score | Total Cost | Smallest Cluster | Largest Cluster |
| --: | ---------------: | ---------: | ---------------: | --------------: |
|   2 |       **0.1657** |   547.3750 |              437 |           1,563 |
|   3 |           0.1444 |   518.6757 |              386 |           1,182 |
|   4 |           0.1007 |   496.0873 |              356 |             856 |
|   5 |           0.0925 |   479.2149 |              290 |             682 |
|   6 |           0.0956 |   465.6280 |              187 |             680 |
|   7 |           0.0724 |   454.5081 |              182 |             431 |
|   8 |           0.0750 |   445.3983 |              131 |             412 |

The revised feature setup improved the best silhouette score from approximately `0.096` to approximately `0.166`.

The highest silhouette score occurred at (k=2), so (k=2) was selected as the **provisional Version 1 PAM baseline result**.

The silhouette value is still not very high. Therefore, the correct interpretation is not that the data contains two perfectly separated natural groups. Instead, the correct interpretation is that the revised data contains a **dominant two-cluster partition with substantial overlap**.

---

##### 13. Final Version 1 PAM cluster profile

The final PAM model used:

| Component               | Final Version 1 Decision                    |
| ----------------------- | ------------------------------------------- |
| Algorithm               | PAM / K-medoids                             |
| Python implementation   | `kmedoids.pam()`                            |
| Input sample size       | 2,000 incidents                             |
| Random seed             | 42                                          |
| Distance type           | Custom Gower-style mixed-data dissimilarity |
| Fitted feature count    | 32                                          |
| Excluded fitted feature | `time_of_day`                               |
| Selected (k)            | 2                                           |
| Silhouette score        | 0.1657                                      |
| Total PAM cost          | 547.3750                                    |
| Medoid sample indices   | 816 and 1745                                |

###### Final cluster sizes

|   Cluster | Number of Sampled Incidents | Percent of Sample |
| --------: | --------------------------: | ----------------: |
| Cluster 0 |                       1,563 |            78.15% |
| Cluster 1 |                         437 |            21.85% |

###### Numeric feature medians

| Feature                    | Overall Median | Cluster 0 Median | Cluster 1 Median |
| -------------------------- | -------------: | ---------------: | ---------------: |
| `Train Speed`              |           22.0 |             20.0 |             30.0 |
| `Estimated Vehicle Speed`  |            5.0 |              8.0 |              0.0 |
| `Number Vehicle Occupants` |            1.0 |              1.0 |              1.0 |
| `Number of Cars`           |           26.0 |             19.0 |             55.0 |
| `Temperature`              |           60.0 |             57.0 |             61.0 |

Cluster 1 tended to involve higher train speeds, more rail cars, and vehicle speeds recorded at zero, which is consistent with incidents involving stopped or stalled highway users.

###### Cluster 0 interpretation

Cluster 0 contained most of the sampled incidents and was characterized by:

* Highway user did not stop: 67.95%, compared with 53.55% overall.
* Highway user moving over the crossing: 86.24%, compared with 72.80% overall.
* Driver in vehicle: 95.52%, compared with 83.90% overall.
* Day visibility was somewhat more common.
* Gates, flashing lights, and audible warning devices were much less common than overall.

Key warning-device differences:

| Device Feature     | Cluster 0 | Overall |    Difference |
| ------------------ | --------: | ------: | ------------: |
| `has_gate`         |     9.98% |  21.45% | -11.47 points |
| `has_standard_fls` |    24.31% |  34.55% | -10.24 points |
| `has_audible`      |    10.62% |  20.00% |  -9.38 points |
| `has_crossbucks`   |    69.10% |  62.15% |  +6.95 points |

Working interpretation: **incidents involving highway users moving through crossings, often without stopping, with fewer active warning devices and more passive crossbuck-style warnings.**

###### Cluster 1 interpretation

Cluster 1 was smaller and characterized by:

* Highway user stopped on crossing: 63.84%, compared with 25.55% overall.
* Driver not in vehicle: 49.66%, compared with 13.65% overall.
* Highway user stopped or stalled on crossing was highly overrepresented.
* Rail equipment struck the highway user: 92.22%, compared with 75.95% overall.
* Dark visibility was more common.
* Main track incidents were more common.
* Gates, flashing lights, and audible warning devices were strongly overrepresented.

Key warning-device differences:

| Device Feature     | Cluster 1 | Overall |    Difference |
| ------------------ | --------: | ------: | ------------: |
| `has_gate`         |    62.47% |  21.45% | +41.02 points |
| `has_standard_fls` |    71.17% |  34.55% | +36.62 points |
| `has_audible`      |    53.55% |  20.00% | +33.55 points |
| `has_crossbucks`   |    37.30% |  62.15% | -24.85 points |

Working interpretation: **incidents involving highway users stopped or stalled at crossings with more active warning infrastructure, where the train more commonly struck the highway user.**

---

##### 14. Outcome variables as post-cluster descriptions

Outcome variables were never used in fitting the clusters. They were inspected only afterward to describe whether the resulting incident-characteristic groups differed in observed outcomes.

|   Cluster | Fatality Present | Injury Present |
| --------: | ---------------: | -------------: |
| Cluster 0 |            7.74% |         28.34% |
| Cluster 1 |            8.47% |         14.65% |

The fatality rates were relatively similar across the two clusters. The injury rate was lower in Cluster 1.

This should not be interpreted as proof that the Cluster 1 incident pattern reduces injury risk. Clustering is descriptive, and these outcome differences may be influenced by many factors not isolated in this analysis.

The correct conclusion is simply that the two incident-characteristic clusters had different observed injury proportions in this pilot sample.

---

##### 15. What was accomplished in this Version 1 baseline

This K-medoids baseline stage accomplished the following:

1. Created a clean separation between clustering features and outcome/profile-only variables.
2. Validated required columns and duplicate handling.
3. Prepared mixed-type clustering inputs with explicit handling for numeric, categorical, and binary variables.
4. Created a reproducible 2,000-incident pilot sample.
5. Constructed a Gower-style distance matrix suitable for mixed incident data.
6. Implemented PAM using a library implementation that accepts precomputed distances.
7. Evaluated multiple values of (k) using silhouette score and total PAM cost.
8. Profiled candidate cluster solutions to determine what incident patterns they represented.
9. Performed feature sensitivity testing to investigate possible redundancy or feature dominance.
10. Removed `time_of_day` from the fitted K-medoids feature space because it appeared redundant or noisy relative to `Visibility`.
11. Selected (k=2) as the provisional Version 1 PAM baseline based on the strongest revised silhouette score.
12. Interpreted the final two clusters as incident-characteristic groups rather than outcome-defined groups.

---

##### 16. Remaining limitations and future clustering work

The K-medoids Version 1 baseline is complete, but the full clustering analysis is not complete.

Remaining items include:

* Confirm whether K-medoids is acceptable as the course-approved classical baseline.
* Run a second substantially different clustering technique for comparison.
* Test whether the K-medoids result is stable across multiple random pilot samples or random seeds.
* Decide whether the final experiment should use a larger sample or a scalable medoid-based approach.
* Consider whether missing-value treatment changes cluster structure.
* Consider whether the warning-device block requires weighting or further sensitivity analysis.
* Compare final clustering methods using consistent quality measures and interpretation procedures.

The Version 1 baseline is therefore best viewed as a successful diagnostic and starting point for the final clustering comparison, not as the completed clustering portion of the project.

### DecisionTree

I originally planned to use association rule mining as one of my two major tasks. The goal was to identify combinations of incident characteristics associated with injury or fatality outcomes. However, as I worked through the setup and initial reasoning, it became clear that this task could produce many weak, repetitive, or difficult-to-interpret rules. Finding rules does not guarantee that the rules will reveal strong or useful patterns, especially in a complex real-world incident dataset.

Rather than continuing down a path that may be difficult to evaluate and explain clearly, I decided to pivot to classification. Classification provides a more direct question: whether incident characteristics can be used to predict if an injury occurred. The cleaned `v1.csv` dataset is still usable because the contextual incident features already prepared for the earlier tasks can now serve as predictors, while `injury_present` becomes the target variable.

For the first classification baseline, I chose a Decision Tree because it is a classical method covered in class, is straightforward to explain, and can provide interpretable decision rules. This pivot should create a clearer experimental pipeline with more direct evaluation measures such as precision, recall, F1 score, precision-recall AUC, and a confusion matrix.

#### Classification Task: Predicting Whether a Recorded Incident Included an Injury

##### Classification Objective

The supervised-learning portion of this project was defined as a binary classification task using the cleaned Version 1 incident-level dataset, `data/v1.csv`. The target variable was:

```python
TARGET = "injury_present"
```

The goal of this task was to classify whether a recorded highway–rail grade crossing incident resulted in at least one reported injury:

```text
injury_present = 1  → at least one reported injury occurred
injury_present = 0  → no reported injury occurred
```

This framing is important. The model is intended to classify injury outcomes among already-recorded incidents using incident-context variables. It is not being presented as a model that predicts whether a future crossing will experience an incident or as a causal model explaining why injuries occur.

The first method selected for this task was a `DecisionTreeClassifier`. A Decision Tree was selected because it is a classical supervised-learning method, is straightforward to explain, can capture nonlinear relationships, can use mixed numeric and categorical information after preprocessing, and provides interpretable decision rules and feature-importance values.

---

##### Use of the Cleaned Version 1 Dataset

The original preprocessing pipeline was not rerun for this classification analysis. Instead, the previously cleaned dataset was treated as the starting point:

```python
DATA_PATH = Path("data/v1.csv")
```

This was appropriate because the saved dataset already contained fixed data-cleaning and feature-construction decisions, including:

* cleaned numeric incident-context values;
* derived time-related categories such as `season` and `time_of_day`;
* binary warning-device indicator features;
* outcome variables retained for auditing but available to exclude from modeling.

The classification pipeline therefore focused on model-specific tasks: defining safe predictors, preventing target leakage, splitting training and testing data, handling missing values within the model pipeline, encoding categorical variables, training a Decision Tree, and evaluating its performance.

---

##### Target Validation and Class Distribution

The cleaned dataset contained:

| Quantity                                   |   Value |
| ------------------------------------------ | ------: |
| Total incidents                            | 250,806 |
| Injury incidents (`injury_present = 1`)    |  68,617 |
| No-injury incidents (`injury_present = 0`) | 182,189 |
| Injury prevalence                          |  27.36% |
| No-injury prevalence                       |  72.64% |

The target variable contained no missing values, so no rows needed to be removed because of an undefined classification label.

The positive injury class was not extremely rare, since more than 68,000 injury incidents were available. However, the dataset was still meaningfully imbalanced because approximately 72.64% of the incidents contained no reported injury. This established an important evaluation baseline:

```text
A model predicting "no injury" for every incident would achieve 72.64% accuracy.
```

Because of this imbalance, accuracy alone was not sufficient for evaluating the classifier. Precision, recall, F1 score, and average precision were also recorded, with particular attention given to recall because a model that rarely identifies actual injury cases would not be very informative even if its accuracy appeared acceptable.

---

##### Predictor Selection and Leakage Prevention

The model predictors were intentionally selected from incident-context features rather than being created by simply dropping a few columns from the full dataset. This reduced the risk that an outcome-related field could accidentally enter the model.

The original proposed predictor space contained:

| Predictor Group                  | Number of Features |
| -------------------------------- | -----------------: |
| Numeric predictors               |                  5 |
| Categorical predictors           |                 16 |
| Binary warning-device predictors |                 12 |
| Total proposed predictors        |                 33 |

The numeric predictors were:

```python
NUMERIC_FEATURES = [
    "Train Speed",
    "Estimated Vehicle Speed",
    "Number Vehicle Occupants",
    "Number of Cars",
    "Temperature"
]
```

The categorical predictors were:

```python
CATEGORICAL_FEATURES = [
    "season",
    "time_of_day",
    "Highway User",
    "Highway User Position",
    "Equipment Involved",
    "Equipment Struck",
    "Equipment Type",
    "Track Type",
    "Warning Connected To Signal",
    "Crossing Illuminated",
    "Visibility",
    "Weather Condition",
    "View Obstruction",
    "Highway User Action",
    "Driver Passed Vehicle",
    "Driver In Vehicle"
]
```

The original binary warning-device predictors were:

```python
BINARY_FEATURES = [
    "has_gate",
    "has_cantilever_fls",
    "has_standard_fls",
    "has_wig_wags",
    "has_highway_traffic_signals",
    "has_audible",
    "has_crossbucks",
    "has_stop_signs",
    "has_watchman",
    "has_flagged_by_crew",
    "has_other_warning",
    "has_no_warning_device"
]
```

The following fields were deliberately excluded from predictor input because they were identifiers, audit-only variables, post-incident severity variables, or fields that directly revealed or overlapped with the target:

```python
LEAKAGE_OR_EXCLUDED_FEATURES = [
    "Report Key",
    "Date",
    "year",
    "Crossing Users Killed",
    "Crossing Users Injured",
    "Employees Killed",
    "Employees Injured",
    "Passengers Killed",
    "Passengers Injured",
    "Total Killed Form 55A",
    "Total Injured Form 55A",
    "Total Killed Form 57",
    "Total Injured Form 57",
    "fatality_present",
    "injury_present",
    "Vehicle Damage Cost"
]
```

The reasoning for these exclusions was:

* `injury_present` was the target and therefore could not be an input.
* Injury count fields directly described whether injuries occurred and would create target leakage.
* Fatality count fields and `fatality_present` represented closely related post-incident severity outcomes and were also excluded to avoid using outcome information to predict another outcome.
* `Report Key` was an identifier and had no valid predictive interpretation.
* `Date` and `year` were retained for data auditing rather than Version 1 modeling.
* `Vehicle Damage Cost` represented post-incident severity and was not inflation-adjusted, so it was excluded from this baseline model.

The required-column audit confirmed that all proposed target and predictor columns existed in `v1.csv`, and the overlap check confirmed that none of the known excluded fields were selected as predictors.

---

##### Removal of a Constant Predictor

During the predictor audit, the warning-device variable:

```python
"has_no_warning_device"
```

was found to have the same value for every incident:

```text
has_no_warning_device: {0: 250806}
```

Because this variable had no variation, it could not help distinguish between injury and no-injury incidents. It was therefore removed from the modeling predictor list.

After removing this constant variable, the final raw feature space for the Decision Tree contained:

| Predictor Group                  | Number of Features |
| -------------------------------- | -----------------: |
| Numeric predictors               |                  5 |
| Categorical predictors           |                 16 |
| Binary warning-device predictors |                 11 |
| Total predictors before encoding |                 32 |

---

##### Predictor Missingness and Data-Type Audit

The predictor audit confirmed that all numeric and binary predictors had numeric stored data types. The binary warning-device predictors also contained the expected binary values of `0` and `1`.

Most predictors contained very little missing data. The two predictors with the largest missingness were:

| Predictor                     | Missing Values | Missing Percentage |
| ----------------------------- | -------------: | -----------------: |
| `Estimated Vehicle Speed`     |         28,072 |             11.19% |
| `Warning Connected To Signal` |         26,570 |             10.59% |

Other examples of missingness included:

| Predictor               | Missing Percentage |
| ----------------------- | -----------------: |
| `Crossing Illuminated`  |              3.61% |
| `Driver In Vehicle`     |              2.43% |
| `Driver Passed Vehicle` |              2.28% |
| `Highway User Action`   |              1.24% |
| `Train Speed`           |              1.02% |

These missing values did not justify removing the corresponding predictors from the initial baseline. Instead, missing-value handling was placed inside the scikit-learn preprocessing pipeline so that imputation would be learned from training data only.

---

##### Audit of Rare Categorical Levels

Before modeling, the categorical predictors were reviewed for rare category levels that could generate overly specific Decision Tree branches after one-hot encoding.

Several predictors contained uncommon levels. Examples included:

| Predictor             | Rare Level                             | Count | Percentage of Incidents |
| --------------------- | -------------------------------------- | ----: | ----------------------: |
| `Driver In Vehicle`   | `Unknown`                              |     2 |                 0.0008% |
| `Equipment Involved`  | `Train standing - RCL`                 |    11 |                 0.0044% |
| `Highway User Action` | `Went around/thru temporary barricade` |    62 |                 0.0247% |
| `Equipment Type`      | `Single Car`                           |    99 |                 0.0395% |
| `Highway User`        | `School bus`                           |   183 |                 0.0730% |

Some rare levels also represented missing values rather than meaningful categories.

The decision was made not to combine rare categories into an `"Other"` group for the first Decision Tree baseline. This avoided introducing unnecessary additional preprocessing decisions before observing actual model behavior. Instead, the Decision Tree was constrained using a minimum leaf-size requirement, which prevented extremely small subsets from independently defining model predictions.

---

##### Audit of Strong Incident-Context Predictors

After the initial Decision Tree was inspected, `Driver In Vehicle_No` appeared as the most influential split feature. Because it contributed strongly to the tree, the relationship between several important incident-context predictors and the target was audited directly.

For `Driver In Vehicle`, the observed injury rates were:

| Driver In Vehicle Value | Incidents | Injury Cases | Injury Rate |
| ----------------------- | --------: | -----------: | ----------: |
| `Yes`                   |   208,773 |       64,856 |      31.07% |
| `No`                    |    35,932 |        1,825 |       5.08% |
| `<MISSING>`             |     6,099 |        1,935 |      31.73% |
| `Unknown`               |         2 |            1 |      50.00% |

This explained why the tree strongly used `Driver In Vehicle_No`: incidents in which the driver was not in the vehicle had a substantially lower observed injury rate than incidents in which the driver remained in the vehicle. This did not directly reveal the target, so the feature was retained as a legitimate incident-context predictor.

Additional useful separation appeared in `Highway User Position`:

| Highway User Position          | Incidents | Injury Cases | Injury Rate |
| ------------------------------ | --------: | -----------: | ----------: |
| `Moving over crossing`         |   179,360 |       58,052 |      32.37% |
| `Stopped on crossing`          |    43,293 |        7,736 |      17.87% |
| `Stalled or stuck on crossing` |    26,258 |        2,428 |       9.25% |

These results supported the interpretation that the Decision Tree was learning meaningful differences among recorded incident situations rather than relying on obvious outcome leakage.

---

##### Training and Testing Split

The dataset was divided into training and testing sets using a stratified random split:

```python
TEST_SIZE = 0.20
RANDOM_STATE = 42
```

The split used:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification was important because it preserved the original injury prevalence in both partitions.

| Split        |    Rows | Injury Cases | No-Injury Cases | Injury Percentage | Majority Baseline Accuracy |
| ------------ | ------: | -----------: | --------------: | ----------------: | -------------------------: |
| Full dataset | 250,806 |       68,617 |         182,189 |            27.36% |                     72.64% |
| Training set | 200,644 |       54,893 |         145,751 |            27.36% |                     72.64% |
| Testing set  |  50,162 |       13,724 |          36,438 |            27.36% |                     72.64% |

This split established a held-out testing set for later evaluation of the selected Decision Tree. After splitting, all learned preprocessing and model fitting were performed using the training data only.

---

##### Leakage-Safe Preprocessing Pipeline

A scikit-learn `ColumnTransformer` was used to apply different preprocessing steps to different predictor groups.

The numeric preprocessing branch used median imputation:

```python
SimpleImputer(strategy="median")
```

Median imputation was selected because it provides a simple baseline treatment for missing numeric values and is less sensitive than the mean to unusually large or small measurements.

The categorical preprocessing branch used most-frequent imputation followed by one-hot encoding:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

Most-frequent imputation provided a simple baseline approach for missing category values. One-hot encoding converted each category into model-readable indicator columns. The option:

```python
handle_unknown="ignore"
```

ensured that prediction would not fail if a category appeared in the testing set that was not present in the training set.

The binary warning-device predictors were already stored as `0` and `1` values without missing entries, so they were passed directly through the transformer without additional processing.

No feature scaling was applied because Decision Trees split based on thresholds and do not require predictors to be placed on the same numerical scale.

The preprocessing audit produced:

| Quantity                              |   Value |
| ------------------------------------- | ------: |
| Training rows after preprocessing     | 200,644 |
| Testing rows after preprocessing      |  50,162 |
| Raw predictors before encoding        |      32 |
| Model-ready predictors after encoding |     111 |
| Numeric output features               |       5 |
| Binary output features                |      11 |
| Encoded categorical output features   |      95 |

The preprocessor was fitted on the training set and then applied to the testing set. This ensured that imputation values and encoded category mappings were not learned from held-out testing data.

---

##### Initial Constrained Decision Tree Baseline

The first Decision Tree baseline was intentionally kept simple and interpretable. It used:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=100,
    class_weight=None,
    random_state=42
)
```

The main design decisions were:

* `criterion="gini"`: standard impurity measure for classification trees.
* `max_depth=5`: limited the tree to a small, readable structure.
* `min_samples_leaf=100`: prevented branches based on very small groups or extremely rare categories.
* `class_weight=None`: allowed the initial model to reflect the observed class distribution before introducing class weighting.
* `random_state=42`: ensured reproducible results.

The initial Decision Tree produced the following confusion matrices:

```text
Training confusion matrix:
[[137343   8408]
 [ 44510  10383]]

Testing confusion matrix:
[[34365  2073]
 [11115  2609]]
```

For the testing set, this means:

| Outcome                                                   |  Count |
| --------------------------------------------------------- | -----: |
| True negatives: correctly predicted no injury             | 34,365 |
| False positives: predicted injury when no injury occurred |  2,073 |
| False negatives: missed injury cases                      | 11,115 |
| True positives: correctly identified injury cases         |  2,609 |

The initial model performance was:

| Split    | Accuracy | Precision | Recall |     F1 | Average Precision |
| -------- | -------: | --------: | -----: | -----: | ----------------: |
| Training |   0.7363 |    0.5526 | 0.1891 | 0.2818 |            0.4419 |
| Testing  |   0.7371 |    0.5572 | 0.1901 | 0.2835 |            0.4410 |

The testing accuracy of `0.7371` was only slightly better than the no-skill majority-class accuracy of `0.7264`. However, average precision increased from the no-skill reference of `0.2736` to `0.4410`, indicating that the model had learned meaningful ranking information about injury occurrence.

The main weakness of this initial model was recall:

```text
Testing recall = 0.1901
```

This means the model identified only approximately 19% of the actual injury incidents. The model was stable, since its training and testing performance were very similar, but it was too conservative in predicting the injury class.

The initial tree reached its permitted maximum depth of 5 and created 31 leaves, indicating that the imposed depth constraint was actively limiting the learned tree.

---

##### Initial Tree Interpretation

The initial constrained tree used only 13 of the 111 processed predictor columns in its splits. Its most important split features were:

| Processed Predictor                          | Importance |
| -------------------------------------------- | ---------: |
| `Driver In Vehicle_No`                       |   0.355201 |
| `Train Speed`                                |   0.277328 |
| `Number Vehicle Occupants`                   |   0.185079 |
| `Estimated Vehicle Speed`                    |   0.126023 |
| `Highway User_Pedestrian`                    |   0.024244 |
| `Highway User Position_Moving over crossing` |   0.016368 |

The model's top decision rule first separated incidents based on whether the driver was not in the vehicle, followed by splits involving train speed, vehicle speed, and number of occupants. These were considered interpretable incident-context variables rather than direct outcome indicators.

It is important to interpret these feature importances carefully. A high feature importance means that a predictor helped the Decision Tree reduce classification impurity in this dataset. It does not establish that the predictor causes injuries.

---

##### Controlled Decision Tree Comparisons Using Training Data Only

Because the initial Decision Tree had very low injury recall, a small set of controlled variations was evaluated. These comparisons were performed using only the training set through three-fold stratified cross-validation:

```python
StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
```

Using cross-validation on the training data allowed model settings to be compared without repeatedly selecting decisions based on the held-out testing set.

The first comparison examined tree depth and the use of automatically balanced class weights:

| Model                | Maximum Depth | Minimum Leaf Size | Class Weight | Accuracy | Precision | Recall |     F1 | Average Precision |
| -------------------- | ------------: | ----------------: | ------------ | -------: | --------: | -----: | -----: | ----------------: |
| Current baseline     |             5 |               100 | `None`       |   0.7363 |    0.5519 | 0.1916 | 0.2844 |            0.4400 |
| Deeper tree          |             8 |               100 | `None`       |   0.7371 |    0.5600 | 0.1816 | 0.2742 |            0.4635 |
| Balanced baseline    |             5 |               100 | `"balanced"` |   0.5977 |    0.3874 | 0.8037 | 0.5221 |            0.4355 |
| Balanced deeper tree |             8 |               100 | `"balanced"` |   0.6151 |    0.3977 | 0.7905 | 0.5292 |            0.4625 |

This comparison showed that simply allowing a deeper tree did not solve the low-recall problem. The unweighted depth-8 tree still identified fewer than 20% of injury cases. In contrast, class weighting caused the tree to identify substantially more injury cases, increasing recall to approximately 79–80%.

However, the fully balanced class weighting also reduced precision and accuracy because the model became much more willing to predict the injury class. Rather than immediately selecting the fully balanced tree, moderate class-weight values were tested to search for a more balanced precision-recall tradeoff.

---

##### Moderate Class-Weight Comparison

A second controlled comparison evaluated intermediate injury-class weights using a depth-8 tree with a minimum leaf size of 100.

| Model              | Maximum Depth | Minimum Leaf Size | Class Weight     | Accuracy | Precision | Recall |     F1 | Average Precision |
| ------------------ | ------------: | ----------------: | ---------------- | -------: | --------: | -----: | -----: | ----------------: |
| Current baseline   |             5 |               100 | `None`           |   0.7363 |    0.5519 | 0.1916 | 0.2844 |            0.4400 |
| Depth 8 unweighted |             8 |               100 | `None`           |   0.7371 |    0.5600 | 0.1816 | 0.2742 |            0.4635 |
| Depth 8 weight 1.5 |             8 |               100 | `{0: 1, 1: 1.5}` |   0.7213 |    0.4873 | 0.3545 | 0.4103 |            0.4613 |
| Depth 8 weight 2.0 |             8 |               100 | `{0: 1, 1: 2.0}` |   0.6723 |    0.4312 | 0.6180 | 0.5078 |            0.4613 |
| Depth 8 balanced   |             8 |               100 | `"balanced"`     |   0.6151 |    0.3977 | 0.7905 | 0.5292 |            0.4625 |

These results showed a clear tradeoff:

* The unweighted models had higher precision and accuracy but failed to detect most injury cases.
* The fully balanced tree achieved the greatest recall and F1 score, but at the cost of lower precision and considerably more positive predictions.
* The weight-1.5 tree improved recall but still detected only about 35% of injuries.
* The weight-2.0 tree achieved a substantial recall improvement while remaining less aggressive than the fully balanced model.

The selected refined Decision Tree therefore used:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=8,
    min_samples_leaf=100,
    class_weight={0: 1, 1: 2.0},
    random_state=42
)
```

This selection was made as a practical compromise. Although the fully balanced tree achieved a slightly higher F1 score in cross-validation, the weight-2.0 tree provided substantially improved injury detection without reducing precision as severely as the fully balanced alternative.

---

##### Final Selected Decision Tree Evaluation

After model settings were selected from training-set cross-validation, the refined Decision Tree was trained on the full training set and evaluated on the held-out testing set.

The selected model used:

| Setting                  | Value            |
| ------------------------ | ---------------- |
| Criterion                | Gini impurity    |
| Maximum depth            | 8                |
| Minimum samples per leaf | 100              |
| Class weighting          | `{0: 1, 1: 2.0}` |
| Random seed              | 42               |

The resulting confusion matrices were:

```text
Training confusion matrix:
[[100382  45369]
 [ 19862  35031]]

Testing confusion matrix:
[[24968 11470]
 [ 4927  8797]]
```

For the held-out testing set:

| Outcome                                                   |  Count |
| --------------------------------------------------------- | -----: |
| True negatives: correctly predicted no injury             | 24,968 |
| False positives: predicted injury when no injury occurred | 11,470 |
| False negatives: missed injury cases                      |  4,927 |
| True positives: correctly identified injury cases         |  8,797 |

The selected model performance was:

| Split    | Accuracy | Precision | Recall |     F1 | Average Precision |
| -------- | -------: | --------: | -----: | -----: | ----------------: |
| Training |   0.6749 |    0.4357 | 0.6382 | 0.5179 |            0.4719 |
| Testing  |   0.6731 |    0.4341 | 0.6410 | 0.5176 |            0.4655 |

The selected model identified:

```text
8,797 of 13,724 held-out injury incidents
```

This corresponds to a testing recall of:

```text
0.6410, or 64.10%
```

Compared with the initial Decision Tree, the selected weighted tree substantially improved its ability to detect injury incidents:

| Metric                    | Initial Constrained Tree | Selected Weighted Tree |
| ------------------------- | -----------------------: | ---------------------: |
| Testing accuracy          |                   0.7371 |                 0.6731 |
| Testing precision         |                   0.5572 |                 0.4341 |
| Testing recall            |                   0.1901 |                 0.6410 |
| Testing F1                |                   0.2835 |                 0.5176 |
| Testing average precision |                   0.4410 |                 0.4655 |

The selected model's lower accuracy was expected. The majority class accounted for 72.64% of the dataset, so a model optimized mainly for accuracy would tend to predict no injury frequently. Introducing injury-class weighting intentionally shifted the model toward detecting more injury cases. This increased false positives, reducing overall accuracy, but substantially improved recall and F1 score.

Average precision also increased from `0.4410` in the initial model to `0.4655` in the selected weighted model, remaining clearly above the no-skill reference value of `0.2736`.

---

##### Overfitting Assessment

The selected model did not show strong evidence of overfitting because its training and testing performance remained very similar:

| Metric            | Training | Testing | Difference |
| ----------------- | -------: | ------: | ---------: |
| Accuracy          |   0.6749 |  0.6731 |     0.0018 |
| Precision         |   0.4357 |  0.4341 |     0.0016 |
| Recall            |   0.6382 |  0.6410 |     0.0028 |
| F1                |   0.5179 |  0.5176 |     0.0003 |
| Average Precision |   0.4719 |  0.4655 |     0.0064 |

The Decision Tree reached its permitted maximum depth of 8 and produced 171 leaves. Although this model was more complex than the initial depth-5 tree, its close training and testing metrics indicated that the added complexity did not create a substantial generalization gap on the held-out data.

---

##### Final Tree Behavior and Important Predictors

The selected tree had access to 111 processed predictors after preprocessing and one-hot encoding. It used 37 of those processed features in its splits.

The most influential processed features were:

| Processed Predictor                                      | Importance |
| -------------------------------------------------------- | ---------: |
| `Driver In Vehicle_No`                                   |   0.352757 |
| `Train Speed`                                            |   0.262711 |
| `Number Vehicle Occupants`                               |   0.143369 |
| `Estimated Vehicle Speed`                                |   0.118793 |
| `Highway User_Pedestrian`                                |   0.037234 |
| `Highway User Position_Moving over crossing`             |   0.024312 |
| `Equipment Struck_Rail equipment struck by highway user` |   0.021421 |
| `Highway User_Truck-trailer`                             |   0.007061 |
| `Number of Cars`                                         |   0.005637 |
| `Visibility_Dark`                                        |   0.004120 |
| `Temperature`                                            |   0.003852 |
| `Equipment Type_Passenger Train - Pulling`               |   0.002730 |
| `Highway User Action_Suicide/attempted suicide`          |   0.002521 |

The selected tree continued to rely primarily on interpretable incident-context variables. The largest contributions came from whether the driver was in the vehicle, train speed, number of vehicle occupants, estimated vehicle speed, pedestrian involvement, and crossing movement position.

These importance values should not be interpreted causally. They only describe which encoded predictors the Decision Tree found most useful for reducing classification impurity while predicting injury status within this dataset.

---

##### Final Decision Tree Conclusion

The Decision Tree classification analysis established that `injury_present` is a viable supervised-learning target for the cleaned Version 1 dataset.

The initial constrained Decision Tree demonstrated that the predictor set contained useful injury-related signal, as shown by an average precision score above the no-skill baseline. However, the unweighted model was too conservative and detected only approximately 19% of injury incidents.

Controlled comparisons using training-set cross-validation showed that class weighting was necessary to meaningfully improve injury recall. A depth-8 Decision Tree with a moderate injury-class weight of `{0: 1, 1: 2.0}` was selected as the refined Decision Tree model because it provided a practical balance between identifying injuries and avoiding the most aggressive false-positive behavior of the fully balanced alternative.

On the held-out testing set, the selected Decision Tree achieved:

| Metric            | Selected Decision Tree Test Result |
| ----------------- | ---------------------------------: |
| Accuracy          |                             0.6731 |
| Precision         |                             0.4341 |
| Recall            |                             0.6410 |
| F1 score          |                             0.5176 |
| Average precision |                             0.4655 |

The selected model detected 8,797 of 13,724 injury incidents in the testing set and demonstrated very similar training and testing performance. This indicates that the Decision Tree provides an interpretable and reasonably stable classical classification method for the project, while also establishing a baseline against which a second substantially different classification technique can later be compared.

---

##### Limitations and Interpretation Boundaries

Several limitations should be kept in mind when interpreting this analysis:

1. The model classifies injury status among recorded incidents. It does not predict whether an incident will occur at a crossing in the future.

2. The model uses variables recorded about the incident itself, including highway-user action and driver status. Therefore, it should not be described as a purely pre-incident prevention model.

3. The binary target only distinguishes whether at least one injury occurred. It does not model the number of injuries or injury severity.

4. The selected class weight reflects a modeling decision that placed greater importance on detecting injury incidents. A different application could justify a different tradeoff between precision and recall.

5. Feature-importance values from a Decision Tree indicate predictive splitting usefulness within this fitted model, not causal relationships.

6. Rare categorical levels were retained for the initial baseline rather than grouped, with tree complexity controls used to reduce the risk of overly specific branches.

7. The held-out testing set was used after the Decision Tree settings were selected through training-set cross-validation. Further Decision Tree tuning should not be based on the same test results, because doing so would weaken the meaning of the held-out evaluation.

---

##### Current Status

The Decision Tree portion of the classification task is complete for the current project stage. The work completed includes:

* validation of the `injury_present` target;
* predictor and leakage auditing;
* removal of a constant predictor;
* missingness and categorical-level diagnostics;
* a stratified training/testing split;
* leakage-safe preprocessing;
* an initial constrained Decision Tree baseline;
* controlled cross-validation comparisons;
* selection of a refined weighted Decision Tree;
* held-out test evaluation;
* inspection of important tree predictors and model behavior.

The selected Decision Tree can now serve as the classical classification method for later comparison with a second substantially different classification technique using the same target, predictor definitions, data split, and evaluation measures.

## Recent models

After running some baseline test we will dive into finding some recent methods to compare against our baseline methods. Time to do some digging.

### HDBSCAN (cluster comparison)

### TabNet

## Evaluation

### Decision Tree metrics

- Average precision
- F1 score

### Clustering metrics

- Gower-distance silhouette score
- Cluster stability

### Hyperparameter ranges