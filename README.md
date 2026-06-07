# ML Data Challenge

This project analyzes the Federal Railroad Administration highway-rail grade crossing incident dataset. The final notebook compares classical and more recent machine learning methods on two tasks:

- **Classification:** predict whether an already-recorded incident involved at least one injury.
- **Clustering:** identify incident profiles from context features without using injury or fatality outcomes to form the groups.

The notebook is the main reproducible artifact. The `src/` folder contains earlier script-based development pipelines.

## Data

Download the FRA Form 57 incident CSV and place it in `data/` with this filename:

```text
data/Highway-Rail_Grade_Crossing_Incident_Data_(Form_57)_20260603.csv
```

Dataset source: <https://data.transportation.gov/stories/s/Form-57-Data-Downloads/i5dw-jvsi/>

The raw dataset has about 250,806 incident rows and 154 original features. The notebook rebuilds the cleaned working dataset, creates the `injury_present` target from Form 55A injury totals, removes leakage-prone fields, and generates modeling features such as warning-device indicators, `season`, and `time_of_day`.

## Methods

### Classification

Target:

```text
injury_present = 1 if Total Injured Form 55A > 0, else 0
```

Models compared:

- **Decision Tree:** classical baseline using imputation and one-hot encoding.
- **TabNet:** recent tabular neural model using categorical embeddings.

Evaluation used a stratified train/validation/test split of 64% / 16% / 20%. Thresholds were selected on the validation split using F1 and then evaluated on the held-out test set. Main metrics were AP / PR-AUC, F1, precision, and recall.

Final test summary:

| Model | Threshold | AP / PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.4947 | 0.4734 | 0.5336 | 0.4010 | 0.7969 |
| TabNet | 0.2642 | 0.4958 | 0.5389 | 0.4034 | 0.8114 |

TabNet performed slightly better, but the improvement was small relative to the added training time and implementation complexity.

### Clustering

Methods compared:

- **K-medoids/PAM:** classical clustering method using a precomputed Gower-style mixed-data distance matrix.
- **HDBSCAN:** density-based comparison method that can label ambiguous records as noise.

Clustering used 2,000-row random samples because both methods depend on a pairwise distance matrix. Sampling was repeated across multiple random seeds to check whether the result depended on one unusual sample.

Final clustering summary:

| Method | Main Result |
|---|---|
| K-medoids/PAM | selected `k=2`, silhouette `0.1454`, total cost `541.7935`; all 2,000 sampled incidents were assigned to a cluster |
| HDBSCAN | selected `eom`, `min_cluster_size=10`, `min_samples=5`; 2 clusters, 251 non-noise records, 87.45% noise, clustered-point silhouette `0.3832` |

The main clustering takeaway is descriptive: K-medoids forced broad groups, while HDBSCAN found smaller dense incident profiles and labeled most records as noise. Injury and fatality outcomes were only used after clustering for interpretation, not as clustering inputs.

## Figures

Final report figures are written to `figures/`:

```text
figures/classification_baseline.png
figures/decision_tree_hyperparameter_search.png
figures/tabnet_hyperparameter_search_table.png
figures/classification_comparison.png
figures/cluster_groups.png
```

## How to Run

Install dependencies from `pyproject.toml`, then open and run:

```text
data_challenge.ipynb
```

If using `uv`:

```powershell
uv sync
uv run jupyter notebook data_challenge.ipynb
```

For the cleanest rerun, restart the notebook kernel and run all cells from the top.

Expected runtime:

- Decision Tree is quick.
- TabNet may take a few minutes on CPU.
- Clustering may take a few minutes because it builds a 2,000 by 2,000 distance matrix.

## Repository Structure

```text
data_challenge.ipynb        Final self-contained notebook
figures/                    Generated report figures
data/                       Raw and generated data files
src/                        Script-based development pipelines
outputs/                    Generated presentation/output artifacts
```

## Interpretation Boundaries

This project models outcomes among already-recorded incidents. It does not predict whether a future crossing incident will occur. Clustering results should also be interpreted descriptively, not causally, because injury and fatality outcomes were excluded from the clustering feature space.
