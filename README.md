# MGM2 embedding temperature-model comparison

This folder prepares a fair, paired temperature-modeling dataset from two MGM2
community-embedding files:

- the original-taxonomy MGM2-small CLS embeddings;
- the MAPseq-taxonomy MGM2-small CLS embeddings.

The preparation stage validates the sources, aligns samples, averages
replicates, reuses the frozen cruise-based split from the earlier ASV
benchmark, and writes machine-readable tables. The modeling stage then compares
CatBoost temperature models trained from the original-taxonomy and MAPseq MGM2
embeddings.

## What the preparation does

1. Validates both HDF5 files (keys, dimensions, IDs, finite values, provenance).
2. Confirms that the original and MAPseq metadata tables are identical.
3. Joins HDF5 rows to metadata by the exact `sample_id` value.
4. Excludes HDF5 rows absent from metadata and rows without a usable material ID.
5. Maps labels such as `Aa`, `Ab`, and `Cb` to canonical biological replicates
   `A`, `A`, and `C`.
6. Averages technical reruns within each material-replicate group.
7. Averages the available A/B/C replicate vectors equally for each material.
8. Aligns the 1,191 temperature-complete materials to the existing frozen
   development/test and five-fold cruise assignments.
9. Writes paired original and MAPseq tables with the same samples, order,
   targets, and split columns.

No PCA, scaling, feature selection, or target-informed transformation is done
at this stage. All 224 MGM2 embedding dimensions are retained.

## Run it

From this folder:

```bash
bash scripts/run_prepare.sh
```

That one command performs both preparation and validation. To exercise the full
pipeline in a disposable temporary directory without keeping prepared tables:

```bash
bash scripts/run_validate.sh
```

The scripts use `python3`, NumPy, and h5py. If needed, install the small set of
dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model preflight (no training)

Before training, validate the two prepared tables, frozen folds, hashes, model
candidate grid, and ASV benchmark alignment:

```bash
bash scripts/run_model_preflight.sh
```

This command must end with:

```text
Preflight passed. No CatBoost model was trained.
```

## Train and compare the temperature models

Choose a new run ID and run:

```bash
bash scripts/run_temperature_comparison.sh temperature_embeddings_cpu_01
```

The default protocol uses CPU CatBoost 1.2.10. It independently tunes the same
six CatBoost candidates for each embedding representation with the frozen five
cruise-grouped development folds. It selects the lowest pooled OOF MAE, refits
that representation's winner on all 946 development samples, and evaluates it
once on the same 245 held-out samples from 8 unseen cruises.

An optional GPU invocation is:

```bash
bash scripts/run_temperature_comparison.sh temperature_embeddings_gpu_01 \
  --task-type GPU --devices 0
```

Do not reuse a run ID: the pipeline refuses to overwrite a nonempty result
directory.

## Modeling outputs

Results are saved under `results/RUN_ID/`. The most useful files are:

| File | Purpose |
|---|---|
| `comparison/report.md` | Plain-language conclusion and main metrics. |
| `comparison/tables/test_metric_summary.csv` | Held-out MAE, RMSE, R², macro-cruise MAE, and intervals. |
| `comparison/tables/paired_test_metric_differences.csv` | Paired MAPseq-versus-original differences. |
| `comparison/tables/paired_test_predictions.csv` | All predictions on the same held-out samples. |
| `comparison/figures/observed_vs_predicted_comparison.png` | Direct prediction comparison. |
| `comparison/figures/test_metric_comparison.png` | Test error with cruise-block uncertainty. |
| `original/models/catboost_temperature.cbm` | Saved original-taxonomy model. |
| `mapseq/models/catboost_temperature.cbm` | Saved MAPseq model. |

Feature importance and SHAP outputs describe the 224 latent MGM2 dimensions;
they do **not** identify individual ASVs or bacterial taxa.

Additional report figures can be regenerated from a completed run without
retraining either model:

```bash
bash scripts/regenerate_result_figures.sh temperature_embedding_cpu_01
```

This adds individual observed-versus-predicted, residual, absolute-error,
cruise-error, development-versus-test, and SHAP figures for each embedding
model. It also adds comparison plots for the full error distributions and the
paired sample-level MAPseq-minus-original error difference.

## Outputs

Prepared files are written to `data/prepared/`:

| File | Purpose |
|---|---|
| `validation_report.md` | Human-readable PASS/FAIL report and warnings. |
| `validation_summary.json` | Machine-readable checks, counts, and warnings. |
| `validation_checks.csv` | One row per validation check. |
| `input_manifest.json` | Source paths, SHA-256 hashes, HDF5 attributes, and output hashes. |
| `sample_alignment_audit.csv` | Status and exclusion reason for every HDF5 sample ID. |
| `replicate_mapping.csv` | Raw-to-canonical replicate mapping and technical-rerun counts. |
| `material_metadata.csv` | One metadata row per averaged material sample. |
| `original_material_embeddings.csv` | One original-taxonomy 224-vector per material. |
| `mapseq_material_embeddings.csv` | One MAPseq 224-vector per material. |
| `original_temperature_training_table.csv` | Frozen temperature cohort and original embeddings. |
| `mapseq_temperature_training_table.csv` | The identical cohort and MAPseq embeddings. |
| `split_assignments.csv` | Validated copy of the frozen split/fold assignments. |
| `feature_columns.txt` | Exact 224-column feature allowlist for later models. |
| `training_schema.json` | Target, grouping, split, and predictor-column roles. |
| `exact_duplicate_embedding_rows.csv` | Audit of exact duplicate vectors within either input. |

The two temperature training tables have this structure:

```text
material_sample,cruise,temperature,outer_partition,development_cv_fold,
training_row,source_asv_processed_matrix_row,embedding_001,...,embedding_224
```

Only columns beginning with `embedding_` will later be used as model features.
`temperature` is the target. `cruise` and the split columns are used only for
grouping and evaluation; they are not predictor features.

The model code reads `feature_columns.txt` as the exact 224-column allowlist. It
must not select all numeric columns, because
`development_cv_fold`, `training_row`, and `source_asv_processed_matrix_row` are
numeric audit/split fields,
not biological predictors.

## Important averaging rule

Technical reruns must not receive extra biological weight. For example, if A
was sequenced twice and B once:

```text
A_mean = mean(Aa, Ab)
material_mean = mean(A_mean, B)
```

The pipeline deliberately does **not** take `mean(Aa, Ab, B)`, because that
would give biological replicate A twice the weight of B.
