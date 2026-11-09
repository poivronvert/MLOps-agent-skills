# Artifact Contract

The repo-wide convention for what each pipeline stage writes and what the next stage
expects. Every path and key below is what the scripts in this repo actually produce or
consume today — treat this file as the interface definition between skills.

## Directory Layout

```
<output-dir>/                        # --output of model-training/scripts/train_model.py
  model/MLmodel                      # official MLflow Model directory
  task_spec.json
  environment.json
  training_report.json               # run metadata + metrics (schema below)

<artifact-root>/<run_name>/<stage>/  # ml-pipeline-orchestration scripts
  model/model.pkl                    #   (default artifact_root: /tmp/ml_pipeline/artifacts)
  evaluation/metrics.json            # flat {metric_name: value} dict
  evaluation/classification_report.txt
```

## Training Outputs

`model-training/scripts/train_model.py --output <dir>` writes:

- **`model/`** — official MLflow model, with complete preprocessing, inference semantics, signature and dependencies. Load using `mlflow.pyfunc.load_model`.
- **`training_report.json`** — keys: `timestamp`, `model_type`, `task`, `input_file`,
  `target_column`, `dataset_rows`, `dataset_features`, `cv_folds`, `cv_strategy`,
  `n_trials`, `best_trial`, `best_cv_score`, `best_params`,
  `train_resubstitution_metrics`, `model_path`, `training_time_seconds`.

Metrics dicts are always flat, with **metric names as keys** and float values:

```json
{"accuracy": 0.94, "f1_weighted": 0.93}            // classification
{"rmse": 3.21, "mae": 2.10, "r2": 0.87}            // regression
```

The orchestration pipelines (`ml-pipeline-orchestration/scripts/{prefect,airflow}_pipeline.py`)
write the same flat shape to `evaluation/metrics.json`
(`accuracy`, `precision`, `recall`, `f1`, `auc_roc`). Use `best_cv_score` for
generalization estimates — `train_resubstitution_metrics` are computed on training data
and are optimistic.

## What Registration Expects

`model-registry/scripts/registry_manager.py --action register` does **not** take a file
path — it registers from an MLflow run:

- `--run-id` — a completed MLflow run that logged the model
  (e.g. via `mlflow.sklearn.log_model(model, "model")`).
- `--artifact-path` — the artifact sub-path within the run (default `model`).

Consequence: log metrics to the run with the same flat metric names as above
(`mlflow.log_metrics(metrics)`), because `--action compare` and `--action lineage` read
`run.data.metrics` — a version registered without run metrics compares as all `N/A`.
Promotion assigns aliases (`--action promote --alias champion`); serving resolves them.

## Dataset input handoff

Training/evaluation records actual data usage in MLflow Dataset inputs. Registration
and the model-card example read those records; no duplicate data-summary Run or Model
Version tags are required or automatically written. Ownership, code and validation
tags retain their separate purposes.

For the integrated trainer, the model's parent run owns refit training and any holdout
test inputs. Direct trial children in the same experiment own CV fold training and
validation inputs, linked by `mlflow.parentRunId`. Model cards enumerate these runs,
including failed/deleted trials, with pagination and explicit run/fold/status labels.
They do not combine overlapping fold row counts or claim every attempted input trained
the final model. Custom nesting beyond this two-level structure needs an explicit scope.

Read names, context, source and digest from inputs, and row counts from the optional
profile when available. Digest is not an external dataset version. Missing metadata
is unknown; absent inputs are unrecorded, without a fallback to legacy tags or backfill.
Lookup failures must be reported as failures, not as missing data.

Dataset inputs identify usage; the existing optional snapshot artifacts provide
downloadable contents. Their manifest, hashing and storage policy are unchanged.
`dataset_inputs: false` (or tracking off) can leave no inputs for a model card.
See [Dataset tracking](../../ml-experiment-tracking/references/DATASET_TRACKING.md)
for producer controls and [the model-card example](../SKILL.md#6-model-card-generation)
for the read-only consumer.

## What Inference Expects

For integrated training outputs use official MLflow directly:

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("output/model")  # or report["tracking"]["model_uri"]
predictions = model.predict(inputs)
```

Or use `mlflow models predict --model-uri output/model --input-path input.csv
--content-type csv --env-manager local` in the model's dependency environment.
Classification/event output contains `prediction` and supported `probability_i` columns;
MLmodel `label_mapping` metadata identifies each class. Regression returns numeric predictions.
Forecast uses keyed entity/time and horizon output, not a one-row-per-input classification contract.

The generic serving/batch scripts also support independent framework-specific files.
Their joblib/ONNX/TorchScript examples are separate interfaces; do not feed the integrated
training directory to a joblib loader. Use the official SDK/CLI for these MLflow models.

## What Monitoring Expects

`model-monitoring/scripts/monitor_model.py --reference ref.parquet --current cur.parquet`:

- **Reference and current files** — Parquet (preferred) or CSV, same feature columns.
  Reference = a scored snapshot of training/validation data; current = recent
  production data, i.e. the direct output of `batch_inference.py`.
- **Prediction column** — named **`prediction`** by default (`--prediction` to override);
  must exist in both files for prediction-distribution checks.
- **Target column** — optional `--target`; when present alongside `prediction`,
  performance metrics are computed with the same names as training
  (`accuracy`/`f1_weighted`/... or `rmse`/`mae`/`r2`/`mape`).
- **Thresholds file** — `--thresholds thresholds.json`, keyed by the same metric names:
  `{"accuracy": 0.9}` (minimum) or `{"rmse": {"max": 5.0}}` (min/max bounds).

`model-drift-detection/scripts/detect_drift.py` uses the identical
`--reference`/`--current` file pair.

## Contract Summary

| Producer | Artifact | Consumer | Key/column it relies on |
|----------|----------|----------|-------------------------|
| train_model.py | official `model/` or returned model URI | MLflow SDK/CLI | signature and documented prediction schema |
| train_model.py | `training_report.json` | humans, CI gates | `best_cv_score`, flat metric names |
| MLflow run (`log_model` + `log_metrics`) | registered version | registry_manager.py | `--run-id`, `run.data.metrics` |
| registry (alias `@champion`) | model URI | serve_model.py | `--model-uri models:/<name>@champion` |
| batch_inference.py | `*_predictions.parquet` | monitor_model.py, detect_drift.py | `prediction` column |
| monitor_model.py | report JSON | setup_alerts.py | flat metric names in thresholds |

## Agent MLOps official model delivery

`train_model.py` keeps its training CLI; output is now the official `model/` directory.
No old-format loader or migration adapter is provided. Training reports contain
`training_status`, `artifact_status`, `model_path`, `tracking`, `evaluation`, `trials`,
data/source hashes, exclusions and split boundaries. Reports are independent of loading.
A model-save failure is an artifact failure even if training completed.

`evaluation` separates resubstitution, CV and holdout; absent holdout is null.
`best_cv_score` uses natural metric units. `--n-trials 1` is one HPO trial;
`--no-hpo` uses fixed parameters. Partial fold models have an explicit role in the report
and are not published as final models.

Remote tracking calls official `log_model` and records its returned URI. It does not
register a version or change aliases. For registration use the returned URI with the
official `mlflow.register_model(model_uri, name)` API as a separately authorized action.
The generic run/artifact-path registry wrapper above is not required for this delivery.

Full configuration: [Agent training](../../model-training/references/AGENT_TRAINING.md).
