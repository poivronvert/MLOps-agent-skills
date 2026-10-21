---
name: ml-experiment-tracking
description: >
  Track and manage ML experiments with MLflow, Weights & Biases, Neptune, and CometML. Covers experiment logging
  (hyperparameters, metrics, artifacts), auto-logging for PyTorch, TensorFlow, scikit-learn, and XGBoost, experiment
  comparison and visualization, artifact versioning, model lineage, reproducibility (environment, code, data versioning),
  remote tracking server setup, team collaboration, experiment search and querying, and integration with training
  pipelines. Use when running experiments, comparing model versions, or setting up experiment infrastructure.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# ML Experiment Tracking

## Tracking for the integrated training CLI

For CSV/Parquet tabular, window-event or recursive forecast jobs, use
[Agent training](../model-training/references/AGENT_TRAINING.md). Its shared
`agent_mlops.tracking.Tracker` is an explicit off/HTTP(S)-remote client with preflight,
parent/trial links, official MLflow models and stop-on-recording-failure behavior.
Use platform-provided URI/authentication; do not deploy a server or silently select a local
store. The administrative wrapper and generic examples below do not replace this contract.
Training saves joblib predictors inside official Models From Code PythonModel artifacts
and uses `mlflow.pyfunc.log_model` for remote runs.
Load the returned model URI with `mlflow.pyfunc.load_model`; registration and aliases are
separate operations. See Agent training for output schemas and failure states.

### Generic wrapper model logging

The standalone `scripts/mlflow_tracker.py` wrapper keeps the selected framework's MLflow
flavor. Its sklearn branch (including unknown-model fallback) follows the SDK default:
MLflow 3.17.1 uses skops outside Databricks. Install this repository's `tracking` extra
alongside the needed model extras (`uv sync --extra tracking ...`); it declares skops.
If using the script independently, supply compatible MLflow and skops dependencies in
that environment. This dependency is separate from the built-in trainer's joblib artifact.

`ExperimentTracker.log_model(..., serialization_format=None, skops_trusted_types=None)`
accepts these keyword-only options for the sklearn branch. Omitting them retains SDK
defaults; they are not forwarded to XGBoost or PyTorch. When skops rejects a type, review
that type and explicitly supply the approved list. Never automatically trust all detected
types, or silently retry using pickle/cloudpickle. An explicit SDK-supported format
remains the caller's choice.
With skops 0.16, even a fitted sklearn tree can require explicit review of
`sklearn.tree._tree.Tree`; installing skops alone does not approve model types.

Model logging failures, including missing skops, serialization and upload errors, are
logged and re-raised. The wrapper does not own the caller's Run lifecycle. A caller that
started a Run must mark it failed on an exception and must not report registration success:

```python
tracker.start_run(run_name="training")
try:
    tracker.log_model(model, input_example=X_example)
except Exception:
    tracker.end_run(status="FAILED")
    raise
else:
    tracker.end_run(status="FINISHED")
```

Optional signature inference may still warn and proceed without a signature; this does
not permit swallowing model logging failures.

## Dataset inputs and snapshots

For dataset lineage recorded during training and evaluation, read
[Dataset tracking](references/DATASET_TRACKING.md).
Dataset inputs are the primary usage record. Do not separately write dataset
summary Run/Model Version tags; ordinary ownership, code and validation tags remain.
Record each actual training, validation, and test split with its name, schema, target,
context, source/version and full-content SHA-256. Include CV fold and trial identity;
do not substitute the original unsplit file or infer membership from a random seed.
The training/preprocessing code supplies the frames it uses. The shared tracking code
must not split or transform them. Identify the boundary (pipeline inputs versus
estimator tensors); preserve preprocessing/model artifacts needed to reproduce it.

When supplying an MLflow Dataset `digest` manually, keep it within the MLflow 3.17.1
limit of 36 characters. For a SHA-256-based digest, use the first 32 hexadecimal
characters; never pass the full 64-character SHA-256 as `digest`. Retain the full
hash in the Dataset input's `data.sha256` tag or the existing dataset manifest for
verification. The shared tracking helper already follows this convention.

Dataset input registration and full-data artifact uploads are independent policies.
The integrated remote trainer defaults to both input registration and snapshots of
training/validation/test splits; metadata-only is `dataset_snapshots: []`.
Respect the selected contexts, storage access and data retention requirements.
Autologging alone does not establish that all evaluation splits were recorded.
Accept completion only after reading back input metadata and verifying selected
snapshots by download, SHA-256, and frame equality. A recording failure stops further
fit/predict work and remains visible in the report.

Production ingestion, durable dataset versions, and retraining decisions have separate
owners: see the dataset reference. A new data version need not launch training.

## Overview

Experiment tracking is the systematic recording of ML experiments - hyperparameters,
metrics, artifacts, code, and environment - to enable reproducibility, comparison,
and collaboration.

## When to Use This Skill

- Starting a new ML project that needs experiment management
- Comparing multiple model architectures or hyperparameter configs
- Setting up team-wide experiment tracking infrastructure
- Ensuring experiment reproducibility
- Migrating between tracking tools

## Step-by-Step Instructions

### 1. MLflow Tracking

#### Basic Setup

```python
import mlflow

# Set tracking URI (local or remote)
mlflow.set_tracking_uri("http://localhost:5000")  # Remote server
# mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite

# Create/set experiment
mlflow.set_experiment("my-recommendation-model")

# Log a run
with mlflow.start_run(run_name="baseline-v1") as run:
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)

    # Log metrics over steps
    for epoch in range(10):
        mlflow.log_metric("loss", loss_val, step=epoch)

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.csv")

    # Log model (MLflow 3: use name= instead of positional artifact_path)
    mlflow.sklearn.log_model(model, name="model")

    # These Dataset objects were constructed from the actual, already-selected splits.
    mlflow.log_input(training_dataset, context="training")
    mlflow.log_input(validation_dataset, context="validation")

    # Run-level metadata
    mlflow.set_tag("team", "recommendation")
```

#### Auto-logging

```python
# scikit-learn
mlflow.sklearn.autolog()
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Automatically logged

# PyTorch
mlflow.pytorch.autolog()

# TensorFlow/Keras
mlflow.tensorflow.autolog()

# XGBoost
mlflow.xgboost.autolog()

# LightGBM
mlflow.lightgbm.autolog()

# HuggingFace Transformers: there is no mlflow.transformers.autolog().
# Use mlflow.autolog() or the transformers MLflowCallback with the Trainer.
mlflow.autolog()
```

### 2. Weights & Biases

```python
import wandb

# Initialize
wandb.init(
    project="recommendation-model",
    name="baseline-v1",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "transformer",
    },
    tags=["baseline", "v1"],
)

# Log metrics
wandb.log({"loss": 0.5, "accuracy": 0.85, "epoch": 1})

# Log plots
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=y_true, preds=y_pred, class_names=class_names
)})

# Log tables
table = wandb.Table(columns=["input", "prediction", "ground_truth"])
for i in range(100):
    table.add_data(inputs[i], preds[i], labels[i])
wandb.log({"predictions": table})

# Log model
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pkl")
wandb.log_artifact(artifact)

wandb.finish()
```

### 3. Experiment Comparison

```python
# MLflow: Query and compare experiments
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9 AND params.model_type = 'random_forest'",
    order_by=["metrics.f1_score DESC"],
    max_results=10,
)

# Compare top runs
comparison = []
for run in runs:
    comparison.append({
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        **run.data.params,
        **run.data.metrics,
    })
comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string())
```

### 4. Reproducibility Setup

```python
import mlflow
import subprocess

def log_reproducibility_info(dataset, context):
    """Log everything needed to reproduce an experiment."""

    # Code version
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    mlflow.set_tag("git_commit", git_hash)

    # Environment
    mlflow.log_artifact("requirements.txt")
    mlflow.log_artifact("conda.yaml")

    # Caller supplies the actual split with its source and digest, during this run.
    mlflow.log_input(dataset, context=context)

    # Random seeds
    mlflow.log_param("random_seed", 42)

    # System info
    import platform
    mlflow.set_tag("python_version", platform.python_version())
    mlflow.set_tag("os", platform.system())
```

### 5. Remote Tracking Server

```bash
# Start MLflow tracking server
mlflow server \
    --backend-store-uri postgresql://user:pass@host/mlflow \
    --default-artifact-root s3://mlflow-artifacts/ \
    --host 0.0.0.0 \
    --port 5000

# Docker deployment
docker run -p 5000:5000 \
    -e MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@host/mlflow \
    -e MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/ \
    ghcr.io/mlflow/mlflow:latest \
    mlflow server --host 0.0.0.0
```

### 6. Nested Runs for Hyperparameter Tuning

```python
with mlflow.start_run(run_name="hpo-sweep") as parent_run:
    mlflow.log_param("search_method", "optuna")

    for trial in study.trials:
        with mlflow.start_run(
            run_name=f"trial-{trial.number}",
            nested=True
        ):
            mlflow.log_params(trial.params)
            mlflow.log_metric("objective", trial.value)

    # Log best result in parent
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_objective", study.best_value)
```

## Best Practices

1. **Log everything** - Parameters, metrics, artifacts, environment, code version
2. **Use auto-logging** as a baseline, add custom logging for specifics
3. **Name runs descriptively** - Include key config in the name
4. **Use tags** for filtering (team, dataset version, experiment phase)
5. **Log data fingerprints** for reproducibility
6. **Use nested runs** for hyperparameter sweeps
7. **Set up a remote server** for team collaboration
8. **Version control experiment configs** alongside code
9. **Clean up** failed/abandoned runs periodically
10. **Standardize metric names** across the team

## Scripts

- `scripts/mlflow_tracker.py` - MLflow tracking wrapper with auto-logging
- `scripts/experiment_compare.py` - Experiment comparison and reporting

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool comparisons.

## Related skills

**Upstream:** `model-training` (runs in parallel — every training job logs params, metrics, artifacts here) · **Downstream:** `model-registry` (the best run gets promoted to a registered version)
**See also:** `ml-pipeline-orchestration` for automatic tracking inside pipelines
