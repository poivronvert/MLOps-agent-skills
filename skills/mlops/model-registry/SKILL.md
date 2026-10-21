---
name: model-registry
description: >
  Manage ML model lifecycle with model registries. Covers MLflow Model Registry, model versioning (semantic, hash),
  model artifact storage, model metadata and tagging, model lifecycle stages (staging, production, archived),
  model promotion workflows, model lineage tracking (data to predictions), model packaging formats (ONNX, TorchScript,
  SavedModel, joblib), model signatures and schemas, model governance and approval, model comparison, rollback
  strategies, CI/CD integration for deployment, and model cards. Use when registering, versioning, promoting,
  or managing model artifacts.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Registry

## Overview

A model registry is the central hub for managing ML model artifacts, versions, metadata,
and lifecycle transitions from development to production.

## When to Use This Skill

- Registering new model versions after training
- Promoting models through staging to production
- Comparing model versions before deployment
- Setting up model governance workflows
- Packaging models for deployment

## Step-by-Step Instructions

### 1. MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model from a run
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "recommendation-model")
print(f"Version: {result.version}")

# Add metadata
client.update_model_version(
    name="recommendation-model",
    version=result.version,
    description="XGBoost model trained on user behavior data v2.1"
)
# Dataset lineage stays on the training run's Dataset inputs.
client.set_model_version_tag(
    name="recommendation-model", version=result.version,
    key="trained_by", value="ml-team"
)

# Promote with aliases (stages API is deprecated since MLflow 2.9)
client.set_registered_model_alias(
    name="recommendation-model",
    alias="staging",
    version=result.version,
)

# After validation passes — moving the alias automatically "demotes"
# whichever version previously held it
client.set_registered_model_alias(
    name="recommendation-model",
    alias="champion",
    version=result.version,
)

# Mark the old version as archived via a tag (aliases have no Archived state)
client.set_model_version_tag(
    name="recommendation-model", version=old_version,
    key="status", value="archived",
)

# Load the current champion for serving
model = mlflow.pyfunc.load_model("models:/recommendation-model@champion")
```

### 2. Model Versioning Strategy

```python
import hashlib
import json
from datetime import datetime, timezone

def generate_model_version(model_path, config, metrics):
    """Generate a unique model version identifier."""
    # Content-based hash
    model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()[:8]

    # Semantic version
    version_info = {
        "major": 2,     # Breaking changes (new architecture)
        "minor": 1,     # New features (added features)
        "patch": 3,     # Bug fixes (retrained with same config)
        "hash": model_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return f"v{version_info['major']}.{version_info['minor']}.{version_info['patch']}-{model_hash}"
```

### 3. Model Packaging

```python
# MLflow model with signature
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(
    model,
    name="model",  # MLflow 3.x: use name= (positional artifact_path is deprecated)
    signature=signature,
    input_example=X_train[:5],
    registered_model_name="recommendation-model",
)

# ONNX export (cross-platform)
import onnx
import torch

dummy_input = torch.randn(1, *input_shape)
torch.onnx.export(
    pytorch_model, dummy_input, "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# TorchScript (PyTorch deployment)
scripted = torch.jit.script(model)
scripted.save("model.pt")

# TensorFlow SavedModel
tf_model.save("saved_model/")

# scikit-learn / XGBoost
import joblib
joblib.dump(model, "model.joblib")
```

### 4. Model Promotion Workflow

```python
def promote_model(model_name, version, target_alias="champion"):
    """Promote model with validation gates."""
    client = MlflowClient()

    # Gate 1: Performance check
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    if metrics.get("test_f1", 0) < 0.85:
        raise ValueError(f"F1 score {metrics['test_f1']} below threshold 0.85")

    # Gate 2: Compare with current champion (if one exists)
    try:
        champion = client.get_model_version_by_alias(model_name, target_alias)
    except Exception:
        champion = None
    if champion:
        prod_run = client.get_run(champion.run_id)
        prod_f1 = prod_run.data.metrics.get("test_f1", 0)
        if metrics["test_f1"] <= prod_f1:
            raise ValueError(f"New model F1 {metrics['test_f1']} not better than champion {prod_f1}")

    # Gate 3: Data validation tag (ModelVersion.tags is a plain dict)
    if model_version.tags.get("data_validated") != "true":
        raise ValueError("Model data not validated")

    # Promote — reassigning the alias replaces the previous holder
    client.set_registered_model_alias(model_name, target_alias, version)
    print(f"Model {model_name} v{version} promoted to @{target_alias}")
```

### 5. Model Lineage

Record Dataset inputs during training/evaluation from the actual splits. Registration
and model cards read those records; they do not recreate inputs or copy dataset summaries
into Run or Model Version tags. See [the data handoff contract](references/ARTIFACT_CONTRACT.md#dataset-input-handoff).
Feature/code metadata and validation gates such as `data_validated` have separate purposes.

```python
# During the active training run, alongside its Dataset inputs:
mlflow.set_tag("features.pipeline_version", feature_pipeline.version)
mlflow.set_tag("features.feature_count", str(len(feature_pipeline.features)))
mlflow.set_tag("code.git_commit", get_git_commit())
mlflow.set_tag("code.git_branch", get_git_branch())
mlflow.log_artifact("requirements.txt")
```

### 6. Model Card Generation

This read-only example covers the model's run and its direct trial children in the
same experiment, including failed/deleted trials. Each run is labeled separately;
fold inputs describe trial usage, not additional rows of the final training set.
The built-in trainer uses this two-level structure. Custom deeper hierarchies need
an explicitly scoped reader. Missing inputs stay unrecorded, even if legacy data tags
exist. Query failures propagate; callers must report failure rather than a complete card.

```python
import html
import json

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


def card_cell(value):
    if value is None or value == "":
        return "unknown"
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", "<br>")


def dataset_sections(client, root):
    runs = [root]
    token = None
    while True:
        page = client.search_runs(
            [root.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{root.info.run_id}'",
            run_view_type=ViewType.ALL,
            max_results=1000,
            page_token=token,
        )
        runs.extend(client.get_run(child.info.run_id) for child in page)
        token = page.token
        if not token:
            break

    sections = []
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        sections.append(
            f"\n### Run {card_cell(name)} ({run.info.run_id})"
            f" — {run.info.status}, {run.info.lifecycle_stage}\n"
        )
        inputs = run.inputs.dataset_inputs if run.inputs else []
        if not inputs:
            sections.append("Dataset inputs not recorded.\n")
            continue
        sections.append(
            "| Dataset / fold | Context | Digest | Rows | Source type | Source |\n"
            "|---|---|---|---|---|---|\n"
        )
        for item in inputs:
            dataset = item.dataset
            tags = {tag.key: tag.value for tag in item.tags}
            try:
                profile = json.loads(dataset.profile or "{}")
            except (ValueError, TypeError):
                profile = {}
            rows = profile.get("num_rows") if isinstance(profile, dict) else None
            if type(rows) is not int or rows < 0:
                rows = None
            values = [
                dataset.name, tags.get("mlflow.data.context"), dataset.digest,
                rows, dataset.source_type, dataset.source,
            ]
            sections.append("| " + " | ".join(card_cell(v) for v in values) + " |\n")
    return "".join(sections)


def generate_model_card(model_name, version):
    """Generate a model card for documentation and governance."""
    client = MlflowClient()
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)

    card = f"""# Model Card: {model_name} v{version}

## Model Details
- **Name**: {model_name}
- **Version**: {version}
- **Created**: {mv.creation_timestamp}
- **Framework**: {run.data.params.get('model_type', 'unknown')}
- **Owner**: {run.data.tags.get('trained_by', 'unknown')}

## Training Data
{dataset_sections(client, run)}

## Performance Metrics
"""
    for metric, value in run.data.metrics.items():
        card += f"- **{metric}**: {value:.4f}\n"

    card += """
## Intended Use
[Describe intended use cases]

## Limitations
[Describe known limitations]

## Ethical Considerations
[Describe ethical considerations]
"""
    return card
```

### 7. Rollback Strategy

```python
def rollback_model(model_name, target_version=None):
    """Rollback the champion alias to a previous model version."""
    client = MlflowClient()
    current = client.get_model_version_by_alias(model_name, "champion")

    if target_version is None:
        # Fall back to the most recent version before the current champion
        all_versions = client.search_model_versions(f"name='{model_name}'")
        candidates = [v for v in all_versions if int(v.version) < int(current.version)]
        if not candidates:
            raise ValueError("No earlier versions available for rollback")
        target_version = max(candidates, key=lambda v: int(v.version)).version

    # Reassigning the alias atomically demotes the current champion
    client.set_registered_model_alias(model_name, "champion", target_version)
    client.set_model_version_tag(model_name, current.version, "status", "rolled_back")
    print(f"Rolled back {model_name} champion to v{target_version}")
```

## Best Practices

1. **Always log model signatures** - Input/output schemas prevent serving errors
2. **Use content-based versioning** - Hash the model for deduplication
3. **Implement promotion gates** - Never promote without automated checks
4. **Generate model cards** - Document every production model
5. **Track full lineage** - Data, features, code, environment
6. **Test before promoting** - Run integration tests at each stage
7. **Keep rollback ready** - Always have a previous version available
8. **Clean up old versions** - Archive/delete unused model versions
9. **Use consistent naming** - `{team}-{task}-{algorithm}` convention

## Scripts

- `scripts/registry_manager.py` - Model registry operations CLI
- `scripts/model_packager.py` - Model format conversion and packaging

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool and format comparisons.

## Related skills

**Upstream:** `model-training` + `ml-experiment-tracking` (candidate artifact with lineage) · **Downstream:** `ml-cicd` (a registered/approved version triggers the promotion pipeline)
**See also:** `references/ARTIFACT_CONTRACT.md` (in this skill) defines what training must hand over and what serving/monitoring may assume · applies equally to fine-tuned LLMs from `llm-fine-tuning`
