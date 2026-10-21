# Dataset inputs and snapshots

## Responsibilities and defaults

Training/preprocessing code owns split membership, order, features and labels.
Shared `agent_mlops.dataset_tracking` accepts those frames unchanged. The skill
chooses the policy and checks the returned evidence.

Dataset inputs are the primary record of which data a run used. Do not automatically
copy data source/version/row counts/hashes into Run or Model Version summary tags.
The generic wrapper no longer derives `data_version` from `DATA_VERSION`; callers
must associate external versions with specific datasets through immutable source
references and the existing provenance fields. A dataset digest is not a table
snapshot ID, DVC commit or other external version.

Dataset input tags such as `mlflow.data.context`, `data.stage` and `data.sha256`
belong to the input relationship, not to `run.data.tags`. Ordinary ownership/code
tags and validation gates such as `data_validated` retain their separate roles.
Selected snapshot artifacts and their existing manifests still describe downloadable
files; their upload, verification and hash formats are unchanged.

The integrated `model-training` runtime uses:

```yaml
tracking: remote
dataset_inputs: true
dataset_snapshots: [training, validation, test]
```

`dataset_snapshots: []` registers descriptions without uploading split data.
`dataset_inputs: false` disables Dataset inputs while still allowing selected snapshots.
Disable both to omit this additional tracking. `tracking: off` makes no dataset calls.
Metadata-only does not disable existing model input examples or prediction/label values
in reports; it is not a global data-redaction policy.

CLI equivalents are `--dataset-inputs` / `--no-dataset-inputs` and
`--dataset-snapshots training validation test`. Pass `--dataset-snapshots` with no values
to omit snapshots. Use the installed training launcher's `train` command; URI and
authentication remain supplied by the environment.

Each trial owns its fold train/validation inputs. The parent owns final refit and
holdout test inputs. No holdout means no invented test dataset. The parent report's
`tracking.datasets` receipts link all these run IDs. An input recorded immediately
before a failed fit indicates an attempted input, not successful training.

Readers use `get_run(...).inputs.dataset_inputs`. For a complete built-in job, read
the parent plus its direct trial children in that experiment (paginate the search),
label each run/fold and preserve its status. Do not total overlapping fold row counts
or copy child inputs to the parent. The model-registry model card follows this contract.
No inputs means "Dataset inputs not recorded", including when registration was disabled
or an older run only has tags; no tag fallback or backfill is performed. Optional
schema/profile fields may be absent. Failed queries must surface as read failures,
not as empty inputs. Run tag filters do not automatically query dataset metadata.

The built-in boundary is the prediction pipeline input:

- Tabular/event: ordered features plus internal `__label`. The manifest records the
  original target; event labels may be derived from future windows.
- Forecast: entity/time keys, target history and declared feature roles supplied to
  the forecast pipeline. Lag generation, history removal and preprocessing happen
  inside the model. Evaluation frames also contain scoring truth; truth is not passed
  as a known-future feature.

These are not post-transform estimator matrices. Reproducing those requires saved
task, code/environment and fitted preprocessing. Custom training needing matrices
should capture them at its estimator boundary and explicitly name that stage.

## Shared Python API and acceptance

Run in the packaged training environment or with `agent_mlops` installed:

```python
from agent_mlops.dataset_tracking import DatasetSplit, record_split

# These must already be the aligned, actual inputs. Do not silently realign labels.
assert X_train.index.equals(y_train.index)
assert "label" not in X_train.columns
frame = X_train.copy()
frame["label"] = y_train
receipt = record_split(
    client, run_id,
    DatasetSplit(
        name="fold-0-train", context="training", frame=frame, target="label",
        source=immutable_source_uri, source_sha256=verified_source_sha256,
        stage="pipeline_input",
    ),
    log_inputs=True, snapshot=True,
)
```

Supply a verified upstream SHA-256 and a credential-free source reference. For a
database/lake table, pin an immutable extract and record its query, table version
and extraction boundaries in `provenance`; a mutable table name is insufficient.
The source is upstream lineage, not a promise that loading it returns the derived split.

The helper uses `MlflowClient.log_inputs`, full-frame Arrow IPC SHA-256 including
index/schema/order, and optional Parquet artifacts. It reads back name, schema,
digest, source and context, then downloads snapshots, checks their byte SHA-256,
and compares frames including index/dtypes/order. It also reads back the manifest.
Arrow/Pandas versions affect serialization; use the recorded environment.
MLflow's shortened digest differs from both full-content and artifact hashes.

Receipts include `snapshot_artifact`, `snapshot_sha256`, `frame_sha256`, source,
context, row count, target and schema. Files live under
`datasets/<name>/<frame_sha256>/`; metadata-only leaves only a manifest there.
Failures stop subsequent computation. Partial remote writes may remain; inspect
them before an explicitly initiated retry.

## Production updates and retraining

| Layer | Destination / responsibility |
|---|---|
| Operational data | Ingestion writes a database, object storage or lake; retain event time, arrival time and source identity. |
| Dataset version | Pin object versions, lake snapshots or DVC/LakeFS commits; track corrections and late labels. A version does not require training. |
| Serving | Prediction services consume fresh features; an online feature store is optional for shared low-latency features. |
| Monitoring | Check distributions and, when labels arrive, quality; record model version and data window. |
| Retraining decision | An orchestrator combines schedules/events, mature labels, quality gates, cooldown and budgets to launch a job on a pinned version. |
| MLflow | Records that job's inputs, metrics and models; promotion and deployment remain separate decisions. |

For example, observations may land daily and be validated/versioned and used for
prediction immediately, while retraining runs weekly or after confirmed quality
decline. A trigger should identify dataset version, window and reason; deduplicate
the request and evaluate the candidate before promotion.

MLflow Dataset inputs do not watch production tables or allocate training compute.
Webhooks notify external systems for supported registry/gateway events; they do not
make data ingestion a built-in retraining scheduler. Use available `data-ingestion`,
`data-validation`, `model-monitoring` / `model-drift-detection`, and
`ml-pipeline-orchestration` skills when implementing these parts. Do not deploy
infrastructure merely to add run lineage.

Primary references:
[Dataset tracking](https://mlflow.org/docs/latest/dataset/),
[MLflow client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html),
[Webhook events](https://mlflow.org/docs/latest/self-hosting/webhooks/).
