# Agent MLOps testing and acceptance

Run from the repository root with Python 3.11–3.12 and uv 0.11.6.
Tests, rubrics and small deterministic fixtures belong in Git. Execution reports,
JUnit outputs, environment receipts, actual agent sessions and generated artifacts
remain local or in CI artifacts. `/docs/validation/agent-mlops/` is ignored;
ordinary tests must work without it. Historical evidence remains in older commits.

## Local regression and runtime consistency

```bash
uv sync --locked --extra forecasting --extra tracking --extra xgboost --extra lightgbm --group dev
uv run --locked --extra forecasting --extra tracking --extra xgboost --extra lightgbm --group dev \
  pytest tests/agent_mlops -m 'not remote'
.venv/bin/python scripts/package_model_training.py --check
.venv/bin/ruff check src/ tests/agent_mlops/
```

The complete extras environment exercises the optional model families. A core-only
environment has a smaller scope; report optional-dependency skips separately.
Keep generated runtime files synchronized with canonical source and dependency locks.
Copied-skill portability tests use temporary installations and environments; they
do not replace the user's installed skills.

`test_evaluation_datasets.py` exercises official MLflow evaluation APIs against a
temporary local SQL store. Its synthetic observations verify storage and scoring
contracts, including raw-evidence hashes and downloaded artifacts. They do not
establish that an acting agent followed a prompt.

## Replay saved capability observations

Keep the evidence directory's `traces.json` and referenced `sessions/*.jsonl` together.
Choose a directory containing reviewed CAP01–CAP12 observations compatible with
`tests/fixtures/agent_mlops/capabilities.json`.

```bash
export AGENT_MLOPS_CAPABILITY_EVIDENCE_DIR="/absolute/path/to/capabilities"
.venv/bin/python -m pytest \
  tests/agent_mlops/replay_capability_observations.py -q
```

This explicitly selected test is outside ordinary `test_*.py` discovery. A missing
setting or evidence file fails the replay rather than silently passing or skipping.
The adapter verifies case identity, raw-file paths and SHA-256 before evaluating;
it requires all 12 saved cases to pass. Replay uses a temporary local SQL store,
does not execute an acting agent, and does not contact a remote tracking server.
Do not present replay as fresh agent acceptance or the SQL store as a production
training fallback.

Fresh agent acceptance requires ordinary prompts, visible inputs, actual assistant
and tool traces, and reviewed evidence locators. Keep missing observations
`unverified` and retain unsuccessful attempts separately.

## Official evaluation dataset CLI

For an explicitly selected evaluation store, publish the version-controlled rubric:

```bash
.venv/bin/python tests/manage_agent_evaluations.py \
  --tracking-uri "$MLFLOW_TRACKING_URI" --experiment-id "$EVAL_EXPERIMENT_ID" \
  publish --fixtures tests/fixtures/agent_mlops/capabilities.json --name capability-cases
```

Use the returned dataset ID and content SHA-256 when evaluating reviewed observations:

```bash
.venv/bin/python tests/manage_agent_evaluations.py \
  --tracking-uri "$MLFLOW_TRACKING_URI" --experiment-id "$EVAL_EXPERIMENT_ID" \
  evaluate --dataset-id "$DATASET_ID" --dataset-sha256 "$DATASET_SHA256" \
  --traces "$AGENT_MLOPS_CAPABILITY_EVIDENCE_DIR/traces.json" \
  --evidence-dir "$AGENT_MLOPS_CAPABILITY_EVIDENCE_DIR"
```

These commands write to the chosen store. The official dataset is mutable: its ID
or native digest is not an immutable version. The adapter checks the selected content
hash and saves the evaluated snapshot, reviewed traces, raw evidence and scorer code
in a separate evaluation Run. Updating rubric fields for unchanged inputs uses
`publish --dataset-id`; changed case inputs require a new dataset. Missing evidence
does not become success. The deterministic scorer checks reviewed annotations and
locators; it does not authenticate traces or judge conversation semantics.

## Remote model acceptance

Provide `AGENT_MLOPS_TEST_TRACKING_URI` as an HTTP(S) platform service, optionally
`AGENT_MLOPS_TEST_EXPERIMENT`, and authentication through the existing environment.
Set `AGENT_MLOPS_CLEAN_CLIENT_PYTHON` to a separate locked client with the necessary
forecasting and tracking dependencies. Preserve the platform's
`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD` setting; presigned storage URLs must be
reachable by the client.

```bash
.venv/bin/python -m pytest tests/agent_mlops/test_remote.py -m remote -q
```

Remote acceptance creates Runs and uploads artifacts. It must verify actual model
download, official loading, prediction parity and Dataset snapshots. A local store,
fault injection or absent service settings cannot establish remote success. The CI
workflow supports explicitly dispatched remote acceptance and uploads JUnit results
as CI artifacts. Do not commit these outputs or treat old Run receipts as current
service acceptance.
