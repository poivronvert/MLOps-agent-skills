# Built-in training entrypoint: use, extend and report

This reference covers the training entrypoint bundled with this skill (`scripts/run.py train`)
and how to work beyond it. For framework-independent temporal design (prediction time,
feature availability, label maturity, split choice) see [Temporal validation](TEMPORAL_VALIDATION.md).

The entrypoint is a reusable starting point, not a list of what the agent may do. A request
that it does not cover is a reason to compose or implement, not to refuse or to quietly
shrink the goal. Ask the user only about choices that change the goal, scope or permissions.

## 1. Decide what can run as-is

Query the implementation before promising a shipped workflow:

```bash
python3 /absolute/path/to/model-training/scripts/run.py train --describe-builtins
```

`agent_mlops.capabilities.describe_builtins()` derives this inventory from the code: modes,
tasks, models, input formats, split and baseline behavior, forecast strategy and prediction
interface. It does not train, check the current environment or verify any run.

Summary of what is built in:

| Mode | Task | Models | Development evaluation | Final test |
|---|---|---|---|---|
| `tabular` | classification, regression | rf, svm, xgb, lgbm | shuffled (stratified) K-fold | optional random holdout |
| `future_event` | classification of an event in `(t, t+window]` | rf, svm, xgb, lgbm | configured time origins, mature labels only | rows from `test_start` |
| `forecast` | regression, recursive multi-step | rf | configured origins | one origin at `test_start` |

For each requested behavior, state which of these applies:

- **Runs now** through the entrypoint.
- **Needs additional work**: data preparation, composition, independent evaluation, or new
  implementation. Describe how it will be validated.
- **Planned / executed / verified**: keep these states separate in what you tell the user.

Check the whole proposal, not just the estimator. Examples of additional work:

- "Reissue a 24-hour forecast every hour through the test period with new observations"
  differs from the built-in single forecast issued at `test_start`; the prediction interface
  has no history-update method (`history_update_interface: null`).
- A previous-day same-hour baseline, target rolling statistics, or equal-weight entity
  reporting are not part of the entrypoint. Extra reporting is also different from changing
  the model-selection objective.

When checking APIs, use the installed versions' source and signatures and small experiments;
web examples are not executed evidence. The bundled lock describes this entrypoint only;
custom project work can declare its own reproducible dependencies.

## 2. Intake: compute facts, ask about meaning

Work from what is already available: a continuing conversation, a data file, a data
dictionary or other documents. Neither a formal EDA, a solution-design pass nor an internal
schema from the user is a prerequisite. Record a source locator (document and line, or
conversation turn) for each material semantic decision.

- **Compute facts yourself**: columns, types, missing fractions, entity coverage, time
  ranges. `agent_mlops.data.read_snapshot` and `inspect_data` implement the training-side checks.
- **Ask about material unknowns** the data cannot settle: which target or event, sample
  grain, prediction time, window meaning, feature availability and evaluation boundaries.
  A CSV alone does not establish how faults are generated; see Temporal validation §1 for
  the ambiguity in "one hour ahead".
- **Exclude** a nonessential ambiguous field with a recorded reason. An ambiguous target, a
  conflict between sources, or a feature unavailable at prediction time blocks fitting until
  resolved; list it in `unresolved`, which stops execution.
- **Separate** missing implementation from data problems. An alternative that changes the
  user's target or strategy needs their choice; implementing the requested behavior within
  the existing authorization does not.

Defaults are descriptive, not business decisions: weighted F1 for classification, RMSE for
regression, and a fixed 0.5 event threshold with an explicit positive label. Ask about
false-positive and false-negative costs when an operating threshold is requested. Only
development data may inform thresholds, features or scoring changes.

When a file changes, compute a new SHA-256 and recheck schema, types, time coverage,
availability and splits; keep semantics that still apply. A current user correction
overrides an older document: record the old source and the reason, and do not edit the
source dictionary or old results. Schema validation cannot prove business meaning.

## 3. Run the entrypoint

Resolve `scripts/run.py` beside the loaded `SKILL.md` and call it by absolute path from the
user's working directory, where relative input, config and output paths resolve. The skill
carries its own `runtime/` source, `pyproject.toml`, `uv.lock` and checksum manifest; no
repository checkout, `PYTHONPATH` or package in the user's project is needed.

Prerequisites: Python 3.11 or 3.12 and uv **0.11.6 or newer** (CI validates 0.11.6). The
launcher finds uv on PATH, then `~/.local/bin/uv`, or uses `UV_BIN`. It defaults to Python
3.12; put `--python 3.11` before the command to select 3.11. The first call may download
locked dependencies and Python.

```bash
TRAIN_SKILL="$HOME/.agents/skills/model-training"   # the actual loaded skill path
python3 "$TRAIN_SKILL/scripts/run.py" train \
  --input data.csv --target label --model-type rf --task classification \
  --no-hpo --fixed-params '{"n_estimators":100}' --holdout-fraction 0.2 --output output/run-01

python3 "$TRAIN_SKILL/scripts/run.py" --extra forecasting --extra tracking train \
  --config task.yaml --output output/run-02

python3 "$TRAIN_SKILL/scripts/run.py" python -c \
  'import mlflow.pyfunc; m = mlflow.pyfunc.load_model("output/run-01/model"); print(type(m).__name__)'
```

- **Extras** go before `train` or `python`: `forecasting` (recursive forecasts), `xgboost`,
  `lightgbm`, `tracking` (skops, for the experiment-tracking wrapper). MLflow is core; RF/SVM
  need no extra. Selected extras are recorded. If a config's `extras` asks for more than the
  launcher selected, execution stops and names the needed flags. Group `dev` is always
  included so the environment matches the validated one.
- **Isolation.** The launcher verifies bundled file hashes, then runs
  `uv run --locked --project <runtime>` in an environment under
  `${XDG_CACHE_HOME:-~/.cache}/agent-mlops/envs/`, keyed by install path, bundle, Python and
  extras. The user's project and the installed skill are not modified. `--locked` means an
  incompatible uv stops instead of re-resolving. A damaged or modified bundle is rejected;
  reinstall it.
- **Extending.** Keep new code in the user's workspace. `run.py python your_script.py` runs
  it in the locked environment with `agent_mlops` importable (for example
  `agent_mlops.temporal` or `agent_mlops.dataset_tracking.record_split`). To change the
  runtime itself, copy `runtime/src/agent_mlops` into the workspace project under its own
  dependency definition, and report results as coming from that code.
- **Reproducibility.** Locks do not promise identical numbers across hardware. Reload checks
  use absolute tolerance 1e-10 and relative tolerance 0. `environment.json` records Python,
  uv, extras, lock hash and the hash of the `agent_mlops` package files only.

## 4. Configuration

`--config` accepts JSON or YAML; `agent_mlops.config.TaskSpec.model_json_schema()` gives the
exact schema. Unknown keys are rejected, CLI flags override config values, and file contents
are hashed and read from the same verified bytes. Include `data_sha256` in configs (the
legacy CLI computes it and rechecks before training). New runs require an empty output
directory.

Event prediction example (fill the hash from the checked input):

```yaml
input: devices.csv
data_sha256: <sha256 of checked file>
target: fault
sample_grain: one device per hour
task: classification
mode: future_event
features: [temperature]
availability: {temperature: 0h}
history_lags: {temperature: [1, 2]}
rolling_windows: {temperature: [4]}
provenance:
  target:
    value: fault
    source: conversation/current confirmation
    status: confirmed
assumptions: [Fixed 0.5 threshold is a baseline, not an operational alarm policy]
positive_label: 1
threshold: 0.5
temporal:
  time_column: timestamp
  entity_keys: [device_id]
  timezone: Asia/Taipei
  frequency: h
  origins: ['2025-01-10T00:00:00+08:00', '2025-01-12T00:00:00+08:00']
  validation_steps: 24
  test_start: '2025-01-15T00:00:00+08:00'
  window: 24h
  reporting_delay: 2h
  event_value: 1
  history_policy: exclude
model: rf
fixed_params: {n_estimators: 100}
hpo: true
search_space:
  max_depth: {type: int, low: 2, high: 12}
n_trials: 3
timeout: 60
metric: f1_weighted
direction: maximize
tracking: off
extras: []
groups: [dev]
```

How the temporal modes behave:

- **Time handling.** Naive times use the declared timezone; ambiguous or nonexistent DST
  times are rejected. Comparisons and outputs are UTC. Duplicate entity/time rows, missing
  periods and irregular frequency are rejected for explicit upstream handling, and all
  entities must share one time grid.
- **Cutoffs** are common across entities. Training rows precede each cutoff, and event labels
  must have matured by it (`t + window + reporting_delay <= cutoff`). The label is exactly an
  event in `(t, t+window]`; windows running past the data end are unknown, not negative.
  Development validation labels must mature before `test_start`. The final refit excludes
  the holdout and immature labels.
- **Origins.** A forecast origin is the first predicted timestamp; the last observed row is
  one period earlier. The holdout starts at `test_start`; an event `test_end` is inclusive.
- **History features.** Lags stay within an entity and must be available by the row's
  prediction time (lag × frequency ≥ availability delay). Rolling windows end at the most
  recent available observation, including the current row when the delay is zero. Rows with
  insufficient history follow `history_policy` (`reject` or `exclude`) and are counted.
- **Preprocessing.** Encoders, imputers, scalers, optional `select_k` and estimators are fit
  on fold training rows only. Unknown categories map to all-zero one-hot columns; schema,
  column order and class mappings are saved. Text regression targets are rejected.

For forecasting, set `mode: forecast`, `task: regression`, the measured value as `target`,
`metric: mae`, `direction: minimize`, model `rf` and the `forecasting` extra, with roles
assigned explicitly:

```yaml
temporal:
  # ...time keys, timezone, frequency, origins, test_start as above...
  strategy: recursive
  horizons: [1, 6, 24]
  lags: [1, 24]
  static_features: [capacity]
  known_future: {hour_of_day: 'published calendar, known at origin'}
  historical_features: {temperature: [48]}
```

- Static fields must be constant within an entity.
- `known_future` entries are semantic claims the agent must check; being numeric proves
  nothing. Never pass observed future sensor values as known-future inputs.
- Historical external lags must cover the maximum horizon plus availability delay; shorter
  lags need a real future source or a different feature design.
- Recursive target lags use earlier predictions, never future truth.
- At prediction time the model needs unique entity/time keys and known-future columns for
  **every step** through the maximum horizon; only requested horizons are returned, row order
  does not matter, and missing values are reported before forecasting. Without known-future
  features it accepts an empty DataFrame.

Scoring: each fold and horizon pools eligible entity samples; horizons are averaged equally,
then folds equally. Missing or nonfinite required groups invalidate a trial. Reports include
group scores and counts, actual and predicted values, and time labels. Baselines on the same
samples: training majority (classification), training mean (regression), last observed value
per entity (forecast). A model need not beat its baseline, but say whether it did.

If visible history shows the test period guided any change, set
`holdout_used_for_selection: true`; results are then marked development-only. Unknown prior
use of the test period is disclosed as unknown.

## 5. Outputs, loading and tracking

A run directory contains `model/`, `task_spec.json`, `environment.json` and
`training_report.json`. The report states `training_status` and `artifact_status`; model
save failure sets `artifact_status=failed`, `model_path=null` and a nonzero exit. Partial
models are labeled and never published as final selected models.

`model/` is an official MLflow Model (Models From Code `PythonModel` wrapping one
`model.joblib` with the fitted pipeline, signature excluding the target, and an input
example). Load it with `mlflow.pyfunc.load_model(uri_or_path)` or
`mlflow models predict --model-uri ... --input-path ... --content-type csv --env-manager local`
in a matching environment; no repository import is required. Outputs:

- Classification: `prediction` plus `probability_0`, `probability_1`, … in the order given by
  `label_mapping` metadata. Event models keep threshold, class order and needed history.
- Regression: a NumPy array.
- Forecast: keyed prediction and horizon columns; stored history keeps only inference columns.

Numeric and timezone-aware columns use MLflow `AnyType` where scalar coercion would reject
int64/NaN or strip time zones; the prediction code validates them. Artifacts in older formats
are not converted.

**Tracking.** `tracking: off` (default) saves the model locally with no tracking store.
`tracking: remote` needs `MLFLOW_TRACKING_URI` set to HTTP(S) with authentication from the
environment or provider; URI credentials, query strings, fragments and file/SQLite stores are
rejected. The service is checked before fitting, and no server is started. Each trial is a
child run of the parent; the parent logs the model with `mlflow.pyfunc.log_model` and the
returned model URI. Registry registration and aliases are separate steps.

On a tracking SDK failure, no further fits, evaluations, trials or refits start; a calculation
already finished remains finished, and a successfully saved local model is kept. There is no
automatic retry or replay, remote runs may be left open, and the local report is
authoritative. Errors are sanitized and credentials are never written. Preserve
`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD`; report transport failures rather than disabling
multipart or retraining.

**Dataset inputs.** Remote runs register each actual split as an MLflow Dataset input
(`dataset_inputs: true`) and upload Parquet snapshots (`dataset_snapshots:
[training, validation, test]`; `[]` keeps metadata only). Trials own fold inputs, the parent
owns refit and holdout. Snapshots are pipeline inputs, not post-transform matrices. A
recording or read-back failure stops the next fit or prediction. Custom training should pass
its actual frames to `agent_mlops.dataset_tracking.record_split` or an equivalent verified
implementation. Details, reader rules and the API are in the `ml-experiment-tracking`
skill's `references/DATASET_TRACKING.md`; the registry handoff is in
[ARTIFACT_CONTRACT.md](../../model-registry/references/ARTIFACT_CONTRACT.md#dataset-input-handoff).

## 6. Custom work and execution records

Workspace scripts may normalize time fields, build features known at prediction time,
assemble configs, plot, and run authorized extra evaluation. For independent evaluation, load
the trusted model through its interface (`mlflow.pyfunc.load_model(model_uri).predict(...)`),
write separate outputs, and confirm the original model and report files are unchanged. If the
interface cannot express a needed history or state update, implement it explicitly and test it.

Changing splits, labels, fit boundaries or scoring is substantive development: write explicit
code with tests. Do not patch `training.splits` or `training.evaluate` at runtime to extend
the shipped pipeline; results from such code do not inherit the entrypoint's validation.

For custom code that affects a result, keep a reproducible record next to the scripts, config
and outputs, for example `execution_record.json`:

- Role: preparation, independent evaluation, or core implementation.
- Exact command, working directory, and frozen copies of the scripts with SHA-256 hashes.
- Config snapshot, input hashes, model URI if used, package versions or lock identity, and
  output hashes.
- Execution status (`not_run`, `completed`, `failed`) separate from validation status
  (`unverified`, `passed`, `failed`), with the checks run and log locations.

`scripts/execution_record.py` (standard library only) produces this record: `run` executes the
command, keeps logs and frozen script/config copies, and hashes inputs, outputs and locks;
`plan` records work not yet run; `check` attaches a validation command, and validation becomes
`passed` only when the run completed and every check exited 0; `verify` re-hashes everything
later. It records installed packages only from an interpreter passed with `--python`, never
guessed from the command, since wrappers such as `run.py python`, `uv run` or `conda run`
execute code elsewhere; for the bundled runtime, pass `--lock` with its `runtime/uv.lock`.
Any equivalent record is fine; the fields above are what matters.

Never put secrets in it. A syntax check does not verify an unexecuted run. `environment.json`
does not cover workspace scripts. Report original training and custom results separately;
preparation-only work is not a trained model and cannot borrow evidence from another run.

## 7. Reporting to the user

State the task and window, metric and split, model and output location, assumptions and
limitations. Then report training status, artifact completeness and tracking status
separately. On failure, give the failing stage and what was kept. Do not claim deployment,
cleanup, persistence beyond what the platform guarantees, or a remote URI that was not returned.

Repository maintainers: executable acceptance checks live in `tests/agent_mlops` and are
described in `docs/testing/agent-mlops.md`; they are not installation prerequisites.
