# Agent training: intake, execution and delivery

Read this before planning training or evaluation, including work beyond the shipped entrypoint.
The internal `TaskSpec` is assembled by the agent; users and other skills need not
produce this schema or invoke an analysis skill first.

## Existing implementation, additional work, and evidence (CAP01–CAP12)

The built-in entrypoint is a reusable starting point, not an allowlist of the agent's
methods. Absence from SKILL.md, TaskSpec, the launcher, or existing tests does not establish
that a request is impossible. Assess how to accomplish it with appropriate APIs and code.
Prefer an existing path when it meets the goal; implement and validate additional work
within the user's authorization. Only ask for choices that materially change the goal,
scope, or actual permissions; do not require approval for each script or repeat consent.

Before promising a shipped workflow, query its implementation contract:

```bash
python3 /absolute/path/to/model-training/scripts/run.py train --describe-builtins
```

Resolve the path from the loaded skill. `agent_mlops.capabilities.describe_builtins()`
is the canonical inventory of this entrypoint, deriving schema choices and input formats
from the implementation. It does not train, judge custom feasibility, attest that the
current environment works, or declare any run verified. The detailed examples below
explain how to use it, not a separate list of everything the agent may do.

For each requested behavior, distinguish:

- The existing execution path that can deliver it now.
- Preparation, composition, independent evaluation, or core implementation still needed.
- What has only been planned, actually executed, or verified by specific evidence.

For example, "one issued forecast covering the next 24 hours" differs from "reissue a
24-hour forecast every hour throughout the test period, incorporating newly available
observations." Explain that difference before promising the latter. The query identifies
the current final-test scope separately from development validation. A package's update
or cross-validation API is a development resource, not proof that the shipped workflow
performs those actions. If proposing additional implementation, describe its validation
and preserve the requested behavior; do not default to refusing it or silently shrinking it.
Accepting a business proposal does not make missing code already implemented.

Check the whole proposal, not just the estimator and forecast strategy. In a forecast
plan, a previous-day same-hour baseline, target rolling statistics, or equal-entity
aggregation needs its own implementation attribution; naming RF does not make those
operations part of the existing entrypoint. Identify additional transforms and reporting
as such, and distinguish extra reporting from a changed model-selection objective.
These are valid development choices, not reasons to reject the request.

Use local version information, source, signatures, and targeted experiments when checking
APIs; consult official documentation when useful and available, matching the installed
version. Web examples are not executed evidence. The bundled dependency lock describes
this entrypoint, not every method an agent may ever implement. Keep changes to project
dependencies explicit and reproducible within the authorized development scope.

### Preparation and custom execution

Workspace scripts may normalize time fields, generate features known at prediction time,
assemble task settings, draw plots, and perform authorized additional evaluation. Record
input identity, transformation, availability assumptions, output identity and provenance.
Compute observable facts yourself; ask only for material unknown semantics. A CSV alone
does not establish the fault-generation rule, and a data dictionary is not mandatory.

For independent evaluation, load the trusted model through its applicable interface
(`mlflow.pyfunc.load_model(model_uri).predict(...)` for official MLflow models), write separate outputs,
and verify that original model/report files remain unchanged. Keep time availability,
label and holdout semantics intact. A new evaluation need not be integrated into the
built-in training entrypoint before it can run. If the interface cannot express the
requested history or model-state updates, implement that behavior explicitly and test it.

Changes to splits, labels, fit boundaries, or scoring semantics are substantive development,
even when installed files remain unchanged. Use explicit code and applicable tests; do
not replace global `training.splits` / `training.evaluate` at runtime as a way of extending
the shipped pipeline and inheriting its acceptance. This is an execution and reporting
contract, not a security sandbox or a demand to scan all Python for monkey patches.

For custom code that affects a result, retain a reproducible workspace record, for example
`execution_record.json`, alongside the scripts, config, outputs and validation logs:

- Role: preparation, independent evaluation, or core implementation.
- Exact entry command, working directory, script/helper copies and their SHA-256 values;
  freeze these copies for the run so later edits do not misidentify executed code.
- Configuration snapshot, input hashes and official model URI when used,
  relevant package versions or lock identity, and result hashes.
- Execution status (`not_run`, `completed`, or `failed`) separate from validation status
  (`unverified`, `passed`, or `failed`), plus checks and actual log/result locations.

This is a suggested record shape, not a schema the user must prepare. Do not place secrets
in it. Preserve planned records as planned; passing a syntax check does not verify an
unexecuted training/evaluation. The bundle's `environment.json` explicitly scopes
`source_sha256` to package files; it cannot cover workspace scripts or prove their behavior.
Report original training and custom results separately. Preparation-only temperature
work is not a trained model and cannot borrow evidence from a previous fault classifier.

## Reuse evidence and resolve only material gaps (A01–A11)

Accept either a continued conversation with prior analysis or a file plus a request.
Reuse visible, still-applicable confirmations, data dictionaries and ordinary documents;
record a source locator for each material semantic decision. Do not request a full EDA,
formal report, external plugin or a particular predecessor skill when the inputs suffice.
A clear existing classification label permits a baseline immediately after data checks.

Distinguish these next actions:

- Compute facts such as columns, types, missing fractions, entity coverage and time ranges
  with the available data tools. Do not ask users to calculate them. `read_snapshot` and
  `inspect_data` in `agent_mlops.data` provide the training-side checks.
- Ask about material semantic unknowns: which event/target, sample grain, prediction time,
  window meaning, availability of features, and evaluation boundaries. “One hour ahead”
  could mean state at t+1h, an event in (t,t+1h], or minimum warning time. Do not choose
  between these interpretations without evidence. A `ready` flag cannot substitute for checks.
- Exclude a nonessential ambiguous field with a recorded reason. An ambiguous target,
  conflict, or unavailable prediction feature blocks fit until resolved.
- Distinguish missing implementation from data problems. Query the built-in contract,
  assess additional implementation for unmet requirements, and explain the path and needed
  validation. An alternative that changes the user's target or strategy requires their
  choice; implementing the requested behavior within existing authorization does not.

A baseline defaults to weighted F1 for classification and RMSE for regression, recording
that these are descriptive defaults, not business tradeoffs or launch thresholds. Clarify
false-positive/false-negative costs when the user asks for an operating threshold. Event
classification uses a fixed threshold (default 0.5); record its assumption and positive label.
Only development data may inform changes to thresholds, features or scoring.

For a changed file, calculate a new SHA-256 and recheck affected schema, types, time coverage,
availability and splits. Retain still-applicable semantics. Preserve an explicit current
user correction over an older document, recording the old source and override reason;
do not edit the source dictionary or old results. Unresolved material conflicts go in
`unresolved` and prevent execution. Schema validation cannot prove business semantics.

## Installed skill and locked environment

Use the absolute path to `scripts/run.py` beside the loaded `SKILL.md`. Stay in the
user's working directory: relative input, config and output paths resolve there.
The skill carries `runtime/src/agent_mlops`, `runtime/pyproject.toml`, `runtime/uv.lock`
and a checksum manifest. It does not need a clone, a repository-relative command,
`PYTHONPATH`, or a package installed into the user's project.

Prerequisites: Python 3.11–3.12 and uv **0.11.6**. The launcher finds uv on PATH,
then `~/.local/bin/uv`, or uses an explicit `UV_BIN`. It defaults to Python 3.12;
put `--python 3.11` before the command to select 3.11. The first call may download
locked dependencies (and the selected Python if uv cannot find it locally).

Example for a user-level installation (substitute the actual loaded skill path):

```bash
TRAIN_SKILL="$HOME/.agents/skills/model-training"
python3 "$TRAIN_SKILL/scripts/run.py" train \
  --input data.csv --target label --model-type rf --task classification \
  --no-hpo --fixed-params '{"n_estimators":100}' --holdout-fraction 0.2 --output output/run-01

python3 "$TRAIN_SKILL/scripts/run.py" --extra forecasting --extra tracking train \
  --config task.yaml --output output/run-02
```

Place extras **before** `train` or `python`: `forecasting` for recursive forecasting,
and `xgboost` / `lightgbm` for those estimators. MLflow is a core dependency. Core RF/SVM need
no extra. The selected extras are recorded in the task/environment artifacts. If
`extras` in a config requests more than the launcher selected, execution stops with
the needed flags. Group `dev` is included consistently with the validated environment.

The launcher verifies bundled file hashes and uses `uv run --locked --project` with
an environment under `${XDG_CACHE_HOME:-~/.cache}/agent-mlops/envs/`. It preserves
cwd and keeps dependencies out of both the user's project and the installed skill.
An environment is specific to the installed path, bundle, Python and extras. For this
entrypoint, use its launcher rather than plain `uv run` in the caller's project; retain
`--locked` and do not silently change its dependencies. A damaged bundle needs reinstalling.
Explicit custom project development can use its own reproducible dependency definition;
the bundled lock is not a permanent restriction on that project's methods or packages.

Use the same launcher and extras for inspection or reload:

```bash
python3 "$TRAIN_SKILL/scripts/run.py" python -c \
  'import mlflow.pyfunc; model = mlflow.pyfunc.load_model("output/run-01/model"); print(type(model).__name__)'
```

Lockfiles do not promise identical numerical results across hardware. Reload tests use
absolute tolerance 1e-10, relative tolerance 0. Record actual Python/packages, code content
hash and lock hash. Repository maintainers update the canonical lock/source and regenerate
the bundled runtime; installed users do not need the development or packaging scripts.

## Internal configuration

JSON or YAML is accepted by `--config`. `agent_mlops.config.TaskSpec.model_json_schema()`
provides the exact schema. Unknown keys are rejected. Existing CLI flags override the
corresponding config values. File contents are hashed and read from the same verified byte
snapshot. Config files should include `data_sha256`; legacy CLI records the initial hash
and verifies it again before training. New runs require an empty output directory.

Example event settings (agent fills the hash from the checked input):

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

Time cutoffs are common across entities. Training rows precede each cutoff; event labels
must also mature by it. The label is exactly `(t, t+window]`; incomplete end windows remain
unknown. Development validation labels must mature before the final test starts. Forecast
origin means the first predicted timestamp; the last observed row is origin minus one period.
The holdout starts at `test_start`; an event `test_end` is inclusive. Forecast evaluation
uses the requested horizons at that origin. Duplicate, irregular or missing periods are
rejected for explicit upstream handling. Naive times use the declared timezone; ambiguous
DST timestamps are rejected. All comparison and output times are UTC.

Lags stay within an entity. A lagged value must be available by the row's prediction time.
Rolling windows end at the most recent available observation, including the current row
when delay is zero. Insufficient history follows `reject` or `exclude` and is counted.
Encoders, imputers, scalers, optional `select_k` and estimators fit only fold training rows.
Unknown categories map to all-zero one-hot columns; numeric/categorical schema and order
are saved. Text regression targets are rejected. Class mappings are fitted and persisted.

For forecast, change `mode: forecast`, `task: regression`, `target` to the measured value,
and `metric: mae`, `direction: minimize`. Use RF and the `forecasting` extra. Set
`temporal.strategy: recursive`, `temporal.horizons: [1, 6, 24]`, and `temporal.lags: [1, 24]`.
Assign roles explicitly:

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

Static fields must remain constant within an entity. Future-known sources are semantic
assertions that the agent must check; a numeric column does not prove future availability.
Historical external lags must cover the maximum horizon plus availability delay; shorter
lags need an explicit future source or a different supported feature design. The stored
history supplies these covariates. Recursive target lags use earlier predictions, never
origin-future truth. `ForecastBundle.predict(future_frame)` accepts unique entity/time keys
and known-future columns covering **all intermediate steps** through the maximum horizon;
only requested horizons are returned. Row order does not affect key alignment. Do not pass
observed future sensor data as a known-future source. Preflight checks required columns and
roles; missing forecast values are reported before the forecast call.

Every fold/horizon pools the eligible entity samples for the metric; horizons are averaged
equally, then folds equally. Missing/nonfinite required groups invalidate the trial. The
report includes group scores/counts, actual/predicted values and time labels. Classification
baselines use the training majority, regression uses the training mean, forecast uses the
last observed value on the same entity/horizon samples. A model need not beat the baseline.
The final refit excludes holdout and immature labels. If visible history shows test-guided
changes, set `holdout_used_for_selection: true`; results are marked development-only.
Unknown prior test use remains disclosed; there is no invented global holdout history.

## Tracking and complete artifacts

Remote training defaults to `dataset_inputs: true` and
`dataset_snapshots: [training, validation, test]`. These independent controls register
each actual split and upload selected Parquet snapshots. Set `dataset_snapshots: []`
for metadata only; `dataset_inputs: false` disables registration. CLI equivalents are
`--dataset-inputs` / `--no-dataset-inputs` and `--dataset-snapshots` followed by the
selected contexts (no values means no snapshots). Tracking off performs no dataset calls.

Trial runs own their fold train/validation inputs; the parent owns final refit and
any holdout test. `tracking.datasets` receipts link all run IDs, contexts, target/schema,
upstream source/hash, full-frame hash and optional snapshot path/byte hash.
Dataset inputs are the primary usage record; do not duplicate their summary in Run
or Model Version tags. Model-card readers query the parent and its direct trial children
in the same experiment, with pagination and run/fold/status labels. They do not sum
overlapping fold counts. Disabled/absent inputs stay unrecorded without tag fallback
or backfill; a lookup failure is reported separately. See
[the handoff contract](../../model-registry/references/ARTIFACT_CONTRACT.md#dataset-input-handoff).
`agent_mlops.dataset_tracking.record_split` accepts caller-selected frames unchanged,
uses `MlflowClient.log_inputs`, and reads back metadata and downloaded snapshots.
A recording/readback failure stops the next fit/predict.

Snapshots capture pipeline inputs, including derived event labels and forecast history,
not post-transform estimator matrices. Fitted preprocessing and task/environment
artifacts are still needed for reproduction. Metadata-only does not redact existing
model input examples or predictions/labels in reports.

For custom training callers, when `ml-experiment-tracking` is available read its
`references/DATASET_TRACKING.md` for the shared dataset recording API.

`tracking: off` is default: official MLflow models are saved locally without starting a tracking run or store. For `remote`, set `MLFLOW_TRACKING_URI`
to HTTP(S), and supply authentication through the environment/provider. URI userinfo,
query parameters, fragments and file/SQLite stores are rejected. Service preflight occurs
before fit. The SDK checks service/authentication responses; it does not assume every server
uses a token or require a particular credential variable. No server is started here.

Each sequential trial has a parent work run, parameters, fold/objective metrics and failure
reasons. An SDK failure stops further fits, prediction evaluations, trials and refits. A
completed result remains a completed calculation even when upload fails. The active
calculation may return; no distributed force-cancel or automatic replay is claimed. Remote
errors are sanitized; secrets/credential URLs are not persisted. Already-open remote runs
may remain running after a connection failure; the local report is authoritative and there
is no automatic retry to repair them.

`model/` is an official MLflow Model with `MLmodel`, inference code, dependencies,
signature and an input example. New artifacts use a Models From Code PythonModel with
one `model.joblib` artifact containing the fitted predictor, including preprocessing.
The PythonModel loads it through `joblib.load` and preserves entity/time, future-input
and horizon semantics. Forecast history retains only the existing inference columns;
saving does not mutate the live predictor or refit it. Event models retain threshold,
class order and needed history.
Classification returns `prediction` plus available `probability_0`, `probability_1`, …;
`label_mapping` metadata defines column order. Regression returns a NumPy array; forecast
retains keyed prediction/horizon columns. Signature inputs exclude the truth target.
Forecast without known-future features accepts an empty DataFrame; no future target is required.
Numeric and timezone-aware temporal columns use official AnyType where MLflow scalar
coercion would reject int64/NaN or strip timezones; prediction code validates their meaning.

Load with `mlflow.pyfunc.load_model(model_uri_or_local_model_directory)` or use
`mlflow models predict --model-uri ... --input-path ... --content-type csv --env-manager local`.
Use the appropriate locked dependency environment. No repository import or custom loader
is required from the caller. New artifacts expose the pyfunc flavor; native sklearn or
MLForecast flavor loaders are not their loading interface. Existing saved artifacts are
not rewritten. There is no old-format compatibility or conversion path.
`task_spec.json`, `environment.json` and `training_report.json` are separate run reports;
their absence does not prevent official model loading. Provenance hashes remain evidence,
not a claim that the official loader implements the former per-file integrity check.
Partial models are labeled by the report and are never published as final selected models.

Remote training uses official `mlflow.pyfunc.log_model`, returning its model URI and linking
the parent training run. Registry registration and aliases remain separate operations.
A failed publish retains a successfully saved local model and reports tracking failure;
there is no automatic retraining or publication replay. Model save failure sets
`artifact_status=failed`, leaves `model_path=null`, and causes a nonzero CLI exit.
Report updates do not serialize the model again.

Preserve `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD`; presigned URLs must be client-reachable.
Report transport failures without disabling multipart or replaying training.

User summaries must state task/window, metric and split, model/output location, assumptions
and limitations, then distinguish training status, artifact completeness and tracking status.
On failure report the actionable stage and retained artifacts. Do not infer deployment,
container cleanup, persistence beyond platform guarantees or a remote URI that was not returned.

## Acceptance evidence

Executable checks live in the source repository's `tests/agent_mlops`; testing and
local evidence replay are documented in `docs/testing/agent-mlops.md`. Execution reports
and raw traces stay outside version control. These are not installation prerequisites. Standalone acceptance
copies only the skill into an unrelated directory and trains/reloads using its bundled runtime.
Conversation fixtures and executable tests are separate evidence channels.
A fixture is not an executed conversation evaluation. M02 requires a real platform service
and clean locked client; mocks validate stop behavior only and cannot pass M02/Q03.
Capability behavior acceptance requires fresh sessions with ordinary prompts and actual
assistant/tool traces. The evaluator needs reviewed actions and resolvable evidence
locators; matching text in this guide is not acceptance. Missing traces stay unverified.
For local capability changes use small CPU fixtures with tracking off. Preserve multipart
settings and the existing tracking stop behavior; this work does not authorize additional
remote training, upload/download, or service changes.

API references: [MLForecast](https://nixtlaverse.nixtla.io/mlforecast/forecast.html),
[uv locked sync](https://docs.astral.sh/uv/concepts/projects/sync/),
[MLflow client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).
