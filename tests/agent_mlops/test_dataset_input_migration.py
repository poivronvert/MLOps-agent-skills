"""Execute the documented model card against real, isolated MLflow records."""

import hashlib
import importlib.util
import json
from pathlib import Path

import mlflow
from mlflow.entities import Dataset, DatasetInput, InputTag
from mlflow.tracking import MlflowClient
import pytest


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills/mlops"
SUMMARY_TAGS = {
    "data.source", "data.version", "data.row_count", "data.hash",
    "data_hash", "data_version", "dataset_version",
}


def script(relative):
    path = SKILLS / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_card(client):
    """Compile the actual published example, not a duplicate implementation."""
    path = SKILLS / "model-registry/SKILL.md"
    section = path.read_text().split("### 6. Model Card Generation", 1)[1]
    source = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"MlflowClient": lambda: client}
    exec(compile(source, str(path), "exec"), namespace)
    # The example may explicitly import the SDK; substitute only its client factory.
    namespace["MlflowClient"] = lambda: client
    return namespace["generate_model_card"]


class ReadOnlyClient:
    """Permit only reads, while forcing real search pagination in small fixtures."""

    def __init__(self, client, failure=None):
        self.client = client
        self.failure = failure
        self.calls = []

    def __getattr__(self, name):
        assert name in {"get_model_version", "get_run", "search_runs"}, (
            f"Model card attempted unexpected API: {name}"
        )

        def read(*args, **kwargs):
            self.calls.append(name)
            if name == self.failure:
                raise RuntimeError("injected lookup failure")
            if name == "search_runs":
                kwargs["max_results"] = 1
            return getattr(self.client, name)(*args, **kwargs)

        return read


@pytest.fixture
def store(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    previous = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.create_experiment(
        "input-migration", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    yield client, experiment, tmp_path
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(previous)


def new_run(store, name="work", parent=None):
    client, experiment, _ = store
    tags = {"mlflow.runName": name}
    if parent:
        tags["mlflow.parentRunId"] = parent
    return client.create_run(experiment, tags=tags).info.run_id


def add_input(store, run_id, name, context, rows=4, profile="default", schema="default"):
    client, _, path = store
    profile = json.dumps({"num_rows": rows}) if profile == "default" else profile
    schema = '[{"type":"double","name":"x"}]' if schema == "default" else schema
    dataset = Dataset(
        name=name,
        digest="known-digest",
        source_type="local",
        source=json.dumps({"uri": (path / "source-v7.parquet").as_uri()}),
        schema=schema,
        profile=profile,
    )
    tags = [InputTag("mlflow.data.context", context)] if context else []
    client.log_inputs(run_id, [DatasetInput(dataset, tags)])


def register(store, run_id):
    client, _, path = store
    client.create_registered_model("audit-model")
    version = client.create_model_version(
        "audit-model", source=(path / "fixture-model").as_uri(), run_id=run_id,
        tags={"trained_by": "qa", "data_validated": "true"},
    )
    return version.version


def snapshot(store):
    client, experiment, path = store
    runs = {
        r.info.run_id: client.get_run(r.info.run_id).to_dictionary()
        for r in client.search_runs([experiment])
    }
    artifacts = {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (path / "artifacts").rglob("*") if p.is_file()
    }
    versions = [dict(v) for v in client.search_model_versions()]
    return {"runs": runs, "artifacts": artifacts, "versions": versions}


def test_DQ01_wrapper_no_automatic_summary_tags(store, monkeypatch):
    client, _, _ = store
    monkeypatch.setenv("DATA_VERSION", "must-not-be-copied")
    wrapper = script("ml-experiment-tracking/scripts/mlflow_tracker.py")
    tracker = wrapper.ExperimentTracker("input-migration", mlflow.get_tracking_uri())
    run = tracker.start_run("new-work", tags={"team": "risk", "project": "audit"})
    tracker.set_tags({"trained_by": "qa", "data_validated": "true"})
    tracker.end_run()
    tags = client.get_run(run.info.run_id).data.tags
    assert not SUMMARY_TAGS.intersection(tags)
    assert {k: tags[k] for k in ["team", "project", "trained_by", "data_validated"]} == {
        "team": "risk", "project": "audit", "trained_by": "qa", "data_validated": "true",
    }
    compare = script("ml-experiment-tracking/scripts/experiment_compare.py")
    matches = compare.fetch_runs(
        "input-migration", mlflow.get_tracking_uri(), tag_filters={"team": "risk"}
    )
    assert [r["run_id"] for r in matches] == [run.info.run_id]
    assert compare.fetch_runs(
        "input-migration", mlflow.get_tracking_uri(), tag_filters={"team": "other"}
    ) == []
    assert tracker.list_runs()[0]["tags"]["project"] == "audit"


def test_DQ02_card_uses_inputs_without_summary_tags(store):
    client, _, path = store
    root = new_run(store)
    add_input(store, root, "refit-train", "training", rows=4)
    add_input(store, root, "holdout-test", "test", rows=2)
    version = register(store, root)
    assert not SUMMARY_TAGS.intersection(client.get_run(root).data.tags)
    card = model_card(ReadOnlyClient(client))("audit-model", version)
    assert "| refit-train | training | known-digest | 4 |" in card
    assert "| holdout-test | test | known-digest | 2 |" in card
    assert (path / "source-v7.parquet").as_uri() in card
    assert "Digest" in card
    assert "Dataset version: known-digest" not in card


def test_DQ03_parent_children_paginated_and_unrelated_excluded(store):
    client, _, _ = store
    root = new_run(store)
    add_input(store, root, "refit-train", "training", rows=8)
    add_input(store, root, "holdout-test", "test", rows=2)
    children = [new_run(store, f"trial-{i}", root) for i in range(2)]
    for child in children:
        for fold in range(2):
            add_input(store, child, f"fold-{fold}-train", "training", rows=4)
            add_input(store, child, f"fold-{fold}-validation", "validation", rows=4)
    client.set_terminated(children[0], status="FINISHED")
    client.set_terminated(children[1], status="FAILED")
    client.delete_run(children[1])
    other = new_run(store, "unrelated-work")
    add_input(store, other, "must-not-appear", "training", rows=999)
    unrelated_child = new_run(store, "unrelated-trial", other)
    add_input(store, unrelated_child, "also-must-not-appear", "test")
    guard = ReadOnlyClient(client)
    card = model_card(guard)("audit-model", register(store, root))
    assert card.count("| refit-train |") == 1
    assert card.count("| holdout-test |") == 1
    for fold in range(2):
        assert card.count(f"| fold-{fold}-train | training | known-digest | 4 |") == 2
        assert card.count(f"| fold-{fold}-validation | validation | known-digest | 4 |") == 2
    for run_id in [root, *children]:
        assert run_id in card
    assert "trial-0" in card and "trial-1" in card
    assert "FINISHED, active" in card and "FAILED, deleted" in card
    assert other not in card and unrelated_child not in card
    assert "must-not-appear" not in card
    assert "Total rows" not in card
    assert guard.calls.count("search_runs") >= 2


def test_DQ04_conflicting_tags_do_not_override_inputs(store):
    client, _, _ = store
    root = new_run(store)
    add_input(store, root, "actual-input", "training", rows=4)
    for key in SUMMARY_TAGS:
        client.set_tag(root, key, "incorrect-legacy-summary")
    card = model_card(ReadOnlyClient(client))("audit-model", register(store, root))
    assert "| actual-input | training | known-digest | 4 |" in card
    assert "incorrect-legacy-summary" not in card


def test_DQ05_old_run_without_inputs_is_not_reconstructed(store):
    client, _, _ = store
    root = new_run(store)
    client.set_tag(root, "data.source", "legacy-source")
    version = register(store, root)
    before = snapshot(store)
    card = model_card(ReadOnlyClient(client))("audit-model", version)
    assert "Dataset inputs not recorded" in card
    assert "legacy-source" not in card
    assert snapshot(store) == before


@pytest.mark.parametrize("profile", [None, "{}", "null", "not-json"])
def test_DQ06_missing_optional_metadata_is_unknown(store, profile):
    client, _, _ = store
    root = new_run(store)
    add_input(store, root, "train-only", None, profile=profile, schema=None)
    card = model_card(ReadOnlyClient(client))("audit-model", register(store, root))
    assert "| train-only | unknown | known-digest | unknown |" in card
    assert "holdout-test" not in card and "| test |" not in card


@pytest.mark.parametrize("operation", ["get_model_version", "get_run", "search_runs"])
def test_DQ07_lookup_failure_propagates(store, operation):
    client, _, _ = store
    root = new_run(store)
    version = register(store, root)
    guard = ReadOnlyClient(client, failure=operation)
    with pytest.raises(RuntimeError, match="injected lookup failure"):
        model_card(guard)("audit-model", version)


def test_DQ08_model_card_reads_leave_all_records_unchanged(store):
    client, _, _ = store
    root = new_run(store)
    child = new_run(store, "trial-0", root)
    add_input(store, root, "refit-train", "training")
    add_input(store, child, "fold-0-validation", "validation")
    client.log_dict(root, {"existing": True}, "existing-artifact.json")
    version = register(store, root)
    before = snapshot(store)
    guard = ReadOnlyClient(client)
    model_card(guard)("audit-model", version)
    assert snapshot(store) == before
    assert set(guard.calls) == {"get_model_version", "get_run", "search_runs"}
