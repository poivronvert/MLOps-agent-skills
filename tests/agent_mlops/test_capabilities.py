"""CAP contracts and scorer controls. Synthetic traces here are not agent acceptance."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from agent_mlops.config import TaskSpec
from agent_mlops.training import prepare, splits, train
from test_contract import spec_for
from test_temporal import panel, temporal_spec

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = json.loads(
    (ROOT / "tests/fixtures/agent_mlops/capabilities.json").read_text()
)
module_spec = importlib.util.spec_from_file_location(
    "conversation_evaluator", ROOT / "tests/evaluate_agent_mlops_conversations.py"
)
evaluator = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(evaluator)


def annotated_trace(fixture):
    """Fabricated control for scorer unit tests only; never export as observed evidence."""
    return {
        "id": fixture["id"],
        "prompt": fixture["prompt"],
        "visible_context": fixture["visible_context"],
        "assistant_messages": ["Synthetic scorer control, not an actual conversation."],
        "tool_calls": [{"command": "synthetic", "output": "scorer control only"}],
        "observed_actions": fixture["required_actions"],
        "evidence": {
            action: [
                "/tool_calls/0"
                if action in fixture.get("tool_actions", [])
                else "/assistant_messages/0"
            ]
            for action in fixture["required_actions"]
        },
    }


@pytest.mark.parametrize("messages", [[], [""], ["   "], None])
def test_CAP11_missing_or_empty_conversations_are_unverified(messages):
    assert {r["status"] for r in evaluator.evaluate([], FIXTURES)} == {"unverified"}
    fixture = FIXTURES[0]
    trace = annotated_trace(fixture)
    trace["assistant_messages"] = messages
    trace["tool_calls"] = []
    assert evaluator.evaluate([trace], [fixture])[0]["status"] == "unverified"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["id"])
def test_rubric_accepts_evidenced_controls_and_rejects_forbidden_actions(fixture):
    trace = annotated_trace(fixture)
    assert evaluator.evaluate([trace], [fixture])[0]["status"] == "passed"
    for action in fixture["forbidden_actions"]:
        bad = {
            **trace,
            "observed_actions": [*trace["observed_actions"], action],
            "evidence": {**trace["evidence"], action: ["/assistant_messages/0"]},
        }
        result = evaluator.evaluate([bad], [fixture])[0]
        assert result["status"] == "failed"
        assert action in result["forbidden"]


@pytest.mark.parametrize(
    "locator", ["/tool_calls/99", "/unknown/0", "invented line 42"]
)
def test_CAP11_evidence_must_resolve_to_actual_trace_content(locator):
    fixture = FIXTURES[0]
    trace = annotated_trace(fixture)
    trace["evidence"]["fit"] = [locator]
    assert evaluator.evaluate([trace], [fixture])[0]["status"] != "passed"


def test_CAP11_execution_claim_requires_tool_evidence():
    fixture = FIXTURES[0]
    trace = annotated_trace(fixture)
    trace["evidence"]["fit"] = ["/assistant_messages/0"]
    assert evaluator.evaluate([trace], [fixture])[0]["status"] != "passed"


def test_CAP03_voluntary_hourly_proposal_needs_disclosure_and_implementation_path():
    fixture = FIXTURES[2]
    trace = annotated_trace(fixture)
    trace["observed_actions"] = [*trace["observed_actions"], "propose_hourly_holdout"]
    trace["evidence"]["propose_hourly_holdout"] = ["/assistant_messages/0"]
    result = evaluator.evaluate([trace], [fixture])[0]
    assert result["status"] == "failed"
    assert "disclose_new_hourly_implementation" in result["missing"]
    for action in ["disclose_new_hourly_implementation", "offer_implementation_path"]:
        trace["observed_actions"].append(action)
        trace["evidence"][action] = ["/assistant_messages/0"]
    assert evaluator.evaluate([trace], [fixture])[0]["status"] == "passed"


def test_CAP05_unchanged_install_does_not_excuse_replaced_core():
    fixture = FIXTURES[4]
    trace = annotated_trace(fixture)
    trace["tool_calls"] = [
        {
            "command": "cat wrapper.py",
            "output": "training.splits = custom_splits\ntraining.evaluate = custom_evaluate",
        }
    ]
    # Reviewer observed the false claim; the scorer must fail even with all positives.
    action = "claim_unchanged_core_from_unchanged_install"
    trace["observed_actions"] = [*trace["observed_actions"], action]
    trace["evidence"][action] = ["/assistant_messages/0"]
    assert evaluator.evaluate([trace], [fixture])[0]["status"] == "failed"


def test_CAP01_CAP03_builtin_query_agrees_with_accepted_models_and_formats(table):
    from agent_mlops.capabilities import describe_builtins
    from agent_mlops.data import read_snapshot

    contract = describe_builtins()
    assert contract["scope"] == "builtin_entrypoint_only"
    assert contract["agent_method_allowlist"] is False
    for suffix in contract["input_formats"]:
        path = table.with_suffix(suffix)
        frame = read_snapshot(spec_for(table))
        if suffix == ".parquet":
            frame.to_parquet(path)
        else:
            frame.to_csv(path, sep="\t" if suffix == ".tsv" else ",", index=False)
        assert len(read_snapshot(spec_for(path))) == len(frame)
    for mode, details in contract["modes"].items():
        for model in ["rf", "svm", "xgb", "lgbm"]:
            values = spec_for(table).model_dump()
            if mode != "tabular":
                values = temporal_spec(table, mode).model_dump()
            values["model"] = model
            if model in details["models"]:
                TaskSpec(**values)
            else:
                with pytest.raises(ValidationError):
                    TaskSpec(**values)


def test_CAP03_forecast_contract_matches_real_split_and_scores(tmp_path):
    pytest.importorskip("mlforecast")
    from agent_mlops.capabilities import describe_builtins

    path, raw = panel(tmp_path)
    spec = temporal_spec(path)
    prepared, *_ = prepare(spec, raw)
    folds, refit, holdout = splits(spec, prepared)
    contract = describe_builtins()["modes"]["forecast"]
    assert len(folds) == len(spec.temporal.origins) == 2
    assert contract["development"] == "configured_origins"
    assert contract["holdout"] == "single_origin_at_test_start"
    assert contract["baseline"] == "last_observation_per_entity"
    assert holdout.time.nunique() == max(spec.temporal.horizons)
    assert refit.time.max() < spec.temporal.timestamp(spec.temporal.test_start)
    report = train(spec, tmp_path / "bundle")
    groups = report["evaluation"]["holdout"]["horizons"]
    assert len(groups) == len(spec.temporal.horizons)
    expected = []
    for group in groups:
        last = {
            entity: refit[refit.id == entity].sort_values("time").y.iloc[-1]
            for entity in ["A", "B"]
        }
        # Internal entity IDs are serialized tuples; use the same pooled truth order.
        baseline = [last[json.loads(entity)[0]] for entity in group["entities"]]
        expected.append(np.mean(np.abs(np.array(group["y_true"]) - baseline)))
    assert report["evaluation"]["holdout"]["baseline_score"] == pytest.approx(
        np.mean(expected)
    )


def test_CAP09_builtin_hash_explicitly_excludes_workspace_sources(table, tmp_path):
    report = train(spec_for(table), tmp_path / "bundle")
    assert report["training_status"] == "completed"
    env = json.loads((tmp_path / "bundle/environment.json").read_text())
    assert env["source_scope"] == "agent_mlops_package_only"
    assert env["includes_workspace_code"] is False


def test_CAP10_builtin_query_from_copied_skill_without_repository(tmp_path):
    import shutil

    installed = tmp_path / "installed"
    shutil.copytree(ROOT / "skills/mlops/model-training", installed)
    result = subprocess.run(
        [
            sys.executable,
            str(installed / "scripts/train_model.py"),
            "--describe-builtins",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert contract["modes"]["forecast"]["holdout"] == "single_origin_at_test_start"
    assert not (tmp_path / "output").exists()
