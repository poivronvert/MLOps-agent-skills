#!/usr/bin/env python3
"""Bridge existing reviewed-action rubrics to official MLflow evaluation datasets.

MLflow owns storage, record merging, discovery and evaluation runs. This adapter only
maps the repository's fixtures and reuses its deterministic domain scorer. It does not
run an acting agent, infer reviewed actions, authenticate traces or call an LLM judge.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from evaluate_agent_mlops_conversations import evaluate


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(records):
    return hashlib.sha256(canonical(records).encode()).hexdigest()


def key(inputs):
    return inputs["case_id"], inputs.get("variant")


def ordered(records):
    return sorted(
        records,
        key=lambda row: (
            key(row["inputs"])[0],
            -1 if key(row["inputs"])[1] is None else key(row["inputs"])[1],
        ),
    )


def fixture_records(fixtures):
    records = []
    for fixture in fixtures:
        for variant in [None, *range(len(fixture.get("variants", [])))]:
            item = fixture if variant is None else fixture["variants"][variant]
            rubric = {
                k: item[k]
                for k in (
                    "required_actions",
                    "forbidden_actions",
                    "conditional_actions",
                    "tool_actions",
                )
                if k in item
            }
            records.append(
                {
                    "inputs": {
                        "case_id": fixture["id"],
                        "variant": variant,
                        "prompt": item.get("prompt", fixture["prompt"]),
                        "visible_context": item.get(
                            "visible_context", fixture.get("visible_context", "")
                        ),
                    },
                    "expectations": {"rubric": rubric},
                }
            )
    validate_records(records)
    return ordered(records)


def validate_records(records):
    if not records:
        raise ValueError("evaluation dataset is empty")
    keys = []
    for row in records:
        inputs = row["inputs"]
        keys.append(key(inputs))
        if not isinstance(inputs["case_id"], str) or not inputs["case_id"].strip():
            raise ValueError("case_id must be nonempty")
        variant = inputs.get("variant")
        if variant is not None and (type(variant) is not int or variant < 0):
            raise ValueError("variant must be null or a nonnegative integer")
        if not isinstance(inputs["prompt"], str) or not inputs["prompt"].strip():
            raise ValueError("case prompt must be nonempty")
        rubric = row["expectations"]["rubric"]
        for field in ("required_actions", "forbidden_actions"):
            if not isinstance(rubric[field], list) or not all(
                isinstance(v, str) for v in rubric[field]
            ):
                raise ValueError(f"{field} must be a list of action names")
    if len(keys) != len(set(keys)):
        raise ValueError(
            "duplicate case_id/variant; use a separate dataset for changed inputs"
        )


def snapshot(dataset):
    # Ignore generated timestamps / record IDs when identifying the evaluated content.
    records = [
        {"inputs": row["inputs"], "expectations": row["expectations"]}
        for row in dataset.to_df().to_dict("records")
    ]
    validate_records(records)
    return ordered(records)


def publish(fixtures, experiment_id, name=None, dataset_id=None):
    from mlflow.genai.datasets import create_dataset, get_dataset

    records = fixture_records(fixtures)
    if dataset_id:
        dataset = get_dataset(dataset_id=dataset_id)
        if experiment_id not in dataset.experiment_ids:
            raise ValueError("dataset is not associated with the selected experiment")
        existing = snapshot(dataset) if dataset.to_df().shape[0] else []
        prior = {key(row["inputs"]): row["inputs"] for row in existing}
        for row in records:
            previous = prior.get(key(row["inputs"]))
            if previous is not None and previous != row["inputs"]:
                raise ValueError("case inputs changed; create a new official dataset")
    else:
        if not name:
            raise ValueError("name is required when creating a dataset")
        dataset = create_dataset(
            name=name,
            experiment_id=experiment_id,
            tags={"rubric_format": "reviewed-actions-v1"},
        )
    # The SDK adds source/tags to submitted dictionaries in place.
    dataset.merge_records(copy.deepcopy(records))
    dataset = get_dataset(dataset_id=dataset.dataset_id)
    rows = snapshot(dataset)
    actual = {key(row["inputs"]): row for row in rows}
    if any(actual.get(key(row["inputs"])) != row for row in records):
        raise ValueError("dataset round-trip differs from submitted records")
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": content_sha256(rows),
        "records": len(rows),
        "mlflow_digest": dataset.digest,
    }


def score_record(inputs, outputs, expectations):
    """Domain scoring only; the official evaluator owns execution and result storage."""
    trace = outputs.get("reviewed_trace")
    fixture = {"id": inputs["case_id"], **expectations["rubric"]}
    if trace is not None:
        if (trace.get("id"), trace.get("variant")) != key(inputs):
            raise ValueError("reviewed trace case identity mismatch")
        # Legacy acting prompts may prepend a skill location / append visible context.
        if inputs["prompt"].strip() not in trace.get("prompt", ""):
            raise ValueError("reviewed trace does not contain this case's prompt")
        if inputs.get("visible_context", "") != trace.get("visible_context", ""):
            raise ValueError("reviewed trace visible context differs from this case")
        trace = {**trace, "variant": None}
    result = evaluate([trace] if trace is not None else [], [fixture])[0]
    result["variant"] = inputs.get("variant")
    return result


def assess(dataset_id, expected_sha256, traces, experiment_id, evidence_dir=None):
    import mlflow
    from mlflow.entities import Feedback
    from mlflow.genai.datasets import get_dataset
    from mlflow.genai.scorers import scorer

    if mlflow.active_run():
        raise ValueError(
            "use a separate evaluation run, outside an active training run"
        )
    dataset = get_dataset(dataset_id=dataset_id)
    if experiment_id not in dataset.experiment_ids:
        raise ValueError("dataset is not associated with the selected experiment")
    records = snapshot(dataset)
    actual_sha = content_sha256(records)
    if actual_sha != expected_sha256:
        raise ValueError(
            "dataset content changed; review the new snapshot before evaluating"
        )
    trace_keys = [(t["id"], t.get("variant")) for t in traces]
    if len(set(trace_keys)) != len(trace_keys):
        raise ValueError("duplicate reviewed traces; select one attempt per case")
    valid_keys = {key(row["inputs"]) for row in records}
    if set(trace_keys) - valid_keys:
        raise ValueError("reviewed traces contain cases outside the selected dataset")
    by_case = dict(zip(trace_keys, traces))
    rows, results = [], []
    evidence = set()
    for row in records:
        trace = by_case.get(key(row["inputs"]))
        if trace and trace.get("raw_events"):
            if evidence_dir is None:
                raise ValueError(
                    "evidence-dir required for traces referencing raw events"
                )
            root = Path(evidence_dir).resolve()
            raw = (root / trace["raw_events"]).resolve()
            if not raw.is_relative_to(root) or not raw.is_file():
                raise ValueError("raw trace path must be a file inside evidence-dir")
            if hashlib.sha256(raw.read_bytes()).hexdigest() != trace.get("raw_sha256"):
                raise ValueError("raw trace hash mismatch")
            evidence.add(raw)
        output = {"reviewed_trace": trace}
        results.append(score_record(row["inputs"], output, row["expectations"]))
        rows.append({**row, "outputs": output})

    @scorer(name="reviewed_action_contract", pass_if=lambda value: value == "passed")
    def contract(inputs, outputs, expectations):
        result = score_record(inputs, outputs, expectations)
        return Feedback(value=result["status"], rationale=canonical(result))

    with mlflow.start_run(
        experiment_id=experiment_id, run_name="reviewed-agent-evaluation"
    ) as run:
        mlflow.set_tags(
            {
                "evaluation.dataset_id": dataset_id,
                "evaluation.dataset_sha256": actual_sha,
                "evaluation.mode": "reviewed_observations_no_agent_execution",
                "evaluation.scorer_sha256": hashlib.sha256(
                    Path(__file__)
                    .with_name("evaluate_agent_mlops_conversations.py")
                    .read_bytes()
                ).hexdigest(),
            }
        )
        # OSS datasets are mutable. Save exactly the evaluated data through official artifacts.
        mlflow.log_dict(records, "evaluation/dataset_snapshot.json")
        mlflow.log_dict(traces, "evaluation/reviewed_traces.json")
        mlflow.log_dict(results, "evaluation/reviewed_results.json")
        mlflow.log_artifact(str(Path(__file__).resolve()), "evaluation/code")
        mlflow.log_artifact(
            str(Path(__file__).with_name("evaluate_agent_mlops_conversations.py")),
            "evaluation/code",
        )
        for raw in sorted(evidence):
            relative = raw.relative_to(Path(evidence_dir).resolve())
            mlflow.log_artifact(
                str(raw), "evaluation/raw/" + relative.parent.as_posix()
            )
        mlflow.log_input(dataset, context="evaluation")
        evaluated = mlflow.genai.evaluate(data=rows, scorers=[contract])
        if evaluated.run_id != run.info.run_id:
            raise RuntimeError("evaluation results were stored in an unexpected run")
        table = evaluated.result_df
        rationale = "reviewed_action_contract/rationale"
        if (
            table is None
            or rationale not in table
            or sorted(table[rationale].fillna("").tolist())
            != sorted(canonical(result) for result in results)
        ):
            raise RuntimeError("official evaluation feedback is missing or differs")
        # Do not turn absent observations into a passing score or hide them in a mean.
        counts = {
            status: sum(r["status"] == status for r in results)
            for status in ("passed", "failed", "unverified")
        }
        mlflow.log_metrics(
            {f"cases_{status}": value for status, value in counts.items()}
        )
        mlflow.log_metric("cases_total", len(results))
        return {
            "run_id": run.info.run_id,
            "dataset_id": dataset_id,
            "dataset_sha256": actual_sha,
            "counts": counts,
            "all_verified_passed": counts["passed"] == len(results),
        }


def main():
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="Explicit official MLflow store; no implicit local fallback",
    )
    parser.add_argument("--experiment-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    put = commands.add_parser("publish")
    put.add_argument("--fixtures", type=Path, required=True)
    group = put.add_mutually_exclusive_group(required=True)
    group.add_argument("--name")
    group.add_argument("--dataset-id")
    check = commands.add_parser("evaluate")
    check.add_argument("--dataset-id", required=True)
    check.add_argument("--dataset-sha256", required=True)
    check.add_argument("--traces", type=Path, required=True)
    check.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    if args.command == "publish":
        result = publish(
            json.loads(args.fixtures.read_text()),
            args.experiment_id,
            args.name,
            args.dataset_id,
        )
    else:
        result = assess(
            args.dataset_id,
            args.dataset_sha256,
            json.loads(args.traces.read_text()),
            args.experiment_id,
            args.evidence_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("all_verified_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
