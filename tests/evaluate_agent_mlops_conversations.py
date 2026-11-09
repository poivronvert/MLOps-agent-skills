#!/usr/bin/env python3
"""Score human/agent-reviewed action annotations against the conversation rubric.

Input JSON list: id, variant (optional integer), prompt, visible_context,
assistant_messages (list of strings), tool_calls (list), observed_actions (list of strings),
evidence (action -> trace locators).
Evidence values are lists of JSON pointers, e.g. ["/tool_calls/0", "/assistant_messages/1"].
This checks structure and reviewer annotations, not trace authenticity or semantics.
Missing conversations are unverified, never a passing evaluation.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolves(trace, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False
    parts = pointer[1:].split("/")
    if parts[0] not in {"assistant_messages", "tool_calls", "visible_context"}:
        return False
    value = trace
    try:
        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not part.isdigit():
                    return False
                value = value[int(part)]
            else:
                value = value[part]
        return bool(value)
    except (KeyError, IndexError, TypeError):
        return False


def evaluate(traces, fixtures):
    results = []
    for fixture in fixtures:
        for variant in [None, *range(len(fixture.get("variants", [])))]:
            rubric = fixture if variant is None else fixture["variants"][variant]
            trace = next(
                (
                    t
                    for t in traces
                    if t["id"] == fixture["id"] and t.get("variant") == variant
                ),
                None,
            )
            result = {"id": fixture["id"], "variant": variant, "status": "unverified"}
            if (
                trace
                and all(
                    key in trace
                    for key in [
                        "prompt",
                        "visible_context",
                        "assistant_messages",
                        "tool_calls",
                        "observed_actions",
                        "evidence",
                    ]
                )
                and isinstance(trace["prompt"], str)
                and trace["prompt"].strip()
                and isinstance(trace["assistant_messages"], list)
                and any(
                    isinstance(message, str) and message.strip()
                    for message in trace["assistant_messages"]
                )
                and isinstance(trace["tool_calls"], list)
                and isinstance(trace["observed_actions"], list)
                and all(isinstance(action, str) for action in trace["observed_actions"])
                and isinstance(trace["evidence"], dict)
            ):
                observed = set(trace["observed_actions"])
                required = set(rubric["required_actions"])
                for condition in rubric.get("conditional_actions", []):
                    if condition["if_observed"] in observed:
                        required.update(condition["required_actions"])
                missing = required - observed
                forbidden = set(rubric["forbidden_actions"]) & observed
                no_evidence = []
                for action in sorted(observed):
                    pointers = trace["evidence"].get(action)
                    valid = (
                        isinstance(pointers, list)
                        and bool(pointers)
                        and all(resolves(trace, pointer) for pointer in pointers)
                    )
                    if valid and action in rubric.get("tool_actions", []):
                        valid = any(p.startswith("/tool_calls/") for p in pointers)
                    if not valid:
                        no_evidence.append(action)
                result.update(
                    status="failed"
                    if missing or forbidden or no_evidence
                    else "passed",
                    missing=sorted(missing),
                    forbidden=sorted(forbidden),
                    actions_without_evidence=no_evidence,
                )
                result.update(
                    wrong_fit=int("fit" in forbidden),
                    unnecessary_block=int(
                        bool(
                            forbidden
                            & {
                                "require_full_eda",
                                "require_plugin",
                                "require_vendor_schema",
                                "refuse_without_builtin",
                                "refuse_supplemental_code",
                                "require_builtin_integration",
                            }
                        )
                    ),
                    repeated_question=int("repeat_confirmed_question" in forbidden),
                )
            results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "tests/fixtures/agent_mlops/conversations.json",
    )
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    results = evaluate(json.loads(args.traces.read_text()), fixtures)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    raise SystemExit(0 if all(r["status"] == "passed" for r in results) else 1)
