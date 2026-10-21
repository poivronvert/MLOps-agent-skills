"""Remote-only MLflow client. Never creates a server/store or retries a failed job."""

import os
import json
from pathlib import Path
from urllib.parse import urlsplit


class TrackingError(RuntimeError):
    pass


def remote_uri():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    try:
        parts = urlsplit(uri)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(
                "remote tracking requires MLFLOW_TRACKING_URI with HTTP(S) service URI"
            )
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError(
                "credential-bearing URLs are prohibited; provide authentication through environment"
            )
    except ValueError:
        raise ValueError(
            "remote tracking requires a valid HTTP(S) URI without embedded credentials"
        ) from None
    return uri


class Tracker:
    def __init__(self, spec):
        self.status = "off" if spec.tracking == "off" else "pending"
        self.run_id = None
        self.model_uri = None
        self.failed_stage = None
        self.trial_id = None
        self.client = None
        self.spec = spec
        self.datasets = []

    def dataset(self, frame, features, *, name, context):
        if self.status == "off":
            return
        if not self.spec.dataset_inputs and context not in self.spec.dataset_snapshots:
            return
        from .dataset_tracking import DatasetSplit, record_split
        from .temporal import LABEL

        if self.spec.mode == "forecast":
            t = self.spec.temporal
            columns = list(
                dict.fromkeys(
                    [
                        *t.entity_keys,
                        t.time_column,
                        self.spec.target,
                        *features,
                    ]
                )
            )
            target = self.spec.target
            stage = "forecast_pipeline_input_with_history"
        else:
            columns = [*features, LABEL]
            target = LABEL
            stage = "pipeline_input"
        split = DatasetSplit(
            name=name,
            context=context,
            frame=frame[columns],
            target=target,
            source=self.spec.input,
            source_sha256=self.spec.data_sha256,
            stage=stage,
        )
        receipt = self.call(
            "dataset_recording",
            record_split,
            self.client,
            self.trial_id or self.run_id,
            split,
            log_inputs=self.spec.dataset_inputs,
            snapshot=context in self.spec.dataset_snapshots,
            provenance={
                "mode": self.spec.mode,
                "original_target": self.spec.target,
            },
        )
        self.datasets.append({"run_id": self.trial_id or self.run_id, **receipt})

    def call(self, stage, fn, *args, **kwargs):
        if self.status == "failed":
            raise TrackingError(
                f"tracking already failed at {self.failed_stage}; no further compute or replay"
            )
        try:
            return fn(*args, **kwargs)
        except TrackingError:
            raise
        except Exception:
            self.status = "failed"
            self.failed_stage = stage
            # SDK exceptions can contain Authorization headers and credential URLs.
            raise TrackingError(
                f"MLflow failed during {stage}; inspect service/authentication outside artifacts"
            ) from None

    def start(self):
        if self.status == "off":
            return
        uri = remote_uri()
        import mlflow

        if mlflow.active_run():
            raise ValueError(
                "training requires its own run; end the active MLflow run first"
            )
        from mlflow.tracking import MlflowClient

        self.client = self.call("client", MlflowClient, tracking_uri=uri)
        exp = self.call(
            "preflight", self.client.get_experiment_by_name, self.spec.experiment
        )
        exp_id = (
            exp.experiment_id
            if exp
            else self.call(
                "experiment", self.client.create_experiment, self.spec.experiment
            )
        )
        self.experiment_id = exp_id
        run = self.call(
            "work_run",
            self.client.create_run,
            exp_id,
            tags={
                "mlflow.runName": "agent-mlops",
                "package_format": "mlflow",
            },
        )
        self.run_id = run.info.run_id
        self.status = "active"
        self.call(
            "scoring_spec",
            self.client.log_param,
            self.run_id,
            "metric",
            self.spec.metric,
        )
        self.call(
            "scoring_spec",
            self.client.log_param,
            self.run_id,
            "aggregation",
            self.spec.aggregation,
        )

    def start_trial(self, number, params):
        if self.status == "off":
            return
        run = self.call(
            "trial_start",
            self.client.create_run,
            self.experiment_id,
            tags={
                "mlflow.parentRunId": self.run_id,
                "mlflow.runName": f"trial-{number}",
            },
        )
        self.trial_id = run.info.run_id
        for key, value in params.items():
            self.call(
                "trial_params", self.client.log_param, self.trial_id, key, str(value)
            )

    def fold(self, number, result):
        if self.status == "off":
            return
        self.call(
            "fold_metrics",
            self.client.log_metric,
            self.trial_id,
            "validation_score",
            result["score"],
            step=number,
        )

    def trial(self, result):
        if self.status == "off":
            return
        if result["status"] == "complete":
            self.call(
                "trial_metrics",
                self.client.log_metric,
                self.trial_id,
                "objective",
                result["score"],
            )
        else:
            self.call(
                "trial_failure",
                self.client.set_tag,
                self.trial_id,
                "failure_reason",
                result["error"],
            )
        self.call(
            "trial_finish",
            self.client.set_terminated,
            self.trial_id,
            status="FINISHED" if result["status"] == "complete" else "FAILED",
        )
        self.trial_id = None

    def finish(self, output, flavor, options):
        if self.status == "off":
            return
        import mlflow

        def publish():
            previous_uri = mlflow.get_tracking_uri()
            try:
                mlflow.set_tracking_uri(remote_uri())
                with mlflow.start_run(run_id=self.run_id):
                    info = flavor.log_model(name="model", **options)
                    self.model_uri = info.model_uri
                    for name in ("task_spec.json", "environment.json"):
                        self.call(
                            "provenance_upload",
                            self.client.log_artifact,
                            self.run_id,
                            str(Path(output) / name),
                        )
                    report = self.call(
                        "report_prepare",
                        lambda: json.loads(
                            (Path(output) / "training_report.json").read_text()
                        ),
                    )
                    report["tracking"] = {**self.snapshot(), "status": "complete"}
                    self.call(
                        "report_upload",
                        self.client.log_dict,
                        self.run_id,
                        report,
                        "training_report.json",
                    )
            finally:
                mlflow.set_tracking_uri(previous_uri)

        self.call("model_upload", publish)
        self.status = "complete"

    def snapshot(self):
        return {
            "status": self.status,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "failed_stage": self.failed_stage,
            "format": "mlflow",
            "load_method": "mlflow.pyfunc.load_model(model_uri)",
            "datasets": self.datasets,
        }

    def close_failed(self, output):
        """Record calculation failure without falsely completing the work run."""
        if self.status == "off":
            return
        for name in ("training_report.json", "task_spec.json", "environment.json"):
            self.call(
                "diagnostics_upload",
                self.client.log_artifact,
                self.run_id,
                str(Path(output) / name),
                artifact_path="diagnostics",
            )
        self.call(
            "work_failure", self.client.set_terminated, self.run_id, status="FAILED"
        )
        self.status = "complete"
