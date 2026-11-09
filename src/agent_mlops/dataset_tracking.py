"""Record caller-selected data; never split, shuffle, fit, or transform it."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import pandas as pd
import pyarrow as pa


@dataclass
class DatasetSplit:
    name: str
    context: str
    frame: pd.DataFrame
    target: str
    source: str
    source_sha256: str
    stage: str = "pipeline_input"


def frame_sha256(frame):
    """Hash all values, row order, index, and Arrow schema, without sampling.

    Versioned by the manifest; reproducibility requires the recorded Arrow/Pandas
    environment. This is separate from MLflow's short digest and Parquet byte hash.
    """
    table = pa.Table.from_pandas(frame, preserve_index=True)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def record_split(
    client,
    run_id,
    split,
    *,
    log_inputs=True,
    snapshot=True,
    artifact_root="datasets",
    provenance=None,
):
    """Upload selected artifacts and read them back before reporting success.

    The MLflow source refers to the upstream input, not to a downloadable copy
    of this derived split. Only snapshot_artifact identifies that copy.
    """
    import mlflow.data
    from mlflow.entities import DatasetInput, InputTag

    if not re.fullmatch(r"[A-Za-z0-9_-]+", split.name):
        raise ValueError("dataset name must contain only letters, digits, _ or -")
    if split.context not in {"training", "validation", "test"}:
        raise ValueError("dataset context must be training, validation, or test")
    if split.frame.empty or split.target not in split.frame:
        raise ValueError("dataset requires nonempty rows and the actual target")
    if not split.frame.columns.is_unique:
        raise ValueError("dataset columns must be unique")
    if not re.fullmatch(r"[a-f0-9]{64}", split.source_sha256):
        raise ValueError("dataset requires the verified upstream source SHA-256")
    content_hash = frame_sha256(split.frame)
    dataset = mlflow.data.from_pandas(
        split.frame,
        source=split.source,
        name=split.name,
        targets=split.target,
        digest=content_hash[:32],
    )
    entity = dataset._to_mlflow_entity()
    artifact_dir = f"{artifact_root}/{split.name}/{content_hash}"
    manifest = {
        "format": "agent-mlops-dataset-v1",
        "name": split.name,
        "context": split.context,
        "stage": split.stage,
        "rows": len(split.frame),
        "columns": list(split.frame.columns),
        "target": split.target,
        "schema": entity.schema,
        "source": split.source,
        "source_sha256": split.source_sha256,
        "frame_sha256": content_hash,
        "hash_format": "arrow-ipc-stream-v1-with-index",
        "serialization_versions": {"pandas": pd.__version__, "pyarrow": pa.__version__},
        "mlflow_digest": dataset.digest,
        "inputs_logged": log_inputs,
        "snapshot_artifact": None,
        "snapshot_sha256": None,
        "provenance": provenance or {},
    }
    if log_inputs:
        tags = {
            "mlflow.data.context": split.context,
            "data.stage": split.stage,
            "data.sha256": content_hash,
        }
        client.log_inputs(
            run_id,
            datasets=[DatasetInput(entity, [InputTag(k, v) for k, v in tags.items()])],
        )
        observed = client.get_run(run_id).inputs.dataset_inputs
        if not any(
            item.dataset.name == split.name
            and item.dataset.digest == dataset.digest
            and item.dataset.schema == entity.schema
            and item.dataset.source == entity.source
            and item.dataset.source_type == entity.source_type
            and item.dataset.profile == entity.profile
            and {tag.key: tag.value for tag in item.tags}.items() >= tags.items()
            for item in observed
        ):
            raise RuntimeError("dataset input readback mismatch")
    with tempfile.TemporaryDirectory(prefix="agent-mlops-dataset-") as directory:
        if snapshot:
            path = Path(directory) / "snapshot.parquet"
            split.frame.to_parquet(path, index=True)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            client.log_artifact(run_id, str(path), artifact_path=artifact_dir)
            manifest.update(
                snapshot_artifact=f"{artifact_dir}/snapshot.parquet",
                snapshot_sha256=expected,
            )
            download_dir = Path(directory) / "verify"
            download_dir.mkdir()
            downloaded = Path(
                client.download_artifacts(
                    run_id, manifest["snapshot_artifact"], str(download_dir)
                )
            )
            payload = downloaded.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError("dataset snapshot byte hash mismatch")
            pd.testing.assert_frame_equal(
                split.frame, pd.read_parquet(io.BytesIO(payload)), check_exact=True
            )
        client.log_dict(run_id, manifest, f"{artifact_dir}/manifest.json")
        manifest_dir = Path(directory) / "manifest"
        manifest_dir.mkdir()
        saved = client.download_artifacts(
            run_id, f"{artifact_dir}/manifest.json", str(manifest_dir)
        )
        if json.loads(Path(saved).read_text()) != manifest:
            raise RuntimeError("dataset manifest readback mismatch")
    return manifest
