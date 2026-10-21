# MLOps & LLMOps Agent Skills

A comprehensive collection of **28 Agent Skills** for MLOps and LLMOps, following the
[Agent Skills standard](https://agentskills.io). These skills provide domain-expert
guidance, executable scripts, and reference documentation for the complete ML/LLM lifecycle.

Compatible with: **Claude Code**, **Kiro CLI/IDE**, **Cursor**, **VS Code Copilot**,
**Gemini CLI**, and any agent supporting the Agent Skills format.

## Reproducible agent training

The existing training CLI now supports fold-local preprocessing, window-event classification,
recursive MLForecast, configurable Optuna, official MLflow model packaging and explicit off/remote tracking. Start with
[the training workflow](skills/mlops/model-training/references/AGENT_TRAINING.md).
Installed skills run from any working folder using their `scripts/run.py` launcher and
bundled runtime. Use Python 3.11–3.12 and uv **0.11.6**; the launcher prepares a locked cache
environment with the requested extras. For repository development, use `uv sync --locked --group dev`.
Root `src/agent_mlops`, `pyproject.toml` / `uv.lock` are authoritative; `requirements-test.txt` is an
automatically generated compatibility export. See [testing and acceptance](docs/testing/agent-mlops.md)
for reproducible checks, service prerequisites and local conversation-evidence replay.

After changing the canonical runtime or lock, run `python scripts/package_model_training.py`.
Commit the generated `skills/mlops/model-training/runtime/` with the source changes.
CI runs `python scripts/package_model_training.py --check` to reject drift. Installed users
only need the complete `model-training` folder, including `runtime/`; they do not run packaging.

## Skills Overview

### MLOps Skills (17)

| Skill | Description |
|-------|-------------|
| `ml-solution-design` | Requirements intake, architecture decisions, reference architectures, proposals |
| `data-ingestion` | Batch/streaming ingestion, ETL/ELT, data lake, versioning |
| `data-validation` | Great Expectations, Pandera, data contracts, quality checks |
| `feature-engineering` | Transformations, encoding, selection, sklearn Pipelines |
| `feature-store` | Feast, online/offline stores, point-in-time joins |
| `ml-experiment-tracking` | MLflow, W&B, experiment comparison, reproducibility |
| `model-training` | HPO (Optuna), distributed training (DDP), mixed precision |
| `model-registry` | Versioning, promotion, lineage, model cards, packaging |
| `ml-cicd` | Model CI, GitHub Actions, SageMaker Pipelines, promotion, rollback |
| `model-serving` | FastAPI, BentoML, Triton, K8s, A/B testing, batching |
| `model-monitoring` | Evidently, Whylogs, performance tracking, alerting |
| `model-drift-detection` | PSI, KS test, chi-squared, retraining triggers |
| `model-observability` | SHAP, LIME, tracing, fairness, prediction logging |
| `ml-pipeline-orchestration` | Airflow, Prefect, Dagster, Kubeflow, ZenML |
| `ml-testing` | Behavioral tests, quality gates, regression, CI/CD |
| `ml-security` | Adversarial robustness, differential privacy, RBAC, PII |
| `ml-cost-optimization` | GPU selection, quantization, spot instances, FinOps |

### LLMOps Skills (11)

| Skill | Description |
|-------|-------------|
| `llm-fine-tuning` | LoRA, QLoRA, PEFT, DPO, SFT, dataset preparation |
| `llm-evaluation` | RAGAS, LLM-as-judge, benchmarks, safety evaluation |
| `llm-deployment` | vLLM, TGI, Ollama, quantization (AWQ/GPTQ/GGUF) |
| `llm-prompt-engineering` | Prompt patterns, templates, versioning, injection defense |
| `llm-rag` | Chunking, embeddings, vector stores, hybrid search, reranking |
| `llm-guardrails` | Input/output validation, PII, toxicity, jailbreak prevention |
| `llm-observability` | Token tracking, latency (TTFT/TPS), LangSmith, feedback |
| `llm-agent-orchestration` | Tool use, LangGraph, CrewAI, memory, human-in-the-loop |
| `llm-cost-optimization` | Model routing, semantic caching, prompt compression, batch API |
| `llm-data-preparation` | Synthetic data, annotation (Argilla), deduplication, quality |
| `llm-distillation` | Teacher generation (Bedrock/R1), reasoning distillation, curation, relative quality gates |

## Delivery Lifecycle

The skills compose into two end-to-end delivery chains — **every node below is clickable**
and jumps to that skill's `SKILL.md`. Each SKILL.md also ends with a "Related skills"
section naming its upstream and downstream neighbors.

### MLOps chain

[![MLOps delivery chain](docs/mlops-lifecycle.svg)](docs/mlops-lifecycle.svg)

<sub>Open the SVG directly for clickable cards + animated flows; the Mermaid source below renders inline with the same click-through.</sub>

<details>
<summary>Mermaid source (inline-rendered fallback)</summary>


```mermaid
flowchart TD
    SD(["🎯 ml-solution-design"]) --> DI["📥 data-ingestion"]
    DI --> DV["✅ data-validation"]
    DV --> FE["🧪 feature-engineering"]
    FE --> FS[("🗄️ feature-store")]
    FE --> MT["🏋️ model-training"]
    FS -.-> MT
    ET["📊 ml-experiment-tracking"] -.parallel.- MT
    MT --> MR["📦 model-registry"]
    MR --> CI["🚀 ml-cicd"]
    CI --> MS["🌐 model-serving"]
    MS --> MM["📈 model-monitoring"]
    MM --> DD["🌊 model-drift-detection"]
    DD -->|retraining trigger| MT

    click SD "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/ml-solution-design/SKILL.md" "Requirements intake & architecture decisions"
    click DI "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/data-ingestion/SKILL.md" "Batch/streaming ingestion"
    click DV "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/data-validation/SKILL.md" "Great Expectations, Pandera, contracts"
    click FE "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/feature-engineering/SKILL.md" "Transformations, encoding, selection"
    click FS "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/feature-store/SKILL.md" "Feast, online/offline stores"
    click MT "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/model-training/SKILL.md" "HPO, distributed training"
    click ET "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/ml-experiment-tracking/SKILL.md" "MLflow, W&B"
    click MR "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/model-registry/SKILL.md" "Versioning, aliases, promotion"
    click CI "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/ml-cicd/SKILL.md" "Model CI, GitHub Actions, SageMaker Pipelines"
    click MS "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/model-serving/SKILL.md" "FastAPI, BentoML, SageMaker endpoints"
    click MM "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/model-monitoring/SKILL.md" "Evidently, Model Monitor, runbooks"
    click DD "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/model-drift-detection/SKILL.md" "PSI, KS test, retraining triggers"

    classDef design fill:#7c5cff,stroke:#5a3fd6,color:#ffffff
    classDef data fill:#0ea5e9,stroke:#0284c7,color:#ffffff
    classDef train fill:#f59e0b,stroke:#d97706,color:#1a1a1a
    classDef deploy fill:#10b981,stroke:#059669,color:#ffffff
    classDef ops fill:#ef4444,stroke:#dc2626,color:#ffffff
    class SD design
    class DI,DV,FE,FS data
    class MT,ET,MR train
    class CI,MS deploy
    class MM,DD ops
```

</details>

**Cross-cutting** (apply at every stage):
[`ml-testing`](skills/mlops/ml-testing/SKILL.md) quality gates inside ml-cicd ·
[`ml-security`](skills/mlops/ml-security/SKILL.md) adversarial robustness, privacy, compliance ·
[`ml-cost-optimization`](skills/mlops/ml-cost-optimization/SKILL.md) design-time sizing + operational right-sizing ·
[`ml-pipeline-orchestration`](skills/mlops/ml-pipeline-orchestration/SKILL.md) automates the chain as DAGs ·
[`model-observability`](skills/mlops/model-observability/SKILL.md) explainability companion to serving + monitoring

The training → registry → serving → monitoring handoff is specified in
[`ARTIFACT_CONTRACT.md`](skills/mlops/model-registry/references/ARTIFACT_CONTRACT.md).

### LLMOps chain

[![LLMOps delivery chain](docs/llmops-lifecycle.svg)](docs/llmops-lifecycle.svg)

<sub>Open the SVG directly for clickable cards + animated flows; the Mermaid source below renders inline with the same click-through.</sub>

<details>
<summary>Mermaid source (inline-rendered fallback)</summary>


```mermaid
flowchart TD
    SD(["🎯 ml-solution-design"]) --> DP["🧹 llm-data-preparation"]
    DP --> FT["🎛️ llm-fine-tuning"]
    DP --> RAG["🔍 llm-rag"]
    FT --> EV["🧾 llm-evaluation"]
    RAG --> EV
    EV --> DEP["🌐 llm-deployment"]
    DEP --> AO["🤖 llm-agent-orchestration"]
    DEP --> GR["🛡️ llm-guardrails"]
    AO --> GR
    GR --> OBS["📈 llm-observability"]
    OBS --> CO["💰 llm-cost-optimization"]
    CO -.optimization loop.-> DEP

    click SD "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/mlops/ml-solution-design/SKILL.md" "Requirements intake & architecture decisions"
    click DP "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-data-preparation/SKILL.md" "Synthetic data, dedup, quality"
    click FT "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-fine-tuning/SKILL.md" "LoRA, QLoRA, DPO"
    click RAG "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-rag/SKILL.md" "Chunking, embeddings, hybrid search"
    click EV "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-evaluation/SKILL.md" "RAGAS, LLM-as-judge, safety"
    click DEP "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-deployment/SKILL.md" "vLLM, Bedrock, quantization"
    click AO "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-agent-orchestration/SKILL.md" "Tool use, LangGraph, multi-agent"
    click GR "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-guardrails/SKILL.md" "PII, jailbreak prevention, Bedrock Guardrails"
    click OBS "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-observability/SKILL.md" "Token tracking, latency, tracing"
    click CO "https://github.com/timwukp/MLOps-agent-skills/blob/main/skills/llmops/llm-cost-optimization/SKILL.md" "Model routing, caching, batch API"

    classDef design fill:#7c5cff,stroke:#5a3fd6,color:#ffffff
    classDef data fill:#0ea5e9,stroke:#0284c7,color:#ffffff
    classDef train fill:#f59e0b,stroke:#d97706,color:#1a1a1a
    classDef deploy fill:#10b981,stroke:#059669,color:#ffffff
    classDef ops fill:#ef4444,stroke:#dc2626,color:#ffffff
    class SD design
    class DP data
    class FT,RAG,EV train
    class DEP,AO deploy
    class GR,OBS,CO ops
```

</details>

**Cross-cutting**: [`llm-prompt-engineering`](skills/llmops/llm-prompt-engineering/SKILL.md)
(every stage from RAG generation to agent system prompts).
**Bridges to classic MLOps**: fine-tuned LLMs flow through
[`model-registry`](skills/mlops/model-registry/SKILL.md) /
[`model-serving`](skills/mlops/model-serving/SKILL.md) /
[`model-monitoring`](skills/mlops/model-monitoring/SKILL.md) like any other model.

<sub>Legend: 🟣 design · 🔵 data · 🟠 train/evaluate · 🟢 deploy · 🔴 operate — node
click-through works on github.com (Mermaid); the inline links above are the fallback.</sub>

## Installation

### Codex: standalone model-training

Copy the complete skill into the user skills directory. For a new installation, from this checkout:

```bash
mkdir -p ~/.agents/skills
cp -R skills/mlops/model-training ~/.agents/skills/model-training
```

If `model-training` is already present, replace that existing skill with the complete new
folder rather than copying inside it. Use an actual copy for an installation independent
of this repository; a symlink still depends on the checkout. Open a new Codex conversation
in your data/project directory and invoke `$model-training`. Codex reads the installed
instructions and runs the skill there. Data and outputs stay in your working directory.

To check the launcher without training:

```bash
python3 ~/.agents/skills/model-training/scripts/run.py --help
```

### Claude Code

```bash
# Copy skills to your project
cp -r skills/ .claude/skills/

# Or to global skills
cp -r skills/ ~/.claude/skills/
```

### Kiro CLI/IDE

```bash
# Copy to workspace
cp -r skills/ .kiro/skills/

# Or to global skills
cp -r skills/ ~/.kiro/skills/
```

### Other Compatible Agents

Copy the skill folders to the location specified by your agent's skills documentation.

## Skill Structure

Each skill follows the Agent Skills standard:

```
skill-name/
├── SKILL.md           # Instructions and guidance (required)
├── scripts/           # Executable Python/Bash scripts
│   ├── main_tool.py   # Primary automation script
│   └── helper.py      # Supporting script
└── references/        # Detailed reference documentation
    └── REFERENCE.md   # Tool comparisons, deep-dives
```

## Design Principles

- **Platform-agnostic**: Works with any cloud or on-prem setup
- **Framework-inclusive**: PyTorch, TensorFlow, scikit-learn, XGBoost, HuggingFace
- **Practical**: Real code examples, not just theory
- **Production-ready**: Scripts include error handling, logging, CLI interfaces
- **Progressive disclosure**: Quick guidance in SKILL.md, deep details in references

## Testing & Validation

Every skill is continuously tested — see [TEST_RESULTS.md](TEST_RESULTS.md) for the full report:

- **Schema validation**: all 28 skills validated against the Agent Skills specification (`tests/validate_skills.py`)
- **Script tests**: bundled scripts compile and pass applicable CLI tests (`tests/test_scripts.py`), run automatically in CI on every push
- **Diagram layout**: the lifecycle SVGs above are generated by [`docs/gen_lifecycle_svg.py`](docs/gen_lifecycle_svg.py) and checked by `tests/check_svg_geometry.py`, which flattens every flow wire into segments and fails if any two wires intersect or a wire crosses a card — keeping the diagrams readable as stages are added
- **Accuracy review (2026-07)**: all code examples and technical claims deep-reviewed against current library releases (MLflow 3.x, Evidently 0.7+, Great Expectations 1.x, Airflow 3.x, Prefect 3.x, TRL 1.x, RAGAS 0.2+, LangChain/LangGraph 1.x, SageMaker SDK v3) and current model pricing
- **Live AWS validation**: the model-training, model-registry, and model-serving patterns were executed end-to-end on a real AWS SageMaker account — registry lifecycle 7/7 checks passed, serverless endpoint deploy + invoke 5/5 checks passed. Reproducible scripts in [`tests/aws_validation/`](tests/aws_validation/)

## License

Apache-2.0
