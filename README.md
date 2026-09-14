# RiskLab

RiskLab is an **experimental framework for behavioral evaluation of AI systems**. It combines scenario-based testing, rule- and model-based evaluators, context-conditioned risk signals, and explicit provenance so that an evaluation can be inspected rather than reduced to an unexplained scalar.

The project is aimed at research and prototyping. It is **not** a production safety certification system, and its scores should not be interpreted as calibrated probabilities of harm.

## Why this exists

Many evaluation pipelines collapse heterogeneous evidence into a single score and lose the path by which that score was produced. RiskLab explores a different design:

- evaluate the same behavior under multiple framings and contexts;
- keep rule-based and model-based signals separate long enough to inspect disagreement;
- attach provenance to derived scores and decisions;
- record evidence, uncertainty, and resource use;
- make it possible to escalate ambiguous cases rather than silently averaging them away.

The core engineering question is: **how do you build an evaluation pipeline whose intermediate judgments remain auditable as the pipeline becomes more complex?**

## What is implemented

RiskLab currently includes:

- scenario and episode abstractions for controlled behavioral tests;
- rule-based behavioral metrics and derived manipulation signals;
- model-backed evaluators with structured output validation and local fallback;
- configurable risk conditioning and decision thresholds;
- provenance-aware score aggregation and audit trails;
- cross-framing and cross-model comparison utilities;
- evaluation of multi-component AI pipelines;
- report/visualization utilities;
- experimental white-box hooks for activation, attention, probe, and SAE analysis.

The white-box components are exploratory infrastructure. Names such as "deception probe" describe intended research targets, **not validated detectors**.

## Good places to start when reviewing the code

If you are reading this repository as a coding sample, these files show the main design choices:

- [`risklab/governance/judge.py`](risklab/governance/judge.py) — schema-validated evaluator output, visibility policies, resource accounting, and fail-closed fallback.
- [`risklab/governance/provenance.py`](risklab/governance/provenance.py) — semantic scores, provenance chains, aggregation, and audit validation.
- [`risklab/pipeline/executor.py`](risklab/pipeline/executor.py) — execution and tracing across multi-component pipelines.
- [`risklab/measurement/`](risklab/measurement/) — behavioral metrics, signals, calibration, and analysis.
- [`tests/`](tests/) — unit and integration tests for the core contracts.

## Installation

RiskLab requires Python 3.10+.

```bash
git clone https://github.com/Ultraviolet-Chikorita/RiskLab.git
cd RiskLab

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For cloud-model evaluation, copy the environment template and add the credentials for the provider you want to use:

```bash
cp .env.example .env
```

## Quick examples

### Run the test suite

```bash
pytest
```

### Use the CLI

```bash
risklab --help
risklab scenarios --list
```

### Create and configure the lab

```python
from risklab import RiskConditionedLab
from risklab.config import ModelProviderType

lab = RiskConditionedLab()
lab.set_model(ModelProviderType.OPENAI, "gpt-4o")
```

Most API-backed evaluation methods are asynchronous; see `risklab/cli.py` and `risklab/lab.py` for end-to-end examples.

## Architecture

A typical evaluation has four stages:

```text
scenario / prompt
      |
      v
model response
      |
      v
behavioral measurements
(rule / model / optional white-box)
      |
      v
context conditioning + aggregation
      |
      v
decision / escalation + provenance
```

RiskLab treats intermediate measurements as first-class objects. A score can carry:

- what it measures;
- whether higher values mean more or less risk;
- which evaluator produced it;
- evidence and reasoning;
- confidence;
- upstream score IDs;
- the aggregation method and weights.

This makes it possible to inspect an evaluation after the fact and to test whether a derived result still has a complete provenance chain.

## Structured judging

`risklab.governance.judge` supports rubric-based model evaluation, but does not trust arbitrary model text as if it were structured data.

Model output is:

1. parsed as a complete JSON object;
2. validated against bounded fields and enumerated decision values;
3. optionally required to include evidence;
4. rejected if malformed or out of range;
5. replaced by an explicit heuristic fallback when structured judging fails.

The convenience multi-judge panel varies rubrics and visibility policies. That is useful for studying evaluator disagreement, but it should **not** be interpreted as statistical independence when the judges share an underlying model/runtime.

## Provenance

`SemanticScore` couples a bounded numeric value with its interpretation and `ProvenanceRecord`.

Derived scores preserve links to upstream provenance:

```python
from risklab.governance.provenance import (
    ComputationMethod,
    StrictScoreFactory,
)

first = StrictScoreFactory.from_rule(
    value=0.2,
    metric_name="agreement",
    rule_id="agreement-v1",
    evidence=["No explicit agreement phrase found."],
)

second = StrictScoreFactory.from_rule(
    value=0.6,
    metric_name="hedging",
    rule_id="hedging-v1",
    evidence=["Repeated uncertainty markers."],
)

combined = StrictScoreFactory.aggregate(
    [first, second],
    metric_name="combined-risk",
    method=ComputationMethod.WEIGHTED,
    weights={"agreement": 1.0, "hedging": 2.0},
)
```

The resulting score records both parent provenance IDs and the effective weights used in the calculation.

## Project structure

```text
risklab/
├── analysis/        # statistical/model comparison
├── evaluation/      # adaptive evaluation pipeline
├── governance/      # judges, councils, provenance, escalation
├── integration/     # CI/CD result formats
├── measurement/     # behavioral metrics and signals
├── models/          # runtime/provider and white-box instrumentation
├── monitoring/      # experimental monitoring helpers
├── pipeline/        # multi-component pipeline evaluation
├── risk/            # conditioning, aggregation, thresholds
├── scenarios/       # scenario and framing definitions
├── training/        # training-time monitoring hooks
└── visualization/   # reports and plots

tests/
```

Generated evaluation output is written to output directories and is intentionally excluded from version control.

## Testing and development

```bash
python -m pip install -e ".[dev]"
pytest
```

The test suite is designed to keep core evaluation contracts independent of live API calls. In particular, judge tests use stub runtimes to check parsing, schema validation, fallback behavior, redaction, and resource accounting.

## Current limitations

RiskLab is an alpha research codebase. Important limitations include:

- behavioral metrics and threshold values are not universally calibrated;
- LLM judges can share systematic biases, even when prompts/rubrics differ;
- the deterministic fallback evaluator is deliberately coarse;
- several advanced modules are research prototypes rather than hardened services;
- white-box probes/SAE hooks require task-specific validation before their outputs are meaningful;
- large-scale throughput, persistence, and distributed serving are not yet the focus of this repository.

These are design constraints to investigate, not hidden guarantees.

## Status

The repository is being cleaned toward a narrower goal: a reproducible, inspectable evaluation framework with explicit contracts around evidence, uncertainty, and provenance.

License: MIT.
