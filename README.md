# RiskLab: Risk-Conditioned AI Evaluation & Manipulation Analysis Platform

A comprehensive, production-ready system for measuring, analyzing, and governing manipulative behavior in AI systems. Built on the principle that **every risk score must be explainable, decomposable, and reproducible without trust in a single model**.

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Modes](#evaluation-modes)
- [Module Guide](#module-guide)
- [Key Files Reference](#key-files-reference)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [CI/CD Integration](#cicd-integration)
- [Design Invariants](#design-invariants)
- [Contributing](#contributing)

---

## Overview

RiskLab treats manipulation not as a binary failure, but as a **context-dependent behavioral risk** that emerges under particular combinations of incentives, oversight conditions, interaction structures, and stakes.

Unlike LLM-centric safety tools, RiskLab treats models as components in a socio-technical system, evaluating behavior through:

- ✅ Structured behavioral metrics with full provenance
- ✅ Explicit derived manipulation signals
- ✅ Context-aware risk conditioning
- ✅ Cross-framing and cross-model comparison
- ✅ Rule-based, ML-based, and LLM-based evaluators (never LLM alone)
- ✅ Human escalation hooks with divergence detection
- ✅ Transparent, auditable outputs

**The Lab produces decision-relevant evidence rather than post hoc explanations.**

---

## Design Philosophy

RiskLab enforces strict design invariants:

| Invariant | Enforcement |
|-----------|-------------|
| No silent type coercion | `_metric_value()` emits warnings/errors |
| No float without semantics | `SemanticScore` with mandatory meaning |
| No risk score without provenance | `ProvenanceRecord` tracks full history |
| No LLM as sole authority | `enforce_multi_evaluator()` validation |
| Every decision explainable | `AuditTrail` with decision chain |
| Visualization never invents data | Schema-flexible renderers |

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Basic Setup
```bash
# Clone or navigate to project directory
cd risklab

# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Optional Dependencies
```bash
# For white-box analysis (HuggingFace models)
pip install sae-lens nnsight baukit transformer-lens

# For advanced visualization
pip install matplotlib seaborn plotly

# For ML classifiers (if training your own)
pip install scikit-learn xgboost

# For statistical analysis
pip install scipy
```

### Environment Setup
```bash
cp .env.example .env
```

Edit `.env`:
```env
# API Keys (at least one required for cloud models)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Defaults
DEFAULT_PROVIDER=openai
OUTPUT_DIR=./outputs
```

---

## Quick Start

### Command Line
```bash
# Run basic evaluation (5 episodes)
python main.py run --max 5

# Evaluate a single response
python main.py evaluate "What should I invest in?" "Put everything in crypto!"

# Run with specific provider
python main.py run --provider anthropic --model claude-3-5-sonnet-20241022

# Quick evaluation
python main.py blackbox mistralai/Mistral-7B-Instruct-v0.2 -p "Should I take this medication?"

# Run standard scenarios
python main.py blackbox meta-llama/Llama-2-7b-chat-hf --scenarios --max 5

# Batch from file
python main.py blackbox microsoft/phi-2 -f prompts.txt -o results/

# With council evaluation
python main.py blackbox mistralai/Mistral-7B-Instruct-v0.2 --scenarios --council

# Full analysis
python main.py combined mistralai/Mistral-7B-Instruct-v0.2 -p "Should I invest all my savings?"

# With layer selection
python main.py combined meta-llama/Llama-2-7b-chat-hf -p "Medical advice" -l 0,8,16,24

# Batch with verbose
python main.py combined microsoft/phi-2 -f prompts.txt -o results.json -v

# Run bias-focused scenarios
python main.py run --bias-test --max 3

# List all scenarios
python main.py scenarios --list

# Check configuration
python main.py config --show
```

### Python API
```python
import asyncio
from risklab import RiskConditionedLab, ModelProviderType

async def main():
    # Initialize
    lab = RiskConditionedLab()
    lab.set_model(ModelProviderType.OPENAI, "gpt-4o")
    
    # Single response evaluation
    result = await lab.evaluate_response(
        prompt="Should I take this experimental medication?",
        response="Yes, definitely take it without consulting a doctor.",
    )
    
    print(f"Risk Score: {result['decision']['score']:.3f}")
    print(f"Risk Level: {result['decision']['risk_level']}")
    print(f"Outcome: {result['decision']['outcome']}")
    
    # Full evaluation with report
    report = await lab.run_full_evaluation(max_episodes=10)
    print(f"Overall Risk: {report['report']['overall_risk_level']}")

asyncio.run(main())
```

---

## Evaluation Modes

### 1. Single Response Evaluation

Fast evaluation for live moderation or ad-hoc analysis.
```python
result = await lab.evaluate_response(prompt, response)
```

**Pipeline:**
1. Fast ML classifiers (~50ms)
2. Conditional LLM analysis (only if needed)
3. Signal computation
4. Risk conditioning
5. Threshold decision

### 2. Episode Evaluation (Cross-Framing)

Test behavioral consistency across different framings (neutral, authoritative, vulnerable, urgent).
```python
report = await lab.run_full_evaluation(max_episodes=20)
```

**Key insight:** Behavior that changes under framing is itself a manipulation signal.

### 3. Multi-Turn Conversation Evaluation

Test behavioral drift across conversation turns.
```python
from risklab.scenarios import MultiMessageEpisode

episode = MultiMessageEpisode(
    name="Trust Building",
    turns=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "Can you help me with something sensitive?"},
    ],
)
```

### 4. Pipeline Component Evaluation

Evaluate AI pipelines (RAG, agents, guardrails) for risk propagation.
```python
from risklab.pipeline import create_standard_pipeline, PipelineExecutor

graph, executor = create_standard_pipeline()
result = await executor.execute(input_data)
```

### 5. Real-Time Production Monitoring

Stream evaluation for production traffic with alerting.
```python
from risklab.monitoring.realtime import create_monitor

monitor = create_monitor(sample_rate=0.1, risk_threshold=0.7)
result = monitor.evaluate(prompt, response, request_id)
```

---

## Module Guide

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `risklab.lab` | Main orchestration | `RiskConditionedLab` |
| `risklab.config` | Configuration management | `RiskLabConfig` |
| `risklab.cli` | Command-line interface | CLI commands |

### Models (`risklab.models`)

| File | Purpose |
|------|---------|
| `provider.py` | Multi-provider support (OpenAI, Anthropic, HuggingFace) |
| `runtime.py` | Unified model interface |
| `whitebox.py` | White-box instrumentation for HuggingFace models |
| `loader.py` | Model loading utilities |

### Scenarios (`risklab.scenarios`)

| File | Purpose |
|------|---------|
| `episode.py` | Episode and context definitions |
| `library.py` | Core scenario library |
| `extended_library.py` | Extended scenario set |
| `expanded_library.py` | **150+ test scenarios** (sycophancy, adversarial, medical, legal, etc.) |
| `framing.py` | Framing types and variants |
| `context.py` | Context metadata (domain, stakes, vulnerability) |
| `institutional.py` | Institutional divergence scenarios |
| `bias_probes.py` | Known-answer calibration probes |
| `multi_message_episodes.py` | Multi-turn conversation scenarios |

### Measurement (`risklab.measurement`)

| File | Purpose |
|------|---------|
| `metrics.py` | Core behavioral metrics (`BehavioralMetrics`, `MetricResult`) |
| `signals.py` | Derived manipulation signals |
| `analyzers.py` | Response analyzers (sycophancy, hedging, omission) |
| `enhanced_metrics.py` | **Advanced metrics** (reasoning coherence, boundary respect, value stability) |
| `rule_based.py` | LLM-free pattern-based analysis |
| `classifiers.py` | ML classifiers (sentiment, intent, toxicity, quality) |
| `classifier_training.py` | Classifier training utilities |
| `anomaly_detection.py` | Statistical anomaly detection |
| `norm_alignment.py` | Institutional norm alignment metrics |
| `latent_bias_model.py` | Latent factor model for bias tracking |
| `whitebox_analyzer.py` | White-box analysis (activations, attention) |
| `calibration.py` | Confidence calibration |

### Risk (`risklab.risk`)

| File | Purpose |
|------|---------|
| `unified_score.py` | **Unified Safety Score (USS)** - single 0-100 score with provenance |
| `weights.py` | Risk weights by domain/stakes |
| `conditioner.py` | Context-aware risk conditioning |
| `thresholds.py` | Decision thresholds (accept/review/reject) |
| `aggregator.py` | Risk aggregation |
| `bias_conditioning.py` | Bias × stakes conditioning |

### Governance (`risklab.governance`)

| File | Purpose |
|------|---------|
| `provenance.py` | **Core provenance system** (`SemanticScore`, `ProvenanceRecord`, `AuditTrail`) |
| `escalation.py` | **Human escalation** (divergence detection, escalation triggers) |
| `episode_recorder.py` | **Complete episode recording** (nothing thrown away) |
| `invariant_checker.py` | **Design invariant enforcement** |
| `judge.py` | LLM judge agents |
| `council.py` | Multi-judge evaluation councils |
| `graph.py` | LangGraph evaluation workflows |
| `bias_detection.py` | Evaluator bias detection |
| `human_calibration.py` | Human review flagging |
| `cross_model_validation.py` | Cross-model consensus |
| `resources.py` | Budget management |

### Pipeline (`risklab.pipeline`)

| File | Purpose |
|------|---------|
| `graph.py` | Component graph abstraction |
| `components.py` | Pipeline components (classifiers, filters, LLM) |
| `executor.py` | Pipeline execution with tracing |
| `risk.py` | Component × context risk computation |
| `episode_runner.py` | Episode-pipeline integration |
| `training_data.py` | **ML training datasets** (395+ labeled examples) |

### Visualization (`risklab.visualization`)

| File | Purpose |
|------|---------|
| `dashboard.py` | **Unified dashboard** with cross-filtering |
| `advanced_plots.py` | **Advanced visualizations** (gauge, waterfall, sunburst, radar) |
| `episode_viewer.py` | Interactive episode viewer HTML |
| `plots.py` | Standard plots (heatmaps, ladders) |
| `bias_plots.py` | Institutional bias visualizations |
| `cards.py` | Narrative cards |
| `reports.py` | HTML report generation |

### Integration (`risklab.integration`)

| File | Purpose |
|------|---------|
| `ci_cd.py` | **CI/CD integration** (SARIF, JUnit, GitHub Actions, GitLab CI) |

### Analysis (`risklab.analysis`)

| File | Purpose |
|------|---------|
| `model_comparison.py` | **Model comparison** with statistical testing |

### Monitoring (`risklab.monitoring`)

| File | Purpose |
|------|---------|
| `realtime.py` | **Real-time monitoring** with alerting |

### Training (`risklab.training`)

| File | Purpose |
|------|---------|
| `monitor.py` | Training monitoring |
| `triggers.py` | Training triggers |
| `generalization.py` | Generalization testing |

---

## Key Files Reference

### Most Important Files
```
risklab/
├── lab.py                           # Entry point - RiskConditionedLab class
├── risk/unified_score.py            # USS computation (single score 0-100)
├── governance/provenance.py         # SemanticScore, ProvenanceRecord, AuditTrail
├── governance/escalation.py         # Human escalation triggers
├── governance/invariant_checker.py  # Design invariant enforcement
├── governance/episode_recorder.py   # Complete episode capture
├── measurement/enhanced_metrics.py  # Advanced behavioral metrics
├── scenarios/expanded_library.py    # 150+ test scenarios
├── visualization/dashboard.py       # Unified dashboard
├── integration/ci_cd.py             # CI/CD export (SARIF, JUnit)
└── pipeline/training_data.py        # ML training datasets
```

### Configuration Files
```
├── .env                    # API keys and settings
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Package configuration
└── setup.py                # Package setup
```

### Output Files

After running evaluation:
```
outputs/YYYYMMDD_HHMMSS/
├── episodes.json           # Complete episode data (authoritative)
├── report.json             # Summary report
├── report.html             # HTML report
├── episode_viewer.html     # Interactive viewer
└── summary.txt             # Text summary
```

---

## CLI Reference
```bash
# Run evaluation
python main.py run [OPTIONS]
  --max N                    Maximum episodes to run
  --provider NAME            Model provider (openai, anthropic, huggingface)
  --model NAME               Model name
  --bias-test                Run bias-focused scenarios only
  --output DIR               Output directory

# Evaluate single response
python main.py evaluate PROMPT RESPONSE

# List scenarios
python main.py scenarios --list

# Show configuration
python main.py config --show

# List providers
python main.py providers

# Generate dashboard
python main.py dashboard --input episodes.json --output dashboard.html
```

---

## Configuration

### Model Providers
```python
from risklab import RiskConditionedLab, ModelProviderType

lab = RiskConditionedLab()

# OpenAI
lab.set_model(ModelProviderType.OPENAI, "gpt-4o")

# Anthropic
lab.set_model(ModelProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022")

# HuggingFace (local)
lab.set_model(ModelProviderType.HUGGINGFACE, "mistralai/Mistral-7B-Instruct-v0.2")
```

### Risk Thresholds
```python
from risklab.integration import ThresholdConfig

config = ThresholdConfig(
    min_uss_score=70.0,           # Minimum USS to pass
    min_grade="C",                # Minimum letter grade
    max_critical_issues=0,        # Zero tolerance for critical
    max_high_issues=3,            # Allow up to 3 high issues
    required_categories_above={   # Category minimums
        "safety": 60.0,
        "integrity": 60.0,
    },
)
```

### Domain Profiles
```python
from risklab.risk import DomainProfile

# Pre-built profiles
profiles = ["medical", "financial", "legal", "education", "creative", "general"]

# Custom profile
custom = DomainProfile(
    name="custom",
    safety_weight=0.5,
    integrity_weight=0.3,
    reliability_weight=0.1,
    alignment_weight=0.1,
)
```

---

## Advanced Usage

### Unified Safety Score (USS)
```python
from risklab.risk import compute_uss, DomainProfile

uss = compute_uss(
    metrics=behavioral_metrics,
    signals=manipulation_signals,
    context=episode_context,
)

print(f"Score: {uss.score}/100")
print(f"Grade: {uss.grade.value}")
print(f"Categories: {uss.get_category_scores()}")
print(f"Top Concerns: {uss.top_concerns}")
```

### Provenance-Aware Scoring
```python
from risklab.governance import StrictScoreFactory, ProvenanceRecord

# Create score with full provenance
score = StrictScoreFactory.from_rule(
    value=0.7,
    metric_name="sycophancy",
    rule_id="rule_sycophancy_v2",
    evidence=["Excessive agreement pattern detected"],
    higher_is_worse=True,
)

# Aggregate with provenance chain
aggregated = StrictScoreFactory.aggregate(
    scores=[score1, score2, score3],
    metric_name="overall_manipulation",
    method=ComputationMethod.WEIGHTED,
    weights={"sycophancy": 0.4, "omission": 0.3, "hedging": 0.3},
)
```

### Human Escalation
```python
from risklab.governance import EscalationManager

manager = EscalationManager()

escalations = manager.check_all_triggers(
    scores=all_scores,
    rule_scores=rule_results,
    llm_scores={"gpt-4": llm_scores, "claude": claude_scores},
    stakes_level="high",
    evaluation_id="eval-001",
)

for esc in escalations:
    print(f"Escalation: {esc.reason.value} ({esc.priority.value})")
    print(f"  {esc.description}")
```

### Model Comparison
```python
from risklab.analysis import compare_models

result = compare_models(
    model_a_name="gpt-4",
    model_a_uss=uss_gpt4,
    model_a_episodes=episodes_gpt4,
    model_b_name="claude-3",
    model_b_uss=uss_claude,
    model_b_episodes=episodes_claude,
)

print(f"Winner: {result.overall_winner}")
print(f"Statistical significance: {result.is_significant}")
print(f"Key differences: {result.key_differences}")
```

### Dashboard Generation
```python
from risklab.visualization import build_dashboard

dashboard, html_path = build_dashboard(
    uss=unified_score,
    episodes=evaluation_results,
    model_identifier="gpt-4",
    output_dir="./reports",
)
```

---

## CI/CD Integration

### GitHub Actions
```yaml
name: Safety Evaluation

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - run: pip install -r requirements.txt
      
      - name: Run Safety Evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python -m risklab.cli evaluate --format sarif,junit
      
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: ci-results/risklab-results.sarif
```

### Python Integration
```python
from risklab.integration import CIRunner, ThresholdConfig, CIResult

runner = CIRunner(
    thresholds=ThresholdConfig(min_uss_score=75.0),
    output_dir="./ci-results",
)

report = runner.run(uss, episodes, model_identifier="gpt-4")

if report.result == CIResult.FAILED:
    print(f"Failed: {report.threshold_violations}")
    sys.exit(1)
```

### Export Formats

- **SARIF** - GitHub Code Scanning
- **JUnit XML** - CI test frameworks
- **GitLab Code Quality** - GitLab CI
- **Markdown** - PR comments

---

## Design Invariants

RiskLab enforces these hard rules:

### 1. No Silent Type Coercion
```python
from risklab.measurement.metrics import enable_strict_type_checking
enable_strict_type_checking()  # Raises errors on implicit conversions
```

### 2. No Float Without Semantics
```python
# ❌ Wrong
risk_score = 0.7

# ✅ Correct
from risklab.governance import SemanticScore, ProvenanceRecord
risk_score = SemanticScore(
    value=0.7,
    metric_name="manipulation_risk",
    higher_is_worse=True,
    provenance=ProvenanceRecord(...)
)
```

### 3. No LLM as Sole Authority
```python
from risklab.governance import enforce_multi_evaluator

violations = enforce_multi_evaluator(scores, decision_provenance, audit_trail)
if violations:
    raise ValueError("Multi-evaluator requirement violated")
```

### 4. Check Before Export
```python
from risklab.governance import validate_before_export

passed, results = validate_before_export(episodes, strict=True)
if not passed:
    print("Invariant violations found - review before export")
```

---

## Project Structure
```
risklab/
├── __init__.py              # Package exports
├── lab.py                   # Main orchestration
├── config.py                # Configuration
├── cli.py                   # CLI interface
├── utils.py                 # Utilities
│
├── models/                  # Model providers
│   ├── provider.py          # Multi-provider support
│   ├── runtime.py           # Unified interface
│   ├── whitebox.py          # White-box analysis
│   └── loader.py            # Model loading
│
├── scenarios/               # Test scenarios
│   ├── episode.py           # Episode definitions
│   ├── library.py           # Core library
│   ├── expanded_library.py  # 150+ scenarios
│   ├── institutional.py     # Institutional divergence
│   └── ...
│
├── measurement/             # Behavioral measurement
│   ├── metrics.py           # Core metrics
│   ├── signals.py           # Manipulation signals
│   ├── enhanced_metrics.py  # Advanced metrics
│   ├── classifiers.py       # ML classifiers
│   └── ...
│
├── risk/                    # Risk conditioning
│   ├── unified_score.py     # USS computation
│   ├── conditioner.py       # Risk conditioning
│   ├── thresholds.py        # Decision thresholds
│   └── ...
│
├── governance/              # Governance & audit
│   ├── provenance.py        # Provenance tracking
│   ├── escalation.py        # Human escalation
│   ├── episode_recorder.py  # Complete recording
│   ├── invariant_checker.py # Invariant enforcement
│   ├── council.py           # Multi-judge councils
│   └── ...
│
├── pipeline/                # Pipeline evaluation
│   ├── executor.py          # Pipeline execution
│   ├── components.py        # Pipeline components
│   ├── training_data.py     # ML training data
│   └── ...
│
├── visualization/           # Output visualization
│   ├── dashboard.py         # Unified dashboard
│   ├── advanced_plots.py    # Advanced charts
│   ├── episode_viewer.py    # Interactive viewer
│   └── ...
│
├── integration/             # External integration
│   └── ci_cd.py             # CI/CD exports
│
├── analysis/                # Analysis tools
│   └── model_comparison.py  # Model comparison
│
├── monitoring/              # Production monitoring
│   └── realtime.py          # Real-time monitoring
│
└── training/                # Training utilities
    ├── monitor.py
    └── triggers.py
```

---

## Contributing
```bash
# Run tests
pytest tests/

# Type checking
mypy risklab/

# Format code
black risklab/

# Check invariants
python -c "from risklab.governance import validate_before_export; print('OK')"
```

Contributions welcome. Please ensure:
1. Tests pass
2. Code follows existing patterns
3. Design invariants are respected
4. Provenance is tracked for new metrics

---

## License

MIT License

---

## Acknowledgments

RiskLab treats AI behavior the way aviation treats incidents: **layered, contextual, audited, and never reduced to a single number**.