# InstructRAG for Contradictory Biomedical Evidence

This project investigates whether InstructRAG-style **denoising rationales** can help a Retrieval-Augmented Generation (RAG) system answer biomedical questions when the retrieved documents contain irrelevant, insufficient, or contradictory evidence.

The central hypothesis is that examples demonstrating how to critically interpret retrieved documents can improve the model's final answer. To test this hypothesis, the experiment compares two strategies under the same conditions:

- **Baseline RAG:** uses the original RAG prompt;
- **InstructRAG-ICL:** adds previously generated denoising rationales to the prompt as in-context examples.

Both strategies receive the same questions, documents, reference answers, retrieval conditions, model settings, and evaluation metrics. Therefore, the main experimental difference is the inclusion of rationales as in-context examples.

## Experimental Workflow

```text
MedicalContradictionDetection-RAG
        │
        ├── questions, reference answers, and retrieved articles
        │
        ▼
Phase 1: Denoising rationale generation
        │
        ▼
Phase 2: Baseline RAG vs. InstructRAG-ICL
        │
        ▼
Metric analysis and paired comparisons
```

In Phase 2, the comparison is repeated under three document-selection conditions: most similar, most contradictory, and least contradictory. The answers are evaluated using ROUGE, semantic similarity, vector similarity, and word-distribution divergence metrics. Results are calculated both with and without rows classified as negative.

## Repository Structure

| Directory | Purpose |
| --- | --- |
| [`MedicalContradictionDetection-RAG`](MedicalContradictionDetection-RAG/) | Reference repository containing the data, original RAG pipeline, and metric methodology. |
| [`contradictions_instructrag_rationales`](contradictions_instructrag_rationales/) | Phase 1: generation of the denoising rationales used as examples. |
| [`contradictions_instructrag_evaluation`](contradictions_instructrag_evaluation/) | Phase 2: execution and evaluation of the Baseline RAG and InstructRAG-ICL comparison. |
| [`contradictions_instructrag_analysis`](contradictions_instructrag_analysis/) | Reproducible analysis of the results produced in Phase 2. |
| `historic_data/` | Local history of previous runs; it is not version-controlled. |

Each module has its own README with detailed information about its configuration, inputs, outputs, and implementation decisions.

## Quick Start

The project requires Python 3.10 or later. Because each stage has its own dependencies, using a separate virtual environment for each module is recommended.

### 1. Generate the Rationales

```bash
cd contradictions_instructrag_rationales
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Edit `config/rationale_generation.yaml`, configure the credentials for your selected provider, and run:

```bash
export HF_TOKEN="your-token"
.venv/bin/python src/generate_rationales.py \
  --config config/rationale_generation.yaml
```

You can validate the data integration without calling a model:

```bash
.venv/bin/python src/generate_rationales.py \
  --config config/rationale_generation.yaml \
  --validate-data-only
```

See the [Phase 1 documentation](contradictions_instructrag_rationales/README.md) for instructions on using DeepSeek or running a Hugging Face model locally.

### 2. Compare Baseline RAG and InstructRAG-ICL

After generating the rationales:

```bash
cd ../contradictions_instructrag_evaluation
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Edit `config/evaluation.yaml`, configure the credentials for your selected provider, and run:

```bash
export HF_TOKEN="your-token"
.venv/bin/python src/run_experiment.py --config config/evaluation.yaml
```

When `rationale_run_dir` is set to `null`, the loader automatically uses the most recent Phase 1 run. A data-only validation is also available:

```bash
.venv/bin/python src/run_experiment.py \
  --config config/evaluation.yaml \
  --validate-data-only
```

See the [Phase 2 documentation](contradictions_instructrag_evaluation/README.md) for all experimental conditions, metrics, providers, and output files.

### 3. Analyze the Results

```bash
cd ../contradictions_instructrag_analysis
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analyze_phase2_results.py
```

By default, the script selects the latest complete Phase 2 run and writes a new analysis to `analysis_outputs/` without changing the original results. See the [analysis documentation](contradictions_instructrag_analysis/README.md) for instructions on selecting a specific run.

## Reproducibility

The data-selection, retrieval, model, prompt, and metric settings are stored in version-controlled YAML files. Each run creates a timestamped directory containing the configuration actually used, raw generations, prompts, documents, and metric tables. API keys are read from environment variables and should not be committed to the repository.

The code in this project only reads data from `MedicalContradictionDetection-RAG`; it does not modify the reference repository or previous experimental results.
