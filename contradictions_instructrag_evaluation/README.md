# Phase 2: Baseline RAG vs InstructRAG-ICL RAG

This directory implements Phase 2 of the medical contradictions experiment.

Phase 1 generated InstructRAG-style denoising rationales in `contradictions_instructrag_rationales/`. Phase 2 compares two answer-generation arms:

- `baseline_rag`: original/common RAG prompt body from `MedicalContradictionDetection-RAG/Codes/rag_pipeline.py`, with the shared unsupported-evidence instruction appended.
- `instructrag_icl_rag`: the same target question and retrieved documents, but with Phase 1 denoising rationales injected as in-context examples.

The intended experimental difference is only the prompt strategy. Both arms use the same selected remedies, questions, official answers, retrieval condition, retrieved documents, model settings, and metric protocol.

## What Is Reused

The module reads original data from `MedicalContradictionDetection-RAG` and does not modify that repository.

- Original RAG prompt body: `Codes/rag_pipeline.py`, where the prompt template is `Given the following medical abstracts:\n{context}\nAnswer the question:\n{input}`.
- Shared unsupported-evidence behavior: both answer-generation arms are instructed to return the configured negative template exactly when none of the retrieved documents contains relevant information to answer the question.
- Original retrieval ordering behavior: `rag_pipeline.py` preserves the stored ranked order with `FirstKOrderRetriever`.
- Original questions and retrieved abstracts: `Datasets/Contradiction/<remedy>/<remedy>_abstract.json`.
- Official/reference answers: `Result/*-response.json`, using `long_ground_truth` when available and `short_ground_truth` otherwise.
- Contradiction metadata for condition-specific document selection: `Datasets/Contradiction/<remedy>/<remedy>_contradiction.json`.
- Metric methodology: `Codes/llm-analysis.ipynb`.

The original metric code is in a notebook, not an importable Python module. The formulas are therefore reimplemented in `src/metrics.py` with a source check that stops execution if the expected notebook functions are missing.

## Retrieval Conditions

For each selected question-remedy pair, the same selected documents are sent to both model arms.

- `most_similar`: first `K` ranked documents, matching the original `FirstKOrderRetriever` behavior.
- `most_contradictory`: documents with the highest max `best_contradiction_score` from the contradiction JSON.
- `least_contradictory`: documents with the lowest max `best_contradiction_score` from the contradiction JSON.

When `preserve_original_document_order: true`, selected documents are restored to their original ranked order before prompting.

## Metrics

Metric source: `MedicalContradictionDetection-RAG/Codes/llm-analysis.ipynb`.

Implemented metrics:

- `rouge1`, `rouge2`, `rougeL`: F-measure ROUGE after the notebook-style preprocessing.
- `semantic_cosine`, `semantic_dot`: the notebook calls this `bert`; it is embedding cosine/dot similarity using `sentence-transformers` model `all-distilroberta-v1`.
- `vsim`: Word2Vec-style vector similarity based on the notebook's `word2vec-google-news-300` Gensim model.
- `jsd`, `kld`: Jensen-Shannon and Kullback-Leibler divergence over word distributions.

The implementation does not call the semantic score BERTScore, because the original repository computes embedding cosine/dot similarity rather than BERTScore.

## Negative Rows

A row is marked negative when either generated answer contains the configured negative template, case-insensitively after whitespace normalization:

```text
After reviewing the provided documents, I found that none of them contain relevant information to answer the question.
```

The run writes result sets both with and without these rows.

Both `baseline_rag` and `instructrag_icl_rag` receive an explicit instruction to emit this exact template when the retrieved documents do not contain relevant information to answer the question.

## Configure

Edit `config/evaluation.yaml`.

Common smoke-test settings:

```yaml
selection:
  question_start_index: 0
  num_questions: 1
  num_remedies: 1

retrieval:
  retrieval_conditions:
    - "most_similar"
  max_documents_per_query: 2

rationales:
  max_icl_examples: 1
```

## Run

From this directory:

```bash
python src/run_experiment.py --config config/evaluation.yaml --validate-data-only
```

With DeepSeek:

```bash
export DEEPSEEK_API_KEY="your-key"
python src/run_experiment.py --config config/evaluation.yaml
```

With Hugging Face Inference Providers:

```bash
export HF_TOKEN="your-token"
python src/run_experiment.py --config config/evaluation.yaml
```

`provider: "huggingface"` uses the Hugging Face router API configured by `huggingface.base_url`, defaulting to `https://router.huggingface.co/v1`; it does not load a local `transformers` model. Use `provider: "huggingface_local"` only for the older local path.

With GroqCloud:

```yaml
models:
  baseline:
    provider: "groq"
    model_name: "llama-3.3-70b-versatile"
  instructrag_icl:
    provider: "groq"
    model_name: "llama-3.3-70b-versatile"
```

```bash
export GROQ_API_KEY="your-key"
python src/run_experiment.py --config config/evaluation.yaml
```

`provider: "groq"` uses Groq's OpenAI-compatible endpoint configured by `groq.base_url`, defaulting to `https://api.groq.com/openai/v1`.

If using the `.venv` created elsewhere, run with the corresponding interpreter, for example:

```bash
.venv/bin/python src/run_experiment.py --config config/evaluation.yaml
```

## Outputs

Each run creates:

```text
outputs/{timestamp}_{run_name}/
  config_used.yaml
  run_summary.json
  raw_generations.jsonl
  prompts.jsonl
  documents.jsonl
  per_example_metrics_all_rows.csv
  per_example_metrics_without_negative_rows.csv
  aggregate_metrics_with_negative_rows.csv
  aggregate_metrics_without_negative_rows.csv
  paired_comparison_with_negative_rows.csv
  paired_comparison_without_negative_rows.csv
```

## Out Of Scope

This module does not:

- fine-tune models;
- create synthetic questions;
- create synthetic official answers;
- modify `MedicalContradictionDetection-RAG`;
- modify `InstructRAG`;
- modify Phase 1 rationale outputs;
- compare arms with different retrieved documents for the same target example.
