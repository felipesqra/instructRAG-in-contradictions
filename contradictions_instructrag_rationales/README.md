# InstructRAG-Style Rationale Generation for Medical Contradictions

This module generates denoising rationales for the medical contradictions experiment using real data from `MedicalContradictionDetection-RAG`.

It implements only rationale generation. It does not perform final answer generation, model evaluation, fine-tuning, or Vanilla RAG comparison.

## Purpose

For each selected question-remedy pair, the generator sends a model:

- the original question from the contradictions repository;
- the official/original answer for that question;
- the retrieved PubMed abstracts for the same question and remedy.

The model returns a concise InstructRAG-style rationale explaining how the retrieved documents should be interpreted before producing the official answer.

## Data Sources

The loader reads from `MedicalContradictionDetection-RAG` and leaves that repository untouched.

- Questions and retrieved abstracts come from `Datasets/Contradiction/<remedy>/<remedy>_abstract.json`, specifically `retrieved_ranked_docs[*].query` and `retrieved_ranked_docs[*].retrieved_docs`.
- Official answers come from `Result/*-response.json`, using `long_ground_truth` when available and `short_ground_truth` otherwise.
- Remedy names come from the abstract JSON `medicine` field, with the folder name as fallback.
- PMID, year, retrieval order, titles, authors, MeSH terms, and contradiction/similarity metadata are preserved when available.

## Unsupported Retrieved Evidence

Unsupported cases still receive a denoising rationale. The model is instructed to explain which documents were inspected and why they are irrelevant, insufficient, noisy, off-topic, or contradictory.

The negative sentence is used only when no retrieved document supports any part of the official answer. If at least one document provides partial support, the rationale should explain that support and should not include the negative sentence.

The fixed negative sentence is not used as a replacement rationale. It is appended only at the end of unsupported rationales:

```text
After reviewing the provided documents, I found that none of them contain relevant information to answer the question.
```

External biomedical knowledge fallback is disabled. The prompt explicitly instructs the model to use only the provided documents and not to justify the official answer from outside knowledge.

## Configure

Edit `config/rationale_generation.yaml`.

Key settings:

- `project.contradictions_repo_path`: path to `MedicalContradictionDetection-RAG`.
- `selection.question_start_index` and `selection.num_questions`: which question indices to use for each selected remedy.
- `selection.num_remedies`: how many remedies to process.
- `selection.remedy_selection_strategy`: `sequential` or `random`.
- `documents.max_documents_per_rationale`: number of retrieved abstracts to include in each prompt.
- `model.provider`: `deepseek`, `huggingface`, or `huggingface_local`.
- `prompt.negative_template`: exact sentence required at the end of unsupported rationales.

## Run With DeepSeek

Install dependencies, set the API key, then run from this directory:

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-key"
python src/generate_rationales.py --config config/rationale_generation.yaml
```

The DeepSeek client uses the OpenAI-compatible API, reads `DEEPSEEK_API_KEY` by default, and supports `model_name`, `temperature`, `max_new_tokens`, and `timeout_seconds`.

## Run With Hugging Face

Set `model.provider: "huggingface"` in the YAML and set `HF_TOKEN`. This uses Hugging Face Inference Providers through the router API; it does not load the model locally.

```bash
pip install -r requirements.txt
export HF_TOKEN="your-token"
python src/generate_rationales.py --config config/rationale_generation.yaml
```

The Hugging Face router is configured by `huggingface.base_url`, defaulting to `https://router.huggingface.co/v1`. To force a specific provider or routing policy, encode it in `model.model_name`, for example `openai/gpt-oss-120b:fastest` or a provider suffix supported by Hugging Face.

For the older local `transformers` path, use `model.provider: "huggingface_local"`. That optional path uses `huggingface.model_name`, `device`, `torch_dtype`, and `trust_remote_code`.

## Preflight Check

To verify data discovery and prompt construction without calling a model:

```bash
python src/generate_rationales.py --config config/rationale_generation.yaml --validate-data-only
```

This does not synthesize rationales. It only checks that selected real questions, official answers, and retrieved abstracts can be joined.

## Outputs

Each generation run is saved under:

```text
outputs/{timestamp}_{run_name}/
```

Files:

- `config_used.yaml`: exact config used for the run.
- `rationales.jsonl`: one record per selected question-remedy pair.
- `rationales.csv`: compact summary for spreadsheet inspection.
- `prompts.jsonl`: prompts sent to the model when enabled.
- `documents.jsonl`: retrieved documents used when enabled.
- `run_summary.json`: run counts and data discovery summary.

Each `rationales.jsonl` record includes the question, remedy, official answer, document IDs, rationale, status, traceability metadata, and whether the rationale ends with the configured negative template.

## Out of Scope

This module intentionally does not:

- generate final answers;
- evaluate model answers;
- compare against Vanilla RAG;
- fine-tune a model;
- modify `MedicalContradictionDetection-RAG`;
- modify `InstructRAG`;
- create synthetic questions, answers, or retrieved documents.
