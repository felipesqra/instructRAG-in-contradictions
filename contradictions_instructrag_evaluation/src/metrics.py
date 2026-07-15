from __future__ import annotations

import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from config import MetricsConfig
from utils import mean


class MetricMethodologyError(RuntimeError):
    pass


ORIGINAL_METRIC_NOTEBOOK = "Codes/llm-analysis.ipynb"
ORIGINAL_SEMANTIC_ENCODER = "all-distilroberta-v1"
ORIGINAL_GENSIM_MODEL = "word2vec-google-news-300"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with", "you", "your", "i",
    "am", "do", "does", "did", "from", "have", "has", "had", "while", "what", "why",
    "how", "before", "after", "any", "all", "can", "may", "should", "using", "use",
}


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise MetricMethodologyError("numpy is required to compute Phase 2 metrics.") from exc
    return np


def verify_original_metric_methodology(repo_path: Path) -> dict[str, Any]:
    notebook = repo_path / ORIGINAL_METRIC_NOTEBOOK
    if not notebook.exists():
        raise MetricMethodologyError(
            f"Original metric notebook not found: {notebook}. "
            "Cannot silently replace the MedicalContradictionDetection-RAG metric methodology."
        )
    text = notebook.read_text(encoding="utf-8")
    required = ["def calc_rouge", "def calc_bert", "def calc_gensim", "def cal_prob_score"]
    missing = [item for item in required if item not in text]
    if missing:
        raise MetricMethodologyError(
            f"Original metric notebook is missing expected functions: {missing}. "
            "Cannot silently replace the metric methodology."
        )
    return {
        "source": ORIGINAL_METRIC_NOTEBOOK,
        "status": "found_notebook_logic_reimplemented_in_phase2_metrics_py",
        "original_functions": required,
        "semantic_encoder_used_by_original": ORIGINAL_SEMANTIC_ENCODER,
        "gensim_model_used_by_original": ORIGINAL_GENSIM_MODEL,
        "note": "The original implementation is in a notebook, so Phase 2 reimplements the same formulas in src/metrics.py.",
    }


class MetricComputer:
    def __init__(self, repo_path: Path, config: MetricsConfig) -> None:
        self.methodology = verify_original_metric_methodology(repo_path)
        self.config = config
        self._semantic_model = None
        self._gensim_model = None

    def compute(self, reference: str, prediction: str) -> dict[str, float | None]:
        processed_reference = preprocess_text(reference)
        processed_prediction = preprocess_text(prediction)
        out = {
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
            "semantic_cosine": None,
            "semantic_dot": None,
            "vsim": None,
            "jsd": None,
            "kld": None,
        }
        requested = set(self.config.metrics_to_compute)
        if requested & {"rouge1", "rouge2", "rougeL"}:
            out.update(calc_rouge(processed_reference, processed_prediction))
        if requested & {"semantic_cosine", "semantic_dot"}:
            semantic = self.calc_semantic(processed_reference, processed_prediction)
            out["semantic_cosine"] = semantic["semantic_cosine"]
            out["semantic_dot"] = semantic["semantic_dot"]
        if "vsim" in requested:
            out["vsim"] = self.calc_gensim_vsim(processed_reference, processed_prediction)
        if requested & {"jsd", "kld"}:
            prob = calc_prob_score(processed_reference, processed_prediction)
            out["jsd"] = prob["jsd"]
            out["kld"] = prob["kld"]
        return out

    def calc_semantic(self, reference: str, prediction: str) -> dict[str, float]:
        if not reference or not prediction:
            return {"semantic_cosine": 0.0, "semantic_dot": 0.0}
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as exc:
            raise MetricMethodologyError(
                "sentence-transformers is required for the original notebook's calc_bert metric."
            ) from exc
        if self._semantic_model is None:
            self._semantic_model = SentenceTransformer(ORIGINAL_SEMANTIC_ENCODER)
        ref_embedding = self._semantic_model.encode(reference, convert_to_tensor=True)
        pred_embedding = self._semantic_model.encode(prediction, convert_to_tensor=True)
        return {
            "semantic_cosine": util.cos_sim(ref_embedding, pred_embedding)[0][0].item(),
            "semantic_dot": util.dot_score(ref_embedding, pred_embedding)[0][0].item(),
        }

    def calc_gensim_vsim(self, reference: str, prediction: str) -> float:
        if not reference or not prediction:
            return 0.0
        try:
            import gensim.downloader as api
        except ImportError as exc:
            raise MetricMethodologyError("gensim is required for the original notebook's calc_gensim metric.") from exc
        if self._gensim_model is None:
            self._gensim_model = api.load(ORIGINAL_GENSIM_MODEL)
        np = _np()
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        vocab = set(ref_tokens + pred_tokens)
        if not vocab:
            return 0.0
        ref_counts = Counter(ref_tokens)
        pred_counts = Counter(pred_tokens)
        ref_vec = np.zeros(self._gensim_model.vector_size)
        pred_vec = np.zeros(self._gensim_model.vector_size)
        for token in vocab:
            if token in self._gensim_model:
                ref_vec += ref_counts[token] * self._gensim_model[token]
                pred_vec += pred_counts[token] * self._gensim_model[token]
        denom = np.linalg.norm(ref_vec) * np.linalg.norm(pred_vec)
        return float(np.dot(ref_vec, pred_vec) / denom) if denom else 0.0


def preprocess_text(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", (text or "").lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in STOPWORDS]
    return " ".join(tokens)


def calc_rouge(reference: str, prediction: str) -> dict[str, float]:
    if not reference or not prediction:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise MetricMethodologyError("rouge-score is required to compute ROUGE metrics.") from exc
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_word_distribution(text: str) -> dict[str, float]:
    tokens = text.lower().split()
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def _entropy(p, q) -> float:
    np = _np()
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    return float(np.sum(np.where(p != 0, p * np.log(p / q), 0)))


def calc_prob_score(reference: str, prediction: str) -> dict[str, float]:
    np = _np()
    p_dist = compute_word_distribution(reference)
    q_dist = compute_word_distribution(prediction)
    words = sorted(set(p_dist) | set(q_dist))
    if not words:
        return {"jsd": 0.0, "kld": 0.0}
    p = np.array([p_dist.get(word, 0.0) for word in words])
    q = np.array([q_dist.get(word, 0.0) for word in words])
    m = 0.5 * (p + q)
    jsd = 0.5 * (_entropy(p, m) + _entropy(q, m))
    kld = _entropy(q, p)
    return {"jsd": float(jsd), "kld": float(kld)}


METRIC_COLUMNS = ["rouge1", "rouge2", "rougeL", "semantic_cosine", "semantic_dot", "vsim", "jsd", "kld"]


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["model_arm"], row["provider"], row["model_name"], row["retrieval_condition"])
        groups.setdefault(key, []).append(row)
    out = []
    for (model_arm, provider, model_name, retrieval_condition), group in sorted(groups.items()):
        record = {
            "model_arm": model_arm,
            "provider": provider,
            "model_name": model_name,
            "retrieval_condition": retrieval_condition,
            "n_examples": len(group),
        }
        for metric in METRIC_COLUMNS:
            record[f"{metric}_mean"] = mean(row.get(metric) for row in group)
        out.append(record)
    return out


def paired_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["question_index"], row["question"], row["remedy"], row["retrieval_condition"])
        paired.setdefault(key, {})[row["model_arm"]] = row
    for key, arms in paired.items():
        if "baseline_rag" in arms and "instructrag_icl_rag" in arms:
            by_condition.setdefault(key[3], []).append(arms)

    out = []
    for condition, pairs in sorted(by_condition.items()):
        record = {"retrieval_condition": condition, "n_paired_examples": len(pairs)}
        for metric in METRIC_COLUMNS:
            baseline_mean = mean(pair["baseline_rag"].get(metric) for pair in pairs)
            instruct_mean = mean(pair["instructrag_icl_rag"].get(metric) for pair in pairs)
            record[f"baseline_{metric}_mean"] = baseline_mean
            record[f"instructrag_{metric}_mean"] = instruct_mean
            record[f"delta_{metric}"] = (
                None if baseline_mean is None or instruct_mean is None else instruct_mean - baseline_mean
            )
        out.append(record)
    return out
