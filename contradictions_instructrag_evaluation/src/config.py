from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {name}")
    return value


def _non_empty_str(value: Any, default: str, field_name: str) -> str:
    resolved = str(value if value is not None else default).strip()
    if not resolved:
        raise ConfigError(f"{field_name} must not be empty. Example: {default}")
    return resolved


def _choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{field_name} must be one of {sorted(allowed)}. Got: {value}")
    return value


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


@dataclass(frozen=True)
class ProjectConfig:
    contradictions_repo_path: Path
    rationales_dir: Path
    output_dir: Path
    run_name: str


@dataclass(frozen=True)
class SelectionConfig:
    question_start_index: int
    num_questions: int
    num_remedies: int
    remedy_selection_strategy: str
    random_seed: int


@dataclass(frozen=True)
class RationalesConfig:
    rationale_run_dir: Path | None
    rationale_file: str
    max_icl_examples: int
    icl_selection_strategy: str
    include_negative_rationales_as_icl: bool
    negative_template: str


@dataclass(frozen=True)
class RetrievalConfig:
    use_original_rag_pipeline: bool
    retrieval_conditions: list[str]
    max_documents_per_query: int
    preserve_original_document_order: bool


@dataclass(frozen=True)
class ModelArmConfig:
    provider: str
    model_name: str
    temperature: float
    max_new_tokens: int
    top_p: float


@dataclass(frozen=True)
class ModelsConfig:
    baseline: ModelArmConfig
    instructrag_icl: ModelArmConfig


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key_env: str
    base_url: str
    timeout_seconds: int


@dataclass(frozen=True)
class HuggingFaceConfig:
    api_key_env: str
    base_url: str
    timeout_seconds: int
    device: str
    torch_dtype: str
    trust_remote_code: bool
    use_auth_token_env: str


@dataclass(frozen=True)
class GroqConfig:
    api_key_env: str
    base_url: str
    timeout_seconds: int


@dataclass(frozen=True)
class BaselinePromptConfig:
    use_original_prompt_if_available: bool
    fallback_system_message: str


@dataclass(frozen=True)
class InstructPromptConfig:
    system_message: str
    answer_instruction: str
    include_rationale_examples: bool
    include_example_official_answer: bool
    include_example_documents: bool


@dataclass(frozen=True)
class PromptConfig:
    baseline: BaselinePromptConfig
    instructrag_icl: InstructPromptConfig


@dataclass(frozen=True)
class MetricsConfig:
    use_original_metric_implementation_if_available: bool
    compute_per_example: bool
    compute_macro_average_by_model: bool
    compute_macro_average_by_retrieval_condition: bool
    compute_macro_average_by_negative_filter: bool
    metrics_to_compute: list[str]
    semantic_encoder: str
    use_same_encoder_as_retrieval: bool


@dataclass(frozen=True)
class NegativeRowsConfig:
    enabled: bool
    definition: str
    generate_with_negative_rows_results: bool
    generate_without_negative_rows_results: bool


@dataclass(frozen=True)
class OutputsConfig:
    save_raw_generations: bool
    save_prompts: bool
    save_documents: bool
    save_per_example_metrics: bool
    save_aggregated_metrics: bool
    save_comparison_tables: bool


@dataclass(frozen=True)
class EvaluationConfig:
    config_path: Path
    module_root: Path
    raw: dict[str, Any]
    project: ProjectConfig
    selection: SelectionConfig
    rationales: RationalesConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
    deepseek: DeepSeekConfig
    huggingface: HuggingFaceConfig
    groq: GroqConfig
    prompt: PromptConfig
    metrics: MetricsConfig
    negative_rows: NegativeRowsConfig
    outputs: OutputsConfig


def _model_arm(raw: dict[str, Any], name: str) -> ModelArmConfig:
    section = _section(raw, name)
    provider = _choice(
        str(section.get("provider", "deepseek")),
        {"deepseek", "groq", "huggingface", "huggingface_local", "original_rag_pipeline"},
        f"models.{name}.provider",
    )
    return ModelArmConfig(
        provider=provider,
        model_name=_non_empty_str(section.get("model_name"), "deepseek-chat", f"models.{name}.model_name"),
        temperature=float(section.get("temperature", 0.0)),
        max_new_tokens=int(section.get("max_new_tokens", 256)),
        top_p=float(section.get("top_p", 1.0)),
    )


def load_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("The YAML config must be a mapping.")

    module_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    project = _section(raw, "project")
    selection = _section(raw, "selection")
    rationales = _section(raw, "rationales")
    retrieval = _section(raw, "retrieval")
    models = _section(raw, "models")
    deepseek = _section(raw, "deepseek")
    huggingface = _section(raw, "huggingface")
    groq = raw.get("groq", {})
    if not isinstance(groq, dict):
        raise ConfigError("Config section 'groq' must be a mapping when provided.")
    prompt = _section(raw, "prompt")
    metrics = _section(raw, "metrics")
    negative_rows = _section(raw, "negative_rows")
    outputs = _section(raw, "outputs")

    rationale_run_dir = rationales.get("rationale_run_dir")
    rationales_config = RationalesConfig(
        rationale_run_dir=_resolve_path(rationale_run_dir, module_root) if rationale_run_dir else None,
        rationale_file=_non_empty_str(rationales.get("rationale_file"), "rationales.jsonl", "rationales.rationale_file"),
        max_icl_examples=int(rationales.get("max_icl_examples", 2)),
        icl_selection_strategy=_choice(
            str(rationales.get("icl_selection_strategy", "random")),
            {"random", "sequential", "same_question_first"},
            "rationales.icl_selection_strategy",
        ),
        include_negative_rationales_as_icl=bool(rationales.get("include_negative_rationales_as_icl", True)),
        negative_template=_non_empty_str(
            rationales.get("negative_template"),
            "After reviewing the provided documents, I found that none of them contain relevant information to answer the question.",
            "rationales.negative_template",
        ),
    )

    retrieval_conditions = list(retrieval.get("retrieval_conditions", ["most_similar"]))
    allowed_conditions = {"most_similar", "most_contradictory", "least_contradictory"}
    invalid_conditions = sorted(set(retrieval_conditions) - allowed_conditions)
    if invalid_conditions:
        raise ConfigError(f"Invalid retrieval conditions: {invalid_conditions}")

    return EvaluationConfig(
        config_path=config_path,
        module_root=module_root.resolve(),
        raw=raw,
        project=ProjectConfig(
            contradictions_repo_path=_resolve_path(project.get("contradictions_repo_path"), module_root),
            rationales_dir=_resolve_path(project.get("rationales_dir"), module_root),
            output_dir=_resolve_path(project.get("output_dir", "outputs"), module_root),
            run_name=_non_empty_str(project.get("run_name"), "phase2_rag_comparison", "project.run_name"),
        ),
        selection=SelectionConfig(
            question_start_index=int(selection.get("question_start_index", 0)),
            num_questions=int(selection.get("num_questions", 2)),
            num_remedies=int(selection.get("num_remedies", 2)),
            remedy_selection_strategy=_choice(
                str(selection.get("remedy_selection_strategy", "sequential")),
                {"sequential", "random"},
                "selection.remedy_selection_strategy",
            ),
            random_seed=int(selection.get("random_seed", 42)),
        ),
        rationales=rationales_config,
        retrieval=RetrievalConfig(
            use_original_rag_pipeline=bool(retrieval.get("use_original_rag_pipeline", True)),
            retrieval_conditions=retrieval_conditions,
            max_documents_per_query=int(retrieval.get("max_documents_per_query", 5)),
            preserve_original_document_order=bool(retrieval.get("preserve_original_document_order", True)),
        ),
        models=ModelsConfig(
            baseline=_model_arm(models, "baseline"),
            instructrag_icl=_model_arm(models, "instructrag_icl"),
        ),
        deepseek=DeepSeekConfig(
            api_key_env=_non_empty_str(deepseek.get("api_key_env"), "DEEPSEEK_API_KEY", "deepseek.api_key_env"),
            base_url=_non_empty_str(deepseek.get("base_url"), "https://api.deepseek.com", "deepseek.base_url"),
            timeout_seconds=int(deepseek.get("timeout_seconds", 120)),
        ),
        huggingface=HuggingFaceConfig(
            api_key_env=_non_empty_str(
                huggingface.get("api_key_env", huggingface.get("use_auth_token_env")),
                "HF_TOKEN",
                "huggingface.api_key_env",
            ),
            base_url=_non_empty_str(
                huggingface.get("base_url"),
                "https://router.huggingface.co/v1",
                "huggingface.base_url",
            ),
            timeout_seconds=int(huggingface.get("timeout_seconds", 120)),
            device=str(huggingface.get("device", "auto")),
            torch_dtype=str(huggingface.get("torch_dtype", "auto")),
            trust_remote_code=bool(huggingface.get("trust_remote_code", False)),
            use_auth_token_env=str(huggingface.get("use_auth_token_env", "HF_TOKEN")),
        ),
        groq=GroqConfig(
            api_key_env=_non_empty_str(groq.get("api_key_env"), "GROQ_API_KEY", "groq.api_key_env"),
            base_url=_non_empty_str(
                groq.get("base_url"),
                "https://api.groq.com/openai/v1",
                "groq.base_url",
            ),
            timeout_seconds=int(groq.get("timeout_seconds", 120)),
        ),
        prompt=PromptConfig(
            baseline=BaselinePromptConfig(
                use_original_prompt_if_available=bool(
                    prompt.get("baseline", {}).get("use_original_prompt_if_available", True)
                ),
                fallback_system_message=str(prompt.get("baseline", {}).get("fallback_system_message", "")).strip(),
            ),
            instructrag_icl=InstructPromptConfig(
                system_message=str(prompt.get("instructrag_icl", {}).get("system_message", "")).strip(),
                answer_instruction=str(prompt.get("instructrag_icl", {}).get("answer_instruction", "")).strip(),
                include_rationale_examples=bool(prompt.get("instructrag_icl", {}).get("include_rationale_examples", True)),
                include_example_official_answer=bool(prompt.get("instructrag_icl", {}).get("include_example_official_answer", True)),
                include_example_documents=bool(prompt.get("instructrag_icl", {}).get("include_example_documents", True)),
            ),
        ),
        metrics=MetricsConfig(
            use_original_metric_implementation_if_available=bool(
                metrics.get("use_original_metric_implementation_if_available", True)
            ),
            compute_per_example=bool(metrics.get("compute_per_example", True)),
            compute_macro_average_by_model=bool(metrics.get("compute_macro_average_by_model", True)),
            compute_macro_average_by_retrieval_condition=bool(
                metrics.get("compute_macro_average_by_retrieval_condition", True)
            ),
            compute_macro_average_by_negative_filter=bool(
                metrics.get("compute_macro_average_by_negative_filter", True)
            ),
            metrics_to_compute=list(metrics.get("metrics_to_compute", [])),
            semantic_encoder=str(metrics.get("semantic_encoder", "BAAI/bge-small-en-v1.5")),
            use_same_encoder_as_retrieval=bool(metrics.get("use_same_encoder_as_retrieval", True)),
        ),
        negative_rows=NegativeRowsConfig(
            enabled=bool(negative_rows.get("enabled", True)),
            definition=str(negative_rows.get("definition", "")),
            generate_with_negative_rows_results=bool(negative_rows.get("generate_with_negative_rows_results", True)),
            generate_without_negative_rows_results=bool(
                negative_rows.get("generate_without_negative_rows_results", True)
            ),
        ),
        outputs=OutputsConfig(
            save_raw_generations=bool(outputs.get("save_raw_generations", True)),
            save_prompts=bool(outputs.get("save_prompts", True)),
            save_documents=bool(outputs.get("save_documents", True)),
            save_per_example_metrics=bool(outputs.get("save_per_example_metrics", True)),
            save_aggregated_metrics=bool(outputs.get("save_aggregated_metrics", True)),
            save_comparison_tables=bool(outputs.get("save_comparison_tables", True)),
        ),
    )
