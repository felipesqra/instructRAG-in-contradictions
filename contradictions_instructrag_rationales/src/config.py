from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    pass


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = raw.get(name)
    if section is None:
        raise ConfigError(f"Missing required config section: {name}")
    if not isinstance(section, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping.")
    return section


def _choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ConfigError(f"{field_name} must be one of: {allowed_text}. Got: {value}")
    return value


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _non_empty_str(value: Any, default: str, field_name: str) -> str:
    resolved = str(value if value is not None else default).strip()
    if not resolved:
        raise ConfigError(f"{field_name} must not be empty. Example: {default}")
    return resolved


@dataclass(frozen=True)
class ProjectConfig:
    contradictions_repo_path: Path
    instructrag_repo_path: Path
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
class DocumentsConfig:
    max_documents_per_rationale: int
    document_selection_strategy: str
    include_document_ids: bool
    include_document_scores_if_available: bool


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float
    max_new_tokens: int


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key_env: str
    base_url: str
    timeout_seconds: int


@dataclass(frozen=True)
class HuggingFaceConfig:
    model_name: str
    api_key_env: str
    base_url: str
    timeout_seconds: int
    device: str
    torch_dtype: str
    trust_remote_code: bool
    use_auth_token_env: str


@dataclass(frozen=True)
class PromptConfig:
    system_message: str
    allow_model_knowledge_fallback: bool
    no_relevant_documents_behavior: str
    negative_template: str
    require_grounding: bool
    require_contradiction_handling: bool
    require_no_new_answer: bool


@dataclass(frozen=True)
class OutputsConfig:
    save_prompt: bool
    save_documents: bool
    save_jsonl: bool
    save_csv_summary: bool


@dataclass(frozen=True)
class RationaleGenerationConfig:
    config_path: Path
    module_root: Path
    raw: Dict[str, Any]
    project: ProjectConfig
    selection: SelectionConfig
    documents: DocumentsConfig
    model: ModelConfig
    deepseek: DeepSeekConfig
    huggingface: HuggingFaceConfig
    prompt: PromptConfig
    outputs: OutputsConfig


def load_config(path: str | Path) -> RationaleGenerationConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("The config file must contain a YAML mapping.")

    module_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent

    project = _section(raw, "project")
    selection = _section(raw, "selection")
    documents = _section(raw, "documents")
    model = _section(raw, "model")
    deepseek = _section(raw, "deepseek")
    huggingface = _section(raw, "huggingface")
    prompt = _section(raw, "prompt")
    outputs = _section(raw, "outputs")

    provider = _choice(
        str(model.get("provider", "deepseek")),
        {"deepseek", "huggingface", "huggingface_local"},
        "model.provider",
    )
    remedy_strategy = _choice(
        str(selection.get("remedy_selection_strategy", "sequential")),
        {"sequential", "random"},
        "selection.remedy_selection_strategy",
    )
    document_strategy = _choice(
        str(documents.get("document_selection_strategy", "as_available")),
        {"as_available"},
        "documents.document_selection_strategy",
    )

    project_config = ProjectConfig(
        contradictions_repo_path=_resolve_path(
            str(project.get("contradictions_repo_path", "../MedicalContradictionDetection-RAG")),
            module_root,
        ),
        instructrag_repo_path=_resolve_path(
            str(project.get("instructrag_repo_path", "../InstructRAG")),
            module_root,
        ),
        output_dir=_resolve_path(str(project.get("output_dir", "outputs")), module_root),
        run_name=str(project.get("run_name", "rationale_generation")),
    )

    selection_config = SelectionConfig(
        question_start_index=int(selection.get("question_start_index", 0)),
        num_questions=int(selection.get("num_questions", 2)),
        num_remedies=int(selection.get("num_remedies", 2)),
        remedy_selection_strategy=remedy_strategy,
        random_seed=int(selection.get("random_seed", 42)),
    )
    if selection_config.question_start_index < 0:
        raise ConfigError("selection.question_start_index must be >= 0.")
    if selection_config.num_questions <= 0:
        raise ConfigError("selection.num_questions must be > 0.")
    if selection_config.num_remedies <= 0:
        raise ConfigError("selection.num_remedies must be > 0.")

    documents_config = DocumentsConfig(
        max_documents_per_rationale=int(documents.get("max_documents_per_rationale", 5)),
        document_selection_strategy=document_strategy,
        include_document_ids=bool(documents.get("include_document_ids", True)),
        include_document_scores_if_available=bool(
            documents.get("include_document_scores_if_available", True)
        ),
    )
    if documents_config.max_documents_per_rationale <= 0:
        raise ConfigError("documents.max_documents_per_rationale must be > 0.")

    model_config = ModelConfig(
        provider=provider,
        model_name=str(model.get("model_name", "deepseek-chat")),
        temperature=float(model.get("temperature", 0.0)),
        max_new_tokens=int(model.get("max_new_tokens", 512)),
    )
    if model_config.max_new_tokens <= 0:
        raise ConfigError("model.max_new_tokens must be > 0.")

    prompt_config = PromptConfig(
        system_message=str(prompt.get("system_message", "")).strip(),
        allow_model_knowledge_fallback=bool(prompt.get("allow_model_knowledge_fallback", False)),
        no_relevant_documents_behavior=str(
            prompt.get("no_relevant_documents_behavior", "denoising_rationale_with_negative_template")
        ),
        negative_template=str(prompt.get("negative_template", "")).strip(),
        require_grounding=bool(prompt.get("require_grounding", True)),
        require_contradiction_handling=bool(prompt.get("require_contradiction_handling", True)),
        require_no_new_answer=bool(prompt.get("require_no_new_answer", True)),
    )
    if prompt_config.allow_model_knowledge_fallback:
        raise ConfigError("prompt.allow_model_knowledge_fallback must remain false for this experiment.")
    if not prompt_config.negative_template:
        raise ConfigError("prompt.negative_template must not be empty.")

    return RationaleGenerationConfig(
        config_path=config_path,
        module_root=module_root.resolve(),
        raw=raw,
        project=project_config,
        selection=selection_config,
        documents=documents_config,
        model=model_config,
        deepseek=DeepSeekConfig(
            api_key_env=_non_empty_str(
                deepseek.get("api_key_env"),
                "DEEPSEEK_API_KEY",
                "deepseek.api_key_env",
            ),
            base_url=_non_empty_str(
                deepseek.get("base_url"),
                "https://api.deepseek.com",
                "deepseek.base_url",
            ),
            timeout_seconds=int(deepseek.get("timeout_seconds", 120)),
        ),
        huggingface=HuggingFaceConfig(
            model_name=str(huggingface.get("model_name", model_config.model_name)),
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
        prompt=prompt_config,
        outputs=OutputsConfig(
            save_prompt=bool(outputs.get("save_prompt", True)),
            save_documents=bool(outputs.get("save_documents", True)),
            save_jsonl=bool(outputs.get("save_jsonl", True)),
            save_csv_summary=bool(outputs.get("save_csv_summary", True)),
        ),
    )
