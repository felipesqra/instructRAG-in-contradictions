from __future__ import annotations

from dataclasses import dataclass

from config import BaselinePromptConfig, InstructPromptConfig
from data_loader import EvaluationExample, RetrievedDocument
from rationale_loader import RationaleExample


@dataclass(frozen=True)
class PromptBundle:
    system_message: str
    user_message: str
    prompt_text: str

    def as_chat_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": self.user_message})
        return messages


def build_baseline_prompt(
    example: EvaluationExample,
    original_template: str,
    config: BaselinePromptConfig,
    negative_template: str,
) -> PromptBundle:
    context = render_documents(example.documents)
    user_message = original_template.replace("{context}", context).replace("{input}", example.question)
    user_message = f"{user_message}\n\n{unsupported_evidence_instruction(negative_template)}"
    system_message = "" if config.use_original_prompt_if_available else config.fallback_system_message
    prompt_text = f"System:\n{system_message}\n\nUser:\n{user_message}" if system_message else f"User:\n{user_message}"
    return PromptBundle(system_message=system_message, user_message=user_message, prompt_text=prompt_text)


def build_instructrag_icl_prompt(
    example: EvaluationExample,
    icl_examples: list[RationaleExample],
    config: InstructPromptConfig,
    negative_template: str,
) -> PromptBundle:
    parts = [        
        "Given the following medical abstracts answer the question."
        "Below are some examples of how to answer the question:"

    ]
    if config.include_rationale_examples:
        for index, icl in enumerate(icl_examples, start=1):
            parts.append(f"Example {index}:")
            parts.append("Question:")
            parts.append(icl.question)
            if config.include_example_documents:
                parts.append("Retrieved documents:")
                parts.append(render_documents(icl.documents))
            parts.append("Denoising rationale:")
            parts.append(icl.rationale)
            if config.include_example_official_answer:
                parts.append("Official answer:")
                parts.append(icl.official_answer)
            

    parts.append("Now answer the target question.")
    parts.append("Target question:")
    parts.append(example.question)
    parts.append("Target retrieved documents:")
    parts.append(render_documents(example.documents))
    parts.append("Instruction:")
    parts.append(f"{config.answer_instruction}\n\n{unsupported_evidence_instruction(negative_template)}")
    user_message = "\n\n".join(parts)
    prompt_text = f"System:\n{config.system_message}\n\nUser:\n{user_message}"
    return PromptBundle(system_message=config.system_message, user_message=user_message, prompt_text=prompt_text)


def unsupported_evidence_instruction(negative_template: str) -> str:
    return (
        "If none of the retrieved documents contains relevant information to answer the question, "
        "return exactly this sentence:\n"
        f"{negative_template}"
    )


def render_documents(documents: list[RetrievedDocument]) -> str:
    if not documents:
        return "[No retrieved documents were provided.]"
    blocks = []
    for index, doc in enumerate(documents, start=1):
        lines = [f"[Document {index}]"]
        if doc.doc_id:
            lines.append(f"Document ID: {doc.doc_id}")
        if doc.score is not None:
            lines.append(f"Retrieval score: {doc.score}")
        title = doc.metadata.get("title")
        if title:
            lines.append(f"Title: {title}")
        lines.append(doc.text or "[Empty document text]")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
