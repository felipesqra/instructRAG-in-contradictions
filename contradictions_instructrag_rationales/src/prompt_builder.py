from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config import DocumentsConfig, PromptConfig
from data_loader import RetrievedDocument


@dataclass(frozen=True)
class PromptBundle:
    system_message: str
    user_message: str
    prompt_text: str

    def as_chat_messages(self) -> List[dict]:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        ]


def build_prompt(
    *,
    question: str,
    official_answer: str,
    documents: List[RetrievedDocument],
    prompt_config: PromptConfig,
    documents_config: DocumentsConfig,
) -> PromptBundle:
    rendered_documents = _render_documents(documents, documents_config)
    user_message = f"""Read the following documents relevant to the given question: {question}

Retrieved documents:
{rendered_documents}

Please identify documents that are useful to answer the given question
and explain how the contents lead to the ground-truth answer: {official_answer}.

Return only the rationale. Prefer the positive format whenever at least one document is useful;
use the negative format only when no provided document is useful.

Positive format:
After reviewing the provided documents, I found that only documents /documents/ contain relevant
information to answer the question. Based on the provided contents, the answer is: /answer/.
Then provide concise denoising reasoning grounded only in those documents."

Negative format:
None of the provided documents contain relevant information to answer the question. The explanation is: /explanation/.

In the positive format, replace /documents/ with document numbers, for example: documents [2, 5, 9].
In the positive format, replace /answer/ with the official TGA/CMI answer.
In the negative format, replace /explanation/ with a concise explanation of why the provided
documents do not support the official answer. Do not explain or restate the official answer itself.

Understand that documents of this type may contain outdated medical recommendations, conflicting findings across different studies, and the evolution of clinical consensus over time.

"""

    system_message = prompt_config.system_message.strip()
    prompt_text = f"System:\n{system_message}\n\nUser:\n{user_message}"
    return PromptBundle(
        system_message=system_message,
        user_message=user_message,
        prompt_text=prompt_text,
    )


def _render_documents(
    documents: List[RetrievedDocument],
    documents_config: DocumentsConfig,
) -> str:
    if not documents:
        return "[No retrieved documents were provided.]"

    blocks: List[str] = []
    for index, document in enumerate(documents, start=1):
        lines = [f"[Document {index}]"]
        if documents_config.include_document_ids and document.doc_id:
            lines.append(f"Document ID: {document.doc_id}")
        if (
            documents_config.include_document_scores_if_available
            and document.score is not None
        ):
            lines.append(f"Retrieval score: {document.score}")
        title = document.metadata.get("title")
        if title:
            lines.append(f"Title: {title}")
        lines.append(document.text or "[Empty document text]")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
