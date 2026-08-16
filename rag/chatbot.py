import os
import re
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured in the Render environment."
    )

BASE_DIR = Path(__file__).resolve().parent.parent
SOP_DIR = BASE_DIR / "sop"

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0,
    timeout=60,
    max_retries=0,
)

prompt = ChatPromptTemplate.from_template(
    """
You are a Telecom NOC Expert.

Answer the user's question using ONLY the supplied SOP context.

If the context does not contain enough information, say:
"Insufficient information in the SOP documents."

Keep the answer concise and practical.

SOP Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()
chain = prompt | llm | parser


STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "to",
    "in", "on", "and", "or", "how", "why", "does", "do",
    "with", "about", "can", "please", "tell", "me"
}


def _tokens(text):
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return {
        word for word in words
        if len(word) >= 2 and word not in STOP_WORDS
    }


def _read_sop_file(path):
    suffix = path.suffix.lower()

    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))

            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if suffix == ".docx":
            from docx import Document

            document = Document(str(path))

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            )

    except Exception as e:
        print(
            f"SOP: Failed reading {path.name}: {e}",
            flush=True
        )
        traceback.print_exc()

    return ""


def _load_sop_chunks():
    if not SOP_DIR.exists():
        print(
            f"SOP: Folder not found: {SOP_DIR}",
            flush=True
        )
        return []

    supported = {".txt", ".md", ".pdf", ".docx"}
    chunks = []

    for path in SOP_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in supported:
            continue

        content = _read_sop_file(path)

        if not content.strip():
            continue

        # Split each document into manageable chunks.
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n|\n", content)
            if p.strip()
        ]

        current = ""

        for paragraph in paragraphs:

            if len(current) + len(paragraph) > 1800:

                if current:
                    chunks.append({
                        "name": path.name,
                        "content": current
                    })

                current = paragraph

            else:

                if current:
                    current += "\n"

                current += paragraph

        if current:
            chunks.append({
                "name": path.name,
                "content": current
            })

    print(
        f"SOP: Loaded {len(chunks)} searchable chunks from {SOP_DIR}",
        flush=True
    )

    return chunks


SOP_CHUNKS = _load_sop_chunks()


def _score_chunk(question, content):
    question_lower = question.lower()
    content_lower = content.lower()

    question_tokens = _tokens(question)
    content_tokens = _tokens(content)

    score = 0

    # Token overlap.
    score += len(
        question_tokens.intersection(content_tokens)
    ) * 3

    # Exact phrase match.
    if question_lower in content_lower:
        score += 20

    # Individual phrase components.
    # This helps queries such as "critical alarm" when the SOP says
    # "critical severity" and "alarm".
    for token in question_tokens:

        if token in content_lower:
            score += 2

    return score


def get_relevant_context(question):
    question = str(question).strip()

    if not question:
        return ""

    print(
        "SOP: Starting local SOP chunk search...",
        flush=True
    )

    start = time.time()

    scored = []

    for item in SOP_CHUNKS:

        score = _score_chunk(
            question,
            item["content"]
        )

        if score > 0:
            scored.append(
                (
                    score,
                    item["name"],
                    item["content"]
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = scored[:3]

    elapsed = round(
        time.time() - start,
        2
    )

    print(
        f"SOP: Local search completed in {elapsed}s; "
        f"matches={len(selected)}",
        flush=True
    )

    if not selected:
        return ""

    context_parts = []

    for score, name, content in selected:

        context_parts.append(
            f"SOP Document: {name}\n{content[:3000]}"
        )

    return "\n\n--- SOP SECTION ---\n\n".join(
        context_parts
    )


def ask_question(question):
    question = str(question).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    context = get_relevant_context(
        question
    )

    if not context:

        return (
            "No relevant SOP information was found "
            "for this question."
        )

    print(
        "SOP: Calling OpenAI GPT-4.1-mini...",
        flush=True
    )

    start = time.time()

    try:

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        elapsed = round(
            time.time() - start,
            2
        )

        print(
            f"SOP: OpenAI response completed in {elapsed}s",
            flush=True
        )

        return answer

    except Exception as e:

        print(
            "SOP: OpenAI LLM call FAILED:",
            flush=True
        )

        traceback.print_exc()

        raise RuntimeError(
            f"SOP LLM call failed: {str(e)}"
        ) from e
