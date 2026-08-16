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

# Project structure:
# project/
#   app.py
#   rag/chatbot.py
#   sop/                 <-- SOP documents
#
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

Answer the question using ONLY the supplied SOP context.
If the SOP context does not contain enough information, say:
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


def _tokens(text):
    return set(
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9]+", str(text))
        if len(token) >= 3
    )


def _read_sop_file(path):
    suffix = path.suffix.lower()

    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

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
            f"SOP: Could not read {path.name}: {e}",
            flush=True,
        )

    return ""


def _load_sop_documents():
    if not SOP_DIR.exists():
        print(
            f"SOP: Folder not found: {SOP_DIR}",
            flush=True,
        )
        return []

    supported = {".txt", ".md", ".pdf", ".docx"}
    documents = []

    for path in SOP_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            content = _read_sop_file(path)

            if content.strip():
                documents.append({
                    "name": path.name,
                    "content": content,
                })

    print(
        f"SOP: Loaded {len(documents)} documents from {SOP_DIR}",
        flush=True,
    )

    return documents


# Load SOP documents once at application startup.
SOP_DOCUMENTS = _load_sop_documents()


def _find_relevant_sops(question, max_results=3):
    question_tokens = _tokens(question)

    scored = []

    for document in SOP_DOCUMENTS:
        content = document["content"]
        content_tokens = _tokens(content)

        overlap = len(
            question_tokens.intersection(content_tokens)
        )

        phrase_bonus = (
            20
            if question.lower() in content.lower()
            else 0
        )

        score = overlap + phrase_bonus

        if score > 0:
            scored.append(
                (
                    score,
                    document["name"],
                    content,
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored[:max_results]


def _build_context(matches, max_chars=10000):
    parts = []
    total = 0

    for score, name, content in matches:
        remaining = max_chars - total

        if remaining <= 0:
            break

        content = content[:remaining]

        parts.append(
            f"SOP Document: {name}\n{content}"
        )

        total += len(content)

    return "\n\n---\n\n".join(parts)


def get_relevant_context(question):
    question = str(question).strip()

    if not question:
        return ""

    print(
        "SOP: Starting local SOP search...",
        flush=True,
    )

    start = time.time()

    matches = _find_relevant_sops(
        question,
        max_results=3,
    )

    elapsed = round(time.time() - start, 2)

    print(
        f"SOP: Local search completed in {elapsed}s; "
        f"matches={len(matches)}",
        flush=True,
    )

    return _build_context(matches)


def ask_question(question):
    question = str(question).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    context = get_relevant_context(question)

    if not context:
        return (
            "No relevant SOP information was found for this question."
        )

    print(
        "SOP: Calling OpenAI GPT-4.1-mini...",
        flush=True,
    )

    start = time.time()

    try:
        answer = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        elapsed = round(time.time() - start, 2)

        print(
            f"SOP: OpenAI response completed in {elapsed}s",
            flush=True,
        )

        return answer

    except Exception as e:
        print(
            "SOP: OpenAI LLM call FAILED:",
            flush=True,
        )
        traceback.print_exc()

        raise RuntimeError(
            f"SOP LLM call failed: {str(e)}"
        ) from e
