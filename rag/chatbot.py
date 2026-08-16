import os
import re
import time
import traceback

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured in the Render environment."
    )

# IMPORTANT:
# Do not use OpenAIEmbeddings for the live SOP search.
# The previous implementation was hanging at:
# retriever.invoke(question)
#
# Instead, read the already-stored Chroma documents locally and perform
# lightweight keyword matching. This avoids a second OpenAI network call.

db = Chroma(
    persist_directory="chroma_db"
)

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
If the context does not contain enough information, say:
"Insufficient information in the SOP context."

Keep the answer concise and practical.

SOP Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()
chain = prompt | llm | parser


def _limit_text(text, max_chars=8000):
    text = str(text or "")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[context truncated]"
    return text


def _tokens(text):
    return set(
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9]+", str(text))
        if len(token) >= 3
    )


def get_relevant_context(question):
    """
    Local keyword retrieval from the documents already stored in Chroma.

    This deliberately avoids retriever.invoke(), because that operation
    requires OpenAI embeddings for the query and was hanging on Render.
    """

    question = str(question).strip()

    if not question:
        return ""

    print("SOP: Starting local Chroma document search...", flush=True)
    start = time.time()

    try:
        collection = db._collection

        data = collection.get(
            include=["documents", "metadatas"]
        )

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        if not documents:
            print("SOP: No documents found in Chroma.", flush=True)
            return ""

        query_tokens = _tokens(question)

        scored = []

        for index, document in enumerate(documents):
            document = str(document or "")
            if not document:
                continue

            doc_tokens = _tokens(document)

            if query_tokens:
                overlap = len(query_tokens.intersection(doc_tokens))
            else:
                overlap = 0

            # Give an exact phrase match a strong boost.
            phrase_bonus = 10 if question.lower() in document.lower() else 0

            score = overlap + phrase_bonus

            if score > 0:
                metadata = (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                )

                scored.append(
                    (score, document, metadata)
                )

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        selected = scored[:2]

        elapsed = round(time.time() - start, 2)

        print(
            f"SOP: Local document search completed in {elapsed}s; "
            f"documents={len(documents)}, matches={len(selected)}",
            flush=True,
        )

        if not selected:
            return ""

        context_parts = []

        for score, document, metadata in selected:
            context_parts.append(
                _limit_text(document, 3500)
            )

        return "\n\n--- SOP DOCUMENT ---\n\n".join(
            context_parts
        )

    except Exception as e:
        print("SOP: Local Chroma search FAILED:", flush=True)
        traceback.print_exc()
        raise RuntimeError(
            f"SOP document search failed: {str(e)}"
        ) from e


def ask_question(question):
    question = str(question).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    context = get_relevant_context(question)

    if not context:
        return (
            "No relevant SOP information was found for this question."
        )

    print("SOP: Calling OpenAI GPT-4.1-mini...", flush=True)
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
        print("SOP: OpenAI LLM call FAILED:", flush=True)
        traceback.print_exc()

        raise RuntimeError(
            f"SOP LLM call failed: {str(e)}"
        ) from e
