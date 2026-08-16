import os
import time
import traceback

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured in the Render environment."
    )


# Keep Chroma/OpenAI embedding work small and bounded.
embedding = OpenAIEmbeddings(
    api_key=api_key,
    request_timeout=30,
    max_retries=0,
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
)

retriever = db.as_retriever(
    search_kwargs={"k": 2}
)


# Keep the LLM call bounded.
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

Answer only using the supplied SOP context.
If the context does not contain the answer, clearly say that the SOP
context does not provide enough information.

Keep the answer concise and practical.

Context:
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


def get_relevant_context(question):
    question = str(question).strip()

    if not question:
        return ""

    print("SOP: Starting Chroma/OpenAI embedding retrieval...", flush=True)
    start = time.time()

    try:
        docs = retriever.invoke(question)

        elapsed = round(time.time() - start, 2)

        print(
            f"SOP: Chroma retrieval completed in {elapsed}s "
            f"({len(docs)} documents)",
            flush=True,
        )

    except Exception as e:
        print("SOP: Chroma retrieval FAILED:", flush=True)
        traceback.print_exc()

        raise RuntimeError(
            f"SOP retrieval failed: {str(e)}"
        ) from e

    context = "\n\n".join(
        str(doc.page_content)
        for doc in docs
        if getattr(doc, "page_content", None)
    )

    return _limit_text(context)


def ask_question(question):
    question = str(question).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    context = get_relevant_context(question)

    if not context:
        return "No relevant SOP information was found."

    print("SOP: Calling OpenAI GPT-4.1-mini...", flush=True)
    start = time.time()

    try:
        answer = chain.invoke({
            "context": context,
            "question": question,
        })

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
