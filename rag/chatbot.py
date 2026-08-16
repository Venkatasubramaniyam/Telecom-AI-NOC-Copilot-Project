import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not configured in the Render environment.")

# OpenAI embeddings are used by Chroma for similarity search.
# Keep network calls bounded so a slow OpenAI request does not hang Render.
embedding = OpenAIEmbeddings(
    api_key=api_key,
    request_timeout=60,
    max_retries=1,
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# Bound the LLM request so Render does not wait indefinitely.
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0,
    timeout=90,
    max_retries=1,
)

prompt = ChatPromptTemplate.from_template(
    """
You are a Telecom NOC Expert.

Answer only using the supplied SOP context.
If the context does not contain the answer, clearly say that the SOP
context does not provide enough information.

Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()
chain = prompt | llm | parser


def _limit_text(text, max_chars=12000):
    text = str(text or "")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[context truncated]"
    return text


def get_relevant_context(question):
    question = str(question).strip()

    if not question:
        return ""

    docs = retriever.invoke(question)

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

    return chain.invoke({
        "context": context,
        "question": question
    })
