from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings()

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0
)

# Prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
"""
You are a Telecom NOC Expert.

Answer only using the supplied SOP context.

Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()

# RAG Function

def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    chain = prompt | llm | parser

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer