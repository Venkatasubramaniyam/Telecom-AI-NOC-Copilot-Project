from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

print("API Key Loaded:", os.getenv("OPENAI_API_KEY"))

loader = DirectoryLoader(
    "sop",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embedding = OpenAIEmbeddings()

db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="chroma_db"
)

print("Vector Database Created")