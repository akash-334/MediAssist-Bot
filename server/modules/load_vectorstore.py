import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from server.modules.pdf_handlers import save_uploaded_files

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-assistant")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def _get_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured")

    pinecone = Pinecone(api_key=api_key)
    if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        while not pinecone.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pinecone.Index(PINECONE_INDEX_NAME)


def load_vectorstore(uploaded_files):
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not configured")

    embed_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768,
    )
    index = _get_index()

    for save_path in save_uploaded_files(uploaded_files):
        documents = PyPDFLoader(str(save_path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_model.embed_documents(texts)
        vectors = [
            (
                f"{save_path.stem}-{i}",
                embedding,
                {**chunk.metadata, "text": chunk.page_content},
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=vectors)
            progress.update(len(vectors))