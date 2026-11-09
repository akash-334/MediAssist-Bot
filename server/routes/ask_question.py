from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from server.modules.llm import get_llm_chain
from server.modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from typing import List
from server.logger import logger
import os

router=APIRouter()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-assistant")


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(PINECONE_INDEX_NAME)
        embed_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            output_dimensionality=768,
        )
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        matches = res["matches"] if isinstance(res, dict) else res.matches
        docs = [
            Document(
                page_content=(match.get("metadata") or {}).get("text", "")
                if isinstance(match, dict)
                else (match.metadata or {}).get("text", ""),
                metadata=(match.get("metadata") or {})
                if isinstance(match, dict)
                else (match.metadata or {}),
            ) for match in matches
        ]

        class SimpleRetriever(BaseRetriever):
            documents: List[Document]

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> List[Document]:
                return self.documents

        retriever = SimpleRetriever(documents=docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question, docs)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})