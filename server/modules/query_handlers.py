from typing import Sequence

from langchain_core.documents import Document

from server.logger import logger


def query_chain(chain, user_input: str, source_documents: Sequence[Document]):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        result = chain.invoke(user_input)
        response = {
            "response": result,
            "sources": [
                doc.metadata.get("source", "") for doc in source_documents
            ],
        }
        logger.debug(f"Chain response:{response}")
        return response
    except Exception as e:
        logger.exception("Error on query chain")
        raise