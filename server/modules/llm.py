from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

def get_llm_chain(retriever: BaseRetriever):
    llm=ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b")
    
    multi_query_retriever=MultiQueryRetriever.from_llm(
        retriever=retriever,llm=llm
    )

    prompt=PromptTemplate(
        input_variables=["context","question"],
        template="""You are a MediBot, an AI-powered assistant to
        help user understand medical documents and health-related documents
        Your job is to provide accurate,clear and helpful response
        based **only on the provided context**


        **Context**:
        {context}
        **user question**
        {question}
        """
    )
    rag_chain=(
       {
           "context": multi_query_retriever | format_docs,
           "question":RunnablePassthrough()
       } |
       prompt | llm | StrOutputParser()

   )
    return rag_chain