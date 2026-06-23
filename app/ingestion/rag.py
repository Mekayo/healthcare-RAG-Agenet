from langchain_groq import ChatGroq
from vector_store import VectorStorePipeline
from pubmed_loader import load_pubmed_documents
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


class RAGSearch:

    def __init__(self, llm_model: str = "openai/gpt-oss-120b"):

        self.vector_pipeline = VectorStorePipeline()

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

    def search_and_summarize(self, query: str,top_k: int = 5):

        # Step 1
        docs = load_pubmed_documents(query=query, max_results=100)

        # Step 2
        chunks = self.vector_pipeline.chunk_documents(docs)

        # Step 3
        self.vector_pipeline.create_vector_store(chunks)

        # Step 4
        retriever = self.vector_pipeline.get_retriever(k=top_k)

        # Step 5
        retrieved_docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        prompt = f"""
Answer the question using only the provided context.

Question:
{query}

Context:
{context}

Provide:
1. Summary
2. Key Findings
3. Treatment Information
4. References if available
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)])

        return response.content
    
    
query ="how to treat Cancer patients above age of 60"

res=RAGSearch()

print(res.search_and_summarize(query))
