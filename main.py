from app.ingestion.rag import RAGSearch
if __name__ == "__main__":

    rag = RAGSearch()

    query = input(
        "Enter medical query: "
    )

    answer = rag.search_and_summarize(
        query=query,
        top_k=5
    )

    print(answer)