import pandas as pd
from langchain_core.documents import Document


def save_documents_to_csv(
        docs: list[Document],
        filepath: str
):
    
    rows = []

    for doc in docs:

        row = {
            "content": doc.page_content,
            **doc.metadata
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        filepath,
        index=False
    )

    print(f"Saved {len(df)} documents")