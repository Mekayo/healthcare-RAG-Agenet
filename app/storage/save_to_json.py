import json
from pathlib import Path
from langchain_core.documents import Document


def save_documents_to_json(
    docs: list[Document],
    filepath: str
):

    Path(filepath).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = []

    for doc in docs:

        data.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        

    with open(
        filepath,
        "w",
        encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved {len(data)} documents to {filepath}")