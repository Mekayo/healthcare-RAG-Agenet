from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    if chunk_overlap>= chunk_size:
        raise ValueError("Chunk_overlap must be less than chunk_size")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"]=(f"{chunk.metadata['pmid']}_{index}")
        chunk.metadata["chunk_length"]=len(chunk.page_content)
    return chunks

def inspect_chunks(chunks):
    print(f"Total Chunks: {len(chunks)}")
    print(chunks[0].metadata)
    print(chunks[0].page_content[:300])
    
    