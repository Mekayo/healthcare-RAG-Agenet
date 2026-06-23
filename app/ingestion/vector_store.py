from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from pubmed_loader import load_pubmed_documents
class VectorStorePipeline:

    def __init__(self, persist_directory: str = "data/chroma_db" ,model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1000,chunk_overlap: int = 200):
      
        self.persist_directory =persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device" : "cpu"},
            encode_kwargs={"normalize_embeddings":True},
            )
        self.vector_store =None
        
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: list[Document]) -> list[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            
            pmid = chunk.metadata.get("pmid", "unknown")
            chunk.metadata["chunk_id"] = f"{pmid}_{idx}"

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")

        return chunks


    def create_vector_store(self, chunks: list[Document]):
        self.vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=self.embedding_model,
        persist_directory=self.persist_directory,
        )
        print(f"[INFO] Chroma DB created with {len(chunks)} chunks.")

        return self.vector_store
    
    def load_vector_store(self):
        
        self.vector_store=Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
        )
        
        print(f"[INFO] Load chromaDB from {self.persist_directory}")
        
        return self.vector_store
    
    def get_retriever(self, k: int = 5):

        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
# =================================
# testing
# =================================
docs = load_pubmed_documents(
    query=" Lung Cancer treatment elderly",
    max_results=200
)

pipeline = VectorStorePipeline()

chunks = pipeline.chunk_documents(
    docs
)

pipeline.create_vector_store(
    chunks
)

retriever = pipeline.get_retriever()

results = retriever.invoke(
    "latest treatments for Lung Cancer patients over 60"
)

print(results[0].page_content)