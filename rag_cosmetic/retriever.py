from openai import OpenAI
import chromadb
from chromadb.config import Settings
import os

def get_openai_client():
    """Creates OpenAI-Client when called"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Bitte setze die Environment-Variable OPENAI_API_KEY!")
    return OpenAI(api_key=api_key)

def get_embedding(text: str):
    """Creates embeddings for the text"""
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_retriever():
    """Retriever for ChromaDB"""
    client_chroma = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

    collection = client_chroma.get_collection("cream_docs")

    class Retriever:
        def invoke(self, query):
            embedding = get_embedding(query)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=3,
                include=["metadatas", "documents"]
            )

            class Doc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata

            return [Doc(d, m) for d, m in zip(results['documents'][0], results['metadatas'][0])]

    return Retriever()