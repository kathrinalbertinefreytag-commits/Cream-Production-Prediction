from openai import OpenAI
import os
from vectorstore_client import get_chroma_client

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY!")
    return OpenAI(api_key=api_key)


def get_embedding(text: str):
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def get_retriever():
    client_chroma = get_chroma_client()
    collection = client_chroma.get_collection("cream_docs")

    class Retriever:
        def invoke(self, query):
            embedding = get_embedding(query)

            results = collection.query(
                query_embeddings=[embedding],
                n_results=3,
                include=["metadatas", "documents"]
            )

            return [
                {
                    "content": doc,
                    "metadata": meta
                }
                for doc, meta in zip(
                    results["documents"][0],
                    results["metadatas"][0]
                )
            ]

    return Retriever()