from openai import OpenAI
from .retriever import get_retriever
import os

def get_openai_client():
    """Creates OpenAI-Client when it is called"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Bitte setze die Environment-Variable OPENAI_API_KEY!")
    return OpenAI(api_key=api_key)



def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def ask_rag(question: str):
    retriever = get_retriever()

    # 🔍 relevante Dokumente holen
    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = f"""
You are a cosmetic formulation expert.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    # 🤖 OpenAI Call
    client = get_openai_client()
    response = client.responses.create(
        model="gpt-5.3-mini",
        input=prompt
    )

    answer = response.output_text

    # 📚 Quellen extrahieren
    sources = [
        doc.metadata.get("source", "unknown")
        for doc in docs
    ]

    return {
        "answer": answer,
        "sources": list(set(sources))  # deduplicate
    }