import os
from openai import OpenAI
import chromadb
from ingestion.extract import extract_text_from_pdf
from ingestion.chunk import chunk_text
from dotenv import load_dotenv
from vectorstore_client import get_chroma_client

print("INGEST START")

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# creating Chroma client and Collection 
chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection("cream_docs")

# reading PDFs, chunking and embedding 
pdf_files = [
    "data/sample/Cream41598_2024_Article_57782.pdf",
    "data/sample/creammarinedrugs-21-00618.pdf",
    "data/sample/creammolecules-26-03921.pdf"
]

chunks = []
metadata = []

for pdf_file in pdf_files:
    if not os.path.isfile(pdf_file):
        print(f"⚠️ PDF nicht gefunden: {pdf_file}")
        continue

    # extracting text from PDF
    text = extract_text_from_pdf(pdf_file)

    # chunking the text
    text_chunks = chunk_text(text)
    chunks += text_chunks

    # Metadata for every chunk
    metadata += [{"source": os.path.basename(pdf_file)}] * len(text_chunks)

# creating embeddings and adding it to the embedding
for i, chunk in enumerate(chunks):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    ).data[0].embedding

    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[str(i)],
        metadatas=[metadata[i]]
    )
print("COUNT AFTER INGEST:", collection.count())
print("Ingestion complete!")