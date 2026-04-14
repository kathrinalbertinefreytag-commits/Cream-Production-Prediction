import chromadb

PERSIST_DIR = "./vectorstore/chroma_db"

def get_chroma_client():
    return chromadb.PersistentClient(
        path=PERSIST_DIR
    )