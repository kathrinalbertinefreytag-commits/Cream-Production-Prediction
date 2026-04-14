from vectorstore_client import get_chroma_client

client = get_chroma_client()

print("Collections:", client.list_collections())