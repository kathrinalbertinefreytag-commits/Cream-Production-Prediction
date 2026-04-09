import re

def clean_text(text: str) -> str:
    """
    Cleans PDF text for embedding quality.
    """
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    # Remove references like [1], (1)
    text = re.sub(r"\[\d+\]|\(\d+\)", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove long numbers (page numbers, IDs)
    text = re.sub(r"\d{4,}", "", text)
    # Remove weird characters
    text = re.sub(r"[^a-zA-Z0-9.,;:!?()\- ]", "", text)
    return text.strip()