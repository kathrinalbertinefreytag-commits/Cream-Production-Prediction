from rag_cosmetic.retriever import get_retriever

def ask_rag(query):
    retriever = get_retriever()

    docs = retriever.invoke(query)

    # structure context
    context = []
    sources = []

    for d in docs:
        context.append({
            "text": d["content"],
            "source": d["metadata"].get("source", "unknown"),
            "score": d["metadata"].get("score", 0)
        })
        sources.append(d["metadata"].get("source", "unknown"))

    # generate LLM answer 
    client = get_openai_client()

    combined_context = "\n\n".join([d.page_content for d in docs])

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer based ONLY on the provided context."
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {combined_context}

                Question:
                {query}
                """
            }
        ]
    )

    answer = completion.choices[0].message.content

    # target-output
    return {
        "answer": answer,
        "context": context,
        "sources": list(set(sources))
    }