def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for _, chunk in retrieved_chunks])
    
    prompt = f"""
    You are a helpful AI assistant.
    
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know"
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer
    """
    return prompt
    
    