from build_prompt import build_prompt

def rag_system(query, embedder, vector_db, llm):
    # Convert query -> embedding
    query_embedding = embedder.encode([query])[0]
    
    # Search similar chunks 
    retrieved_chunks = vector_db.search(query_embedding, top_k = 2)
    
    # Build prompt
    prompt = build_prompt(query, retrieved_chunks)
    
    # Generate answer 
    answer = llm.generate(prompt)
    
    return answer