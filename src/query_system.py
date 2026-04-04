def query_system(query, embedder, vector_db, top_k=3):
    # Convert query -> embedding
    query_embedding = embedder.encode([query])[0]
    
    # Search similar chunks 
    results = vector_db.search(query_embedding, top_k = top_k)
    
    return results