import numpy as np
import faiss

class VectorStore:
    def __init__(self, d):
        self.index = faiss.IndexFlatIP(d)
        self.chunks = [] # Store chunks internally to keep them synced
    
    def add(self, embeddings, new_chunks):
        embeddings_matrix = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_matrix) # Normalize for Cosine Similarity  
        
        # Add embeddigs to the database      
        self.index.add(embeddings_matrix)
        self.chunks.extend(new_chunks)
    
    def search(self, query_embedding, top_k=3):
        query_vec = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        # Retrieve context 
        retrieved_chunks = [(distances[0][j], self.chunks[i]) for j, i in enumerate(indices[0]) if i != -1]
        
        return retrieved_chunks
        
    