import numpy as np
from chunking_pipeline import ChunkingPipeline
from text_embedding import EmbeddingModel
from store_vector import VectorStore
from rag_system import rag_system
from llm_generation import LLMModel

with open("sentences.txt", "r") as file:
  sentences = file.read()

# print(sentences)

# Run pipeline 
pipeline = ChunkingPipeline(20, 1)
chunks = pipeline.run_pipeline(sentences, "Human History", "internet")

# print(chunks)

# Extract text only 
chunk_texts = [chunk["text"] for chunk in chunks]

# Generate embeddings 
embedder = EmbeddingModel()
embeddings = np.array(embedder.encode(chunk_texts))

d = embeddings.shape[1]

# Store in vectro DB
vector_db = VectorStore(d)
vector_db.add(embeddings, chunks)

llm = LLMModel()

query = "What gained global mainstream recognition?"

response = rag_system(query, embedder, vector_db, llm)

print(response)
