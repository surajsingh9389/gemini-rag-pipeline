import numpy as np
from google import genai
import os
from dotenv import load_dotenv
import faiss


# 1. Load your .env file
load_dotenv()

# 2. Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# convert text to embedding 
def get_embedding(text):
    
    result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
    ) 

    return np.array(result.embeddings[0].values).astype('float32')


# normalize the vector 
def safe_normalize(v):
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v/magnitude
    

sentences = [
 "I love machine learning",
 "Python is great for backend",
 "AI is the future",
 "Football is a great sport"
]

# Sentences embedding
raw_embeddings = [get_embedding(v) for v in sentences]

# Sentences Normalization 
normalized_embeddings = [safe_normalize(v) for v in raw_embeddings]

# for extreme speed, float32 type slightly less precision(compare to 64), but uses half the memory. Each number takes 4 bytes.
embeddings_matrix = np.array(normalized_embeddings).astype('float32')

# dimension of embedding
d = embeddings_matrix.shape[1]  # .shape -> (row, cols)

# container or database that holds all your vectors 
index = faiss.IndexFlatIP(d) # Inner Product (best for cosine)

# Add embeddigs to the database 
index.add(embeddings_matrix)

query = input("Enter your search query: ")
k = int(input("How many top matches you want?: "))

query_embedding = get_embedding(query)
print(query_embedding)

normalized_query = safe_normalize(query_embedding)

normalized_query = np.array([normalized_query]).astype('float32')

distances, indices = index.search(normalized_query, k)

for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {sentences[idx]} (score: {distances[0][i]:.4f})")
