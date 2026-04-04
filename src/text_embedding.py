from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        # Load pre_trained embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def encode(self, texts):
        """
        Convert text (or list of texts) into embeddings
        """
        return self.model.encode(texts)
                
