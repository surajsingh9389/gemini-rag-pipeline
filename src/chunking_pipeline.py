import string

class ChunkingPipeline:
    def __init__(self, chunk_size, overlap_k):
        self.chunk_size = chunk_size
        self.overlap_k = overlap_k
        
    # Function for converting raw text into list of sentences 
    def build_sentences(self, text):
        # Input validation 
        if not text: 
            return []
        
        delimiters = [".", "!", "?"]
        sentences = []
        start = 0
        
        for i in range(len(text)):
            if text[i] in delimiters:
                sentence = text[start: i+1].strip()
                if sentence:
                    sentences.append(sentence)
                start = i+1
        
        # leftover text
        if start < len(text):
            sentence = text[start:].strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    
    def create_chunks(self, sentences):
        # input validation 
        if not sentences: 
            return []
        
        chunks_list = []
        word_limit = self.chunk_size
        
        # Stores current chunk as list of sentences
        current_chunk_list = []
        
        # Tracks total word count of current chunk
        word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # Check if adding this sentence exceeds chunk size
            if word_count + sentence_word_count > word_limit:
                
                # Save current chunk if not empty
                if current_chunk_list:
                    chunks_list.append(" ".join(current_chunk_list))
                    
                # Apply overlap: take last k sentences and add current sentence
                current_chunk_list = current_chunk_list[-self.overlap_k:] + [sentence]
                
                # Recalculate word count for new chunk
                word_count = sum(len(s.split()) for s in current_chunk_list)
            else:
                # Add sentence to current chunk
                current_chunk_list.append(sentence)
                word_count+=sentence_word_count
                
        # Append final chunk        
        if current_chunk_list:
            chunks_list.append(" ".join(current_chunk_list))
            
        return chunks_list
    
    # Function to add metadata on chunks 
    def add_metadata(self, chunks, title, source):   
        # Validat input 
        if not chunks:
            return []
        
        # Handle missing metadata 
        title = title or "Unknown Title"
        source = source or "Unknown Source"
        
        chunked_documents = []
        
        for chunk_id , chunk in enumerate(chunks, 1):
            chunk_info = {
                "chunk_id": chunk_id,
                "text": chunk,
                "word_count": len(chunk.split()),
                "metadata": {
                    "title": title,
                    "source": source
                }
            }
            
            chunked_documents.append(chunk_info)
            
        return chunked_documents
    
    def validate_chunk(self, chunk):
        """
        Validate a chunk before embedding.

        Rules:
        1. Chunk should not be empty or whitespace
        2. Word count should be within acceptable limits
        3. Chunk should not contain excessive noisy symbols
        """
        chunk_size = self.chunk_size
        
        # safe extraction of text 
        text = chunk.get("text", "")
        allowed_symbols = set(string.punctuation)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Reject empty text
        if not text:
            return False
        
        # Count words 
        word_count = len(text.split())
        
        # Rule 1: Check size constraints
        if word_count < 5 or word_count > chunk_size:
            return False
        
        # Count total characters
        total_char = len(text)
        
        # Avoid division by zero
        if total_char == 0:
            return False
        
        # Count noisy symbols
        total_symbols = 0        
        for char in text:
            if not char.isalnum() and char not in allowed_symbols and char != " ":
                total_symbols+=1
        
        # Calculate noise ratio
        ratio_of_symbols = (total_symbols/total_char)*100
        
        # Rule 2: Reject if too noisy
        symbol_threshold = 30
        if ratio_of_symbols > symbol_threshold:
            return False
        
        return True
    
    def analyze_chunks(self, chunks):
        """
        Analyze chunk statistics to understand quality before embedding.

        Returns:
        - total number of chunks
        - average word count
        - minimum word count
        - maximum word count
        """
        
        # Handle empty input 
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_word_count": 0,
                "min_word_count": 0,
                "max_word_count": 0
            }
            
        word_counts = []
        
        # Collect word counts from each chunk 
        for chunk in chunks:
            text = chunk.get("text", "")
            word_counts.append(len(text.split()))
            
            
        total_chunks = len(word_counts)
        avg_word_count = sum(word_counts)/total_chunks
        min_word_count = min(word_counts)
        max_word_count = max(word_counts)
        
        
        return {
            "total_chunks": total_chunks,
            "avg_word_count": round(avg_word_count, 2),
            "min_word_count": min_word_count,
            "max_word_count": max_word_count
            
        }
        
    
    def run_pipeline(self, text, title, source, debug=False):
        
        """
        End-to-end pipeline:
        1. Sentence splitting
        2. Chunk creation with overlap
        3. Metadata enrichment
        4. Chunk validation (filtering)
        """
        
        # Input validation 
        if not text:
            return []
        
        # step 1: Build sentences 
        sentences = self.build_sentences(text)
        
        # step 2: Create chunks 
        chunks = self.create_chunks(sentences)
        
        # step 3: Add metadata 
        chunked_documents = self.add_metadata(chunks, title, source)
        
        # step 4: Validate and filter chunks 
        clean_chunks = [c for c in chunked_documents if self.validate_chunk(c)]
        
        # debugging 
        if debug:
            print(f"Total sentences: {len(sentences)}")
            print(f"Chunks before validation: {len(chunked_documents)}")
            print(f"Chunks after validation: {len(clean_chunks)}")
        
        
        return clean_chunks