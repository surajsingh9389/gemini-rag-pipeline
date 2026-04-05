# 🚀 Rapid-Context-RAG

A high-performance, modular **Retrieval-Augmented Generation (RAG)** pipeline built with **Hugging Face Transformers** and **FAISS**. This project implements a complete local loop: **Intelligent Chunking** → **Vector Embedding** → **Semantic Retrieval** → **Grounded Generation**.

## 🌟 Key Technical Features
*   **End-to-End Hugging Face Integration:** Leverages `sentence-transformers` for local vectorization and `transformers` (FLAN-T5) for neural text generation.
*   **Advanced Chunking Pipeline:** Implements a sliding window strategy with configurable overlap (`overlap_k`) to maintain semantic continuity across segments.
*   **High-Performance Vector Store:** Uses **FAISS** (`IndexFlatIP`) with **L2 Normalization** to achieve high-precision **Cosine Similarity** matching in sub-milliseconds.
*   **Grounded Generation:** A custom prompt engineering framework ensures the LLM answers strictly based on retrieved context, effectively eliminating hallucinations.
*   **Data Quality Control:** Built-in validation logic filters "noisy" chunks based on symbol-to-text ratios and word count constraints.

## 🛠️ Tech Stack
*   **LLM (Generation):** `google/flan-t5-base` (Hugging Face)
*   **Embeddings:** `all-MiniLM-L6-v2` (384-dimensional vectors)
*   **Vector DB:** [FAISS](https://github.com) (Facebook AI Similarity Search)
*   **Data Processing:** [NumPy](https://numpy.org) for matrix operations and L2 Normalization
*   **Language:** Python 3.x

## 📈 System Architecture

### 1. Ingestion & Preprocessing (`chunking_pipeline.py`)
Raw text is transformed into manageable segments using a multi-step pipeline:
- **Sentence Splitting:** Breaks text into individual sentences.
- **Sliding Window Chunking:** Groups sentences into chunks of defined size with overlapping context.
- **Validation:** Filters out low-quality chunks (e.g., too short or high noise-to-text ratio).
- **Metadata Tagging:** Attaches source and title info to every chunk.

### 2. Vectorization & Storage (`text_embedding.py`, `store_vector.py`)
- **Embedding:** Chunks are converted into 384-dimensional dense vectors using a local Transformer model.
- **Indexing:** Vectors are L2-normalized and stored in a FAISS `IndexFlatIP` index, enabling ultra-fast similarity searches.

### 3. RAG Execution (`rag_system.py`, `llm_generation.py`, `build_prompt.py`)
- **Retrieval:** User queries are embedded and searched against the FAISS index to find the `top_k` most relevant context snippets.
- **Augmentation:** A prompt is dynamically constructed by injecting the retrieved context into a structured template.
- **Generation:** The FLAN-T5 model synthesizes a final, human-readable response based **ONLY** on the provided context.

## 🔧 Installation
1.  Clone the repository:

    ```bash
    git clone https://github.com
    ```
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Run Project:

    ```bash
    .venv/Scripts/activate
    python src/main.py
    ```



