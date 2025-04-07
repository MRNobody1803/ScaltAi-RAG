class ModelConfig:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Nom court fonctionnel
    EMBEDDING_DEVICE = "cpu"  # "cuda" si GPU NVIDIA
    EMBEDDING_BATCH_SIZE = 64
    VECTOR_STORE_PATH = "data/vector_store/faiss_index"
    SIMILARITY_TOP_K = 3
