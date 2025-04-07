from typing import List
from sentence_transformers import SentenceTransformer
from configs.model_config import ModelConfig
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Solution alternative directe avec SentenceTransformer"""

    def __init__(self):
        try:
            self.model = SentenceTransformer(
                ModelConfig.EMBEDDING_MODEL,
                device="cpu"  # Changez à "cuda" si disponible
            )
        except Exception as e:
            logger.error(f"Erreur de chargement du modèle: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Version batch optimisée"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embedding pour une seule requête"""
        return self.model.encode(text, convert_to_numpy=True).tolist()