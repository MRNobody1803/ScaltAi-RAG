# src/pipelines/embedding.py
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.core.embeddings import EmbeddingManager
from src.core.retriever import Retriever  # Modification ici
from configs.model_config import ModelConfig
import logging
from dotenv import load_dotenv

# Configuration
load_dotenv()
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """
    Pipeline complet pour le traitement des embeddings avec :
    - Parallélisation
    - Gestion des batches
    - Barre de progression
    - Sauvegarde automatique
    """

    def __init__(self, retriever: Optional[Retriever] = None):  # Modification ici
        self.embedder = EmbeddingManager()
        self.retriever = retriever or Retriever()  # Modification ici
        self.batch_size = ModelConfig.EMBEDDING_BATCH_SIZE

    def process_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Traite une liste de documents et retourne leurs embeddings

        Args:
            documents: Liste de textes à embedder

        Returns:
            List[List[float]]: Embeddings générés
        """
        try:
            # Découpage en batches pour optimisation mémoire
            batches = [
                documents[i:i + self.batch_size]
                for i in range(0, len(documents), self.batch_size)
            ]

            embeddings = []
            with ThreadPoolExecutor() as executor:
                # Traitement parallèle avec barre de progression
                results = list(tqdm(
                    executor.map(self._process_batch, batches),
                    total=len(batches),
                    desc="Génération des embeddings"
                ))

                for batch_result in results:
                    embeddings.extend(batch_result)

            return embeddings

        except Exception as e:
            logger.error(f"Erreur lors du embedding: {str(e)}")
            raise

    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Traite un batch de documents"""
        return self.embedder.embed_documents(batch)

    def run_pipeline(self, documents: List[str], save: bool = True) -> Retriever:  # Modification ici
        """
        Exécute le pipeline complet :
        1. Génère les embeddings
        2. Ajoute au vector store
        3. Sauvegarde sur disque

        Args:
            documents: Liste de textes à traiter
            save: Sauvegarder le résultat sur disque

        Returns:
            VectorRetriever: Retriever mis à jour
        """
        try:
            logger.info(f"Début du pipeline pour {len(documents)} documents")

            # Étape 1: Génération des embeddings
            embeddings = self.process_documents(documents)

            # Étape 2: Ajout au vector store
            self.retriever.add_documents(documents)

            # Étape 3: Sauvegarde
            if save:
                self.retriever.save_to_disk()
                logger.info(f"Embeddings sauvegardés dans {ModelConfig.VECTOR_STORE_PATH}")

            return self.retriever

        except Exception as e:
            logger.critical(f"Échec du pipeline: {str(e)}")
            raise

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration de base
    logging.basicConfig(level=logging.INFO)

    # Documents d'exemple
    sample_docs = [
        "La décomposition de tâches est une technique clé en IA",
        "DeepSeek est un modèle de langage avancé",
        "Le RAG combine recherche et génération"
    ]

    # Exécution du pipeline
    pipeline = EmbeddingPipeline()
    retriever = pipeline.run_pipeline(sample_docs)

    # Test de récupération
    results = retriever.retrieve("Qu'est-ce que la décomposition ?")
    print("Résultats:", results)