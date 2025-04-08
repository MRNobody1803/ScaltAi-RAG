#!/usr/bin/env python3
"""
Point d'entrée principal pour le pipeline RAG
Usage:
  python main.py query --query "Votre question"
  python main.py ingest fichier.pdf
"""

import argparse
from pathlib import Path
from src.pipelines.ingestion import DocumentIngester
from src.pipelines.embedding import EmbeddingPipeline
from src.pipelines.rag_chain import build_rag_chain
from src.core.llm import DeepSeekLLM  # Nouvel import
from src.utils.logger import setup_logger
import warnings
import os

# Configuration
warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Système RAG avec DeepSeek")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Mode Question-Réponse
    qa_parser = subparsers.add_parser("query", help="Poser une question")
    qa_parser.add_argument("--query", type=str, required=True)
    qa_parser.add_argument("--top-k", type=int, default=3)

    # Mode Ingestion
    ingest_parser = subparsers.add_parser("ingest", help="Ingérer des documents")
    ingest_parser.add_argument("paths", nargs="+", type=str)
    ingest_parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.command == "query":
        try:
            llm = DeepSeekLLM()
            rag_chain = build_rag_chain()
            response = rag_chain.invoke({"query": args.query})
            print("\nRéponse RAG:")
            print("─" * 50)
            print(response)
            print("─" * 50)
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            logger.info("Vérifiez que vous avez bien ingéré des documents avec 'python main.py ingest'")

    elif args.command == "ingest":
        try:
            pipeline = EmbeddingPipeline()
            valid_extensions = DocumentIngester.SUPPORTED_TYPES

            for path in args.paths:
                path_obj = Path(path)
                if path_obj.suffix.lower() not in valid_extensions:  # Case insensitive
                    logger.warning(f"Format non supporté: {path_obj.suffix} - Fichier ignoré")
                    continue

                if not path_obj.exists():
                    logger.warning(f"Fichier introuvable: {path} - Ignoré")
                    continue

                logger.info(f"Traitement de {path}...")
                documents = DocumentIngester.process_file(str(path_obj))
                pipeline.run_pipeline(documents)

            logger.info("Ingestion terminée avec succès!")
        except Exception as e:
            logger.critical(f"Échec de l'ingestion: {str(e)}")

if __name__ == "__main__":
    main()