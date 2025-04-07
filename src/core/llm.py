import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Charge les variables d'environnement
load_dotenv()

logger = logging.getLogger(__name__)

class DeepSeekLLM:

    """
    Wrapper sécurisé pour l'API DeepSeek avec configuration via .env
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_env_config()
        self._validate_config()

    def _load_env_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis les variables d'environnement"""
        return {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3")),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "timeout": int(os.getenv("DEEPSEEK_TIMEOUT", "30"))
        }

    def _validate_config(self):
        """Valide les paramètres essentiels"""
        if not self.config["api_key"]:
            raise RuntimeError(
                "Configuration DeepSeek manquante. "
                "Créez un fichier .env avec DEEPSEEK_API_KEY"
            )

        if self.config["temperature"] < 0 or self.config["temperature"] > 1:
            raise ValueError("La température doit être entre 0 et 1")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir du prompt

        Args:
            prompt: Texte d'entrée
            **kwargs: Paramètres optionnels (override)
                - temperature: float
                - max_tokens: int

        Returns:
            str: Réponse générée
        """
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": kwargs.get("model", self.config["model"]),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            **{k: v for k, v in kwargs.items() if k in ["max_tokens", "top_p"]}
        }

        try:
            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur API: {str(e)}")
            raise RuntimeError(f"Échec de la génération: {str(e)}") from e

# Utilisation de base
if __name__ == "__main__":
    try:
        llm = DeepSeekLLM()
        print(llm.generate("Explique-moi les LLMs comme si j'avais 5 ans"))
    except Exception as e:
        logger.critical(f"Erreur critique: {str(e)}")
        raise