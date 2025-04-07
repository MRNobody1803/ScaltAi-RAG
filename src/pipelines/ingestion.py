from unstructured.partition.auto import partition
from typing import List
import os

class DocumentIngester:
    """Charge et préprocesse les documents bruts"""

    SUPPORTED_TYPES = ('.pdf', '.html', '.docx')

    @classmethod
    def process_file(cls, file_path: str) -> List[str]:
        if not file_path.endswith(cls.SUPPORTED_TYPES):
            raise ValueError(f"Format non supporté: {os.path.splitext(file_path)[1]}")

        elements = partition(filename=file_path)
        return [str(el) for el in elements if el.text.strip()]