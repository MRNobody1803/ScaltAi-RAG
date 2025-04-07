from langchain_core.runnables import Runnable, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
import os

class Retriever(Runnable):
    """Implémentation simplifiée et fonctionnelle du Retriever"""

    def __init__(self):
        self.vectorstore = FAISS.from_texts(
            texts=[""],
            embedding=HuggingFaceEmbeddings()
        )

    @classmethod
    def load_from_disk(cls):
        instance = cls()
        if os.path.exists("data/vector_store"):
            instance.vectorstore = FAISS.load_local(
                "data/vector_store",
                HuggingFaceEmbeddings(),
                allow_dangerous_deserialization=True
            )
        return instance

    def add_documents(self, documents: List[str]):
        self.vectorstore.add_texts(documents)
        self.vectorstore.save_local("data/vector_store")

    def invoke(self, input: Dict[str, Any], **kwargs):
        return self.vectorstore.similarity_search(input["query"])

    def as_runnable(self):
        """Convertit le retriever en Runnable compatible"""
        return RunnableLambda(self.invoke)