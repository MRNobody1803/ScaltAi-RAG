from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(llm):
    from src.core.retriever import Retriever

    retriever = Retriever.load_from_disk()
    prompt = ChatPromptTemplate.from_template(
        "Réponds en français avec ce contexte:\n{context}\n\nQuestion: {query}"
    )

    return (
            {"context": retriever.as_runnable(), "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )