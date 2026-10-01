from langchain_pinecone import PineconeVectorStore
from rag.indexing import embeddings
from utils.config import PINECONE_INDEX_NAME

def get_retriever(namespace):
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})
