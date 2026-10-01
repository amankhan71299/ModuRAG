import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL

# Initialize models
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

# Initialize Pinecone
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    pc = None

def init_pinecone_index():
    if not pc:
        return None
    if not pc.has_index(PINECONE_INDEX_NAME):
        # Auto-detect dimension
        embedding_dimension = len(embeddings.embed_query("dimension check"))
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
            
    return pc.Index(PINECONE_INDEX_NAME)

def index_documents(documents, namespace):
    if not pc:
        raise ValueError("Pinecone API key not configured.")
        
    init_pinecone_index()
    
    # Try to clear namespace if it exists (though it shouldn't for unique doc IDs)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass
        
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )
    vectorstore.add_documents(documents)
    return vectorstore
