from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from groq import Groq
from utils.config import TEXT_MODEL, VISION_MODEL, GROQ_API_KEY
from rag.vision import image_to_data_uri

if GROQ_API_KEY:
    text_llm = ChatGroq(
        model=TEXT_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY
    )
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    text_llm = None
    groq_client = None

rag_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about an uploaded document.

Use ONLY the retrieved context below.

If the answer is not available in the context, say:
"I could not find enough information in the uploaded document to answer this question."

Mention page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

if text_llm:
    text_rag_chain = (
        rag_prompt
        | text_llm
        | StrOutputParser()
    )
else:
    text_rag_chain = None

def estimate_tokens(text):
    if not text: return 0
    # Rough heuristic: 1 token ~ 4 characters
    return len(text) // 4

def build_bounded_context(documents, max_tokens=2800):
    from langchain_core.documents import Document
    selected_docs = []
    current_tokens = 0
    
    for doc in documents:
        content = doc.page_content
        modality = doc.metadata.get("modality")
        
        if modality == "visual" and len(content) > 1000:
            content = content[:1000] + "... [truncated]"
            
        doc_tokens = estimate_tokens(content)
        metadata_overhead = 20 # Roughly "[Page X | MODALITY]\n\n"
        
        if current_tokens + doc_tokens + metadata_overhead <= max_tokens:
            new_doc = Document(page_content=content, metadata=doc.metadata.copy())
            selected_docs.append(new_doc)
            current_tokens += doc_tokens + metadata_overhead
        else:
            if not selected_docs:
                # Force include but truncated if first chunk is too big
                allowed_chars = (max_tokens - metadata_overhead) * 4
                if allowed_chars > 0:
                    truncated_content = content[:allowed_chars] + "... [truncated]"
                    new_doc = Document(page_content=truncated_content, metadata=doc.metadata.copy())
                    selected_docs.append(new_doc)
                    current_tokens = max_tokens
            break
            
    return selected_docs, current_tokens

def format_context(retrieved_docs):
    parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        modality = doc.metadata.get("modality")
        parts.append(
            f"[Page {page} | {modality.upper()}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)

def answer_with_vision(question, context, image_paths):
    if not groq_client:
        return "GROQ_API_KEY is not configured."
        
    image_paths = image_paths[:3] # Limit to avoid exceeding tokens
    content = [
        {
            "type": "text",
            "text": f"""
You are answering questions about an uploaded document.

Use ONLY the retrieved context and the attached retrieved visuals.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers.
- If the information is missing, say "I could not find enough information in the uploaded document to answer this question."
""",
        }
    ]

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_uri(image_path)
                },
            }
        )

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0,
        max_completion_tokens=900,
    )

    return response.choices[0].message.content.strip()
