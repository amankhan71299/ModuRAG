import time
import json
from datetime import datetime
from pathlib import Path
from rag.retrieval import get_retriever
from rag.llm import format_context, answer_with_vision, text_rag_chain, build_bounded_context, estimate_tokens

def answer_question(question, namespace):
    t0 = time.perf_counter()
    
    # Retrieval
    retriever = get_retriever(namespace)
    retrieved_docs = retriever.invoke(question)
    t1 = time.perf_counter()
    retrieval_latency_ms = (t1 - t0) * 1000
    
    if not retrieved_docs:
        return {
            "answer": "I could not find enough information in the uploaded document to answer this question.",
            "sources": [],
            "metrics": None
        }

    # Context Building
    selected_docs, _ = build_bounded_context(retrieved_docs, max_tokens=2800)
    
    def get_estimated_tokens(docs):
        # collect images
        img_paths = []
        for doc in docs:
            modality = doc.metadata.get("modality")
            image_path = doc.metadata.get("image_path")
            if modality == "visual" and image_path and Path(image_path).exists():
                img_paths.append(image_path)
        
        img_paths = img_paths[:3] # LLM only takes up to 3 images
        
        ctx = format_context(docs)
        if img_paths:
            prompt_text = f"You are answering questions about an uploaded document.\nUse ONLY the retrieved context and the attached retrieved visuals.\nRETRIEVED CONTEXT:\n{ctx}\nQUESTION:\n{question}\nInstructions:\n- Answer factually.\n- Use the attached visuals when relevant.\n- Mention page numbers.\n- If the information is missing, say \"I could not find enough information in the uploaded document to answer this question.\""
        else:
            prompt_text = f"You are a helpful assistant answering questions about an uploaded document.\nUse ONLY the retrieved context below.\nIf the answer is not available in the context, say:\n\"I could not find enough information in the uploaded document to answer this question.\"\nMention page numbers when possible.\nCONTEXT:\n{ctx}\nQUESTION:\n{question}\nANSWER:"
            
        estimated = estimate_tokens(prompt_text)
        # Add conservative overhead for vision model images (e.g. 1200 tokens per image)
        estimated += len(img_paths) * 1200
        return estimated, ctx, img_paths
        
    initial_chunks = len(selected_docs)
    final_prompt_tokens, context, image_paths = get_estimated_tokens(selected_docs)
    initial_prompt_tokens = final_prompt_tokens
    token_safety_triggered = False

    # Final Safety Check (< 6000 tokens)
    while final_prompt_tokens > 6000 and len(selected_docs) > 1:
        selected_docs.pop()
        token_safety_triggered = True
        final_prompt_tokens, context, image_paths = get_estimated_tokens(selected_docs)

    if final_prompt_tokens > 6000 and len(selected_docs) == 1:
        token_safety_triggered = True
        doc = selected_docs[0]
        img_tokens = 1200 if (doc.metadata.get("modality") == "visual" and doc.metadata.get("image_path") and Path(doc.metadata.get("image_path")).exists()) else 0
        allowed_chars = (6000 - img_tokens - estimate_tokens(question) - 200) * 4
        if allowed_chars > 0 and len(doc.page_content) > allowed_chars:
            doc.page_content = doc.page_content[:allowed_chars] + "... [truncated]"
        final_prompt_tokens, context, image_paths = get_estimated_tokens(selected_docs)

    t2 = time.perf_counter()
    context_building_latency_ms = (t2 - t1) * 1000
    
    print("========================================")
    print("RAG TOKEN SAFETY")
    print("========================================")
    print(f"Initial estimated prompt tokens: {initial_prompt_tokens}")
    print(f"Final estimated prompt tokens: {final_prompt_tokens}")
    print(f"Chunks before safety: {initial_chunks}")
    print(f"Chunks after safety: {len(selected_docs)}")
    print("========================================")
    
    sources = []
    for doc in selected_docs:
        sources.append({
            "page": doc.metadata.get("page"),
            "modality": doc.metadata.get("modality"),
            "content": doc.page_content,
            "image_path": doc.metadata.get("image_path")
        })


    # LLM Generation
    t3 = time.perf_counter()
    
    visual_keywords = [
        "graph",
        "chart",
        "diagram",
        "image",
        "picture",
        "visual",
        "photo",
        "figure",
        "plot"
    ]
    question_lower = question.lower()
    needs_vision = any(keyword in question_lower for keyword in visual_keywords)
    
    if image_paths and needs_vision:
        answer = answer_with_vision(
            question=question,
            context=context,
            image_paths=image_paths,
        )
    elif text_rag_chain:
        answer = text_rag_chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )
    else:
        answer = "GROQ_API_KEY is not configured."
    
    t4 = time.perf_counter()
    llm_latency_ms = (t4 - t3) * 1000
    total_latency_ms = (t4 - t0) * 1000

    print("========================================")
    print("RAG PERFORMANCE")
    print("========================================")
    print(f"Retrieval latency:        {retrieval_latency_ms:.2f} ms")
    print(f"Context building:         {context_building_latency_ms:.2f} ms")
    print(f"LLM latency:              {llm_latency_ms:.2f} ms")
    print(f"Total latency:            {total_latency_ms:.2f} ms")
    print(f"Retrieved chunks:         {len(retrieved_docs)}")
    print(f"Selected chunks:          {len(selected_docs)}")
    print(f"Final prompt tokens:      {final_prompt_tokens}")
    print("========================================")
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "retrieval_latency_ms": retrieval_latency_ms,
        "context_building_latency_ms": context_building_latency_ms,
        "llm_latency_ms": llm_latency_ms,
        "total_latency_ms": total_latency_ms,
        "retrieved_chunks": len(retrieved_docs),
        "selected_chunks": len(selected_docs),
        "estimated_context_tokens": estimate_tokens(context),
        "estimated_final_prompt_tokens": final_prompt_tokens,
        "token_safety_triggered": token_safety_triggered
    }
    
    PROJECT_ROOT = Path(__file__).parent.parent
    METRICS_FILE = PROJECT_ROOT / "rag_metrics_optimized.jsonl"
    
    with open(METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\\n")
        
    return {
        "answer": answer,
        "sources": sources,
        "metrics": metrics
    }

