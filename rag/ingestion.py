import fitz
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from utils.config import IMAGE_DIR
from rag.vision import summarize_visual

def extract_multimodal_documents(pdf_path, document_id):
    pdf = fitz.open(str(pdf_path))
    documents = []
    extracted_xrefs = set()
    
    # Create specific image directory for this document
    doc_image_dir = IMAGE_DIR / document_id
    doc_image_dir.mkdir(exist_ok=True, parents=True)

    text_chunks = 0
    tables_count = 0
    visuals_count = 0

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_number = page_index + 1

        # 1. TEXT
        text = page.get_text("text").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "modality": "text",
                        "source": pdf_path.name,
                    },
                )
            )
            text_chunks += 1

        # 2. TABLES
        try:
            tables = page.find_tables().tables
            for table_number, table in enumerate(tables, start=1):
                df = table.to_pandas()
                if not df.empty:
                    table_text = df.to_markdown(index=False)
                    documents.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "page": page_number,
                                "modality": "table",
                                "table_number": table_number,
                                "source": pdf_path.name,
                            },
                        )
                    )
                    tables_count += 1
        except Exception as error:
            print(f"Table extraction warning on page {page_number}:", error)

        # 3. IMAGES
        for image_number, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            if xref in extracted_xrefs:
                continue
            extracted_xrefs.add(xref)
            
            image_data = pdf.extract_image(xref)
            image_bytes = image_data["image"]
            image_extension = image_data["ext"]
            
            image_path = doc_image_dir / f"page_{page_number}_image_{image_number}.{image_extension}"
            image_path.write_bytes(image_bytes)
            
            try:
                summary = summarize_visual(image_path=image_path, page_number=page_number)
                documents.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "page": page_number,
                            "modality": "visual",
                            "image_path": str(image_path),
                            "source": pdf_path.name,
                        },
                    )
                )
                visuals_count += 1
            except Exception as error:
                print(f"Vision warning on page {page_number}:", error)

    pdf.close()
    return documents, {"text_chunks": text_chunks, "tables": tables_count, "visuals": visuals_count}
