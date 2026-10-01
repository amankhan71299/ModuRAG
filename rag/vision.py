import io
import base64
from pathlib import Path
from PIL import Image
from groq import Groq
from utils.config import GROQ_API_KEY, VISION_MODEL

# Ensure client uses key if available
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

def image_to_data_uri(image_path):
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def summarize_visual(image_path, page_number):
    if not groq_client:
        return f"[Visual on page {page_number}]"

    image_data = image_to_data_uri(image_path)

    prompt = f"""
This visual was extracted from page {page_number} of a document.

Describe the useful business information visible in the visual.

If it is a chart or graph:
- mention important values
- mention highest/lowest values
- mention the main trend

If it is a diagram:
- identify important components
- explain the flow or relationships

If it is a normal business image:
- describe the useful factual information

Keep the description concise and factual.
"""
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data},
                    },
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()
