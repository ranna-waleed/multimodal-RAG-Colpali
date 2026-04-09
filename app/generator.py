import base64
import io
from groq import Groq
from PIL import Image
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_answer(query, retrieved_pages):
    if not retrieved_pages:
        return "No relevant pages found. Please ingest a document first."

    print("Generating answer with Groq...")

    content = []
    content.append({
        "type": "text",
        "text": f"""You are a document analysis assistant.
You have been given {len(retrieved_pages)} relevant page(s) from a document.

Answer the following question based ONLY on the provided pages:
Question: {query}

Rules:
- Be specific and detailed
- Always mention which page number your answer comes from
- If the answer involves a table or chart, describe the key numbers
- If the answer cannot be found, say so clearly
- Use citations like (Page X)
"""
    })

    for page in retrieved_pages:
        if page["image"] is not None:
            content.append({
                "type": "text",
                "text": f"\n--- Page {page['page_number']} from '{page['doc_name']}' ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(page['image'])}"
                }
            })

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content}],
        max_tokens=1024
    )

    return response.choices[0].message.content


def format_citations(retrieved_pages):
    citations = []
    for page in retrieved_pages:
        citations.append(
            f" {page['doc_name']} — Page {page['page_number']} (score: {page['score']:.3f})"
        )
    return "\n".join(citations)


if __name__ == "__main__":
    print("Testing Groq connection...")
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50
    )
    print("Groq response:", response.choices[0].message.content)
    print(" Groq connection working")