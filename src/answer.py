"""Vision answer step- Gemini 3.5 Flash reads the retrieved page images."""

from pathlib import Path
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL

_PROMPT = (
    "You are given one or more pages from a document as images. "
    "Answer the question using only what is visible in these pages- including "
    "charts, tables, and scanned text. If the answer is a number from a chart or "
    "table, state it exactly and name the page it came from."
    "If the pages do not contain the answer, say so.\n\nQuestion:{question}"
)

def _image_part(image_path: Path) -> types.Part:
    """Load a saved page PNG as an inline image part for the model."""
    data = Path(image_path).read_bytes()
    return types.Part.from_bytes(data=data, mime_type="image/png")


def answer(question: str, pages: list[dict]) -> str:
    """Ask Gemini the question against the retrieved page images."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents: list = []
    for page in pages:
        contents.append(f"---{page['pdf']}- page{page['page_number']} ---")
        contents.append(_image_part(Path(page["image_path"])))
    contents.append(_PROMPT.format(question=question))
    response = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    return response.text