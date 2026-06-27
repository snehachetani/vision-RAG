"""Vision answer step- Gemini 3.5 Flash reads the retrieved page images and highlights citations."""

from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

from src.config import GEMINI_API_KEY, GEMINI_MODEL

class Citation(BaseModel):
    pdf: str = Field(description="The filename of the PDF.")
    page_number: int = Field(description="The page number where the answer is found.")
    box_2d: list[int] = Field(description="Bounding box coordinates containing the evidence/answer [ymin, xmin, ymax, xmax] normalized to 0-1000.")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The detailed answer to the question.")
    citations: list[Citation] = Field(description="List of citations with bounding boxes indicating where the answer is located in the images.")

_PROMPT = (
    "You are given one or more pages from a document as images. "
    "Answer the question using only what is visible in these pages - including "
    "charts, tables, and scanned text. If the answer is a number from a chart or "
    "table, state it exactly.\n\n"
    "Identify the exact bounding box region on the page image that contains the evidence/answer. "
    "Return the coordinates normalized to 0-1000 scale: [ymin, xmin, ymax, xmax]. "
    "If the pages do not contain the answer, say so.\n\n"
    "Question: {question}"
)

def _image_part(image_path: Path) -> types.Part:
    """Load a saved page PNG as an inline image part for the model."""
    data = Path(image_path).read_bytes()
    return types.Part.from_bytes(data=data, mime_type="image/png")

def draw_citation_box(image_path: Path, box_2d: list[int]) -> Path:
    """Draw a blue rectangle on the page image based on normalized coordinates [ymin, xmin, ymax, xmax] (0-1000)."""
    with Image.open(image_path) as img:
        w, h = img.size
        ymin, xmin, ymax, xmax = box_2d
        ImageDraw.Draw(img).rectangle(
            [xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000],
            outline="blue", width=5
        )
        annotated_path = image_path.parent / f"annotated_{image_path.name}"
        img.save(annotated_path)
        return annotated_path

def answer(question: str, pages: list[dict]) -> str:
    """Ask Gemini the question against the retrieved page images and annotate the citation boxes."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents: list = []
    for page in pages:
        contents.append(f"--- {page['pdf']} - page {page['page_number']} ---")
        contents.append(_image_part(Path(page["image_path"])))
    contents.append(_PROMPT.format(question=question))
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AnswerWithCitations,
        )
    )
    
    try:
        result = AnswerWithCitations.model_validate_json(response.text)
        annotated_info = []
        for citation in result.citations:
            matching_page = next((p for p in pages if Path(p["pdf"]).name == Path(citation.pdf).name and int(p["page_number"]) == int(citation.page_number)), None)
            if matching_page:
                annotated_path = draw_citation_box(Path(matching_page["image_path"]), citation.box_2d)
                annotated_info.append(f"- {citation.pdf} (page {citation.page_number}) -> Highlighted file: {annotated_path.name}")
        
        final_answer = result.answer
        if annotated_info:
            final_answer += "\n\nCitations (with bounding boxes):\n" + "\n".join(annotated_info)
        return final_answer
    except Exception as e:
        print(f"Error parsing structured response or drawing box: {e}")
        return response.text

