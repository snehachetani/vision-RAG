"""Turn PDF pages into images- the only parsing this RAG ever does."""

from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image


from src.config import PAGE_IMAGES_DIR, POPPLER_PATH, RENDER_DPI

def pdf_to_images(pdf_path: Path, dpi: int = RENDER_DPI) -> list[Image.Image]:
    """Render every page of a PDF to an RGB PIL image, in page order."""
    poppler = POPPLER_PATH if POPPLER_PATH else None
    return convert_from_path(str(pdf_path), dpi=dpi, fmt="RGB", poppler_path=poppler)

def save_page_image(image: Image.Image, pdf_name: str, page_number: int) -> Path:
    """Save one rendered page as a PNG and return its path."""
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAGE_IMAGES_DIR / f"{Path(pdf_name).stem}_page_{page_number}.png"
    image.save(out_path, format="PNG")
    return out_path