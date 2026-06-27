"""Ingest CLI: PDF pages -> page PNGs -> ColPali multivectors -> Qdrant."""

import sys
from pathlib import Path

from src.config import PDFS_DIR
from src.embedder import embed_image
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import close_client, ensure_collection, upsert_page

def ingest_pdf(pdf_path: Path, start_id: int) -> int:
    """Render, embed, and store every page of one PDF."""
    print(f"\nRendering{pdf_path.name} ...")
    pages = pdf_to_images(pdf_path)
    print(f"{len(pages)} pages")
    for offset, page in enumerate(pages):
        page_number = offset + 1
        image_path = save_page_image(page, pdf_path.name, page_number)
        multivector = embed_image(page)
        upsert_page(start_id + offset, multivector,pdf_path.name, page_number, image_path)
        print(f" embedded + stored page{page_number}")
    return start_id + len(pages)


def main(pdf_args: list[str]) -> None:
    """Resolve PDFs, then build the Qdrant collection fresh."""
    pdfs = [Path(p) for p in pdf_args] if pdf_args else sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Put a PDF in{PDFS_DIR} or pass a path.")
        sys.exit(1)
    ensure_collection(reset=True)
    next_id = 0
    try:
        for pdf in pdfs:
            if pdf.exists():
                next_id = ingest_pdf(pdf, next_id)
    finally:
        close_client()
    print(f"\nDone. Indexed{next_id} pages into Qdrant.")

if __name__ == "__main__":
    main(sys.argv[1:])


        
