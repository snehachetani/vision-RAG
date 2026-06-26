"""Central configuration: env vars, model names, paths, and Qdrant settings."""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
PDFS_DIR = ROOT_DIR / "pdfs"
PAGE_IMAGES_DIR = ROOT_DIR / "page_images"
QDRANT_PATH = ROOT_DIR / "qdrant_data"

# pdf2image
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

# COLPALI_MODEL = "vidore/colqwen2-v1.0"
# Note: Since the GPU is an MX450 (2 GB VRAM), we'll run this model on the CPU 
# to avoid Out-Of-Memory (OOM) issues, as a 2B parameter model requires 4GB+ VRAM.
COLPALI_MODEL = "vidore/colqwen2-v1.0"
RENDER_DPI = 150

COLLECTION_NAME = "pdf_pages"
VECTOR_DIM = 128 # ColQwen emits one 128-d vector per patch
TOP_K = 3
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")