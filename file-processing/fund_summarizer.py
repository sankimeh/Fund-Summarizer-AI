import os
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------- Ingestion Utilities -----------

def classify_file(file_path: str) -> str:
    lower_name = os.path.basename(file_path).lower()
    for doc_type in ['prospectus', 'slide', 'factsheet']:
        if doc_type in lower_name:
            return doc_type
    return "unknown"

def extract_pdf_text_by_page(file_path: str, image_output_dir: str = "pdf_images") -> List[Dict[str, Any]]:
    os.makedirs(image_output_dir, exist_ok=True)
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_data = {"page_number": i, "text": "", "tables": []}

            # Text extraction
            text = page.extract_text()
            if not text or text.strip() == "":
                text = ocr_pdf_page(file_path, i - 1, image_output_dir)
            page_data["text"] = text.strip() if text else ""

            # Table extraction
            tables = page.extract_tables()
            for table in tables:
                if table:
                    page_data["tables"].append(table)

            # Save rendered image for graphs
            image_path = os.path.join(image_output_dir, f"{os.path.basename(file_path)}_page_{i}.png")
            save_pdf_page_as_image(file_path, i - 1, image_path)

            pages.append(page_data)

    return pages

def ocr_pdf_page(file_path: str, page_index: int, image_output_dir: str) -> str:
    with fitz.open(file_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img_path = os.path.join(image_output_dir, f"{os.path.basename(file_path)}_page_{page_index+1}_ocr.png")
        img.save(img_path)
    return pytesseract.image_to_string(img)

def save_pdf_page_as_image(file_path: str, page_index: int, output_path: str):
    with fitz.open(file_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.save(output_path)

def extract_ppt_text_by_slide(file_path: str) -> List[Dict[str, Any]]:
    prs = Presentation(file_path)
    slides = []

    for i, slide in enumerate(prs.slides, start=1):
        slide_data = {"page_number": i, "text": "", "tables": []}
        text_lines = [f"Slide {i}:"]

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_lines.append(shape.text.strip())

            # Extract tables
            if shape.has_table:
                table_data = []
                for row in shape.table.rows:
                    row_text = [cell.text_frame.text if cell.text_frame else "" for cell in row.cells]
                    table_data.append(row_text)
                slide_data["tables"].append(table_data)

        slide_data["text"] = "\n".join(text_lines).strip()
        slides.append(slide_data)

    return slides

# ----------- Chunking Utilities -----------

def extract_section_title(text: str) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines and len(lines[0].split()) <= 20:
        return lines[0]
    return "Unknown Section"

def chunk_pages(
    pages: List[Dict[str, Any]],
    fund_name: str,
    source_file: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for page in pages:
        text_chunks = splitter.split_text(page["text"])
        for chunk_text in text_chunks:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "fund_name": fund_name,
                    "section_title": extract_section_title(chunk_text),
                    "page_number": page["page_number"],
                    "source_file": source_file,
                    # Option 1: exclude tables
                    # "tables": None
                    # Option 2: serialize tables as string (beware of size)
                    # "tables": json.dumps(page.get("tables", []))
                }
            })
    return chunks

# ----------- Main Controller -----------

def process_files(fund_name: str, file_paths: List[str], image_output_dir: str = "pdf_images") -> List[Dict[str, Any]]:
    all_chunks = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        source_file = os.path.basename(file_path)

        if ext == ".pdf":
            pages = extract_pdf_text_by_page(file_path, image_output_dir)
        elif ext in {".ppt", ".pptx"}:
            pages = extract_ppt_text_by_slide(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            continue

        chunks = chunk_pages(pages, fund_name, source_file)
        all_chunks.extend(chunks)

    return all_chunks
