# Required libraries:
# pip install pdf2image pytesseract pillow
from pdf2image import convert_from_path
import pytesseract, unicodedata, re
from multiprocessing import Pool

# Path to Tesseract binary (adjust this path if needed)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

def clean_text(text: str) -> str:
    """
    Normalize and clean OCR text:
    - Convert Unicode to NFC form
    - Replace multiple spaces/newlines with a single space
    - Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ocr_page(args):
    """
    Perform OCR on a single page.
    Args:
        args: (index, page_image)
    Returns:
        A string with page header and recognized text
    """
    idx, page = args
    text = pytesseract.image_to_string(page, lang="vie")  # OCR in Vietnamese
    text = clean_text(text)
    return f"\n--- Page {idx+1} ---\n{text}"

def main(pdf_path, first_page=1, last_page=20, n_proc=12):
    """
    Convert PDF pages to images and perform OCR in parallel.

    Args:
        pdf_path (str): Path to the PDF file
        first_page (int): First page number to process
        last_page (int): Last page number to process
        n_proc (int): Number of parallel processes for OCR
    """
    print(f"Converting {last_page - first_page + 1} pages from PDF to images...")
    pages = convert_from_path(pdf_path, dpi=200, first_page=first_page)

    print(f"Running OCR with {n_proc} parallel processes...")
    with Pool(processes=n_proc) as p:
        results = p.map(ocr_page, list(enumerate(pages, start=first_page)))

    out_file = f"{pdf_path}_p{first_page}-{last_page}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"OCR results saved from page {first_page} to {last_page} into: {out_file}")

if __name__ == "__main__":
    pdf_path = "SGK_LOP4_canhdieu.pdf"
    main(pdf_path, first_page=1, last_page=20, n_proc=12)
