import logging
import os
import sys
from pathlib import Path
from typing import List

import pytesseract
from langchain.document_loaders import PyPDFium2Loader, UnstructuredFileLoader
from langchain.schema import Document
from pdf2image import convert_from_path

# ========================
# CONFIGURATION
# ========================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================
# PATH RESOLUTION
# ========================

def get_installation_path(executable: str) -> str:
    """Find executable in common installation locations"""
    search_paths = [
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
        r"C:\Program Files\poppler\Library\bin",
        *os.environ["PATH"].split(os.pathsep)
    ]
    
    for path in search_paths:
        full_path = Path(path) / executable
        if full_path.exists():
            return str(full_path)
    
    raise FileNotFoundError(
        f"{executable} not found. Please install and add to PATH.\n"
        f"Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
        f"Poppler: https://blog.alivate.com.au/poppler-windows/"
    )

# ========================
# PDF PROCESSOR CLASS
# ========================

class PDFExtractor:
    def __init__(self):
        """Initialize with comprehensive dependency checking"""
        try:
            # Find Tesseract
            self.tesseract_path = get_installation_path("tesseract.exe")
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            # Find Poppler
            self.poppler_path = os.path.dirname(get_installation_path("pdftoppm.exe"))
            
            # Verify installations
            tesseract_version = pytesseract.get_tesseract_version()
            logger.info(f"Initialized Tesseract v{tesseract_version} at {self.tesseract_path}")
            logger.info(f"Using Poppler from {self.poppler_path}")
            
        except Exception as e:
            logger.error("Dependency initialization failed")
            logger.error(str(e))
            raise

    def process_directory(self, input_dir: str, output_dir: str = "extracted_results") -> None:
        """Process all PDFs in directory and subdirectories"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(exist_ok=True)
        
        pdf_files = list(input_path.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDFs to process")
        
        for pdf_file in pdf_files:
            try:
                # Create matching output subdirectory
                relative_path = pdf_file.relative_to(input_path)
                output_subdir = output_path / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Process file
                docs = self._process_pdf(pdf_file)
                
                # Save results
                output_file = output_subdir / f"{pdf_file.stem}.txt"
                self._save_results(docs, output_file)
                
                logger.info(f"Processed: {relative_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")

    def _process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF with fallback strategies"""
        try:
            # First try native text extraction
            loader = UnstructuredFileLoader(
                str(pdf_path),
                strategy="hi_res",
                post_processors=["ocr"],
            )
            docs = loader.load()
            
            if not self._is_quality_acceptable(docs):
                logger.debug(f"Trying PyPDFium2 for better table extraction")
                loader = PyPDFium2Loader(str(pdf_path))
                docs = loader.load()
                
                if self._needs_ocr(docs):
                    logger.debug(f"Falling back to OCR")
                    return self._run_ocr(pdf_path)
            
            return docs
            
        except Exception as e:
            logger.debug(f"Standard extraction failed, using OCR: {str(e)}")
            return self._run_ocr(pdf_path)

    def _run_ocr(self, pdf_path: Path) -> List[Document]:
        """Process PDF through OCR"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=300,
                poppler_path=self.poppler_path
            )
            
            return [
                Document(
                    page_content=pytesseract.image_to_string(img),
                    metadata={
                        "source": str(pdf_path),
                        "page": i+1,
                        "method": "OCR"
                    }
                ) for i, img in enumerate(images)
            ]
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return []

    def _save_results(self, docs: List[Document], output_file: Path) -> None:
        """Save extracted text to file"""
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(f"\n=== Page {doc.metadata.get('page', 'N/A')} ===\n")
                f.write(doc.page_content)
                f.write("\n\n")

    def _is_quality_acceptable(self, docs: List[Document]) -> bool:
        """Check if extracted text meets quality standards"""
        return bool(docs) and any(len(doc.page_content.strip()) > 100 for doc in docs)

    def _needs_ocr(self, docs: List[Document]) -> bool:
        """Determine if OCR should be attempted"""
        return not docs or sum(len(doc.page_content.strip()) for doc in docs[:3]) < 300