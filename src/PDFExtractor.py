import logging
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib
import pytesseract
from langchain_community.document_loaders import PyPDFium2Loader, UnstructuredFileLoader
from langchain.schema import Document
from pdf2image import convert_from_path
from tqdm import tqdm

# ========================
# CONFIGURATION
# ========================

@dataclass
class ExtractorConfig:
    """Configuration settings for PDF extractor"""
    # OCR Settings
    ocr_dpi: int = 400
    ocr_language: str = "eng"
    ocr_timeout: int = 30
    
    # Quality thresholds
    min_text_length: int = 100
    quality_check_pages: int = 3
    min_quality_score: float = 0.3
    
    # Performance settings
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: str = ".pdf_cache"
    
    # Processing options
    enable_retry: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Output settings
    output_format: str = "txt"  # txt, json, markdown
    include_metadata: bool = True
    preserve_structure: bool = True

# Configure logging with structured format
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
# UTILITIES
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

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file for caching"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def retry_operation(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retry logic"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator

# ========================
# PROCESSING METRICS
# ========================

@dataclass
class ProcessingMetrics:
    """Track processing statistics"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    ocr_files: int = 0
    cached_files: int = 0
    total_pages: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def success_rate(self) -> float:
        return (self.successful_files / self.total_files) if self.total_files > 0 else 0.0

# ========================
# ENHANCED PDF PROCESSOR CLASS
# ========================

class EnhancedPDFExtractor:
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """Initialize with comprehensive dependency checking and configuration"""
        self.config = config or ExtractorConfig()
        self.metrics = ProcessingMetrics()
        self._cache = {}
        
        try:
            # Find Tesseract
            self.tesseract_path = get_installation_path("tesseract.exe")
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            # Find Poppler
            self.poppler_path = os.path.dirname(get_installation_path("pdftoppm.exe"))
            
            # Setup cache directory
            if self.config.enable_caching:
                self.cache_dir = Path(self.config.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                self._load_cache()
            
            # Verify installations
            tesseract_version = pytesseract.get_tesseract_version()
            logger.info(f"Initialized Enhanced PDF Extractor")
            logger.info(f"Tesseract v{tesseract_version} at {self.tesseract_path}")
            logger.info(f"Poppler from {self.poppler_path}")
            logger.info(f"Configuration: {self.config}")
            
        except Exception as e:
            logger.error("Dependency initialization failed")
            logger.error(str(e))
            raise

    def _load_cache(self) -> None:
        """Load processing cache from disk"""
        cache_file = self.cache_dir / "processing_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded cache with {len(self._cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save processing cache to disk"""
        if not self.config.enable_caching:
            return
            
        cache_file = self.cache_dir / "processing_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def process_directory(self, input_dir: str, output_dir: str = "extracted_results") -> ProcessingMetrics:
        """Process all PDFs in directory with enhanced features"""
        start_time = time.time()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(exist_ok=True)
        
        pdf_files = list(input_path.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return self.metrics

        self.metrics.total_files = len(pdf_files)
        logger.info(f"Found {len(pdf_files)} PDFs to process")
        
        # Process files with parallel execution
        if self.config.max_workers > 1:
            self._process_parallel(pdf_files, input_path, output_path)
        else:
            self._process_sequential(pdf_files, input_path, output_path)
        
        # Save final metrics and cache
        self.metrics.processing_time = time.time() - start_time
        self._save_cache()
        self._save_metrics(output_path)
        
        logger.info(f"Processing complete! Success rate: {self.metrics.success_rate():.1%}")
        return self.metrics

    def _process_parallel(self, pdf_files: List[Path], input_path: Path, output_path: Path) -> None:
        """Process files in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_file, pdf_file, input_path, output_path): pdf_file
                for pdf_file in pdf_files
            }
            
            with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
                for future in as_completed(futures):
                    pdf_file = futures[future]
                    try:
                        future.result()
                        self.metrics.successful_files += 1
                    except Exception as e:
                        self.metrics.failed_files += 1
                        logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                    pbar.update(1)

    def _process_sequential(self, pdf_files: List[Path], input_path: Path, output_path: Path) -> None:
        """Process files sequentially with progress bar"""
        with tqdm(pdf_files, desc="Processing PDFs") as pbar:
            for pdf_file in pbar:
                try:
                    self._process_single_file(pdf_file, input_path, output_path)
                    self.metrics.successful_files += 1
                except Exception as e:
                    self.metrics.failed_files += 1
                    logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                pbar.set_postfix({"Success": f"{self.metrics.success_rate():.1%}"})

    def _process_single_file(self, pdf_file: Path, input_path: Path, output_path: Path) -> None:
        """Process a single PDF file with caching"""
        relative_path = pdf_file.relative_to(input_path)
        output_subdir = output_path / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        file_hash = calculate_file_hash(pdf_file) if self.config.enable_caching else None
        if file_hash and file_hash in self._cache:
            logger.debug(f"Using cached result for {relative_path}")
            self.metrics.cached_files += 1
            return
        
        # Process file
        docs = self._process_pdf(pdf_file)
        
        if docs:
            # Save results in specified format
            if self.config.output_format == "json":
                output_file = output_subdir / f"{pdf_file.stem}.json"
                self._save_results_json(docs, output_file)
            elif self.config.output_format == "markdown":
                output_file = output_subdir / f"{pdf_file.stem}.md"
                self._save_results_markdown(docs, output_file)
            else:
                output_file = output_subdir / f"{pdf_file.stem}.txt"
                self._save_results(docs, output_file)
            
            # Update cache
            if file_hash:
                self._cache[file_hash] = {
                    "file": str(relative_path),
                    "processed_at": time.time(),
                    "pages": len(docs)
                }
            
            self.metrics.total_pages += len(docs)
            logger.info(f"Processed: {relative_path} ({len(docs)} pages)")
        else:
            raise Exception("No content extracted")

    @retry_operation(max_retries=3, delay=1.0)
    def _process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF with enhanced fallback strategies"""
        try:
            # First try native text extraction
            loader = UnstructuredFileLoader(
                str(pdf_path),
                strategy="hi_res",
                post_processors=["ocr"] if self.config.enable_retry else [],
            )
            docs = loader.load()
            
            if not self._is_quality_acceptable(docs):
                logger.debug(f"Trying PyPDFium2 for better extraction")
                loader = PyPDFium2Loader(str(pdf_path))
                docs = loader.load()
                
                if self._needs_ocr(docs):
                    logger.debug(f"Falling back to OCR")
                    docs = self._run_ocr(pdf_path)
                    self.metrics.ocr_files += 1
            
            return docs
            
        except Exception as e:
            logger.debug(f"Standard extraction failed, using OCR: {str(e)}")
            docs = self._run_ocr(pdf_path)
            self.metrics.ocr_files += 1
            return docs

    def _run_ocr(self, pdf_path: Path) -> List[Document]:
        """Enhanced OCR processing with better configuration"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.config.ocr_dpi,
                poppler_path=self.poppler_path
            )
            
            # Configure OCR
            ocr_config = f"-l {self.config.ocr_language} --oem 3 --psm 6"
            
            docs = []
            for i, img in enumerate(images):
                try:
                    text = pytesseract.image_to_string(
                        img, 
                        config=ocr_config,
                        timeout=self.config.ocr_timeout
                    )
                    
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "page": i+1,
                            "method": "OCR",
                            "quality_score": self._calculate_text_quality(text)
                        }
                    ))
                except Exception as e:
                    logger.warning(f"OCR failed for page {i+1}: {str(e)}")
                    
            return docs
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return []

    def _calculate_text_quality(self, text: str) -> float:
        """Calculate text quality score based on multiple metrics"""
        if not text.strip():
            return 0.0
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        space_ratio = sum(c.isspace() for c in text) / len(text)
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        
        # Detect common OCR artifacts
        ocr_artifacts = ['|||', '___', '...', '???', '~~~']
        artifact_penalty = sum(text.count(artifact) for artifact in ocr_artifacts) / len(text)
        
        # Calculate quality score
        quality_score = (alpha_ratio * 0.6 + space_ratio * 0.2 + digit_ratio * 0.1) - artifact_penalty
        return max(0.0, min(1.0, quality_score))

    def _save_results(self, docs: List[Document], output_file: Path) -> None:
        """Save extracted text to file with enhanced formatting"""
        with open(output_file, "w", encoding="utf-8") as f:
            if self.config.include_metadata:
                f.write(f"=== DOCUMENT METADATA ===\n")
                f.write(f"Source: {docs[0].metadata.get('source', 'Unknown')}\n")
                f.write(f"Total Pages: {len(docs)}\n")
                f.write(f"Extraction Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*50}\n\n")
            
            for doc in docs:
                f.write(f"\n=== Page {doc.metadata.get('page', 'N/A')} ===\n")
                if self.config.include_metadata:
                    f.write(f"Method: {doc.metadata.get('method', 'Unknown')}\n")
                    if 'quality_score' in doc.metadata:
                        f.write(f"Quality Score: {doc.metadata['quality_score']:.2f}\n")
                    f.write("-" * 30 + "\n")
                f.write(doc.page_content)
                f.write("\n\n")

    def _save_results_json(self, docs: List[Document], output_file: Path) -> None:
        """Save results in JSON format"""
        data = {
            "metadata": {
                "source": docs[0].metadata.get('source', 'Unknown'),
                "total_pages": len(docs),
                "extraction_time": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "pages": [
                {
                    "page_number": doc.metadata.get('page', i+1),
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata)
                }
                for i, doc in enumerate(docs)
            ]
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_results_markdown(self, docs: List[Document], output_file: Path) -> None:
        """Save results in Markdown format"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Document: {docs[0].metadata.get('source', 'Unknown')}\n\n")
            
            if self.config.include_metadata:
                f.write("## Metadata\n")
                f.write(f"- **Total Pages:** {len(docs)}\n")
                f.write(f"- **Extraction Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for doc in docs:
                f.write(f"## Page {doc.metadata.get('page', 'N/A')}\n\n")
                f.write(doc.page_content)
                f.write("\n\n---\n\n")

    def _save_metrics(self, output_path: Path) -> None:
        """Save processing metrics"""
        metrics_file = output_path / "processing_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

    def _is_quality_acceptable(self, docs: List[Document]) -> bool:
        """Enhanced quality checking"""
        if not docs:
            return False
        
        # Check multiple pages for better assessment
        pages_to_check = min(self.config.quality_check_pages, len(docs))
        total_quality = 0
        
        for doc in docs[:pages_to_check]:
            if len(doc.page_content.strip()) < self.config.min_text_length:
                continue
            quality = self._calculate_text_quality(doc.page_content)
            total_quality += quality
        
        avg_quality = total_quality / pages_to_check if pages_to_check > 0 else 0
        return avg_quality >= self.config.min_quality_score

    def _needs_ocr(self, docs: List[Document]) -> bool:
        """Enhanced OCR decision logic"""
        if not docs:
            return True
        
        # Check first few pages
        pages_to_check = min(self.config.quality_check_pages, len(docs))
        total_length = sum(len(doc.page_content.strip()) for doc in docs[:pages_to_check])
        
        return total_length < (self.config.min_text_length * pages_to_check)

# ========================
# BACKWARD COMPATIBILITY
# ========================

class PDFExtractor(EnhancedPDFExtractor):
    """Backward compatible wrapper for original PDFExtractor"""
    def __init__(self):
        # Use default config for backward compatibility
        super().__init__(ExtractorConfig())

# ========================
# USAGE EXAMPLE
# ========================

def main():
    """Example usage of the enhanced PDF extractor"""
    # Create custom configuration
    config = ExtractorConfig(
        ocr_dpi=400,
        max_workers=2,
        enable_caching=True,
        output_format="json",
        include_metadata=True
    )
    
    # Initialize extractor
    extractor = EnhancedPDFExtractor(config)
    
    # Process directory
    try:
        metrics = extractor.process_directory("input_pdfs", "output_results")
        print(f"Processing completed with {metrics.success_rate():.1%} success rate")
        print(f"Total pages processed: {metrics.total_pages}")
        print(f"Processing time: {metrics.processing_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()