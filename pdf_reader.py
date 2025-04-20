"""
Advanced PDF text extraction with OCR capabilities and content structuring.
Supports multiple extraction strategies and content organization.
"""
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import PyPDF2
import fitz  # PyMuPDF
import logging
import io
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFReader:
    """
    Enhanced PDF text extraction with multiple strategies and content structuring.
    Implements fallback mechanisms and content organization.
    """
    def __init__(self, ocr_enabled: bool = True, dpi: int = 300, language: str = 'eng'):
        """
        Initialize the PDF reader with configurable options.
        
        Args:
            ocr_enabled: Whether to use OCR for text extraction
            dpi: Resolution for PDF-to-image conversion when using OCR
            language: OCR language for pytesseract
        """
        self.ocr_enabled = ocr_enabled
        self.dpi = dpi
        self.language = language
        self.last_error = None
        
    def extract_text(self, pdf_path: str, verbose: bool = True, 
                    use_ocr_fallback: bool = True, 
                    extract_structure: bool = True,
                    extract_metadata: bool = True) -> Dict[str, Any]:
        """
        Extract text from a PDF using multiple strategies with fallback.
        
        Args:
            pdf_path: Path to the PDF file
            verbose: Whether to print progress information
            use_ocr_fallback: Whether to use OCR as a fallback if native extraction fails
            extract_structure: Whether to attempt extracting document structure
            extract_metadata: Whether to extract PDF metadata
            
        Returns:
            Dictionary containing extracted text, metadata, and structure
        """
        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
            
        result = {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "success": False,
            "extraction_method": None,
            "page_count": 0,
            "text": "",
            "pages": []
        }
        
        # Add metadata if requested
        if extract_metadata:
            result["metadata"] = self._extract_metadata(pdf_path)
        
        # Try PyMuPDF first (usually best quality)
        try:
            if verbose:
                logger.info(f"Attempting PyMuPDF extraction for {pdf_path}")
                
            text_by_page, page_count, structure = self._extract_with_pymupdf(pdf_path, extract_structure)
            
            if text_by_page and all(len(page.strip()) > 0 for page in text_by_page):
                result["success"] = True
                result["extraction_method"] = "pymupdf"
                result["page_count"] = page_count
                result["text"] = "\n\n".join(text_by_page)
                result["pages"] = [{"number": i+1, "text": text} for i, text in enumerate(text_by_page)]
                
                if extract_structure and structure:
                    result["structure"] = structure
                    
                return result
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Try PyPDF2 as second option
        try:
            if verbose:
                logger.info(f"Attempting PyPDF2 extraction for {pdf_path}")
                
            text_by_page, page_count = self._extract_with_pypdf2(pdf_path)
            
            if text_by_page and all(len(page.strip()) > 0 for page in text_by_page):
                result["success"] = True
                result["extraction_method"] = "pypdf2"
                result["page_count"] = page_count
                result["text"] = "\n\n".join(text_by_page)
                result["pages"] = [{"number": i+1, "text": text} for i, text in enumerate(text_by_page)]
                return result
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Fall back to OCR if enabled and other methods failed
        if self.ocr_enabled and use_ocr_fallback:
            try:
                if verbose:
                    logger.info(f"Attempting OCR extraction for {pdf_path}")
                    
                text_by_page, page_count = self._extract_with_ocr(pdf_path, verbose)
                
                if text_by_page:
                    result["success"] = True
                    result["extraction_method"] = "ocr"
                    result["page_count"] = page_count
                    result["text"] = "\n\n".join(text_by_page)
                    result["pages"] = [{"number": i+1, "text": text} for i, text in enumerate(text_by_page)]
                    return result
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"OCR extraction failed: {e}")
        
        # If we get here, all extraction methods failed
        result["error"] = f"Text extraction failed with all methods. Last error: {self.last_error}"
        return result
    
    def _extract_with_pymupdf(self, pdf_path: str, extract_structure: bool = True) -> Tuple[List[str], int, Optional[Dict[str, Any]]]:
        """
        Extract text using PyMuPDF (fitz).
        
        Args:
            pdf_path: Path to the PDF file
            extract_structure: Whether to attempt extracting document structure
            
        Returns:
            Tuple of (list of page texts, page count, optional structure dict)
        """
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        text_by_page = []
        structure = None
        
        # Extract text from each page
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_by_page.append(text)
        
        # Extract document structure if requested
        if extract_structure:
            structure = self._extract_document_structure(doc)
            
        doc.close()
        return text_by_page, page_count, structure
        
    def _extract_with_pypdf2(self, pdf_path: str) -> Tuple[List[str], int]:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (list of page texts, page count)
        """
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            page_count = len(reader.pages)
            text_by_page = []
            
            for page_num in range(page_count):
                page = reader.pages[page_num]
                text = page.extract_text()
                text_by_page.append(text)
                
        return text_by_page, page_count
        
    def _extract_with_ocr(self, pdf_path: str, verbose: bool = True) -> Tuple[List[str], int]:
        """
        Extract text using OCR via pdf2image and pytesseract.
        
        Args:
            pdf_path: Path to the PDF file
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (list of page texts, page count)
        """
        images = convert_from_path(pdf_path, dpi=self.dpi)
        page_count = len(images)
        text_by_page = []
        
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang=self.language)
            if verbose:
                char_count = len(text)
                logger.info(f"[OCR] Page {i+1}/{page_count}: {char_count} characters")
            text_by_page.append(text.strip())
        
        return text_by_page, page_count
        
    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract PDF metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary of metadata fields
        """
        metadata = {}
        
        # Try PyMuPDF first
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
        except Exception:
            # Fall back to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    if reader.metadata:
                        for key, value in reader.metadata.items():
                            # Remove the leading slash from keys
                            clean_key = key[1:] if key.startswith('/') else key
                            metadata[clean_key] = value
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
        
    def _extract_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract document structure including TOC and potential sections.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            "toc": [],
            "sections": []
        }
        
        # Extract table of contents
        toc = doc.get_toc()
        if toc:
            structure["toc"] = toc
            
            # Convert TOC to sections
            current_section = None
            for item in toc:
                level, title, page = item
                if level == 1:
                    current_section = {
                        "title": title,
                        "start_page": page,
                        "subsections": []
                    }
                    structure["sections"].append(current_section)
                elif level == 2 and current_section:
                    current_section["subsections"].append({
                        "title": title,
                        "page": page
                    })
        
        return structure
    
    def extract_images(self, pdf_path: str, output_dir: Optional[str] = None,
                      min_size: int = 100) -> List[Dict[str, Any]]:
        """
        Extract images from the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            min_size: Minimum width or height for extracted images
            
        Returns:
            List of dictionaries with image information
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
            
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        images_info = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                
                for image_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Try to load as an image to get dimensions
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        width, height = image.size
                    except Exception:
                        width, height = 0, 0
                    
                    # Skip small images
                    if width < min_size or height < min_size:
                        continue
                    
                    image_info = {
                        "page": page_index + 1,
                        "index": image_index,
                        "width": width,
                        "height": height,
                        "format": image_ext
                    }
                    
                    # Save image if output directory provided
                    if output_dir:
                        image_filename = f"page{page_index+1}_img{image_index}.{image_ext}"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                            
                        image_info["path"] = image_path
                        
                    images_info.append(image_info)
                    
            doc.close()
        except Exception as e:
            logger.error(f"Failed to extract images: {e}")
            
        return images_info
        
    def extract_with_pattern(self, pdf_path: str, pattern: str, 
                            flags: int = re.IGNORECASE) -> List[Dict[str, Any]]:
        """
        Extract text matching a specific pattern from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            pattern: Regex pattern to match
            flags: Regex flags
            
        Returns:
            List of dictionaries with matched content
        """
        # First get all the text
        extraction_result = self.extract_text(pdf_path, verbose=False)
        
        if not extraction_result["success"]:
            return []
            
        matches = []
        
        # Compile the pattern
        regex = re.compile(pattern, flags)
        
        # Search page by page
        for page_info in extraction_result["pages"]:
            page_num = page_info["number"]
            text = page_info["text"]
            
            for match in regex.finditer(text):
                matches.append({
                    "page": page_num,
                    "match": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "groups": match.groups() if match.groups() else None
                })
                
        return matches

def extract_text_from_pdf(pdf_path: str, dpi: int = 200, verbose: bool = True) -> str:
    """
    Extracts OCR'd text from all pages of a PDF file.
    Legacy function for backward compatibility.
    
    Args:
        pdf_path (str): Full path to the PDF file.
        dpi (int): DPI used to render PDF pages as images.
        verbose (bool): Whether to print per-page progress.

    Returns:
        str: Concatenated OCR text from all pages.
    """
    try:
        reader = PDFReader(ocr_enabled=True, dpi=dpi)
        result = reader.extract_text(pdf_path, verbose=verbose, use_ocr_fallback=True)
        
        if result["success"]:
            return result["text"]
        else:
            return f"[OCR ERROR] {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"[OCR ERROR] {str(e)}"

def extract_kernel_from_text(text: str, domain: str = "general") -> Dict[str, Any]:
    """
    Extracts symbolic elements from raw PDF text to form a symbolic kernel.
    
    Args:
        text: Raw text from PDF
        domain: Target domain (law, finance, etc.)
    
    Returns:
        Dict with 'symbols', 'paradoxes', and 'frames'
    """
    symbols = []
    paradoxes = []
    frames = []

    lines = text.split("\n")
    for line in lines:
        l = line.lower()
        if any(term in l for term in ["vs.", "versus", "tension between"]):
            paradoxes.append(line.strip())
        elif any(term in l for term in ["defined as", "refers to", "is known as", "means that"]):
            symbols.append(line.strip())
        elif any(term in l for term in ["principle", "framework", "approach", "model"]):
            frames.append(line.strip())

    return {
        "domain": domain,
        "symbols": list(set(symbols)),
        "paradoxes": list(set(paradoxes)),
        "frames": list(set(frames))
    }

if __name__ == "__main__":
    # Example usage if run as a script
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract text from PDF")
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("--method", choices=["auto", "pymupdf", "pypdf2", "ocr"], 
                        default="auto", help="Extraction method")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for OCR")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    reader = PDFReader(ocr_enabled=True, dpi=args.dpi)
    
    # Choose extraction method
    if args.method == "pymupdf":
        try:
            text_by_page, page_count, _ = reader._extract_with_pymupdf(args.pdf_file)
            text = "\n\n".join(text_by_page)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.method == "pypdf2":
        try:
            text_by_page, page_count = reader._extract_with_pypdf2(args.pdf_file)
            text = "\n\n".join(text_by_page)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.method == "ocr":
        try:
            text_by_page, page_count = reader._extract_with_ocr(args.pdf_file, args.verbose)
            text = "\n\n".join(text_by_page)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:  # auto
        result = reader.extract_text(args.pdf_file, verbose=args.verbose)
        if result["success"]:
            text = result["text"]
            if args.verbose:
                print(f"Extraction method: {result['extraction_method']}")
                print(f"Page count: {result['page_count']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
    
    # Output the text
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        print(text)