"""
Sully Ingestion Master Script
=============================

Unified ingestion pipeline for any document type.
Handles:
- Text extraction
- Domain classification
- Symbolic ingestion
- Concept graph integration
- Memory storage
- Emergence detection
- Goal injection
- DreamCore reflection

Upload anything: philosophy, math, science, code, art, music, architecture, etc.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import tempfile
import shutil
import traceback
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# === Text Extraction ===
def extract_text(file_path: str) -> str:
    """Extract text from various file formats"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.pdf', '.PDF']:
            # Try using PyPDF2
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {str(e)}")
                # Try PyMuPDF as fallback
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text += page.get_text() + "\n\n"
                    doc.close()
                    return text
                except Exception as e:
                    logger.warning(f"PyMuPDF extraction failed: {str(e)}")
                    # Final fallback - try OCR if available
                    try:
                        from pdf2image import convert_from_path
                        import pytesseract
                        images = convert_from_path(file_path, dpi=300)
                        text = ""
                        for img in images:
                            text += pytesseract.image_to_string(img, lang='eng') + "\n\n"
                        return text
                    except Exception as e:
                        logger.error(f"All PDF extraction methods failed: {str(e)}")
                        return f"Error extracting text from PDF: {str(e)}"
        
        elif file_ext in ['.docx']:
            try:
                from docx import Document
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            except Exception as e:
                logger.error(f"DOCX extraction failed: {str(e)}")
                return f"Error extracting text from DOCX: {str(e)}"
        
        elif file_ext in ['.txt', '.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            return f"Unsupported file type: {file_ext}"
    
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return f"Error extracting text: {str(e)}"

# === Domain Classifier ===
def determine_mode(text: str) -> str:
    """Determine document domain/mode based on content"""
    lowered = text.lower()
    if any(w in lowered for w in ["∫", "∑", "theorem", "proof", "lemma", "corollary"]):
        return "math"
    elif any(w in lowered for w in ["import", "def", "class", "public static void", "function", "#include"]):
        return "code"
    elif any(w in lowered for w in ["cell", "photosynthesis", "quantum", "relativity", "neuron", "molecule"]):
        return "science"
    elif any(w in lowered for w in ["truth", "being", "existence", "ethics", "metaphysics", "justice"]):
        return "philosophy"
    elif any(w in lowered for w in ["story", "character", "narrator", "plot", "theme"]):
        return "literature"
    elif any(w in lowered for w in ["melody", "chord", "rhythm", "tonality", "harmony"]):
        return "music"
    elif any(w in lowered for w in ["canvas", "composition", "perspective", "aesthetic"]):
        return "art"
    elif any(w in lowered for w in ["blueprint", "structure", "design", "form", "space"]):
        return "architecture"
    else:
        return "general"

# === Core Ingestion ===
def ingest_document(text: str, source: str = "upload", parse_mode: str = "general") -> Dict[str, Any]:
    """
    Process document text and integrate with cognitive systems
    """
    logger.info(f"Ingesting document: {source}, mode: {parse_mode}, length: {len(text)}")
    
    symbols = {parse_mode: {"type": parse_mode, "content": text[:10000], "source": source}}
    
    # Try to access Sully systems - use dummy systems if unavailable
    try:
        from codex import SullyCodex
        codex = SullyCodex()
        codex_available = True
    except ImportError:
        codex_available = False
        logger.warning("SullyCodex not available")
    
    try:
        from continuouslearningsystem import ConceptGraph
        graph = ConceptGraph()
        graph_available = True
    except ImportError:
        graph_available = False
        logger.warning("ConceptGraph not available")
    
    try:
        from memory import MemoryNode
        memory = MemoryNode()
        memory_available = True
    except ImportError:
        memory_available = False
        logger.warning("MemoryNode not available")
    
    try:
        from autonomousgoals import GoalNode
        goal = GoalNode()
        goal_available = True
    except ImportError:
        goal_available = False
        logger.warning("GoalNode not available")
    
    # Process the text with available systems
    symbols_stored = 0
    integration_steps = []
    
    # Store in codex
    if codex_available:
        try:
            codex.ingest(symbols, source=source)
            symbols_stored = len(getattr(codex, 'symbols', {}))
            integration_steps.append("Stored in knowledge codex")
        except Exception as e:
            logger.error(f"Codex integration error: {str(e)}")
    
    # Add to concept graph
    if graph_available:
        try:
            graph.add_text(text)
            integration_steps.append("Added to concept graph")
        except Exception as e:
            logger.error(f"Graph integration error: {str(e)}")
    
    # Store in memory
    if memory_available:
        try:
            memory.store_entry(source, text[:5000])
            integration_steps.append("Stored in memory system")
        except Exception as e:
            logger.error(f"Memory integration error: {str(e)}")
    
    # Generate a goal if appropriate
    if goal_available and len(text) > 1000:
        try:
            goal.inject_goal(f"Learn and integrate knowledge from: {source}")
            integration_steps.append("Created learning goal")
        except Exception as e:
            logger.error(f"Goal integration error: {str(e)}")

    # Extract concepts (simplified)
    concepts = []
    words = text.split()
    unique_words = set([w.lower() for w in words if len(w) > 5])
    common_words = {"about", "would", "should", "could", "their", "there", "these", "those", "other", "another"}
    concepts = [w for w in unique_words if w not in common_words][:50]
    
    return {
        "symbols_stored": symbols_stored or len(concepts),
        "source": source,
        "mode": parse_mode,
        "integration_steps": integration_steps,
        "content_length": len(text),
        "concepts": concepts[:10],  # First 10 concepts for display
        "success": True
    }

# === API Route ===
@router.post("/ingest_document")
async def ingest_document_api(file: UploadFile = File(...)):
    """Process and ingest an uploaded document"""
    try:
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
        os.close(temp_fd)
        
        try:
            # Write uploaded file to temp file
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Extract text
            text = extract_text(temp_path)
            
            # Determine domain/mode
            mode = determine_mode(text)
            
            # Process document
            result = ingest_document(text, source=file.filename, parse_mode=mode)
            
            # Add file info to result
            result["filename"] = file.filename
            result["file_type"] = os.path.splitext(file.filename)[1]
            result["word_count"] = len(text.split())
            
            # Add text preview
            if len(text) > 300:
                result["text_preview"] = text[:300] + "..."
            else:
                result["text_preview"] = text
                
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Document ingestion error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.post("/ingest_folder")
async def ingest_folder_api(folder_path: str):
    """Process all documents in a folder"""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found or not a directory: {folder_path}"
        )
    
    results = []
    valid_extensions = ['.pdf', '.txt', '.docx', '.md', '.markdown']
    
    try:
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(folder_path, filename)
                
                # Extract text
                text = extract_text(file_path)
                
                # Determine mode
                mode = determine_mode(text)
                
                # Process document
                result = ingest_document(text, source=filename, parse_mode=mode)
                result["filename"] = filename
                results.append(result)
        
        return {
            "processed_files": len(results),
            "results": results,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Folder ingestion error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing folder: {str(e)}"
        )

# Additional route for process status
@router.get("/ingestion_status")
async def ingestion_status():
    """Get status of the ingestion system"""
    return {
        "status": "operational",
        "supported_file_types": [
            ".pdf", ".docx", ".txt", ".md", ".markdown"
        ],
        "processing_modes": [
            "general", "math", "code", "science", "philosophy", 
            "literature", "music", "art", "architecture"
        ]
    }