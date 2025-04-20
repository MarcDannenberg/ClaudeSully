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

from codex import SullyCodex
from continuouslearningsystem import ConceptGraph
from emergenceframework import EmergenceFramework
from memory import MemoryNode
from autonomousgoals import GoalNode
from fusion import SymbolFusionEngine
from persona import PersonaManager
from dream import DreamCore
from neuralmodification import NeuralModification
from reasoning import SymbolicReasoningNode
from docx import Document
from PyPDF2 import PdfReader
from fastapi import APIRouter, UploadFile
import os

router = APIRouter()

# === Text Extraction ===
def extract_text(file_path):
    if file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    return ""

# === Domain Classifier ===
def determine_mode(text: str) -> str:
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
        return "raw"

# === Core Ingestion ===
def ingest_document(text, source="upload", parse_mode="raw"):
    codex = SullyCodex()
    graph = ConceptGraph()
    fusion = SymbolFusionEngine()
    emergence = EmergenceFramework()
    memory = MemoryNode()
    goal = GoalNode()
    persona = PersonaManager()
    dream = DreamCore()
    modify = NeuralModification()

    symbols = {parse_mode: {"type": parse_mode, "content": text[:3000], "source": source}}

    codex.ingest(symbols, source=source)
    graph.add_text(text)
    memory.store_entry(source, text[:1000])
    persona.activate_context("reflective")
    dream.reflect(text)

    for symbol in list(codex.symbols.keys()):
        result = fusion.synthesize([symbol, text])
        if result and emergence.is_emergent(result):
            codex.store_emergent_idea(result, context=source)
            modify.propose_update(result)
            goal.inject_goal(f"Pursue insight from: {result[:60]}")

    return {"symbols_stored": len(codex.symbols), "source": source, "mode": parse_mode}

# === API Route ===
@router.post("/ingest_document")
def ingest_document_api(file: UploadFile):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())

    text = extract_text(temp_path)
    mode = determine_mode(text)
    result = ingest_document(text, source=file.filename, parse_mode=mode)
    os.remove(temp_path)
    return result
