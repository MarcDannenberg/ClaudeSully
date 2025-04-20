"""
Symbolic Ingestion Module (API Bindable)
========================================

This module wraps symbolic ingestion from DEV4 into reusable functions.
Intended to be imported by FastAPI routes (e.g., /ingest_document).
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
import re

# Optional PDF/DOCX support
from docx import Document
from PyPDF2 import PdfReader

def extract_text(file_path):
    if file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    return ""

def extract_bible_chapters(text):
    pattern = re.compile(r"(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|1\\sSamuel|2\\sSamuel|"
                         r"1\\sKings|2\\sKings|1\\sChronicles|2\\sChronicles|Ezra|Nehemiah|Esther|Job|Psalms|Proverbs|"
                         r"Ecclesiastes|Song\\sOf\\sSongs|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|"
                         r"Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Mark|Luke|John|"
                         r"Acts|Romans|1\\sCorinthians|2\\sCorinthians|Galatians|Ephesians|Philippians|Colossians|"
                         r"1\\sThessalonians|2\\sThessalonians|1\\sTimothy|2\\sTimothy|Titus|Philemon|Hebrews|James|"
                         r"1\\sPeter|2\\sPeter|1\\sJohn|2\\sJohn|3\\sJohn|Jude|Revelation)\\s+\\d+", re.IGNORECASE)
    parts = re.split(pattern, text)
    chapters = {}
    for i in range(1, len(parts), 2):
        label = parts[i].strip().title()
        content = parts[i+1].strip()
        if label:
            chapters[label] = {
                "type": "scripture",
                "content": content[:3000],
                "source": "upload"
            }
    return chapters

def ingest_document(text, source="upload", mode="bible"):
    codex = SullyCodex()
    graph = ConceptGraph()
    fusion = SymbolFusionEngine()
    emergence = EmergenceFramework()
    memory = MemoryNode()
    goal = GoalNode()
    persona = PersonaManager()
    dream = DreamCore()
    modify = NeuralModification()

    # Ingest symbols
    if mode == "bible":
        symbols = extract_bible_chapters(text)
        if not symbols:
            symbols = {"raw_text": {"type": "text", "content": text[:3000], "source": source}}
    else:
        symbols = {"raw_text": {"type": "text", "content": text[:3000], "source": source}}

    codex.ingest(symbols, source=source)
    graph.add_text(text)
    memory.store_entry(source, text[:1000])
    persona.activate_context("mythic")
    dream.reflect(text)

    for symbol in list(codex.symbols.keys()):
        result = fusion.synthesize([symbol, text])
        if result and emergence.is_emergent(result):
            codex.store_emergent_idea(result, context=source)
            modify.propose_update(result)
            goal.inject_goal(f"Pursue insight from: {result[:60]}")

    return {"symbols_stored": len(codex.symbols), "source": source}