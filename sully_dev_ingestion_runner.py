"""
Sully Master Cognition Script
=============================

This is the full symbolic ingestion and synthesis pipeline.
It handles ingestion, semantic linking, symbolic fusion, memory, goal formation, persona shaping, dreaming, neural self-modification, and reasoning.

Use it as a unified dev runner **OR** as a backend script powering `/upload/ingest`.
"""

import re
from PyPDF2 import PdfReader

# --- Core Modules ---
class SullyCodex:
    def __init__(self):
        self.symbols = {}

    def ingest(self, symbol_dict, source=None):
        for key, data in symbol_dict.items():
            self.symbols[key.lower()] = data

    def lookup(self, name):
        return self.symbols.get(name.lower(), None)

    def store_emergent_idea(self, idea, context="unknown"):
        self.symbols[f"emergent_{len(self.symbols)}"] = {
            "type": "emergent",
            "summary": idea,
            "context": context
        }

class ConceptGraph:
    def __init__(self):
        self.data = []
    def add_text(self, text):
        self.data.append(text)

class FusionEngine:
    def synthesize(self, items):
        return f"Fusion between {items[0]} and {items[1][:40]}..." if len(items) == 2 else None

class EmergenceFramework:
    def is_emergent(self, output):
        return "Fusion" in output

class MemoryNode:
    def __init__(self):
        self.logs = {}
    def store_entry(self, key, text):
        self.logs[key] = text

class GoalNode:
    def inject_goal(self, text):
        print(f"[GOAL] {text}")

class PersonaManager:
    def activate_context(self, name, dominant=False):
        print(f"[Persona] {name} mode activated.")

class DreamCore:
    def reflect(self, text):
        print(f"[Dream] Interpreting: {text[:100]}...")

class NeuralModification:
    def propose_update(self, idea):
        print(f"[Neural Update Proposal] Based on: {idea[:60]}...")

class SymbolicReasoningNode:
    def __init__(self, codex, translator, memory=None):
        self.codex = codex
        self.translator = translator
        self.memory = memory

    def reason(self, prompt, tone="analytical"):
        if "who is" in prompt.lower():
            name = prompt.split("is")[-1].strip().title()
            info = self.codex.lookup(name)
            if info:
                return f"{name} is a {info.get('title', 'symbol')} known for {info.get('known_for', ['unknown'])[0]}."
        return self.translator(prompt)

# --- Extraction Functions ---
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def extract_oxford_entries(text, max_entries=None, offset=0):
    pattern = re.compile(r"\n([A-Za-z\-']{2,})\n(.*?)(?=\n[A-Za-z\-']{2,}\n|\Z)", re.DOTALL)
    entries = pattern.findall(text)
    structured = {}
    for i, (word, block) in enumerate(entries):
        if max_entries and (i < offset or i >= offset + max_entries):
            continue
        # Fix: Properly handle digit patterns in regex
        definitions = [d.strip() for d in re.split(r"\d\)", block) if d.strip()]
        structured[word.lower()] = {
            "type": "lexical",
            "definitions": definitions[:3],
            "source": "Oxford Dictionary"
        }
    return structured

def extract_bible_chapters(text):
    # Fix: Properly escape backslashes in regex pattern
    pattern = re.compile(r"(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|1 Samuel|2 Samuel|"
                         r"1 Kings|2 Kings|1 Chronicles|2 Chronicles|Ezra|Nehemiah|Esther|Job|Psalms|Proverbs|"
                         r"Ecclesiastes|Song Of Songs|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|"
                         r"Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Mark|Luke|John|"
                         r"Acts|Romans|1 Corinthians|2 Corinthians|Galatians|Ephesians|Philippians|Colossians|"
                         r"1 Thessalonians|2 Thessalonians|1 Timothy|2 Timothy|Titus|Philemon|Hebrews|James|"
                         r"1 Peter|2 Peter|1 John|2 John|3 John|Jude|Revelation) \d+", re.IGNORECASE)
    parts = re.split(pattern, text)
    chapters = {}
    
    # Fix: Handle edge case where the pattern doesn't match anything
    if len(parts) <= 1:
        return chapters
        
    for i in range(1, len(parts), 2):
        # Fix: Handle index out of range error
        if i+1 >= len(parts):
            break
            
        label = parts[i].strip().title()
        content = parts[i+1].strip()
        if label:
            chapters[label] = {
                "type": "scripture",
                "content": content[:3000],
                "source": "Bible PDF"
            }
    return chapters

# --- Unified Ingestion ---
def ingest_and_synthesize(text, codex, graph, fusion, emergence, memory, goal, persona, dream, modify, context):
    graph.add_text(text)
    memory.store_entry(context, text[:1000])
    persona.activate_context("mythic")
    dream.reflect(text)

    for symbol in list(codex.symbols.keys()):
        result = fusion.synthesize([symbol, text])
        if result and emergence.is_emergent(result):
            codex.store_emergent_idea(result, context)
            modify.propose_update(result)
            goal.inject_goal(f"Pursue insight from: {result[:50]}")

# --- CLI Runner ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sully_master_cognition.py <file.pdf> <bible|oxford>")
        sys.exit(1)

    path, mode = sys.argv[1], sys.argv[2].lower()

    codex = SullyCodex()
    graph = ConceptGraph()
    fusion = FusionEngine()
    emergence = EmergenceFramework()
    memory = MemoryNode()
    goal = GoalNode()
    persona = PersonaManager()
    dream = DreamCore()
    modify = NeuralModification()
    translate = lambda text: f"⟶Symbolic({text})"
    reasoner = SymbolicReasoningNode(codex, translate, memory)

    try:
        text = load_pdf_text(path)
        data = extract_bible_chapters(text) if mode == "bible" else extract_oxford_entries(text)
        
        codex.ingest(data, source=path)
        ingest_and_synthesize(text, codex, graph, fusion, emergence, memory, goal, persona, dream, modify, context=mode)
        
        # Fix: Handle case when "Moses" isn't found
        result = reasoner.reason("Who is Moses?")
        print(result)
        print(f"✅ Complete: {len(codex.symbols)} symbols stored")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")