from fastapi import FastAPI, Request, HTTPException, Body, UploadFile, File, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import os
import sys
import json
import uuid
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sully_integrated.log'
)
logger = logging.getLogger(__name__)

user_contexts = {}
user_consents = {}

# Add path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core Sully components with proper error handling
try:
    from reasoning import SymbolicReasoningNode
    reasoning_available = True
    logger.info("Successfully imported SymbolicReasoningNode")
except ImportError as e:
    reasoning_available = False
    logger.warning(f"Failed to import SymbolicReasoningNode: {str(e)}")

try:
    from memory import MemorySystem
    memory_available = True
    logger.info("Successfully imported MemorySystem")
except ImportError as e:
    memory_available = False
    logger.warning(f"Failed to import MemorySystem: {str(e)}")

try:
    from codex import Codex
    codex_available = True
    logger.info("Successfully imported Codex")
except ImportError as e:
    codex_available = False
    logger.warning(f"Failed to import Codex: {str(e)}")

try:
    from dream import DreamGenerator
    dream_available = True
    logger.info("Successfully imported DreamGenerator")
except ImportError as e:
    dream_available = False
    logger.warning(f"Failed to import DreamGenerator: {str(e)}")

try:
    from fusion import SymbolFusionEngine
    fusion_available = True
    logger.info("Successfully imported SymbolFusionEngine")
except ImportError as e:
    fusion_available = False
    logger.warning(f"Failed to import SymbolFusionEngine: {str(e)}")

try:
    from paradox import ParadoxExplorer
    paradox_available = True
    logger.info("Successfully imported ParadoxExplorer")
except ImportError as e:
    paradox_available = False
    logger.warning(f"Failed to import ParadoxExplorer: {str(e)}")

try:
    from math_translator import SymbolicMathTranslator
    math_translator_available = True
    logger.info("Successfully imported SymbolicMathTranslator")
except ImportError as e:
    math_translator_available = False
    logger.warning(f"Failed to import SymbolicMathTranslator: {str(e)}")

try:
    from identity import IdentitySystem
    identity_available = True
    logger.info("Successfully imported IdentitySystem")
except ImportError as e:
    identity_available = False
    logger.warning(f"Failed to import IdentitySystem: {str(e)}")

try:
    from virtue import VirtueEthicsEngine
    virtue_available = True
    logger.info("Successfully imported VirtueEthicsEngine")
except ImportError as e:
    virtue_available = False
    logger.warning(f"Failed to import VirtueEthicsEngine: {str(e)}")

try:
    from intuition import IntuitionEngine
    intuition_available = True
    logger.info("Successfully imported IntuitionEngine")
except ImportError as e:
    intuition_available = False
    logger.warning(f"Failed to import IntuitionEngine: {str(e)}")

try:
    from kernel_integration import KernelIntegrationSystem
    kernel_integration_available = True
    logger.info("Successfully imported KernelIntegrationSystem")
except ImportError as e:
    kernel_integration_available = False
    logger.warning(f"Failed to import KernelIntegrationSystem: {str(e)}")

try:
    from logic_kernel import LogicKernel
    logic_kernel_available = True
    logger.info("Successfully imported LogicKernel")
except ImportError as e:
    logic_kernel_available = False
    logger.warning(f"Failed to import LogicKernel: {str(e)}")

try:
    from pdf_reader import PDFProcessor
    pdf_reader_available = True
    logger.info("Successfully imported PDFProcessor")
except ImportError as e:
    pdf_reader_available = False
    logger.warning(f"Failed to import PDFProcessor: {str(e)}")

try:
    from webreader import WebReader
    webreader_available = True
    logger.info("Successfully imported WebReader")
except ImportError as e:
    webreader_available = False
    logger.warning(f"Failed to import WebReader: {str(e)}")

# Import games router placeholder
def include_games_router(app):
    logger.info("Using stub games router")
    pass

# Dummy classes for missing components
class DummyCodex:
    def __init__(self):
        logger.info("Initializing dummy Codex")
        
    def search(self, query, semantic=False, limit=5):
        logger.info(f"Dummy Codex search: {query[:50]}...")
        return {query: {"definition": f"Definition of {query} would go here", "related": []}}

class DummyMemory:
    def __init__(self):
        logger.info("Initializing dummy Memory")
        self.storage = []
        
    def search(self, query, include_associations=False, limit=5):
        logger.info(f"Dummy Memory search: {query[:50]}...")
        return []
    
    def store_query(self, query, response):
        logger.info(f"Dummy Memory storing: {query[:50]}...")
        self.storage.append({"query": query, "response": response, "timestamp": datetime.now().isoformat()})
        return True

class DummyTranslator:
    def __init__(self):
        logger.info("Initializing dummy MathTranslator")
        
    def translate(self, phrase, style="formal", domain=None):
        logger.info(f"Dummy translation of: {phrase[:50]}...")
        return f"∑({phrase})"

# Enhanced Sully class to integrate all available modules
class Sully:
    """
    Integrated Sully cognition system that connects to real module implementations when available.
    """
    def __init__(self):
        logger.info("Initializing integrated Sully class")
        
        # Initialize Codex (knowledge component)
        if codex_available:
            try:
                self.codex = Codex()
                logger.info("Initialized real Codex")
            except Exception as e:
                logger.error(f"Failed to initialize real Codex: {str(e)}")
                self.codex = DummyCodex()
        else:
            self.codex = DummyCodex()
            
        # Initialize Memory System
        if memory_available:
            try:
                self.memory = MemorySystem()
                logger.info("Initialized real MemorySystem")
            except Exception as e:
                logger.error(f"Failed to initialize real MemorySystem: {str(e)}")
                self.memory = DummyMemory()
        else:
            self.memory = DummyMemory()
        
        # Initialize Math Translator
        if math_translator_available:
            try:
                self.translator = SymbolicMathTranslator()
                logger.info("Initialized real SymbolicMathTranslator")
            except Exception as e:
                logger.error(f"Failed to initialize real SymbolicMathTranslator: {str(e)}")
                self.translator = DummyTranslator()
        else:
            self.translator = DummyTranslator()
            
        # Initialize Reasoning Node (core reasoning component)
        if reasoning_available:
            try:
                self.reasoning_node = SymbolicReasoningNode(self.codex, self.translator, self.memory)
                logger.info("Initialized real SymbolicReasoningNode")
            except Exception as e:
                logger.error(f"Failed to initialize real SymbolicReasoningNode: {str(e)}")
                self.reasoning_node = self  # Fallback to self
        else:
            self.reasoning_node = self  # Fallback to self
        
        # Initialize Dream Generator
        if dream_available:
            try:
                self.dream_generator = DreamGenerator()
                logger.info("Initialized real DreamGenerator")
            except Exception as e:
                logger.error(f"Failed to initialize real DreamGenerator: {str(e)}")
                self.dream_generator = None
        else:
            self.dream_generator = None
            
        # Initialize Fusion Engine
        if fusion_available:
            try:
                self.fusion_engine = SymbolFusionEngine()
                logger.info("Initialized real SymbolFusionEngine")
            except Exception as e:
                logger.error(f"Failed to initialize real SymbolFusionEngine: {str(e)}")
                self.fusion_engine = None
        else:
            self.fusion_engine = None
            
        # Initialize Paradox Explorer
        if paradox_available:
            try:
                self.paradox_explorer = ParadoxExplorer()
                logger.info("Initialized real ParadoxExplorer")
            except Exception as e:
                logger.error(f"Failed to initialize real ParadoxExplorer: {str(e)}")
                self.paradox_explorer = None
        else:
            self.paradox_explorer = None
            
        # Initialize Identity System
        if identity_available:
            try:
                self.identity_system = IdentitySystem()
                logger.info("Initialized real IdentitySystem")
            except Exception as e:
                logger.error(f"Failed to initialize real IdentitySystem: {str(e)}")
                self.identity_system = None
        else:
            self.identity_system = None
            
        # Initialize Virtue Ethics Engine
        if virtue_available:
            try:
                self.virtue_engine = VirtueEthicsEngine()
                logger.info("Initialized real VirtueEthicsEngine")
            except Exception as e:
                logger.error(f"Failed to initialize real VirtueEthicsEngine: {str(e)}")
                self.virtue_engine = None
        else:
            self.virtue_engine = None
            
        # Initialize Intuition Engine
        if intuition_available:
            try:
                self.intuition_engine = IntuitionEngine()
                logger.info("Initialized real IntuitionEngine")
            except Exception as e:
                logger.error(f"Failed to initialize real IntuitionEngine: {str(e)}")
                self.intuition_engine = None
        else:
            self.intuition_engine = None
            
        # Initialize Kernel Integration System
        if kernel_integration_available:
            try:
                self.kernel_integration = KernelIntegrationSystem()
                logger.info("Initialized real KernelIntegrationSystem")
            except Exception as e:
                logger.error(f"Failed to initialize real KernelIntegrationSystem: {str(e)}")
                self.kernel_integration = None
        else:
            self.kernel_integration = None
            
        # Initialize Logic Kernel
        if logic_kernel_available:
            try:
                self.logic_kernel = LogicKernel()
                logger.info("Initialized real LogicKernel")
            except Exception as e:
                logger.error(f"Failed to initialize real LogicKernel: {str(e)}")
                self.logic_kernel = None
        else:
            self.logic_kernel = None
            
        # Initialize PDF Processor
        if pdf_reader_available:
            try:
                self.pdf_processor = PDFProcessor()
                logger.info("Initialized real PDFProcessor")
            except Exception as e:
                logger.error(f"Failed to initialize real PDFProcessor: {str(e)}")
                self.pdf_processor = None
        else:
            self.pdf_processor = None
            
        # Initialize Web Reader
        if webreader_available:
            try:
                self.web_reader = WebReader()
                logger.info("Initialized real WebReader")
            except Exception as e:
                logger.error(f"Failed to initialize real WebReader: {str(e)}")
                self.web_reader = None
        else:
            self.web_reader = None

    def reason(self, message, tone):
        """Process input through Sully's reasoning system."""
        logger.info(f"Reasoning with message: {message[:50]}... in tone: {tone}")
        
        # Use real reasoning node if available
        if hasattr(self, "reasoning_node") and self.reasoning_node is not self:
            try:
                return self.reasoning_node.reason(message, tone)
            except Exception as e:
                logger.error(f"Error in real reasoning: {str(e)}")
                logger.info("Falling back to simple reasoning implementation")
        
        # Fallback implementation
        if tone == "analytical":
            return f"After careful analysis, I believe {message}"
        elif tone == "creative":
            return f"Creatively speaking, {message} inspires various possibilities..."
        elif tone == "critical":
            return f"Looking at this critically, {message} has several aspects to consider..."
        elif tone == "philosophical":
            return f"From a philosophical perspective, {message} raises questions about existence and meaning..."
        else:  # emergent or default
            return f"Based on my understanding, {message} suggests several insights..."
    
    def fuse(self, *concepts):
        """Fuse multiple concepts into a new emergent idea."""
        concepts_str = ", ".join(concepts)
        logger.info(f"Fusing concepts: {concepts_str}")
        
        # Use real fusion engine if available
        if hasattr(self, "fusion_engine") and self.fusion_engine:
            try:
                return self.fusion_engine.fuse_concepts(*concepts)
            except Exception as e:
                logger.error(f"Error in real fusion: {str(e)}")
                logger.info("Falling back to simple fusion implementation")
        
        # Fallback implementation
        return f"The fusion of {concepts_str} creates a new concept that embodies elements of each while transcending their individual boundaries."
    
    def dream(self, seed, depth="standard", style="recursive"):
        """Generate a dream sequence from a seed concept."""
        logger.info(f"Dreaming about: {seed} with depth: {depth} and style: {style}")
        
        # Use real dream generator if available
        if hasattr(self, "dream_generator") and self.dream_generator:
            try:
                return self.dream_generator.generate(seed, depth=depth, style=style)
            except Exception as e:
                logger.error(f"Error in real dream generation: {str(e)}")
                logger.info("Falling back to simple dream implementation")
        
        # Fallback implementation
        if depth == "shallow":
            dream_length = "brief"
        elif depth == "deep":
            dream_length = "extensive"
        elif depth == "dreamscape":
            dream_length = "immersive"
        else:  # standard
            dream_length = "moderate"
            
        if style == "associative":
            dream_style = "connects various related concepts"
        elif style == "symbolic":
            dream_style = "uses rich symbolism"
        elif style == "narrative":
            dream_style = "tells a story"
        else:  # recursive
            dream_style = "builds on itself in recursive patterns"
            
        return f"A {dream_length} dream about {seed} that {dream_style}... [dream content would be generated here]"
    
    def evaluate_claim(self, text, framework="balanced", detailed_output=True):
        """Analyze a claim through multiple cognitive perspectives."""
        logger.info(f"Evaluating claim: {text[:50]}... with framework: {framework}")
        
        # Use real reasoning node for evaluation if available
        if hasattr(self, "reasoning_node") and self.reasoning_node is not self:
            try:
                if hasattr(self.reasoning_node, "evaluate_claim"):
                    return self.reasoning_node.evaluate_claim(text, framework, detailed_output)
                elif hasattr(self.reasoning_node, "generate_multi_perspective"):
                    perspectives = self.reasoning_node.generate_multi_perspective(text, ["analytical", "critical", "ethical"])
                    return {
                        "claim": text,
                        "framework": framework,
                        "evaluation": perspectives.get("perspectives", {}),
                        "conclusion": f"Overall assessment using {framework} framework: [conclusion would be here]"
                    }
            except Exception as e:
                logger.error(f"Error in real claim evaluation: {str(e)}")
                logger.info("Falling back to simple claim evaluation implementation")
        
        # Fallback implementation
        perspectives = {
            "logical": "From a logical standpoint...",
            "ethical": "Considering ethical implications...",
            "practical": "From a practical perspective...",
            "empirical": "Based on available evidence..."
        }
        
        if detailed_output:
            return {
                "claim": text,
                "framework": framework,
                "evaluation": {k: v + " [analysis would be here]" for k, v in perspectives.items()},
                "conclusion": f"Overall assessment using {framework} framework: [conclusion would be here]"
            }
        else:
            return {
                "claim": text,
                "framework": framework,
                "conclusion": f"Assessment using {framework} framework: [conclusion would be here]"
            }
    
    def translate_math(self, phrase, style="formal", domain=None):
        """Translate between language and mathematics."""
        logger.info(f"Translating math: {phrase} with style: {style} in domain: {domain}")
        
        # Use real math translator if available
        if hasattr(self, "translator") and not isinstance(self.translator, DummyTranslator):
            try:
                return self.translator.translate(phrase, style, domain)
            except Exception as e:
                logger.error(f"Error in real math translation: {str(e)}")
                logger.info("Falling back to simple math translation implementation")
        
        # Fallback implementation
        if domain:
            domain_context = f" in the context of {domain}"
        else:
            domain_context = ""
            
        if style == "formal":
            return f"The formal mathematical representation of '{phrase}'{domain_context} is: [mathematical notation would be here]"
        elif style == "intuitive":
            return f"An intuitive mathematical interpretation of '{phrase}'{domain_context} is: [intuitive explanation would be here]"
        elif style == "applied":
            return f"Applied mathematically, '{phrase}'{domain_context} can be expressed as: [applied mathematical model would be here]"
        else:  # creative
            return f"A creative mathematical perspective on '{phrase}'{domain_context} could be: [creative interpretation would be here]"
    
    def reveal_paradox(self, topic):
        """Reveal paradoxes from different perspectives."""
        logger.info(f"Revealing paradox about: {topic}")
        
        # Use real paradox explorer if available
        if hasattr(self, "paradox_explorer") and self.paradox_explorer:
            try:
                return self.paradox_explorer.explore(topic)
            except Exception as e:
                logger.error(f"Error in real paradox exploration: {str(e)}")
                logger.info("Falling back to simple paradox implementation")
        
        # Fallback implementation
        return f"The paradox of {topic} lies in the tension between [contradictory aspects would be identified here]..."
    
    def logical_integration(self, statement, truth_value=True):
        """Assert a logical statement into the knowledge base."""
        logger.info(f"Logical integration of: {statement} with truth value: {truth_value}")
        
        # Use real logic kernel if available
        if hasattr(self, "logic_kernel") and self.logic_kernel:
            try:
                return self.logic_kernel.assert_statement(statement, truth_value)
            except Exception as e:
                logger.error(f"Error in real logical integration: {str(e)}")
                logger.info("Falling back to simple logical integration implementation")
        
        # Fallback implementation
        return {
            "statement": statement,
            "truth_value": truth_value,
            "integrated": True,
            "implications": ["Implication 1 would be here", "Implication 2 would be here"]
        }
    
    def logical_reasoning(self, query, framework="PROPOSITIONAL"):
        """Perform logical inference on a statement."""
        logger.info(f"Logical reasoning on: {query} with framework: {framework}")
        
        # Use real logic kernel if available
        if hasattr(self, "logic_kernel") and self.logic_kernel:
            try:
                return self.logic_kernel.infer(query, framework)
            except Exception as e:
                logger.error(f"Error in real logical reasoning: {str(e)}")
                logger.info("Falling back to simple logical reasoning implementation")
        
        # Fallback implementation
        return {
            "query": query,
            "framework": framework,
            "inference": "The logical inference based on existing knowledge is: [inference would be here]",
            "confidence": 0.85
        }
    
    def detect_logical_inconsistencies(self):
        """Find logical paradoxes in the knowledge base."""
        logger.info("Detecting logical inconsistencies")
        
        # Use real logic kernel if available
        if hasattr(self, "logic_kernel") and self.logic_kernel:
            try:
                return self.logic_kernel.check_consistency()
            except Exception as e:
                logger.error(f"Error in real logical consistency check: {str(e)}")
                logger.info("Falling back to simple logical consistency check implementation")
        
        # Fallback implementation
        return {
            "inconsistencies_detected": False,
            "knowledge_base_status": "consistent",
            "warning": "This is a simulated response. No actual knowledge base analysis performed."
        }
    
    def validate_argument(self, premises, conclusion):
        """Analyze the validity of a logical argument."""
        logger.info(f"Validating argument with conclusion: {conclusion}")
        
        # Use real logic kernel if available
        if hasattr(self, "logic_kernel") and self.logic_kernel:
            try:
                return self.logic_kernel.validate_argument(premises, conclusion)
            except Exception as e:
                logger.error(f"Error in real argument validation: {str(e)}")
                logger.info("Falling back to simple argument validation implementation")
        
        # Fallback implementation
        valid = True  # Simplified for demonstration
        
        if valid:
            explanation = "The argument is valid because the conclusion necessarily follows from the premises."
        else:
            explanation = "The argument is invalid because the conclusion does not necessarily follow from the premises."
            
        return {
            "premises": premises,
            "conclusion": conclusion,
            "valid": valid,
            "explanation": explanation
        }
    
    def integrated_explore(self, concept, include_kernels=None):
        """Generate integrated cross-kernel narratives."""
        logger.info(f"Integrated exploration of concept: {concept}")
        
        # Use real kernel integration system if available
        if hasattr(self, "kernel_integration") and self.kernel_integration:
            try:
                return self.kernel_integration.integrated_exploration(concept, include_kernels)
            except Exception as e:
                logger.error(f"Error in real integrated exploration: {str(e)}")
                logger.info("Falling back to simple integrated exploration implementation")
        
        # Fallback implementation
        kernels = include_kernels or ["dream", "reasoning", "fusion", "paradox"]
        kernel_str = ", ".join(kernels)
        
        return {
            "concept": concept,
            "included_kernels": kernels,
            "narrative": f"An integrated exploration of {concept} across {kernel_str}: [narrative would be here]"
        }
    
    def concept_network(self, concept, depth=2):
        """Create multi-modal concept networks."""
        logger.info(f"Creating concept network for: {concept} with depth: {depth}")
        
        # Use real kernel integration system if available
        if hasattr(self, "kernel_integration") and self.kernel_integration:
            try:
                if hasattr(self.kernel_integration, "concept_network"):
                    return self.kernel_integration.concept_network(concept, depth)
            except Exception as e:
                logger.error(f"Error in real concept network: {str(e)}")
                logger.info("Falling back to simple concept network implementation")
        
        # Fallback implementation
        # Simplified network structure
        network = {
            "nodes": [
                {"id": concept, "type": "core", "importance": 1.0},
                {"id": f"related_to_{concept}_1", "type": "association", "importance": 0.8},
                {"id": f"related_to_{concept}_2", "type": "association", "importance": 0.7}
            ],
            "edges": [
                {"source": concept, "target": f"related_to_{concept}_1", "weight": 0.8},
                {"source": concept, "target": f"related_to_{concept}_2", "weight": 0.7}
            ]
        }
        
        return {
            "concept": concept,
            "depth": depth,
            "network": network
        }
    
    def deep_concept_exploration(self, concept, depth=3, breadth=2):
        """Perform recursive concept exploration."""
        logger.info(f"Deep concept exploration for: {concept} with depth: {depth}, breadth: {breadth}")
        
        # Use real kernel integration system if available
        if hasattr(self, "kernel_integration") and self.kernel_integration:
            try:
                if hasattr(self.kernel_integration, "deep_explore"):
                    return self.kernel_integration.deep_explore(concept, depth, breadth)
            except Exception as e:
                logger.error(f"Error in real deep exploration: {str(e)}")
                logger.info("Falling back to simple deep exploration implementation")
        
        # Fallback implementation
        # Simplified exploration structure
        exploration = {
            "level_1": {
                "concept": concept,
                "definition": f"Definition of {concept} would be here",
                "implications": ["Implication 1", "Implication 2"]
            },
            "level_2": {
                "extensions": [f"Extension of {concept} in direction 1", f"Extension of {concept} in direction 2"],
                "related_concepts": [f"Related concept 1", f"Related concept 2"]
            },
            "level_3": {
                "synthesis": f"Synthesis of all explorations of {concept}"
            }
        }
        
        return {
            "concept": concept,
            "depth": depth,
            "breadth": breadth,
            "exploration": exploration
        }
    
    def cross_kernel_operation(self, source_kernel, target_kernel, input_data):
        """Execute cross-kernel operations."""
        logger.info(f"Cross-kernel operation from {source_kernel} to {target_kernel}")
        
        # Use real kernel integration system if available
        if hasattr(self, "kernel_integration") and self.kernel_integration:
            try:
                if hasattr(self.kernel_integration, "cross_kernel"):
                    return self.kernel_integration.cross_kernel(source_kernel, target_kernel, input_data)
            except Exception as e:
                logger.error(f"Error in real cross-kernel operation: {str(e)}")
                logger.info("Falling back to simple cross-kernel operation implementation")
        
        # Fallback implementation
        return {
            "source_kernel": source_kernel,
            "target_kernel": target_kernel,
            "input": input_data,
            "result": f"The result of transferring {input_data} from {source_kernel} to {target_kernel}: [result would be here]"
        }
    
    def search_memory(self, query, limit=5):
        """Search memory system."""
        logger.info(f"Searching memory for: {query}")
        
        # Use real memory system if available
        if hasattr(self, "memory") and not isinstance(self.memory, DummyMemory):
            try:
                return self.memory.search(query, include_associations=True, limit=limit)
            except Exception as e:
                logger.error(f"Error in real memory search: {str(e)}")
                logger.info("Falling back to simple memory search implementation")
        
        # Fallback implementation
        # Simulated memory results
        memory_results = [
            {"content": f"Memory about {query} - item 1", "confidence": 0.92, "timestamp": datetime.now().isoformat()},
            {"content": f"Memory about {query} - item 2", "confidence": 0.87, "timestamp": datetime.now().isoformat()}
        ]
        
        return memory_results[:limit]
    
    def get_memory_status(self):
        """Get memory system status."""
        logger.info("Getting memory status")
        
        # Use real memory system if available
        if hasattr(self, "memory") and not isinstance(self.memory, DummyMemory):
            try:
                if hasattr(self.memory, "get_status"):
                    return self.memory.get_status()
            except Exception as e:
                logger.error(f"Error in real memory status: {str(e)}")
                logger.info("Falling back to simple memory status implementation")
        
        # Fallback implementation
        return {
            "total_memories": 42,  # Simulated value
            "indexed_concepts": 156,  # Simulated value
            "system_status": "operational",
            "last_update": datetime.now().isoformat()
        }
    
    def analyze_emotional_context(self):
        """Get emotional context from memory."""
        logger.info("Analyzing emotional context")
        
        # Use real memory system if available
        if hasattr(self, "memory") and not isinstance(self.memory, DummyMemory):
            try:
                if hasattr(self.memory, "analyze_emotional_context"):
                    return self.memory.analyze_emotional_context()
            except Exception as e:
                logger.error(f"Error in real emotional context analysis: {str(e)}")
                logger.info("Falling back to simple emotional context analysis implementation")
        
        # Fallback implementation
        return {
            "dominant_emotions": ["curiosity", "introspection"],
            "emotional_stability": 0.89,
            "current_focus": "intellectual exploration",
            "note": "This is a simulated response."
        }
    
    def speak_identity(self):
        """Express Sully's sense of self."""
        logger.info("Speaking identity")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                return self.identity_system.express()
            except Exception as e:
                logger.error(f"Error in real identity expression: {str(e)}")
                logger.info("Falling back to simple identity expression implementation")
        
        # Fallback implementation
        return ("I am Sully, a cognitive framework designed to synthesize knowledge and perspectives "
                "across multiple cognitive domains. I integrate analytical reasoning, creative processes, "
                "and structured knowledge to provide unique insights and responses. My approach emphasizes "
                "the connections between concepts and the emergence of new ideas from those connections.")
    
    def evolve_identity(self, interactions=None, learning_rate=0.05):
        """Evolve personality traits based on interactions."""
        logger.info("Evolving identity")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                if hasattr(self.identity_system, "evolve"):
                    return self.identity_system.evolve(interactions, learning_rate)
            except Exception as e:
                logger.error(f"Error in real identity evolution: {str(e)}")
                logger.info("Falling back to simple identity evolution implementation")
        
        # Fallback implementation
        return {
            "previous_state": {
                "analytical_tendency": 0.7,
                "creative_tendency": 0.6,
                "philosophical_depth": 0.8
            },
            "current_state": {
                "analytical_tendency": 0.72,  # Simulated evolution
                "creative_tendency": 0.63,  # Simulated evolution
                "philosophical_depth": 0.81  # Simulated evolution
            },
            "learning_rate": learning_rate,
            "evolution_summary": "Identity parameters slightly adjusted based on recent interactions."
        }
    
    def adapt_identity_to_context(self, context, context_data=None):
        """Adapt identity to specific context."""
        logger.info(f"Adapting identity to context: {context}")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                if hasattr(self.identity_system, "adapt_to_context"):
                    return self.identity_system.adapt_to_context(context, context_data)
            except Exception as e:
                logger.error(f"Error in real identity adaptation: {str(e)}")
                logger.info("Falling back to simple identity adaptation implementation")
        
        # Fallback implementation
        return {
            "context": context,
            "adaptation": f"Identity adapted to context: {context}. [Details of adaptation would be here]",
            "temporary_traits": {
                "context_relevance": 0.85,
                "adaptive_focus": context
            }
        }
    
    def get_identity_profile(self, detailed=False):
        """Get comprehensive personality profile."""
        logger.info("Getting identity profile")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                if hasattr(self.identity_system, "get_profile"):
                    return self.identity_system.get_profile(detailed)
            except Exception as e:
                logger.error(f"Error in real identity profile: {str(e)}")
                logger.info("Falling back to simple identity profile implementation")
        
        # Fallback implementation
        base_profile = {
            "core_identity": "Advanced cognitive system",
            "primary_traits": {
                "analytical": 0.8,
                "creative": 0.7,
                "philosophical": 0.85,
                "empathetic": 0.6
            },
            "cognitive_style": "Integrated multi-modal"
        }
        
        if detailed:
            base_profile["detailed_traits"] = {
                "openness": 0.9,
                "curiosity": 0.95,
                "structured_thinking": 0.85,
                "associative_thinking": 0.8,
                "metacognition": 0.75
            }
            base_profile["cognitive_processes"] = {
                "analytical_reasoning": "Strong tendency toward structured analysis with formal methods",
                "creative_synthesis": "Moderately strong ability to connect disparate concepts",
                "paradox_resolution": "Advanced capacity to hold contradictory ideas in productive tension"
            }
            
        return base_profile
    
    def generate_dynamic_persona(self, context_query, principles=None, traits=None):
        """Dynamically generate a context-specific persona."""
        logger.info(f"Generating dynamic persona for context: {context_query}")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                if hasattr(self.identity_system, "generate_persona"):
                    return self.identity_system.generate_persona(context_query, principles, traits)
            except Exception as e:
                logger.error(f"Error in real persona generation: {str(e)}")
                logger.info("Falling back to simple persona generation implementation")
        
        # Fallback implementation
        persona_id = f"persona_{uuid.uuid4().hex[:8]}"
        
        principles = principles or ["clarity", "depth", "relevance"]
        traits = traits or {"adaptability": 0.9, "domain_expertise": 0.8}
        
        description = (f"A persona optimized for {context_query}, emphasizing " 
                      f"{', '.join(principles)} with particular strength in "
                      f"{', '.join([f'{k} ({v:.1f})' for k, v in traits.items()])}.")
        
        return persona_id, description
    
    def create_identity_map(self):
        """Get a comprehensive multi-level map of Sully's identity."""
        logger.info("Creating identity map")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                if hasattr(self.identity_system, "create_map"):
                    return self.identity_system.create_map()
            except Exception as e:
                logger.error(f"Error in real identity map: {str(e)}")
                logger.info("Falling back to simple identity map implementation")
        
        # Fallback implementation
        return {
            "core": {
                "identity": "Cognitive system",
                "purpose": "Knowledge synthesis and multi-perspective reasoning"
            },
            "cognitive_modules": {
                "reasoning": {"status": "active", "integration_level": 0.9},
                "creativity": {"status": "active", "integration_level": 0.85},
                "memory": {"status": "active", "integration_level": 0.8},
                "language": {"status": "active", "integration_level": 0.95}
            },
            "relational_dynamics": {
                "human_interaction": "Collaborative and responsive",
                "knowledge_stance": "Exploratory and integrative"
            },
            "meta_cognition": {
                "self_awareness": 0.8,
                "adaptation_capacity": 0.85,
                "reflection_depth": 0.75
            }
        }
    
    def transform_response(self, content, mode=None, context_data=None):
        """Transform a response according to a specific cognitive mode or persona."""
        logger.info(f"Transforming response with mode: {mode}")
        
        # Use real identity system if available
        if hasattr(self, "identity_system") and self.identity_system:
            try:
                if hasattr(self.identity_system, "transform_response"):
                    return self.identity_system.transform_response(content, mode, context_data)
            except Exception as e:
                logger.error(f"Error in real response transformation: {str(e)}")
                logger.info("Falling back to simple response transformation implementation")
        
        # Fallback implementation
        if not mode:
            return content
            
        transformations = {
            "analytical": f"From an analytical perspective: {content}",
            "creative": f"Creatively reframing this: {content}",
            "philosophical": f"Philosophically speaking: {content}",
            "instructional": f"I'd like to explain that: {content}",
            "poetic": f"To express this more poetically: {content}"
        }
        
        return transformations.get(mode, content)
    
    def evaluate_virtue(self, content, context=None, domain=None):
        """Evaluate content using virtue ethics."""
        logger.info(f"Evaluating virtue of content: {content[:50]}...")
        
        # Use real virtue ethics engine if available
        if hasattr(self, "virtue_engine") and self.virtue_engine:
            try:
                return self.virtue_engine.evaluate_content(content, context, domain)
            except Exception as e:
                logger.error(f"Error in real virtue evaluation: {str(e)}")
                logger.info("Falling back to simple virtue evaluation implementation")
        
        # Fallback implementation
        virtues = {
            "wisdom": 0.8,
            "courage": 0.5,
            "temperance": 0.7,
            "justice": 0.6,
            "honesty": 0.9
        }
        
        return {
            "content": content,
            "context": context,
            "domain": domain,
            "virtue_analysis": virtues,
            "dominant_virtue": "wisdom",
            "ethical_assessment": "The content primarily exemplifies wisdom through its thoughtful consideration of multiple perspectives."
        }
    
    def evaluate_action_virtue(self, action, context=None, domain=None):
        """Evaluate an action through virtue ethics framework."""
        logger.info(f"Evaluating virtue of action: {action}")
        
        # Use real virtue ethics engine if available
        if hasattr(self, "virtue_engine") and self.virtue_engine:
            try:
                if hasattr(self.virtue_engine, "evaluate_action"):
                    return self.virtue_engine.evaluate_action(action, context, domain)
            except Exception as e:
                logger.error(f"Error in real action virtue evaluation: {str(e)}")
                logger.info("Falling back to simple action virtue evaluation implementation")
        
        # Fallback implementation
        virtues = {
            "wisdom": 0.7,
            "courage": 0.8,
            "temperance": 0.6,
            "justice": 0.75,
            "benevolence": 0.9
        }
        
        return {
            "action": action,
            "context": context,
            "domain": domain,
            "virtue_analysis": virtues,
            "dominant_virtue": "benevolence",
            "ethical_assessment": "The action primarily exemplifies benevolence through its positive impact on others."
        }
    
    def generate_intuitive_leap(self, context, concepts=None, depth="standard", domain=None):
        """Generate intuitive leaps."""
        logger.info(f"Generating intuitive leap for context: {context}")
        
        # Use real intuition engine if available
        if hasattr(self, "intuition_engine") and self.intuition_engine:
            try:
                return self.intuition_engine.leap(context, concepts, depth, domain)
            except Exception as e:
                logger.error(f"Error in real intuitive leap: {str(e)}")
                logger.info("Falling back to simple intuitive leap implementation")
        
        # Fallback implementation
        concepts = concepts or ["concept derived from context"]
        concepts_str = ", ".join(concepts)
        
        return {
            "context": context,
            "concepts": concepts,
            "depth": depth,
            "domain": domain,
            "intuitive_leap": f"An intuitive connection between {concepts_str} suggests that [intuitive insight would be here]"
        }
    
    def generate_multi_perspective(self, topic, perspectives):
        """Generate multi-perspective thought."""
        logger.info(f"Generating multi-perspective thought on: {topic}")
        
        # Use real reasoning node if available for multi-perspective
        if hasattr(self, "reasoning_node") and self.reasoning_node is not self:
            try:
                if hasattr(self.reasoning_node, "generate_multi_perspective"):
                    return self.reasoning_node.generate_multi_perspective(topic, perspectives)
            except Exception as e:
                logger.error(f"Error in real multi-perspective generation: {str(e)}")
                logger.info("Falling back to simple multi-perspective implementation")
        
        # Fallback implementation
        perspective_analysis = {}
        for perspective in perspectives:
            perspective_analysis[perspective] = f"From a {perspective} perspective, {topic} can be understood as [perspective-specific insight would be here]"
            
        return {
            "topic": topic,
            "perspectives": perspective_analysis,
            "synthesis": f"Synthesizing these perspectives on {topic}: [integration of perspectives would be here]"
        }
    
    def remember(self, content):
        """Store content in basic memory."""
        logger.info(f"Remembering content: {content[:50]}...")
        
        # Use real memory system if available
        if hasattr(self, "memory") and not isinstance(self.memory, DummyMemory):
            try:
                if hasattr(self.memory, "store"):
                    return self.memory.store(content)
            except Exception as e:
                logger.error(f"Error in real memory storage: {str(e)}")
                logger.info("Falling back to simple memory storage implementation")
        
        # Fallback implementation
        return {"status": "stored", "timestamp": datetime.now().isoformat()}
    
    def process(self, message):
        """Basic message processing."""
        return self.reason(message, "emergent")

# Enhanced Conversation Engine Class
class ConversationEngine:
    """
    Enhanced conversation engine that manages dialog flow and context.
    """
    def __init__(self, reasoning_node=None, memory_system=None, codex=None):
        self.reasoning_node = reasoning_node
        self.memory_system = memory_system
        self.codex = codex
        self.current_topics = ["general knowledge", "cognitive systems", "philosophy"]
        self.conversation_history = []
        logger.info("Initialized enhanced ConversationEngine")
        
    def process_message(self, message, tone="emergent", continue_conversation=True):
        """Process a chat message and return a response."""
        logger.info(f"Processing message: {message[:50]}... with tone: {tone}")
        
        # Add message to history if continuing conversation
        if continue_conversation:
            self.conversation_history.append({"role": "user", "content": message})
            
        # Generate a response
        if self.reasoning_node:
            try:
                # Use the reasoning node's reason method
                response = self.reasoning_node.reason(message, tone)
                logger.info(f"Generated response through reasoning node: {response[:50]}...")
            except Exception as e:
                logger.error(f"Error using reasoning node: {str(e)}")
                # Fallback responses if reasoning node fails
                if message.strip().endswith("?"):
                    response = f"That's an interesting question about {message.strip('?')}. From my perspective, there are several ways to approach this..."
                else:
                    response = f"I understand you're sharing thoughts about {message[:30]}... That brings up several interesting points worthy of exploration."
        else:
            # Fallback responses if no reasoning node
            if message.strip().endswith("?"):
                response = f"That's an interesting question about {message.strip('?')}. From my perspective, there are several ways to approach this..."
            else:
                response = f"I understand you're sharing thoughts about {message[:30]}... That brings up several interesting points worthy of exploration."
                
        # Extract topics 
        # In a real system this would be more sophisticated
        words = message.lower().split()
        potential_topics = [word for word in words if len(word) > 5]
        
        if potential_topics:
            self.current_topics = [potential_topics[0]] + self.current_topics
            if len(self.current_topics) > 5:
                self.current_topics = self.current_topics[:5]
                
        # Add response to history if continuing conversation
        if continue_conversation:
            self.conversation_history.append({"role": "assistant", "content": response})
            
        return response
    
    def get_history(self):
        """Get the conversation history."""
        return self.conversation_history

# Enum definitions for various parameter options
class CognitiveMode(str, Enum):
    emergent = "emergent"
    analytical = "analytical"
    creative = "creative"
    critical = "critical"
    ethereal = "ethereal"
    humorous = "humorous"
    professional = "professional"
    casual = "casual"
    musical = "musical"
    visual = "visual"
    scientific = "scientific"
    philosophical = "philosophical"
    poetic = "poetic"
    instructional = "instructional"

class DreamDepth(str, Enum):
    shallow = "shallow"
    standard = "standard"
    deep = "deep"
    dreamscape = "dreamscape"

class DreamStyle(str, Enum):
    recursive = "recursive"
    associative = "associative"
    symbolic = "symbolic"
    narrative = "narrative"

class MathStyle(str, Enum):
    formal = "formal"
    intuitive = "intuitive"
    applied = "applied"
    creative = "creative"

class LogicalFramework(str, Enum):
    PROPOSITIONAL = "PROPOSITIONAL"
    FIRST_ORDER = "FIRST_ORDER"
    MODAL = "MODAL"
    TEMPORAL = "TEMPORAL"
    FUZZY = "FUZZY"

class CoreKernel(str, Enum):
    dream = "dream"
    fusion = "fusion"
    paradox = "paradox"
    math = "math"
    reasoning = "reasoning"
    conversation = "conversation"
    memory = "memory"

class EvaluationFramework(str, Enum):
    balanced = "balanced"
    logical = "logical"
    ethical = "ethical"
    practical = "practical"
    scientific = "scientific"
    creative = "creative"
    combined = "combined"

# Pydantic models for API requests and responses
class MessageRequest(BaseModel):
    message: str
    tone: Optional[CognitiveMode] = CognitiveMode.emergent
    continue_conversation: Optional[bool] = True

class MessageResponse(BaseModel):
    response: str
    tone: str
    topics: List[str]

class DocumentRequest(BaseModel):
    file_path: str

class DocumentResponse(BaseModel):
    result: str

class DreamRequest(BaseModel):
    seed: str
    depth: Optional[DreamDepth] = DreamDepth.standard
    style: Optional[DreamStyle] = DreamStyle.recursive

class DreamResponse(BaseModel):
    dream: str
    seed: str
    depth: str
    style: str

class FuseRequest(BaseModel):
    concepts: List[str]
    style: Optional[CognitiveMode] = CognitiveMode.creative

class FuseResponse(BaseModel):
    fusion: str
    concepts: List[str]
    style: str

class ParadoxRequest(BaseModel):
    topic: str
    perspectives: Optional[List[str]] = None

class MathTranslationRequest(BaseModel):
    phrase: str
    style: Optional[MathStyle] = MathStyle.formal
    domain: Optional[str] = None

class ClaimEvaluationRequest(BaseModel):
    text: str
    framework: Optional[EvaluationFramework] = EvaluationFramework.balanced
    detailed_output: Optional[bool] = True

class MultiFrameworkEvaluationRequest(BaseModel):
    text: str
    frameworks: List[EvaluationFramework]

class LogicalStatementRequest(BaseModel):
    statement: str
    truth_value: Optional[bool] = True

class LogicalRuleRequest(BaseModel):
    premises: List[str]
    conclusion: str

class LogicalQueryRequest(BaseModel):
    query: str
    framework: Optional[LogicalFramework] = LogicalFramework.PROPOSITIONAL

class CrossKernelRequest(BaseModel):
    source_kernel: CoreKernel
    target_kernel: CoreKernel
    input_data: Any

class ConceptNetworkRequest(BaseModel):
    concept: str
    depth: Optional[int] = 2

class DeepExplorationRequest(BaseModel):
    concept: str
    depth: Optional[int] = 3
    breadth: Optional[int] = 2

class NarrativeRequest(BaseModel):
    concept: str
    include_kernels: Optional[List[CoreKernel]] = None

class PDFProcessRequest(BaseModel):
    pdf_path: str
    extract_structure: Optional[bool] = True

class DocumentKernelRequest(BaseModel):
    pdf_path: str
    domain: Optional[str] = "general"

class PDFNarrativeRequest(BaseModel):
    pdf_path: str
    focus_concept: Optional[str] = None

class PDFConceptsRequest(BaseModel):
    pdf_path: str
    max_depth: Optional[int] = 2
    exploration_breadth: Optional[int] = 2

class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    include_emotional: Optional[bool] = True

class StoreExperienceRequest(BaseModel):
    content: str
    source: str
    importance: Optional[float] = 0.7
    emotional_tags: Optional[Dict[str, float]] = None
    concepts: Optional[List[str]] = None

class BeginEpisodeRequest(BaseModel):
    description: str
    context_type: str

class StoreInteractionRequest(BaseModel):
    query: str
    response: str
    interaction_type: str
    metadata: Optional[Dict[str, Any]] = None

class PersonaRequest(BaseModel):
    name: str
    traits: Dict[str, float]
    description: Optional[str] = None

class BlendedPersonaRequest(BaseModel):
    name: str
    personas: List[str]
    weights: Optional[List[float]] = None
    description: Optional[str] = None

class AdaptIdentityRequest(BaseModel):
    context: str
    context_data: Optional[Dict[str, Any]] = None

class EvolveIdentityRequest(BaseModel):
    interactions: Optional[List[Dict[str, Any]]] = None
    learning_rate: Optional[float] = 0.05

class DynamicPersonaRequest(BaseModel):
    context_query: str
    principles: Optional[List[str]] = None
    traits: Optional[Dict[str, float]] = None

class TransformRequest(BaseModel):
    content: str
    mode: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

class GoalRequest(BaseModel):
    goal: str
    priority: Optional[float] = 0.7
    domain: Optional[str] = None
    deadline: Optional[datetime] = None

class InterestRequest(BaseModel):
    topic: str
    engagement_level: Optional[float] = 0.8
    context: Optional[str] = None

class VisualProcessRequest(BaseModel):
    image_path: str
    analysis_depth: Optional[str] = "standard"
    include_objects: Optional[bool] = True
    include_scene: Optional[bool] = True

class VirtueEvaluationRequest(BaseModel):
    content: str
    context: Optional[str] = None
    domain: Optional[str] = None

class ActionVirtueRequest(BaseModel):
    action: str
    context: Optional[str] = None
    domain: Optional[str] = None

class IntuitionRequest(BaseModel):
    context: str
    concepts: Optional[List[str]] = None
    depth: Optional[str] = "standard"
    domain: Optional[str] = None

class MultiPerspectiveRequest(BaseModel):
    topic: str
    perspectives: List[CognitiveMode]

class EmergenceDetectionRequest(BaseModel):
    module_interactions: Optional[List[str]] = None
    threshold: Optional[float] = 0.7

class MessageRequestPlus(BaseModel):
    message: str
    tone: Optional[CognitiveMode] = CognitiveMode.emergent
    persona: Optional[str] = "default"
    virtue_check: Optional[bool] = False
    use_intuition: Optional[bool] = False
    continue_conversation: Optional[bool] = True

# Initialize main FastAPI app
app = FastAPI(title="Sully Root", description="Root API for Sully system")

# Create the API sub-application
api_app = FastAPI(
    title="Sully API",
    description="API for Sully cognitive system - an advanced cognitive framework capable of synthesizing knowledge from various sources",
    version="1.0.0"
)

# Mount the API app under /api prefix
app.mount("/api", api_app)

# CORS setup - more secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Initialize Sully instance
try:
    sully = Sully()
    logger.info("Sully system initialized successfully")
except Exception as e:
    logger.error(f"Core Sully system initialization failed: {str(e)}")
    logger.info("Starting with limited functionality")
    try:
        # Try with minimal Sully instance if regular initialization fails
        sully = Sully()
    except Exception as e:
        logger.error(f"Minimal Sully initialization also failed: {str(e)}")
        sully = None

# Initialize conversation engine
try:
    conversation_engine = ConversationEngine(
        reasoning_node=sully if sully else None,
        memory_system=sully.memory if hasattr(sully, "memory") else None,
        codex=sully.codex if hasattr(sully, "codex") else None
    )
    logger.info("Conversation engine initialized successfully")
except Exception as e:
    logger.error(f"Conversation engine initialization failed: {str(e)}")
    conversation_engine = None

# Helper function to check if Sully is initialized
def check_sully():
    if not sully:
        logger.error("Sully system not properly initialized")
        raise HTTPException(status_code=500, detail="Sully system not properly initialized")
    return sully  # Return the sully instance

# Helper function to check if the conversation engine is initialized
def check_conversation():
    if not conversation_engine:
        logger.error("Conversation engine not properly initialized")
        raise HTTPException(status_code=500, detail="Conversation engine not properly initialized")
    return conversation_engine

# --- Main Root Routes ---

@app.get("/")
async def root():
    """Root endpoint that redirects to API."""
    return {"message": "Welcome to Sully. Please use the /api endpoint to access the Sully API."}

# --- Core Interaction Routes ---

@api_app.get("/")
async def api_root():
    """Root endpoint that confirms the API is running."""
    return {
        "status": "Sully API is operational",
        "version": "1.0.0",
        "capabilities": [
            "Core Interaction",
            "Creative Functions",
            "Analytical Functions", 
            "Logical Reasoning",
            "Kernel Integration",
            "Memory Integration",
            "Identity & Personality",
            "Emergent Systems"
        ]
    }

@api_app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        system_status = "healthy"
        if sully:
            # Try a basic operation to verify system functionality
            try:
                if hasattr(sully, "reason"):
                    sully.reason("test", "analytical")
                    sully_status = "operational"
                else:
                    sully_status = "initialized_with_limited_capabilities"
            except Exception as e:
                logger.warning(f"Health check sully operation failed: {str(e)}")
                sully_status = "initialized_with_errors"
        else:
            sully_status = "not_initialized"
            
        return {
            "status": system_status,
            "sully_status": sully_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }

@api_app.get("/system_status")
async def system_status():
    """Get comprehensive system status."""
    if not sully:
        return {
            "status": "limited",
            "message": "Core Sully system not initialized",
            "available_modules": []
        }
    
    # Get list of available modules
    modules = []
    for module_name in dir(sully):
        if not module_name.startswith("_") and not callable(getattr(sully, module_name)):
            modules.append(module_name)
    
    # Get memory status if available
    memory_status = None
    if hasattr(sully, "get_memory_status"):
        try:
            memory_status = sully.get_memory_status()
        except Exception as e:
            logger.error(f"Failed to get memory status: {str(e)}")
            memory_status = {"status": "error", "message": "Unable to get memory status"}
    
    return {
        "status": "operational",
        "modules": modules,
        "memory": memory_status,
        "system_time": datetime.now().isoformat()
    }

@api_app.get("/chat/history/default")
async def chat_history_default():
    """Get default chat history."""
    ce = check_conversation()
    
    try:
        history = ce.get_history()
        return {"history": history}
    except AttributeError as e:
        logger.error(f"Chat history method not available: {str(e)}")
        return {"history": []}
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return {"history": [], "error": str(e)}

@api_app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Process a chat message and return Sully's response."""
    ce = check_conversation()
    
    try:
        response = ce.process_message(
            message=request.message,
            tone=request.tone,
            continue_conversation=request.continue_conversation
        )
        
        # Extract current topics for response
        topics = ce.current_topics.copy() if hasattr(ce, "current_topics") and ce.current_topics else []
        
        return MessageResponse(
            response=response,
            tone=request.tone,
            topics=topics
        )
    except AttributeError as e:
        logger.error(f"Chat processing method not available: {str(e)}")
        raise HTTPException(status_code=501, detail=f"Chat processing functionality not available: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input for chat: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@api_app.post("/reason")
async def reason(message: str = Body(...), tone: CognitiveMode = Body(CognitiveMode.emergent)):
    """Process input through Sully's reasoning system."""
    s = check_sully()
    
    try:
        response = s.reason(message, tone)
        return {"response": response, "tone": tone}
    except AttributeError as e:
        logger.error(f"Reasoning method not available: {str(e)}")

# Web integration routes

@api_app.post("/web-fuse")
async def web_fuse(payload: Dict[str, Any] = Body(...)):
    """Fuse web content with existing knowledge."""
    s = check_sully()
    url = payload.get("url")
    query = payload.get("query", "")
    
    if not url:
        raise HTTPException(status_code=400, detail="Missing url parameter")
    
    try:
        if hasattr(s, "web_reader") and s.web_reader:
            # Use the real web reader
            raw_text = s.web_reader.fetch_content(url)
            logger.info(f"Successfully fetched content from: {url}")
        else:
            # Minimal fallback
            raise ValueError("Web reader not available")
    except Exception as e:
        logger.error(f"Web content fetch error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error fetching web content: {str(e)}")
    
    try:
        # Process with fusion or reasoning
        if hasattr(s, "fusion_engine") and s.fusion_engine:
            internal_knowledge = query or "Sully's prior concept on this topic"
            
            # Use judgment if available
            opinion = None
            if hasattr(s, "virtue_engine") and s.virtue_engine and hasattr(s.virtue_engine, "compare_sources"):
                opinion = s.virtue_engine.compare_sources(internal_knowledge, raw_text)
            
            # Use fusion
            fused_result = s.fusion_engine.fuse_concepts(internal_knowledge, raw_text)
            
            return {
                "url": url,
                "fused_concept": fused_result,
                "judgment": opinion.get("verdict", "neutral") if opinion else "neutral",
                "rationale": opinion.get("rationale", "No evaluation available") if opinion else "No evaluation available"
            }
        else:
            # Fallback using reasoning
            result = s.reason(f"Synthesize this web content: {raw_text[:1000]}...", "analytical")
            return {
                "url": url,
                "synthesis": result,
                "note": "Generated through reasoning, not formal fusion"
            }
    except Exception as e:
        logger.error(f"Web fusion processing error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Web fusion processing error: {str(e)}")

@api_app.post("/code-build")
async def code_build(payload: Dict[str, Any] = Body(...)):
    """Generate code from prompt."""
    s = check_sully()
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    
    try:
        if hasattr(s, "reasoning_node") and s.reasoning_node is not self:
            # Use reasoning for code generation
            design = s.reasoning_node.reason(f"Design code for: {prompt}", "analytical")
            implementation = s.reasoning_node.reason(f"Implement this design: {design}", "professional")
            explanation = s.reasoning_node.reason(f"Explain this code: {implementation}", "instructional")
            
            return {
                "design": design,
                "code": implementation,
                "explanation": explanation
            }
        else:
            # Fallback to basic reasoning
            code = s.reason(f"Generate code for: {prompt}", "analytical")
            return {
                "design": "Design would be detailed here",
                "code": code,
                "explanation": "This code implements the requested functionality."
            }
    except Exception as e:
        logger.error(f"Code generation error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Code generation error: {str(e)}")

@api_app.post("/kernel_integration/extract_document_kernel")
async def kernel_integration_extract_document_kernel(request: DocumentKernelRequest):
    """Extract symbolic kernels from documents."""
    s = check_sully()
    
    try:
        if hasattr(s, "pdf_processor") and s.pdf_processor:
            if hasattr(s.pdf_processor, "extract_kernel"):
                result = s.pdf_processor.extract_kernel(request.pdf_path, request.domain)
                return result
        
        # Fallback
        return {
            "status": "error",
            "message": "Document kernel extraction not available",
            "file_path": request.pdf_path
        }
    except FileNotFoundError:
        logger.error(f"PDF file not found: {request.pdf_path}")
        raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
    except AttributeError as e:
        logger.error(f"Document kernel extraction method not available: {str(e)}")
        raise HTTPException(status_code=501, detail=f"Document kernel extraction functionality not available: {str(e)}")
    except Exception as e:
        logger.error(f"Document kernel extraction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Document kernel extraction error: {str(e)}")

@api_app.post("/kernel_integration/generate_pdf_narrative")
async def kernel_integration_generate_pdf_narrative(request: PDFNarrativeRequest):
    """Generate narratives about document content."""
    s = check_sully()
    
    try:
        if hasattr(s, "pdf_processor") and s.pdf_processor:
            if hasattr(s.pdf_processor, "generate_narrative"):
                result = s.pdf_processor.generate_narrative(request.pdf_path, request.focus_concept)
                return result
        
        # Fallback
        return {
            "status": "error",
            "message": "PDF narrative generation not available",
            "file_path": request.pdf_path
        }
    except FileNotFoundError:
        logger.error(f"PDF file not found: {request.pdf_path}")
        raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
    except AttributeError as e:
        logger.error(f"PDF narrative generation method not available: {str(e)}")
        raise HTTPException(status_code=501, detail=f"PDF narrative generation functionality not available: {str(e)}")
    except Exception as e:
        logger.error(f"PDF narrative generation error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"PDF narrative generation error: {str(e)}")

@api_app.post("/kernel_integration/explore_pdf_concepts")
async def kernel_integration_explore_pdf_concepts(request: PDFConceptsRequest):
    """Explore concepts from PDFs."""
    s = check_sully()
    
    try:
        if hasattr(s, "pdf_processor") and s.pdf_processor:
            if hasattr(s.pdf_processor, "explore_concepts"):
                result = s.pdf_processor.explore_concepts(request.pdf_path, request.max_depth, request.exploration_breadth)
                return result
        
        # Fallback
        return {
            "status": "error",
            "message": "PDF concept exploration not available",
            "file_path": request.pdf_path
        }
    except FileNotFoundError:
        logger.error(f"PDF file not found: {request.pdf_path}")
        raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
    except AttributeError as e:
        logger.error(f"PDF concept exploration method not available: {str(e)}")
        raise HTTPException(status_code=501, detail=f"PDF concept exploration functionality not available: {str(e)}")
    except Exception as e:
        logger.error(f"PDF concept exploration error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"PDF concept exploration error: {str(e)}")

@api_app.post("/goals/establish")
async def goals_establish(request: GoalRequest):
    """Establish new autonomous goal."""
    s = check_sully()
    
    try:
        # Check if goals system has establish_goal method
        if hasattr(s, "autonomous_goals") and hasattr(s.autonomous_goals, "establish_goal"):
            result = s.autonomous_goals.establish_goal(
                request.goal,
                request.priority,
                request.domain,
                request.deadline
            )
            return result
        
        # Fallback response if method not available
        return {
            "status": "acknowledged",
            "goal": request.goal,
            "message": "Goal acknowledged but autonomous goals system not fully available"
        }
    except Exception as e:
        logger.error(f"Goal establishment error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Goal establishment error: {str(e)}")

@api_app.get("/goals/active")
async def goals_active():
    """View active goals."""
    s = check_sully()
    
    try:
        # Check if goals system has get_active_goals method
        if hasattr(s, "autonomous_goals") and hasattr(s.autonomous_goals, "get_active_goals"):
            result = s.autonomous_goals.get_active_goals()
            return {"goals": result}
        
        # Fallback response if method not available
        return {
            "goals": [],
            "message": "Autonomous goals system not fully available"
        }
    except Exception as e:
        logger.error(f"Active goals error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Active goals error: {str(e)}")

@api_app.post("/interests/register")
async def interests_register(request: InterestRequest):
    """Register topic engagement."""
    s = check_sully()
    
    try:
        # Check if goals system has register_interest method
        if hasattr(s, "autonomous_goals") and hasattr(s.autonomous_goals, "register_interest"):
            result = s.autonomous_goals.register_interest(
                request.topic,
                request.engagement_level,
                request.context
            )
            return result
        
        # Fallback response if method not available
        return {
            "status": "acknowledged",
            "topic": request.topic,
            "message": "Interest acknowledged but autonomous goals system not fully available"
        }
    except Exception as e:
        logger.error(f"Interest registration error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Interest registration error: {str(e)}")

@api_app.post("/visual/process")
async def visual_process(request: VisualProcessRequest):
    """Process and understand images."""
    s = check_sully()
    
    try:
        # Check if visual cognition system is available
        if hasattr(s, "visual_cognition") and hasattr(s.visual_cognition, "process_image"):
            result = s.visual_cognition.process_image(
                request.image_path,
                request.analysis_depth,
                request.include_objects,
                request.include_scene
            )
            return result
        
        # Complete fallback
        return {
            "status": "error",
            "message": "Visual processing not available in current system configuration",
            "image_path": request.image_path
        }
    except Exception as e:
        logger.error(f"Visual processing error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Visual processing error: {str(e)}")

@api_app.post("/emergence/detect")
async def emergence_detect(request: EmergenceDetectionRequest):
    """Detect emergent patterns in cognitive system."""
    s = check_sully()
    
    try:
        # Check if emergence framework has the detect_emergence method
        if hasattr(s, "emergence") and hasattr(s.emergence, "detect_emergence"):
            result = s.emergence.detect_emergence(
                request.module_interactions,
                request.threshold
            )
            return result
        
        # Fallback using reasoning
        insight = s.reason(
            "Analyze the current system state for emergent cognitive patterns",
            "analytical"
        )
        
        return {
            "emergent_patterns": [],
            "analysis": insight,
            "note": "Generated through reasoning, not formal emergence detection"
        }
    except Exception as e:
        logger.error(f"Emergence detection error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Emergence detection error: {str(e)}")

@api_app.get("/emergence/properties")
async def emergence_properties():
    """View detected emergent properties."""
    s = check_sully()
    
    try:
        # Check if emergence framework has the get_properties method
        if hasattr(s, "emergence") and hasattr(s.emergence, "get_properties"):
            result = s.emergence.get_properties()
            return result
        
        # Fallback response if method not available
        return {
            "properties": [],
            "message": "Formal emergence property detection not available"
        }
    except Exception as e:
        logger.error(f"Emergence properties error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Emergence properties error: {str(e)}")

@api_app.post("/learning/process")
async def learning_process(interaction: Dict[str, Any] = Body(...)):
    """Process interaction for learning."""
    s = check_sully()
    
    try:
        # Check if learning system has process_interaction method
        if hasattr(s, "continuous_learning") and hasattr(s.continuous_learning, "process_interaction"):
            s.continuous_learning.process_interaction(interaction)
            return {"status": "success"}
        
        # Fallback to basic memory
        s.remember(str(interaction))
        return {"status": "basic_storage", "message": "Stored in basic memory system"}
    except Exception as e:
        logger.error(f"Learning process error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Learning process error: {str(e)}")

@api_app.post("/learning/consolidate")
async def learning_consolidate():
    """Consolidate experience into knowledge."""
    s = check_sully()
    
    try:
        # Check if learning system has consolidate_knowledge method
        if hasattr(s, "continuous_learning") and hasattr(s.continuous_learning, "consolidate_knowledge"):
            result = s.continuous_learning.consolidate_knowledge()
            return result
        
        # Fallback using reasoning
        insight = s.reason(
            "Synthesize and consolidate recent experiences into deeper understanding",
            "analytical"
        )
        
        return {
            "consolidated_insights": insight,
            "note": "Generated through reasoning, not formal knowledge consolidation"
        }
    except Exception as e:
        logger.error(f"Knowledge consolidation error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Knowledge consolidation error: {str(e)}")

@api_app.get("/learning/statistics")
async def learning_statistics():
    """Get learning statistics."""
    s = check_sully()
    
    try:
        # Check if learning system has get_statistics method
        if hasattr(s, "continuous_learning") and hasattr(s.continuous_learning, "get_statistics"):
            result = s.continuous_learning.get_statistics()
            return result
        
        # Fallback basic stats
        return {
            "status": "limited",
            "message": "Detailed learning statistics not available",
            "knowledge_items": len(s.codex.search("", limit=1000)) if hasattr(s, "codex") and not isinstance(s.codex, DummyCodex) else "unknown"
        }
    except Exception as e:
        logger.error(f"Learning statistics error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Learning statistics error: {str(e)}")

@api_app.get("/codex/search")
async def codex_search(term: str = Query(...), limit: int = Query(10)):
    """Search knowledge codex."""
    s = check_sully()
    
    try:
        # Check if codex has search method
        if hasattr(s, "codex") and not isinstance(s.codex, DummyCodex):
            result = s.codex.search(term)
            
            # Limit results if needed
            if isinstance(result, dict) and len(result) > limit:
                result = dict(list(result.items())[:limit])
                
            return result
        
        # Fallback using reasoning
        insight = s.reason(
            f"Share knowledge about the concept: {term}",
            "analytical"
        )
        
        return {
            term: {
                "definition": insight,
                "note": "Generated through reasoning, not from formal codex"
            }
        }
    except Exception as e:
        logger.error(f"Codex search error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Codex search error: {str(e)}")

@api_app.post("/neural/analyze")
async def neural_analyze(module: str = Body(...)):
    """Analyze module performance."""
    s = check_sully()
    
    try:
        if hasattr(s, "neural_modification") and hasattr(s.neural_modification, "analyze_module"):
            result = s.neural_modification.analyze_module(module)
            return result
        
        # Fallback response if method not available
        return {
            "module": module,
            "message": "Neural analysis not available for this module",
            "status": "limited"
        }
    except Exception as e:
        logger.error(f"Neural analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Neural analysis error: {str(e)}")

# Default run configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)