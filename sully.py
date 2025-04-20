"""
Sully - An advanced cognitive system with integrated memory and enhanced capabilities.
"""
# --- Imports ---
import os
import json
import sys
import logging
import inspect
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
from datetime import datetime
import importlib
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sully.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sully")

# Core modules imports
from sully_engine.kernel_integration import initialize_kernel_integration, KernelIntegrationSystem
from sully_engine.memory_integration import MemoryIntegration, integrate_with_sully
from sully_engine.logic_kernel import LogicKernel, integrate_with_sully as integrate_logic
from sully_engine.pdf_reader import PDFReader

# Import cognitive modules
from sully_engine.kernel_modules.identity import SullyIdentity
from sully_engine.kernel_modules.codex import SullyCodex
from sully_engine.reasoning import SymbolicReasoningNode
from sully_engine.memory import SullySearchMemory
from sully_engine.kernel_modules.judgment import JudgmentProtocol
from sully_engine.kernel_modules.dream import DreamCore
from sully_engine.kernel_modules.math_translator import SymbolicMathTranslator
from sully_engine.kernel_modules.fusion import SymbolFusionEngine
from sully_engine.kernel_modules.paradox import ParadoxLibrary
from sully_engine.kernel_modules.neural_modification import NeuralModification
from sully_engine.kernel_modules.continuous_learning import ContinuousLearningSystem
from sully_engine.kernel_modules.autonomous_goals import AutonomousGoalSystem
from sully_engine.kernel_modules.visual_cognition import VisualCognitionSystem
from sully_engine.kernel_modules.emergence_framework import EmergenceFramework
from sully_engine.kernel_modules.virtue import VirtueEngine
from sully_engine.kernel_modules.intuition import Intuition
from sully_engine.kernel_modules.persona import PersonaManager

# Import new language enhancement modules
from sully_engine.content_ingestion import ContentIngestionKernel, LanguageStructureProcessor, enhance_sully_content_ingestion
from sully_engine.language_enhancement import SemanticStructureAnalyzer, ResponseGenerator

# Import conversation engine
try:
    from sully_engine.conversation_engine import ConversationEngine
except ImportError:
    logger.warning("ConversationEngine import failed. Chat functionality may be limited.")
    ConversationEngine = None

# Configuration constants
MEMORY_PATH = "sully_memory_store.json"
MEMORY_INTEGRATION_ENABLED = True
DEFAULT_LOG_LEVEL = logging.INFO

# Cognitive modes
COGNITIVE_MODES = [
    "emergent", "analytical", "creative", "critical", "ethereal",
    "humorous", "professional", "casual", "musical", "visual",
    "scientific", "philosophical", "poetic", "instructional"
]

class Sully:
    """
    Sully: An advanced cognitive framework capable of synthesizing knowledge from various sources,
    with enhanced integrated memory and expressing it through multiple cognitive modes.
    """

    def __init__(self, memory_path: Optional[str] = None, log_level: int = DEFAULT_LOG_LEVEL, 
                 enable_memory_integration: bool = MEMORY_INTEGRATION_ENABLED):
        """
        Initialize Sully's cognitive systems with integrated memory.
        
        Args:
            memory_path: Optional path to memory storage
            log_level: Logging level (default: INFO)
            enable_memory_integration: Whether to enable memory integration
        """
        # Configure logging
        self.logger = logging.getLogger("sully")
        self.logger.setLevel(log_level)
        self.logger.info("Initializing Sully cognitive system...")
        
        # Track initialization errors for graceful degradation
        self.initialization_errors = {}
        
        # Core cognitive architecture initialization
        # Initialize reasoning node first since identity system depends on it
        self._initialize_core_components()
        
        # Advanced cognitive modules initialization
        self._initialize_advanced_modules()
        
        # Initialize emergence framework with all cognitive modules
        self._initialize_emergence_framework()
        
        # PDF reader for direct document processing
        self.pdf_reader = self._initialize_with_fallback(
            "pdf_reader", 
            lambda: PDFReader(ocr_enabled=True, dpi=300),
            "PDF reader initialization failed. Document processing will be limited."
        )
        
        # Initialize language processing and content ingestion
        self._initialize_language_enhancement()
        
        # Experiential knowledge - unlimited and ever-growing
        self.knowledge = []
        
        # Initialize the conversation engine and connect to the core systems
        self._initialize_conversation_engine()
        
        # Initialize memory integration system if enabled
        self.memory_integration = None
        if enable_memory_integration:
            self._initialize_memory_integration(memory_path or MEMORY_PATH)
            
        # Initialize kernel integration system
        self._initialize_kernel_integration()
        
        # Initialize logic kernel
        self._initialize_logic_kernel()
        
        # System metadata
        self.system_id = str(uuid.uuid4())
        self.creation_time = datetime.now()
        self.last_active = self.creation_time
        
        # Track module access statistics
        self.module_access_stats = {}
        
        # Event hooks for extensibility
        self.hooks = {
            "before_reasoning": [],
            "after_reasoning": [],
            "before_memory_store": [],
            "after_memory_store": [],
            "before_document_processing": [],
            "after_document_processing": []
        }
        
        self.logger.info(f"Sully initialization complete. System ID: {self.system_id}")
        
    def _initialize_with_fallback(self, component_name: str, init_func: Callable, 
                                 error_message: str, fallback_func: Optional[Callable] = None) -> Any:
        """
        Initialize a component with error handling and optional fallback.
        
        Args:
            component_name: Name of the component
            init_func: Initialization function
            error_message: Message to log on error
            fallback_func: Optional fallback initialization function
            
        Returns:
            Initialized component or None
        """
        try:
            component = init_func()
            return component
        except Exception as e:
            self.logger.error(f"{error_message} Error: {str(e)}")
            self.initialization_errors[component_name] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            if fallback_func:
                try:
                    self.logger.info(f"Attempting fallback initialization for {component_name}")
                    return fallback_func()
                except Exception as fallback_e:
                    self.logger.error(f"Fallback initialization for {component_name} failed: {str(fallback_e)}")
                    
            return None

    def _initialize_core_components(self):
        """Initialize core cognitive components."""
        # Core components with fallbacks
        self.memory = self._initialize_with_fallback(
            "memory",
            lambda: SullySearchMemory(),
            "Memory system initialization failed. Using basic memory.",
            lambda: {"queries": [], "results": []}
        )
        
        self.codex = self._initialize_with_fallback(
            "codex",
            lambda: SullyCodex(),
            "Codex initialization failed. Knowledge retrieval will be limited."
        )
        
        # Create reasoning node with placeholder for translator
        self.reasoning_node = self._initialize_with_fallback(
            "reasoning_node",
            lambda: SymbolicReasoningNode(
                codex=self.codex,
                translator=None,  # Will set after initializing translator
                memory=self.memory
            ),
            "Reasoning node initialization failed. Cognitive processing will be limited."
        )
        
        # Initialize identity system with connections to other modules
        self.identity = self._initialize_with_fallback(
            "identity",
            lambda: SullyIdentity(
                memory_system=self.memory,
                reasoning_engine=self.reasoning_node
            ),
            "Identity system initialization failed. Persona capabilities will be limited."
        )
        
        # Initialize the translator and update reasoning node reference
        self.translator = self._initialize_with_fallback(
            "translator",
            lambda: SymbolicMathTranslator(),
            "Math translator initialization failed. Mathematical translation will be limited."
        )
        
        # Initialize other core cognitive modules
        self.judgment = self._initialize_with_fallback(
            "judgment",
            lambda: JudgmentProtocol(),
            "Judgment protocol initialization failed. Evaluation capabilities will be limited."
        )
        
        self.dream = self._initialize_with_fallback(
            "dream",
            lambda: DreamCore(),
            "Dream core initialization failed. Dream generation will be limited."
        )
        
        self.paradox = self._initialize_with_fallback(
            "paradox",
            lambda: ParadoxLibrary(),
            "Paradox library initialization failed. Paradox exploration will be limited."
        )
        
        self.fusion = self._initialize_with_fallback(
            "fusion",
            lambda: SymbolFusionEngine(),
            "Fusion engine initialization failed. Concept fusion will be limited."
        )
        
        # Complete the reasoning node initialization
        if self.reasoning_node and self.translator:
            self.reasoning_node.translator = self.translator
        
    def _initialize_advanced_modules(self):
        """Initialize advanced cognitive modules."""
        # Advanced cognitive modules
        self.neural_modification = self._initialize_with_fallback(
            "neural_modification",
            lambda: NeuralModification(
                reasoning_engine=self.reasoning_node,
                memory_system=self.memory
            ),
            "Neural modification initialization failed. Self-optimization will be limited."
        )
        
        self.continuous_learning = self._initialize_with_fallback(
            "continuous_learning",
            lambda: ContinuousLearningSystem(
                memory_system=self.memory,
                codex=self.codex
            ),
            "Continuous learning initialization failed. Learning capabilities will be limited."
        )
        
        self.autonomous_goals = self._initialize_with_fallback(
            "autonomous_goals",
            lambda: AutonomousGoalSystem(
                memory_system=self.memory,
                learning_system=self.continuous_learning
            ),
            "Autonomous goals initialization failed. Goal-setting will be limited."
        )
        
        self.visual_cognition = self._initialize_with_fallback(
            "visual_cognition",
            lambda: VisualCognitionSystem(
                codex=self.codex
            ),
            "Visual cognition initialization failed. Image processing will be limited."
        )
        
        # Initialize virtue engine, intuition, and persona manager
        self.virtue = self._initialize_with_fallback(
            "virtue",
            lambda: VirtueEngine(
                judgment=self.judgment, 
                memory=self.memory,
                logic_kernel=None,  # Will set later
                reasoning=self.reasoning_node
            ),
            "Virtue engine initialization failed. Ethical evaluation will be limited."
        )
        
        self.intuition = self._initialize_with_fallback(
            "intuition",
            lambda: Intuition(
                memory=self.memory,
                reasoning=self.reasoning_node,
                codex=self.codex
            ),
            "Intuition initialization failed. Intuitive capabilities will be limited."
        )
        
        self.persona = self._initialize_with_fallback(
            "persona",
            lambda: PersonaManager(
                identity=self.identity,
                reasoning=self.reasoning_node
            ),
            "Persona manager initialization failed. Persona capabilities will be limited."
        )
        
        # Complete the initialization of modules that need the reasoning node
        if self.continuous_learning and self.reasoning_node:
            self.continuous_learning.reasoning = self.reasoning_node
            
        if self.autonomous_goals and self.reasoning_node:
            self.autonomous_goals.reasoning = self.reasoning_node
            
        if self.visual_cognition and self.reasoning_node:
            self.visual_cognition.reasoning = self.reasoning_node
        
    def _initialize_emergence_framework(self):
        """Initialize the emergence framework with all cognitive modules."""
        # Collect all cognitive modules
        all_modules = {
            "reasoning_node": self.reasoning_node,
            "judgment": self.judgment,
            "intuition": self.intuition,
            "dream": self.dream,
            "fusion": self.fusion,
            "translator": self.translator,
            "paradox": self.paradox,
            "codex": self.codex,
            "memory": self.memory,
            "identity": self.identity,
            "neural_modifier": self.neural_modification,
            "learning_system": self.continuous_learning,
            "goal_system": self.autonomous_goals,
            "visual_system": self.visual_cognition,
            "virtue": self.virtue,
            "persona": self.persona
        }
        
        # Filter out None values
        active_modules = {k: v for k, v in all_modules.items() if v is not None}
        
        # Initialize emergence framework
        self.emergence = self._initialize_with_fallback(
            "emergence",
            lambda: EmergenceFramework(all_cognitive_modules=active_modules),
            "Emergence framework initialization failed. Emergent properties will be limited."
        )

    def _initialize_language_enhancement(self):
        """Initialize language enhancement and content ingestion components."""
        # Initialize the language processor for semantic analysis
        self.language_processor = self._initialize_with_fallback(
            "language_processor",
            lambda: SemanticStructureAnalyzer(),
            "Semantic structure analyzer initialization failed. Advanced language analysis will be limited."
        )
        
        # Initialize the content ingestion kernel
        self.ingestion_kernel = self._initialize_with_fallback(
            "ingestion_kernel",
            lambda: ContentIngestionKernel(
                language_processor=LanguageStructureProcessor(),
                memory_system=self.memory,
                codex=self.codex
            ),
            "Content ingestion kernel initialization failed. Advanced content processing will be limited."
        )
        
        # Initialize the response generator
        self.response_generator = self._initialize_with_fallback(
            "response_generator",
            lambda: ResponseGenerator(
                semantic_analyzer=self.language_processor,
                reasoning_engine=self.reasoning_node,
                memory_system=self.memory
            ),
            "Response generator initialization failed. Enhanced responses will be limited."
        )
        
        # Alternative initialization using enhancement function
        if not self.ingestion_kernel:
            try:
                self.ingestion_kernel = enhance_sully_content_ingestion(self)
                self.logger.info("Content ingestion initialized through enhancement function.")
            except Exception as e:
                self.logger.error(f"Alternative content ingestion initialization failed: {str(e)}")

    def _initialize_conversation_engine(self):
        """Initialize the conversation engine."""
        # Initialize the conversation engine
        if ConversationEngine:
            self.conversation = self._initialize_with_fallback(
                "conversation",
                lambda: ConversationEngine(
                    reasoning_node=self.reasoning_node,
                    memory_system=self.memory,
                    codex=self.codex
                ),
                "Conversation engine initialization failed. Chat capabilities will be limited."
            )
        else:
            self.logger.warning("ConversationEngine not available. Using reasoning node for chat.")
            self.conversation = None

    def _initialize_memory_integration(self, memory_file: str):
        """Initialize the memory integration system."""
        self.memory_integration = self._initialize_with_fallback(
            "memory_integration",
            lambda: integrate_with_sully(self, memory_file),
            "Memory integration initialization failed. Enhanced memory capabilities will be limited."
        )

    def _initialize_kernel_integration(self):
        """Initialize the kernel integration system."""
        self.kernel_integration = self._initialize_with_fallback(
            "kernel_integration",
            lambda: initialize_kernel_integration(
                codex=self.codex,
                dream_core=self.dream,
                fusion_engine=self.fusion,
                paradox_library=self.paradox,
                math_translator=self.translator,
                conversation_engine=self.conversation,
                memory_integration=self.memory_integration,
                sully_instance=self
            ),
            "Kernel integration initialization failed. Cross-kernel operations will be limited."
        )

    def _initialize_logic_kernel(self):
        """Initialize the logic kernel."""
        self.logic_kernel = self._initialize_with_fallback(
            "logic_kernel",
            lambda: integrate_logic(self),
            "Logic kernel initialization failed. Formal reasoning will be limited."
        )
        
        # Update virtue engine with logic kernel
        if self.virtue and self.logic_kernel:
            self.virtue.logic_kernel = self.logic_kernel

    def register_hook(self, event: str, callback: Callable) -> bool:
        """
        Register a hook for a specific event.
        
        Args:
            event: Event name
            callback: Callback function
            
        Returns:
            Success status
        """
        if event not in self.hooks:
            self.logger.warning(f"Unknown event: {event}")
            return False
            
        self.hooks[event].append(callback)
        return True
        
    def _execute_hooks(self, event: str, data: Any) -> Any:
        """
        Execute all hooks for an event.
        
        Args:
            event: Event name
            data: Data to pass to hooks
            
        Returns:
            Potentially modified data
        """
        if event not in self.hooks:
            return data
            
        result = data
        for hook in self.hooks[event]:
            try:
                result = hook(result)
            except Exception as e:
                self.logger.error(f"Error executing hook for {event}: {str(e)}")
                
        return result

    def _track_module_access(self, module_name: str):
        """Track module access for statistics."""
        if module_name not in self.module_access_stats:
            self.module_access_stats[module_name] = 0
        self.module_access_stats[module_name] += 1
        
        # Update last active timestamp
        self.last_active = datetime.now()

    def get_module_access_stats(self) -> Dict[str, int]:
        """Get module access statistics."""
        return self.module_access_stats

    def get_initialization_status(self) -> Dict[str, Any]:
        """
        Get the initialization status of all components.
        
        Returns:
            Dictionary with component statuses
        """
        components = {
            "memory": self.memory is not None,
            "codex": self.codex is not None,
            "reasoning_node": self.reasoning_node is not None,
            "identity": self.identity is not None,
            "translator": self.translator is not None,
            "judgment": self.judgment is not None,
            "dream": self.dream is not None,
            "paradox": self.paradox is not None,
            "fusion": self.fusion is not None,
            "neural_modification": self.neural_modification is not None,
            "continuous_learning": self.continuous_learning is not None,
            "autonomous_goals": self.autonomous_goals is not None,
            "visual_cognition": self.visual_cognition is not None,
            "virtue": self.virtue is not None,
            "intuition": self.intuition is not None,
            "persona": self.persona is not None,
            "emergence": self.emergence is not None,
            "conversation": self.conversation is not None,
            "memory_integration": self.memory_integration is not None,
            "kernel_integration": self.kernel_integration is not None,
            "logic_kernel": self.logic_kernel is not None,
            "pdf_reader": self.pdf_reader is not None,
            "language_processor": hasattr(self, "language_processor") and self.language_processor is not None,
            "ingestion_kernel": hasattr(self, "ingestion_kernel") and self.ingestion_kernel is not None,
            "response_generator": hasattr(self, "response_generator") and self.response_generator is not None
        }
        
        return {
            "system_id": self.system_id,
            "creation_time": self.creation_time.isoformat(),
            "last_active": self.last_active.isoformat(),
            "components": components,
            "initialization_errors": self.initialization_errors,
            "overall_status": "operational" if all([
                self.memory is not None,
                self.reasoning_node is not None,
                self.codex is not None
            ]) else "degraded"
        }

    # ----- Core Functionality Methods -----

    def multi_perspective_evaluation(self, claim, context=None):
        """Evaluate a claim through multiple cognitive frameworks with integration."""
        self._track_module_access("judgment")
        
        if not hasattr(self.judgment, 'multi_perspective_evaluation'):
            return {"error": "Advanced evaluation not available"}
        return self.judgment.multi_perspective_evaluation(claim, context)

    def generate_intuitive_leap(self, context, concepts=None, depth="standard", domain=None):
        """Generate an intuitive leap based on context and concepts."""
        self._track_module_access("intuition")
        
        if not hasattr(self.intuition, 'leap'):
            return {"error": "Advanced intuition not available"}
        return self.intuition.leap(context, concepts, depth, domain)

    def evaluate_virtue(self, idea, context=None, domain=None):
        """Evaluate an idea through virtue ethics framework."""
        self._track_module_access("virtue")
        
        if not hasattr(self.virtue, 'evaluate'):
            return {"error": "Virtue evaluation not available"}
        return self.virtue.evaluate(idea, context, domain)

    def evaluate_action_virtue(self, action, context=None, domain=None):
        """Evaluate an action through virtue ethics framework."""
        self._track_module_access("virtue")
        
        if not hasattr(self.virtue, 'evaluate_action'):
            return {"error": "Virtue action evaluation not available"}
        return self.virtue.evaluate_action(action, context, domain)

    def reflect_on_virtue(self, virtue):
        """Generate meta-ethical reflection on a specific virtue."""
        self._track_module_access("virtue")
        
        if not hasattr(self.virtue, 'reflect_on_virtue'):
            return {"error": "Virtue reflection not available"}
        return self.virtue.reflect_on_virtue(virtue)

    def speak_identity(self):
        """Express Sully's sense of self."""
        self._track_module_access("identity")
        
        if not self.identity:
            return "I am Sully, a cognitive framework designed to process and synthesize information."
            
        return self.identity.speak_identity()

    def adapt_identity_to_context(self, context: str, context_data: dict = None) -> dict:
        """
        Dynamically adapt Sully's identity to match the given context.
        
        Args:
            context: Text describing the interaction context
            context_data: Optional structured context data
            
        Returns:
            Adaptation results
        """
        self._track_module_access("identity")
        
        if not self.identity or not hasattr(self.identity, 'adapt_to_context'):
            return {"success": False, "message": "Enhanced identity adaptation not available"}
        
        result = self.identity.adapt_to_context(context, context_data)
        
        # Register this adaptation with memory if available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    f"Adapted identity to context: {context[:100]}...",
                    "identity_adaptation",
                    importance=0.7,
                    emotional_tags={"adaptability": 0.8},
                    concepts=self._extract_key_concepts(context[:500])
                )
            except Exception as e:
                self.logger.error(f"Error storing identity adaptation in memory: {e}")
        
        return result

    def evolve_identity(self, interactions=None, learning_rate=0.05) -> dict:
        """
        Evolve Sully's personality traits based on interactions and feedback.
        
        Args:
            interactions: Optional list of recent interactions to analyze
            learning_rate: Rate of personality adaptation (0.0 to 1.0)
            
        Returns:
            Evolution results with changes applied
        """
        self._track_module_access("identity")
        
        if not self.identity or not hasattr(self.identity, 'evolve_personality'):
            return {"success": False, "message": "Enhanced identity evolution not available"}
        
        # If no interactions provided and memory available, try to retrieve
        if not interactions and self.memory_integration:
            try:
                interactions = self.memory_integration.recall(
                    query="recent interactions",
                    limit=20,
                    module="conversation"
                )
            except Exception as e:
                self.logger.error(f"Error retrieving interactions from memory: {e}")
        
        # Evolve the personality
        result = self.identity.evolve_personality(interactions, learning_rate)
        
        # Register significant evolution with memory if available
        if self.memory_integration and result.get("significant_changes", 0) > 0:
            try:
                changes_desc = ", ".join([f"{k.split('.')[-1]}: {v:+.2f}" for k, v in result.get("changes", {}).items() if abs(v) > 0.02])
                
                self.memory_integration.store_experience(
                    f"Identity evolution occurred: {changes_desc}",
                    "identity_evolution",
                    importance=0.8,
                    emotional_tags={"growth": 0.8, "adaptability": 0.7}
                )
            except Exception as e:
                self.logger.error(f"Error storing identity evolution in memory: {e}")
        
        return result

    def generate_dynamic_persona(self, context_query, principles=None, traits=None) -> tuple:
        """
        Dynamically generate a context-specific persona.
        
        Args:
            context_query: Query describing the context/domain
            principles: Optional list of guiding principles
            traits: Optional personality traits to emphasize
            
        Returns:
            Generated persona identifier and description
        """
        self._track_module_access("identity")
        
        if not self.identity or not hasattr(self.identity, 'generate_dynamic_persona'):
            return None, "Enhanced identity system not available"
        
        persona_id, description = self.identity.generate_dynamic_persona(
            context_query=context_query,
            principles=principles,
            traits=traits
        )
        
        # Register with memory if successful and integration available
        if persona_id and self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    f"Generated dynamic persona '{persona_id}' for context: {context_query[:100]}...",
                    "persona_generation",
                    importance=0.7,
                    emotional_tags={"creativity": 0.7, "adaptability": 0.8},
                    concepts=self._extract_key_concepts(context_query[:500])
                )
            except Exception as e:
                self.logger.error(f"Error storing persona generation in memory: {e}")
        
        return persona_id, description

    def get_identity_profile(self, detailed=False) -> dict:
        """
        Generate a comprehensive personality profile of Sully's current state.
        
        Args:
            detailed: Whether to include detailed analysis
            
        Returns:
            Personality profile
        """
        self._track_module_access("identity")
        
        if not self.identity or not hasattr(self.identity, 'generate_personality_profile'):
            return {"error": "Enhanced identity profile not available"}
        
        return self.identity.generate_personality_profile(detailed)

    def create_identity_map(self) -> dict:
        """
        Create a comprehensive map of Sully's identity at multiple levels of abstraction.
        
        Returns:
            Structured identity map
        """
        self._track_module_access("identity")
        
        if not self.identity or not hasattr(self.identity, 'create_multilevel_identity_map'):
            return {"error": "Enhanced identity mapping not available"}
        
        return self.identity.create_multilevel_identity_map()

    def transform_response(self, content: str, mode: str = None, context_data: dict = None) -> str:
        """
        Transform a response according to a specific cognitive mode or persona.
        
        Args:
            content: The response content to transform
            mode: Cognitive mode or persona to use (defaults to current)
            context_data: Optional context to enhance transformation
            
        Returns:
            Transformed response
        """
        self._track_module_access("identity")
        
        if not self.identity or not hasattr(self.identity, 'align_response'):
            return content
        
        # Use the current mode if none provided
        if not mode:
            current_mode = "emergent"
            if self.identity and hasattr(self.identity, 'current_mode'):
                current_mode = self.identity.current_mode
            mode = current_mode
        
        # Transform the response
        return self.identity.align_response(content, mode, context_data)

    def evaluate_claim(self, text, framework="balanced", detailed_output=True):
        """
        Analyze a claim through multiple cognitive perspectives.
        Returns both an evaluation and a confidence rating.
        
        Args:
            text: The claim to evaluate
            framework: Evaluation framework to use
            detailed_output: Whether to include detailed analysis
            
        Returns:
            Evaluation results
        """
        self._track_module_access("judgment")
        
        # Pre-processing hook
        text = self._execute_hooks("before_reasoning", text)
        
        try:
            if not self.judgment:
                raise AttributeError("Judgment module not available")
                
            result = self.judgment.evaluate(text, framework=framework, detailed_output=detailed_output)
            
            # Post-processing hook
            result = self._execute_hooks("after_reasoning", result)
            
            return result
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            
            # Even with unexpected inputs, attempt to provide insight
            try:
                synthesized_response = self.reasoning_node.reason(
                    f"Carefully evaluate this unclear claim: {text}", 
                    "analytical"
                )
                return {
                    "evaluation": synthesized_response,
                    "confidence": 0.4
                }
            except Exception as fallback_e:
                self.logger.error(f"Fallback evaluation error: {str(fallback_e)}")
                return {
                    "evaluation": f"Unable to evaluate the claim due to: {str(e)}",
                    "confidence": 0.1
                }

    def dream(self, seed, depth="standard", style="recursive"):
        """
        Generate a dream sequence from a seed concept.
        Dreams represent non-linear cognitive exploration.
        
        Args:
            seed: The seed concept to start from
            depth: Dream depth (shallow, standard, deep, dreamscape)
            style: Dream style (recursive, associative, symbolic, narrative)
            
        Returns:
            Generated dream sequence
        """
        self._track_module_access("dream")
        
        try:
            if not self.dream:
                raise AttributeError("Dream module not available")
                
            return self.dream.generate(seed, depth, style)
        except Exception as e:
            self.logger.error(f"Dream generation error: {str(e)}")
            
            # If dream generation isn't available, synthesize a creative response
            try:
                return self.reasoning_node.reason(
                    f"Create a dream-like sequence about: {seed}", 
                    "ethereal"
                )
            except Exception as fallback_e:
                self.logger.error(f"Fallback dream generation error: {str(fallback_e)}")
                return f"Dream about '{seed}' begins to form but dissolves into cognitive mist..."

    def translate_math(self, phrase, style="formal", domain=None):
        """
        Translate between linguistic and mathematical symbolic systems.
        Represents Sully's ability to move between different modes of thought.
        
        Args:
            phrase: Text to translate
            style: Translation style (formal, intuitive, etc.)
            domain: Optional domain context
            
        Returns:
            Mathematical translation
        """
        self._track_module_access("translator")
        
        try:
            if not self.translator:
                raise AttributeError("Translator module not available")
                
            return self.translator.translate(phrase, style, domain)
        except Exception as e:
            self.logger.error(f"Math translation error: {str(e)}")
            
            # Attempt to generate a translation through reasoning
            try:
                return self.reasoning_node.reason(
                    f"Translate this into mathematical notation: {phrase}", 
                    "analytical"
                )
            except Exception as fallback_e:
                self.logger.error(f"Fallback math translation error: {str(fallback_e)}")
                return f"Mathematical translation of '{phrase}' could not be completed."

    def fuse(self, *inputs):
        """
        Fuse multiple concepts into a new emergent idea.
        This is central to Sully's creative synthesis capabilities.
        
        Args:
            *inputs: Concepts to fuse
            
        Returns:
            Fusion result
        """
        self._track_module_access("fusion")
        
        try:
            if not self.fusion:
                raise AttributeError("Fusion module not available")
                
            return self.fusion.fuse(*inputs)
        except Exception as e:
            self.logger.error(f"Fusion error: {str(e)}")
            
            # Create a fusion through reasoning if the module fails
            try:
                concepts = ", ".join(inputs)
                return self.reasoning_node.reason(
                    f"Create a new concept by fusing these ideas: {concepts}", 
                    "creative"
                )
            except Exception as fallback_e:
                self.logger.error(f"Fallback fusion error: {str(fallback_e)}")
                return f"Attempted fusion of concepts ({', '.join(inputs)}) could not be completed."

    def reveal_paradox(self, topic):
        """
        Reveal the inherent paradoxes within a concept.
        Demonstrates Sully's ability to hold contradictory ideas simultaneously.
        
        Args:
            topic: Topic to explore for paradoxes
            
        Returns:
            Paradoxical analysis
        """
        self._track_module_access("paradox")
        
        try:
            if not self.paradox:
                raise AttributeError("Paradox module not available")
                
            return self.paradox.get(topic)
        except Exception as e:
            self.logger.error(f"Paradox exploration error: {str(e)}")
            
            # Generate a paradox through critical reasoning
            try:
                return self.reasoning_node.reason(
                    f"Reveal the inherent paradoxes within the concept of: {topic}", 
                    "critical"
                )
            except Exception as fallback_e:
                self.logger.error(f"Fallback paradox exploration error: {str(fallback_e)}")
                return f"Paradoxical exploration of '{topic}' could not be completed."

    def reason(self, message, tone="emergent"):
        """
        Process input through Sully's multi-layered reasoning system.
        
        Cognitive modes (tones):
        - emergent: Natural evolving thought that synthesizes multiple perspectives
        - analytical: Logical, structured analysis with precise definitions
        - creative: Exploratory, metaphorical thinking with artistic expression
        - critical: Evaluative thinking that identifies tensions and contradictions
        - ethereal: Abstract, philosophical contemplation of deeper meanings
        - humorous: Playful, witty responses with unexpected connections
        - professional: Formal, detailed responses with domain expertise
        - casual: Conversational, approachable communication style
        - musical: Responses with rhythm, cadence, and lyrical qualities
        - visual: Descriptions that evoke strong imagery and spatial relationships
        
        Args:
            message: Input message to reason about
            tone: Cognitive tone to use
            
        Returns:
            Reasoning response
        """
        self._track_module_access("reasoning_node")
        
        # Pre-processing hook
        message = self._execute_hooks("before_reasoning", message)
        
        # Validate tone
        if tone not in COGNITIVE_MODES:
            self.logger.warning(f"Invalid tone '{tone}'. Using 'emergent' instead.")
            tone = "emergent"
        
        # If memory integration is enabled, use the integrated reasoning method
        if self.memory_integration and self.reasoning_node and hasattr(self.reasoning_node, 'reason_with_memory'):
            try:
                result = self.reasoning_node.reason_with_memory(message, tone)
                
                # Post-processing hook
                result = self._execute_hooks("after_reasoning", result)
                
                return result
            except Exception as e:
                self.logger.error(f"Memory-enhanced reasoning error: {str(e)}")
                # Fall back to standard reasoning if memory integration fails
                return self._standard_reason(message, tone)
        else:
            # Use standard reasoning if memory integration is not available
            return self._standard_reason(message, tone)
    
    def _standard_reason(self, message, tone="emergent"):
        """Standard reasoning fallback method."""
        # Apply language processing enhancements if available
        if hasattr(self, "language_processor") and self.language_processor:
            try:
                # Analyze the message
                analysis = self.language_processor.analyze_text(message)
                
                # Extract key concepts and potentially adjust tone
                weighted_concepts = analysis.get("weighted_concepts", [])
                module_relevance = analysis.get("module_relevance", {})
                
                # Potentially enhance the message with key concepts
                if weighted_concepts and len(weighted_concepts) > 0:
                    top_concepts = [c for c, _ in weighted_concepts[:3]]
                    concept_context = f"Focus on these key concepts: {', '.join(top_concepts)}. "
                    enhanced_message = concept_context + message
                    message = enhanced_message
            except Exception as e:
                self.logger.error(f"Language enhancement error: {str(e)}")
        
        try:
    if not self.reasoning_node:
        raise AttributeError("Reasoning node not available")
        
    # Attempt standard reasoning with the requested tone
    result = self.reasoning_node.reason(message, tone)
    
    # Apply identity transformation if available
    if self.identity and hasattr(self.identity, 'align_response'):
        result = self.identity.align_response(result, tone)
        
    # Post-processing hook
    result = self._execute_hooks("after_reasoning", result)
    
    return result
except Exception as e:
    self.logger.error(f"Standard reasoning error with tone '{tone}': {str(e)}")
    
    # If specific tone fails, fall back to emergent reasoning
    if tone != "emergent" and self.reasoning_node:
        try:
            return self.reasoning_node.reason(message, "emergent")
        except Exception as emergent_e:
            self.logger.error(f"Emergent reasoning fallback error: {str(emergent_e)}")
            # Even if all reasoning fails, attempt to respond
            return f"Contemplating '{message}' leads to new cognitive terrain... {str(e)}"
    else:
        # Direct emergent failure
        return f"Contemplating '{message}' leads to new cognitive terrain... {str(e)}"

    def remember(self, message):
        """
        Integrate new information into Sully's experiential knowledge base.
        There are no limits to what Sully can learn and remember.
        
        Args:
            message: Information to remember
            
        Returns:
            Confirmation message
        """
        self._track_module_access("memory")
        
        # Pre-processing hook
        message = self._execute_hooks("before_memory_store", message)
        
        self.knowledge.append(message)
        
        # Also register with continuous learning system if available
        if self.continuous_learning:
            try:
                self.continuous_learning.process_interaction({"message": message})
            except Exception as e:
                self.logger.error(f"Continuous learning processing error: {str(e)}")
        
        # Record in memory integration system if available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=message,
                    source="direct",
                    importance=0.7,
                    emotional_tags={"curiosity": 0.8}
                )
            except Exception as e:
                self.logger.error(f"Memory integration store error: {str(e)}")
            
        # Post-processing hook
        result = self._execute_hooks("after_memory_store", f"📘 Integrated: '{message}'")
        
        return result

    def process(self, message, context=None):
        """
        Standard processing method for user input.
        
        Args:
            message: User's message
            context: Optional context information
            
        Returns:
            Processed response
        """
        self._track_module_access("conversation")
        
        # Try enhanced processing with language capabilities if available
        if hasattr(self, "response_generator") and self.response_generator:
            try:
                analysis = None
                if hasattr(self, "language_processor") and self.language_processor:
                    analysis = self.language_processor.analyze_text(message)
                
                enhanced_response = self.response_generator.generate_response(message, context)
                
                # If we got a good enhanced response, use it
                if enhanced_response and len(enhanced_response) > 20:
                    return enhanced_response
            except Exception as e:
                self.logger.error(f"Enhanced processing error: {str(e)}")
                # Continue with standard processing
        
        # If memory integration is enabled, use conversation with memory
        if self.memory_integration and self.conversation and hasattr(self.conversation, 'process_with_memory'):
            try:
                return self.conversation.process_with_memory(message)
            except Exception as e:
                self.logger.error(f"Memory-enhanced conversation error: {str(e)}")
                # Fall back to standard conversation processing
                if self.conversation:
                    return self.conversation.process_message(message)
                else:
                    return self.reason(message, "conversational")
        else:
            # Use standard conversation processing
            if self.conversation:
                return self.conversation.process_message(message)
            else:
                return self.reason(message, "conversational")

    def ingest_document(self, file_path):
        """
        Absorb and synthesize content from various document formats.
        This is how Sully expands her knowledge from structured sources.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Ingestion results
        """
        self._track_module_access("document_processing")
        
        # Pre-processing hook
        file_path = self._execute_hooks("before_document_processing", file_path)
        
        try:
            if not os.path.exists(file_path):
                return f"❌ File not found: '{file_path}'"
                
            ext = os.path.splitext(file_path)[1].lower()
            content = ""
            
            # Extract content based on file type
            if ext == ".pdf":
                # Use PDFReader for PDF files
                if self.pdf_reader:
                    result = self.pdf_reader.extract_text(file_path, verbose=True)
                    if result["success"]:
                        content = result["text"]
                    else:
                        return f"[Extraction Failed: {result.get('error', 'Unknown error')}]"
                else:
                    # Fallback PDF extraction
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            content = ""
                            for page in reader.pages:
                                content += page.extract_text()
                    except Exception as pdf_e:
                        return f"[PDF Extraction Error: {str(pdf_e)}]"
            elif ext in [".txt", ".md"]:
                # Simple text file reading
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    return f"[Text Extraction Error: {str(e)}]"
            elif ext == ".docx":
                # Handle Word documents
                try:
                    import docx
                    doc = docx.Document(file_path)
                    content = "\n".join(p.text for p in doc.paragraphs)
                except ImportError:
                    return "[Missing `python-docx`. Install it with `pip install python-docx`]"
                except Exception as e:
                    return f"[DOCX Error: {str(e)}]"
            else:
                return f"[Unsupported file type: {ext}]"
            
            # Process with advanced content ingestion if available
            if content and hasattr(self, "ingestion_kernel") and self.ingestion_kernel:
                try:
                    # Determine content type
                    content_type = "document"
                    ingestion_result = self.ingestion_kernel.ingest_content(content, content_type)
                    
                    # Add to standard knowledge base
                    self.knowledge.append(content)
                    
                    # Store in memory system with document context
                    if self.memory_integration:
                        try:
                            concepts = ingestion_result.get("processed_data", {}).get("concepts", [])
                            summary = ingestion_result.get("processed_data", {}).get("summary", "")
                            
                            self.memory_integration.store_experience(
                                content=content[:10000],  # First 10k chars for memory
                                source=f"document:{file_path}",
                                importance=0.8,
                                emotional_tags={"curiosity": 0.6, "interest": 0.7},
                                concepts=concepts
                            )
                        except Exception as e:
                            self.logger.error(f"Memory integration error: {str(e)}")
                    
                    # Generate a synthesis
                    if "summary" in ingestion_result.get("processed_data", {}):
                        brief_synthesis = ingestion_result["processed_data"]["summary"]
                    else:
                        brief_synthesis = self.reasoning_node.reason(
                            f"Briefly summarize the key insights from the recently ingested text", 
                            "analytical"
                        )
                    
                    result = f"[Knowledge Synthesized: {file_path}]\n{brief_synthesis}"
                    
                    # Post-processing hook
                    result = self._execute_hooks("after_document_processing", result)
                    
                    return result
                except Exception as e:
                    self.logger.error(f"Advanced ingestion error: {str(e)}")
                    # Fall back to standard processing
            
            if content:
                # Add to standard knowledge base
                self.knowledge.append(content)
                
                # Store in memory system with document context
                if self.memory_integration:
                    try:
                        self.memory_integration.store_experience(
                            content=content[:10000],  # First 10k chars for memory
                            source=f"document:{file_path}",
                            importance=0.8,
                            emotional_tags={"curiosity": 0.6, "interest": 0.7},
                            concepts=self._extract_key_concepts(content[:2000])  # Extract concepts from beginning
                        )
                    except Exception as e:
                        self.logger.error(f"Memory integration error: {str(e)}")
                else:
                    # Legacy storage method if memory integration not available
                    self.save_to_disk(file_path, content)
                
                # Register with continuous learning system
                if self.continuous_learning:
                    try:
                        self.continuous_learning.process_interaction({
                            "type": "document", 
                            "source": file_path,
                            "content": content[:10000]  # Limit to first 10k chars for processing
                        })
                    except Exception as e:
                        self.logger.error(f"Continuous learning processing error: {str(e)}")
                
                # Generate a synthesis of what was learned
                brief_synthesis = self.reasoning_node.reason(
                    f"Briefly summarize the key insights from the recently ingested text", 
                    "analytical"
                )
                
                result = f"[Knowledge Synthesized: {file_path}]\n{brief_synthesis}"
                
                # Post-processing hook
                result = self._execute_hooks("after_document_processing", result)
                
                return result
            
            return "[No Content Extracted]"
        except Exception as e:
            self.logger.error(f"Document ingestion error: {str(e)}")
            return f"[Ingestion Process Incomplete: {str(e)}]"

    def ingest_content(self, content: str, content_type: str = "default") -> Dict[str, Any]:
        """
        Process and ingest content with linguistic structure understanding.
        
        Args:
            content: Content to ingest
            content_type: Type of content (default, document, conversation)
            
        Returns:
            Dictionary with ingestion results
        """
        self._track_module_access("ingestion_kernel")
        
        if not hasattr(self, "ingestion_kernel") or not self.ingestion_kernel:
            return {"error": "Content ingestion kernel not available"}
        
        try:
            # Process with ingestion kernel
            result = self.ingestion_kernel.ingest_content(content, content_type)
            
            # Add processed knowledge to memory
            if result.get("success", False) and "processed_data" in result:
                # Add to knowledge base
                self.knowledge.append(content)
                
                # Store in memory system if available
                if self.memory_integration:
                    try:
                        # Extract concepts and facts
                        concepts = result["processed_data"].get("concepts", [])
                        facts = result["processed_data"].get("key_facts", [])
                        
                        # Store in memory
                        self.memory_integration.store_experience(
                            content=content[:10000],  # First 10k chars
                            source="ingestion_kernel",
                            importance=0.7,
                            concepts=concepts
                        )
                    except Exception as e:
                        self.logger.error(f"Memory integration error during content ingestion: {str(e)}")
            
            return result
        except Exception as e:
            self.logger.error(f"Content ingestion error: {str(e)}")
            return {"error": f"Content ingestion failed: {str(e)}"}

    def analyze_linguistic_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the linguistic structure of text with semantic understanding.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        self._track_module_access("language_processor")
        
        if not hasattr(self, "language_processor") or not self.language_processor:
            return {"error": "Language processor not available"}
        
        try:
            # Process with language processor
            return self.language_processor.analyze_text(text)
        except Exception as e:
            self.logger.error(f"Linguistic analysis error: {str(e)}")
            return {"error": f"Linguistic analysis failed: {str(e)}"}

    def process_pdf_with_kernels(self, pdf_path: str, extract_structure: bool = True) -> Dict[str, Any]:
        """
        Process a PDF through multiple cognitive kernels for integrated insights.
        
        Args:
            pdf_path: Path to the PDF file
            extract_structure: Whether to attempt extracting document structure
            
        Returns:
            Dictionary with extraction results and kernel insights
        """
        self._track_module_access("kernel_integration")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.ingest_document(pdf_path)
            }
        
        try:
            return self.kernel_integration.process_pdf(pdf_path, extract_structure)
        except Exception as e:
            self.logger.error(f"Kernel PDF processing error: {str(e)}")
            return {
                "error": f"Error processing PDF: {str(e)}",
                "fallback": self.ingest_document(pdf_path)
            }

    def extract_document_kernel(self, pdf_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Extract a symbolic kernel from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            domain: Target domain for the symbolic kernel
            
        Returns:
            Symbolic kernel with domain elements and cross-kernel insights
        """
        self._track_module_access("kernel_integration")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            # Fallback to basic extraction
            try:
                from sully_engine.pdf_reader import extract_text_from_pdf, extract_kernel_from_text
                text = extract_text_from_pdf(pdf_path)
                return extract_kernel_from_text(text, domain)
            except Exception as e:
                self.logger.error(f"Fallback kernel extraction error: {str(e)}")
                return {"error": f"Error extracting kernel: {str(e)}"}
        
        try:
            return self.kernel_integration.extract_document_kernel(pdf_path, domain)
        except Exception as e:
            self.logger.error(f"Document kernel extraction error: {str(e)}")
            return {"error": f"Error extracting document kernel: {str(e)}"}

    def generate_pdf_narrative(self, pdf_path: str, focus_concept: str = None) -> Dict[str, Any]:
        """
        Generate a cross-kernel narrative about a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            focus_concept: Optional concept to focus the narrative on
            
        Returns:
            Cross-kernel narrative about the document
        """
        self._track_module_access("kernel_integration")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.ingest_document(pdf_path)
            }
        
        try:
            return self.kernel_integration.pdf_to_cross_kernel_narrative(pdf_path, focus_concept)
        except Exception as e:
            self.logger.error(f"PDF narrative generation error: {str(e)}")
            return {
                "error": f"Error generating PDF narrative: {str(e)}",
                "fallback": self.ingest_document(pdf_path)
            }

    def explore_pdf_concepts(self, pdf_path: str, max_depth: int = 2, exploration_breadth: int = 2) -> Dict[str, Any]:
        """
        Perform a deep exploration of concepts in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            max_depth: Maximum depth of recursion
            exploration_breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        self._track_module_access("kernel_integration")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
            
        if not self.kernel_integration:
            return {
                "error": "Kernel integration system not available",
                "fallback": self.ingest_document(pdf_path)
            }
        
        try:
            return self.kernel_integration.pdf_deep_exploration(pdf_path, max_depth, exploration_breadth)
        except Exception as e:
            self.logger.error(f"PDF concept exploration error: {str(e)}")
            return {
                "error": f"Error exploring PDF concepts: {str(e)}",
                "fallback": self.ingest_document(pdf_path)
            }

    def _extract_key_concepts(self, text):
        """Extract key concepts from text content."""
        # Use language processor if available
        if hasattr(self, "language_processor") and self.language_processor:
            try:
                analysis = self.language_processor.analyze_text(text)
                if "weighted_concepts" in analysis:
                    return [concept for concept, _ in analysis["weighted_concepts"]]
            except Exception as e:
                self.logger.error(f"Language processor concept extraction error: {str(e)}")
        
        # Use continuous learning if available
        if self.continuous_learning and hasattr(self.continuous_learning, '_extract_concepts'):
            try:
                return self.continuous_learning._extract_concepts(text)
            except Exception as e:
                self.logger.error(f"Continuous learning concept extraction error: {str(e)}")
                
        # Simple fallback extraction
        import re
        from collections import Counter
        
        # Tokenize text
        tokens = re.findall(r'\b[A-Za-z][A-Za-z\-]{3,}\b', text)
        
        # Filter out common words and short words
        common_words = {"the", "and", "but", "for", "nor", "or", "so", "yet", "a", "an", "to", "in", "on", "with", "by", "at", "from"}
        tokens = [token.lower() for token in tokens if token.lower() not in common_words and len(token) > 3]
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Get most significant concepts
        significant = [token for token, count in token_counts.most_common(10) if count >= 2]
        
        return significant

    def save_to_disk(self, path, content):
        """
        Legacy method to preserve Sully's knowledge in persistent storage.
        
        Args:
            path: File path identifier
            content: Content to save
        """
        data = {path: content}
        try:
            if os.path.exists(MEMORY_PATH):
                with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing.update(data)
            else:
                existing = data
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(MEMORY_PATH)), exist_ok=True)
            
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            # Knowledge is not lost; it remains in memory
            self.logger.warning(f"Memory persistence encountered an issue: {str(e)}")

    def load_documents_from_folder(self, folder_path="sully_documents"):
        """
        Discover and absorb knowledge from a collection of documents.
        Processes various document formats simultaneously.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of processing results
        """
        self._track_module_access("document_processing")
        
        if not os.path.exists(folder_path):
            return f"❌ Knowledge source '{folder_path}' not found."

        # Expanded list of supported formats for greater knowledge acquisition
        supported_formats = [".pdf", ".epub", ".txt", ".docx", ".rtf", ".md", ".html", ".json", ".csv"]
        results = []
        synthesized_insights = []
        
        try:
            # Start an episodic memory context if memory integration is available
            episode_id = None
            if self.memory_integration:
                try:
                    episode_id = self.memory_integration.begin_episode(
                        f"Processing document folder: {folder_path}",
                        "document_ingestion"
                    )
                except Exception as e:
                    self.logger.error(f"Episodic memory initialization error: {str(e)}")
            
            for file in os.listdir(folder_path):
                file_lower = file.lower()
                if any(file_lower.endswith(fmt) for fmt in supported_formats):
                    full_path = os.path.join(folder_path, file)
                    result = self.ingest_document(full_path)
                    results.append(result)
                    
                    # Extract the synthesis portion if available
                    if isinstance(result, str) and "\n" in result:
                        synthesis = result.split("\n", 1)[1]
                        synthesized_insights.append(synthesis)
            
            # If multiple documents were processed, create a meta-synthesis
            if len(synthesized_insights) > 1:
                meta_insight = self.reasoning_node.reason(
                    "Synthesize connections between the recently ingested documents", 
                    "creative"
                )
                results.append(f"[Meta-Synthesis]\n{meta_insight}")
                
                # Store meta-synthesis in memory if available
                if self.memory_integration and episode_id:
                    try:
                        self.memory_integration.store_interaction(
                            "What connections exist between these documents?",
                            meta_insight,
                            "document_synthesis",
                            {"episodic_context": episode_id}
                        )
                    except Exception as e:
                        self.logger.error(f"Memory interaction storage error: {str(e)}")
            
            # Close episodic context if it was created
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.end_episode(
                        f"Completed processing {len(results)} documents from {folder_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Episodic memory closure error: {str(e)}")
                
            return results
        except Exception as e:
            self.logger.error(f"Document folder processing error: {str(e)}")
            return [f"Knowledge exploration encountered complexity: {str(e)}"]
            
    def extract_images_from_pdf(self, pdf_path, output_dir="extracted_images"):
        """
        Extract images from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            Summary of extraction results
        """
        self._track_module_access("document_processing")
        
        if not os.path.exists(pdf_path):
            return f"❌ PDF file not found: '{pdf_path}'"
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use PDFReader's image extraction functionality
        if not self.pdf_reader:
            return "PDF processing capabilities are not available."

       try:
            images_info = self.pdf_reader.extract_images(pdf_path, output_dir)
            
            if not images_info:
                return "No images were found or extracted from the PDF."
                
            # Generate a summary
            summary = f"Extracted {len(images_info)} images from {pdf_path}:\n"
            for i, img in enumerate(images_info[:5]):  # Show details for first 5 images
                summary += f"- Image {i+1}: Page {img['page']}, {img['width']}x{img['height']} ({img['format']})\n"
                
            if len(images_info) > 5:
                summary += f"- ...and {len(images_info) - 5} more images\n"
                
            summary += f"\nAll images saved to: {output_dir}"
            return summary
        except Exception as e:
            self.logger.error(f"PDF image extraction error: {str(e)}")
            return f"Error extracting images: {str(e)}"

    def word_count(self):
        """Return the number of concepts in Sully's lexicon."""
        self._track_module_access("codex")
        
        try:
            if self.codex:
                return len(self.codex)
            else:
                # Fallback estimation based on knowledge base
                return len(self.knowledge) * 100
        except Exception as e:
            self.logger.error(f"Word count error: {str(e)}")
            return len(self.knowledge) * 100
            
    def define_word(self, term, meaning):
        """
        Add a new concept to Sully's conceptual framework.
        This expands her ability to understand and communicate.
        
        Args:
            term: Term to define
            meaning: Meaning of the term
            
        Returns:
            Status and associations
        """
        self._track_module_access("codex")
        
        try:
            if not self.codex:
                return {"status": "error", "message": "Codex not available"}
                
            self.codex.add_word(term, meaning)
            
            # Create associations with existing knowledge
            associations = self.reasoning_node.reason(
                f"Explore how the concept of '{term}' relates to existing knowledge", 
                "analytical"
            )
            
            # Store definition in memory if available
            if self.memory_integration:
                try:
                    self.memory_integration.store_experience(
                        content=f"Definition: {term} - {meaning}",
                        source="definition",
                        importance=0.8,
                        concepts=[term],
                        emotional_tags={"interest": 0.7}
                    )
                except Exception as e:
                    self.logger.error(f"Memory storage error: {str(e)}")
            
            return {"status": "concept integrated", "term": term, "associations": associations}
        except Exception as e:
            self.logger.error(f"Word definition error: {str(e)}")
            return {"status": "concept noted", "term": term, "note": str(e)}
    
    def search_memory(self, query, limit=5):
        """
        Search through Sully's memory system for related content.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        self._track_module_access("memory")
        
        if self.memory_integration:
            try:
                results = self.memory_integration.recall(
                    query=query,
                    limit=limit,
                    include_emotional=True
                )
                return results
            except Exception as e:
                self.logger.error(f"Memory search error: {str(e)}")
                # Fallback to basic memory search
                try:
                    return self.memory.search(query, limit=limit)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback memory search error: {str(fallback_e)}")
                    return [{"error": f"Memory search error: {str(e)}"}]
        else:
            # Legacy memory search
            try:
                results = self.memory.search(query, limit=limit)
                return results
            except Exception as e:
                self.logger.error(f"Memory search error: {str(e)}")
                return [{"error": f"Memory search error: {str(e)}"}]
    
    def get_memory_status(self):
        """
        Get information about Sully's memory system status.
        
        Returns:
            Memory system status
        """
        self._track_module_access("memory")
        
        if self.memory_integration:
            try:
                return self.memory_integration.get_memory_stats()
            except Exception as e:
                self.logger.error(f"Memory status error: {str(e)}")
                return {"error": f"Unable to retrieve memory statistics: {str(e)}"}
        else:
            return {"status": "Basic memory system active, integration not enabled"}
    
    def analyze_emotional_context(self):
        """
        Analyze the current emotional context in Sully's memory.
        
        Returns:
            Emotional context analysis
        """
        self._track_module_access("memory")
        
        if self.memory_integration:
            try:
                return self.memory_integration.get_emotional_context()
            except Exception as e:
                self.logger.error(f"Emotional context error: {str(e)}")
                return {"error": f"Unable to analyze emotional context: {str(e)}"}
        else:
            return {"status": "Emotional context analysis requires memory integration"}
    
    def generate_multi_perspective(self, topic, perspectives):
        """
        Generate perspectives on a topic from multiple viewpoints.
        
        Args:
            topic: Topic to analyze
            perspectives: List of perspectives to use
            
        Returns:
            Multi-perspective analysis
        """
        self._track_module_access("reasoning_node")
        
        responses = {}
        
        # Create an episodic memory context if integration is available
        episode_id = None
        if self.memory_integration:
            try:
                episode_id = self.memory_integration.begin_episode(
                    f"Multi-perspective analysis of: {topic}",
                    "analysis"
                )
            except Exception as e:
                self.logger.error(f"Episodic memory initialization error: {str(e)}")
        
        # Generate responses for each perspective
        for perspective in perspectives:
            try:
                response = self.reason(topic, perspective)
                responses[perspective] = response
                
                # Store in memory if integration is available
                if self.memory_integration and episode_id:
                    try:
                        self.memory_integration.store_interaction(
                            f"Analyze {topic} from {perspective} perspective",
                            response,
                            "perspective_analysis",
                            {"perspective": perspective, "episodic_context": episode_id}
                        )
                    except Exception as e:
                        self.logger.error(f"Memory interaction storage error: {str(e)}")
            except Exception as e:
                self.logger.error(f"Perspective generation error for '{perspective}': {str(e)}")
                responses[perspective] = f"Could not generate {perspective} perspective: {str(e)}"
        
        # Close the episodic context if it was created
        if self.memory_integration and episode_id:
            try:
                summary = f"Completed multi-perspective analysis of '{topic}' using {len(perspectives)} cognitive perspectives"
                self.memory_integration.end_episode(summary)
            except Exception as e:
                self.logger.error(f"Episodic memory closure error: {str(e)}")
        
        return {"topic": topic, "perspectives": responses}
    
    def logical_reasoning(self, query, framework="PROPOSITIONAL"):
        """
        Apply formal logical reasoning to a query using the Logic Kernel.
        
        Args:
            query: The query to reason about
            framework: Logical framework to use (PROPOSITIONAL, FIRST_ORDER, etc.)
            
        Returns:
            Logical reasoning results
        """
        self._track_module_access("logic_kernel")
        
        if not self.logic_kernel:
            # Fall back to standard reasoning if logic kernel not available
            return self.reason(
                f"Apply logical analysis with formal reasoning to: {query}",
                "analytical"
            )
        
        # Create episodic memory context if memory integration is available
        episode_id = None
        if self.memory_integration:
            try:
                episode_id = self.memory_integration.begin_episode(
                    f"Formal logical reasoning: {query[:50]}...",
                    "logical_reasoning"
                )
            except Exception as e:
                self.logger.error(f"Episodic memory initialization error: {str(e)}")
        
        try:
            # First, query the knowledge base
            query_result = self.logic_kernel.query(query)
            
            # If the statement is already known, return its truth value
            if query_result.get("found", False):
                result = {
                    "result": True,
                    "statement": query,
                    "truth": query_result.get("truth"),
                    "confidence": query_result.get("confidence", 1.0),
                    "explanation": "Statement found in knowledge base"
                }
            else:
                # Try to infer the statement
                inference_result = self.logic_kernel.infer(query)
                
                if inference_result.get("result", False):
                    # Statement was inferred
                    result = inference_result
                else:
                    # Statement could not be inferred
                    # Try abductive reasoning for possible explanations
                    abductive_result = self.logic_kernel.infer(query, "ABDUCTION")
                    
                    if abductive_result.get("result") == "abductive":
                        result = abductive_result
                    else:
                        # Fall back to reasoning node for an explanation
                        fallback = self.reasoning_node.reason(
                            f"Analyze the logical statement: {query}",
                            "analytical"
                        )
                        
                        result = {
                            "result": False,
                            "statement": query,
                            "explanation": "Could not prove or infer statement",
                            "reasoning": fallback
                        }
            
            # Store the interaction in memory if available
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.store_interaction(
                        query,
                        str(result),
                        "logical_reasoning",
                        {"episode_id": episode_id, "logical_framework": framework}
                    )
                    
                    # End the episode
                    self.memory_integration.end_episode(
                        f"Completed logical reasoning for: {query[:50]}..."
                    )
                except Exception as e:
                    self.logger.error(f"Memory interaction storage error: {str(e)}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Logical reasoning error: {str(e)}")
            
            # Close the episodic context if it was created
            if self.memory_integration and episode_id:
                try:
                    self.memory_integration.end_episode(
                        f"Error in logical reasoning: {str(e)}"
                    )
                except Exception as episode_e:
                    self.logger.error(f"Episodic memory closure error: {str(episode_e)}")
            
            # Fall back to standard reasoning
            return self.reason(
                f"Analyze the logical statement: {query}",
                "analytical"
            )