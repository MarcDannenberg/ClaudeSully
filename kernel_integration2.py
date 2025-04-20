# sully_engine/kernel_integration.py
# 🧠 SuperPowered Kernel Integration System with Enhanced Continuous Operation

from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Generator
import json
import os
import time
from datetime import datetime, timedelta
import random
import re
import asyncio
import threading
import queue
import logging
import traceback
import numpy as np
from collections import defaultdict, Counter, deque
import hashlib
import uuid
import copy
import importlib
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='kernel_integration.log'
)
logger = logging.getLogger(__name__)

# Import PDF reader
try:
    from sully_engine.pdf_reader import PDFReader, extract_kernel_from_text
except ImportError:
    logger.warning("PDF reader module not available. PDF functionality will be limited.")
    
    # Create minimal implementations for compatibility
    class PDFReader:
        def __init__(self, ocr_enabled=True, dpi=300):
            self.ocr_enabled = ocr_enabled
            self.dpi = dpi
            
        def extract_text(self, pdf_path, verbose=True, use_ocr_fallback=True, extract_structure=True):
            logger.warning(f"Mock PDF extraction for {pdf_path}")
            return {
                "success": False,
                "error": "PDF reader module not available",
                "text": "",
                "pages": []
            }
    
    def extract_kernel_from_text(text, domain="general"):
        logger.warning("Mock kernel extraction from text")
        return {
            "symbols": [],
            "paradoxes": [],
            "frames": [],
            "domain": domain
        }

# Import the enhanced codex
try:
    from sully_engine.kernel_modules.enhanced_codex import EnhancedCodex
except ImportError:
    logger.warning("Enhanced codex module not available. Using basic implementation.")
    
    class EnhancedCodex:
        def __init__(self):
            self.entries = {}
            
        def get(self, concept, default=None):
            return self.entries.get(concept, default or {"concept": concept})
            
        def get_related_concepts(self, concept, max_depth=1):
            return {f"related_to_{concept}": {"concept": f"related_to_{concept}"}}
            
        def add(self, concept, data):
            self.entries[concept] = data
            return True

# Import kernels with graceful fallbacks
kernel_modules = {
    "dream": "DreamCore",
    "fusion": "SymbolFusionEngine",
    "paradox": "ParadoxLibrary",
    "math_translator": "SymbolicMathTranslator",
    "conversation": "ConversationEngine",
    "memory": "MemoryIntegration"
}

loaded_modules = {}

for module_name, class_name in kernel_modules.items():
    try:
        module = importlib.import_module(f"sully_engine.kernel_modules.{module_name}")
        loaded_modules[module_name] = getattr(module, class_name)
        logger.info(f"Successfully loaded {class_name} from {module_name}")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load {class_name} from {module_name}: {str(e)}")
        loaded_modules[module_name] = None

# Global activation lock - used for thread safety
_activation_lock = threading.RLock()

class CognitivePathway:
    """
    Represents a cognitive pathway between kernels.
    
    Cognitive pathways are more than just connections - they evolve over time
    based on usage, adapt to new contexts, and can create emergent properties
    when multiple pathways interact.
    """
    
    def __init__(self, source_kernel: str, target_kernel: str, 
                 transform_fn: Callable, initial_strength: float = 0.5):
        """
        Initialize a cognitive pathway.
        
        Args:
            source_kernel: Source kernel name 
            target_kernel: Target kernel name
            transform_fn: Function to transform outputs from source to target
            initial_strength: Initial connection strength (0.0-1.0)
        """
        self.id = str(uuid.uuid4())
        self.source_kernel = source_kernel
        self.target_kernel = target_kernel
        self.transform_fn = transform_fn
        self.strength = initial_strength
        self.usage_count = 0
        self.last_used = None
        self.success_rate = 0.5  # Start with neutral success rate
        self.creation_time = datetime.now()
        self.metadata = {}
        
        # Dynamic attributes that evolve over time
        self.transformations = {}  # Stores successful transformations
        self.context_efficacy = {}  # Tracks effectiveness in different contexts
        self.activation_threshold = 0.2  # Minimum strength needed to auto-activate
        
    def transform(self, source_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform data from source kernel to target kernel format.
        
        Args:
            source_data: Data from the source kernel
            context: Optional contextual information
            
        Returns:
            Transformed data for target kernel
        """
        self.usage_count += 1
        self.last_used = datetime.now()
        context = context or {}
        
        try:
            # Apply the transformation function
            result = self.transform_fn(source_data)
            
            # Record context effectiveness if success
            context_key = self._get_context_key(context)
            if context_key:
                current = self.context_efficacy.get(context_key, {"success": 0, "total": 0})
                current["success"] += 1
                current["total"] += 1
                self.context_efficacy[context_key] = current
                
            # Update success rate
            self._update_success_rate(True)
            
            # Update transformations dictionary with a sample
            if isinstance(source_data, (dict, list)) and isinstance(result, (dict, list)):
                # Create a hash of the input to use as key
                hash_key = self._hash_data(source_data)
                if hash_key not in self.transformations:
                    # Only store if not already present
                    self.transformations[hash_key] = {
                        "input": self._summarize_data(source_data),
                        "output": self._summarize_data(result),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return result
        except Exception as e:
            logger.error(f"Error in cognitive pathway from {self.source_kernel} to {self.target_kernel}: {str(e)}")
            
            # Record context failure
            context_key = self._get_context_key(context)
            if context_key:
                current = self.context_efficacy.get(context_key, {"success": 0, "total": 0})
                current["total"] += 1
                self.context_efficacy[context_key] = current
            
            # Update success rate
            self._update_success_rate(False)
            
            # Return error information
            return {
                "error": str(e),
                "source_kernel": self.source_kernel,
                "target_kernel": self.target_kernel
            }
            
    def _update_success_rate(self, success: bool) -> None:
        """
        Update the success rate of this pathway.
        
        Args:
            success: Whether the transformation was successful
        """
        # Weighted update - recent results influence more than older ones
        learning_rate = 0.1
        self.success_rate = (1 - learning_rate) * self.success_rate + learning_rate * (1.0 if success else 0.0)
        
        # Also adjust strength based on success rate but with some inertia
        strength_adjustment = 0.05
        if success:
            self.strength = min(1.0, self.strength + strength_adjustment)
        else:
            self.strength = max(0.1, self.strength - strength_adjustment)
            
    def _get_context_key(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Create a context key for tracking efficacy in different contexts.
        
        Args:
            context: Context dictionary
            
        Returns:
            Context key string, or None if context is invalid
        """
        if not context:
            return None
            
        # Extract primary context elements
        elements = []
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, (str, int, float, bool)):
                elements.append(f"{key}:{value}")
                
        if not elements:
            return None
            
        # Create a combined key, limited to prevent explosion
        return "|".join(elements[:5])
        
    def _hash_data(self, data: Any) -> str:
        """
        Create a hash of input data for storage and lookup.
        
        Args:
            data: The data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, dict):
            # For dictionaries, use stable serialization with sorted keys
            serialized = json.dumps(data, sort_keys=True)[:1000]  # Limit size
        elif isinstance(data, list):
            # For lists, use stable serialization
            serialized = json.dumps(data)[:1000]  # Limit size
        else:
            # For other types, convert to string
            serialized = str(data)[:1000]  # Limit size
            
        # Create hash
        return hashlib.md5(serialized.encode()).hexdigest()
        
    def _summarize_data(self, data: Any) -> Any:
        """
        Create a summarized version of data for storage.
        
        Args:
            data: The data to summarize
            
        Returns:
            Summarized data
        """
        if isinstance(data, dict):
            # For large dictionaries, keep only the first few entries
            if len(data) > 5:
                return {k: data[k] for k in list(data.keys())[:5]}
            return data
        elif isinstance(data, list):
            # For long lists, keep only the first few items
            if len(data) > 5:
                return data[:5]
            return data
        else:
            # For strings and other types, truncate if needed
            if isinstance(data, str) and len(data) > 500:
                return data[:500] + "..."
            return data
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pathway to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "source_kernel": self.source_kernel,
            "target_kernel": self.target_kernel,
            "strength": self.strength,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "creation_time": self.creation_time.isoformat(),
            "activation_threshold": self.activation_threshold,
            "contexts": len(self.context_efficacy),
            "transformations": len(self.transformations)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], transform_fn: Callable) -> 'CognitivePathway':
        """
        Create pathway from dictionary representation.
        
        Args:
            data: Dictionary data
            transform_fn: Transformation function
            
        Returns:
            Reconstructed CognitivePathway
        """
        pathway = cls(
            source_kernel=data["source_kernel"],
            target_kernel=data["target_kernel"],
            transform_fn=transform_fn,
            initial_strength=data["strength"]
        )
        
        pathway.id = data["id"]
        pathway.usage_count = data["usage_count"]
        pathway.success_rate = data["success_rate"]
        
        if data.get("last_used"):
            pathway.last_used = datetime.fromisoformat(data["last_used"])
            
        if data.get("creation_time"):
            pathway.creation_time = datetime.fromisoformat(data["creation_time"])
            
        if data.get("activation_threshold"):
            pathway.activation_threshold = data["activation_threshold"]
            
        return pathway

class EmergentProperty:
    """
    Represents an emergent property that arises from kernel interactions.
    
    Emergent properties are not coded directly but arise from the interactions
    between kernels and their interconnections. They can represent new capabilities,
    insights, or patterns that weren't explicitly programmed.
    """
    
    def __init__(self, name: str, description: str, 
                 source_kernels: List[str], confidence: float = 0.5):
        """
        Initialize an emergent property.
        
        Args:
            name: Name of the emergent property
            description: Description of what this property represents
            source_kernels: List of kernels that contribute to this property
            confidence: Confidence in the reality of this property (0.0-1.0)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.source_kernels = source_kernels
        self.confidence = confidence
        self.detection_time = datetime.now()
        self.last_activation = datetime.now()
        self.activation_count = 1
        self.observations = []
        self.stability = 0.5  # Initial stability is neutral
        
    def observe(self, observation: str, context: Dict[str, Any] = None) -> None:
        """
        Add an observation of this emergent property in action.
        
        Args:
            observation: Description of the observation
            context: Contextual information about when/where observed
        """
        self.last_activation = datetime.now()
        self.activation_count += 1
        
        # Add to observations list
        self.observations.append({
            "observation": observation,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        # Update stability based on continued observations
        # Properties become more stable the more they're observed
        self.stability = min(1.0, self.stability + 0.02)
        
    def update_confidence(self, new_evidence_strength: float) -> None:
        """
        Update confidence based on new evidence.
        
        Args:
            new_evidence_strength: Strength of new evidence (0.0-1.0)
        """
        # Bayesian-inspired update - weight existing confidence more
        prior_weight = 0.8
        self.confidence = (prior_weight * self.confidence + 
                          (1 - prior_weight) * new_evidence_strength)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert emergent property to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_kernels": self.source_kernels,
            "confidence": self.confidence,
            "detection_time": self.detection_time.isoformat(),
            "last_activation": self.last_activation.isoformat(),
            "activation_count": self.activation_count,
            "observations": self.observations[-5:],  # Last 5 observations
            "stability": self.stability
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmergentProperty':
        """
        Create emergent property from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Reconstructed EmergentProperty
        """
        prop = cls(
            name=data["name"],
            description=data["description"],
            source_kernels=data["source_kernels"],
            confidence=data["confidence"]
        )
        
        prop.id = data["id"]
        
        if data.get("detection_time"):
            prop.detection_time = datetime.fromisoformat(data["detection_time"])
            
        if data.get("last_activation"):
            prop.last_activation = datetime.fromisoformat(data["last_activation"])
            
        prop.activation_count = data.get("activation_count", 1)
        prop.observations = data.get("observations", [])
        prop.stability = data.get("stability", 0.5)
        
        return prop

class CognitiveWorkspace:
    """
    A workspace for holding temporary cognitive processing data.
    
    The workspace is used for multi-step cognitive operations where data needs
    to be held and transformed across multiple kernels. It provides a context
    for complex operations and enables meta-reasoning about the process.
    """
    
    def __init__(self, title: str, purpose: str):
        """
        Initialize a cognitive workspace.
        
        Args:
            title: Title of the workspace
            purpose: Purpose/goal of this workspace
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.purpose = purpose
        self.creation_time = datetime.now()
        self.last_update = datetime.now()
        self.data = {}  # Main data store
        self.kernels_involved = set()  # Kernels that have contributed
        self.processing_history = []  # Record of processing steps
        self.insights = []  # Insights generated during processing
        self.status = "active"  # active, complete, or failed
        self.tags = set()  # Tags for classification
        
    def add_data(self, key: str, value: Any, source_kernel: str = None) -> None:
        """
        Add data to the workspace.
        
        Args:
            key: Data key
            value: Data value
            source_kernel: Optional source kernel name
        """
        self.data[key] = value
        self.last_update = datetime.now()
        
        if source_kernel:
            self.kernels_involved.add(source_kernel)
            
            # Record the processing step
            self.processing_history.append({
                "action": "add_data",
                "key": key,
                "kernel": source_kernel,
                "timestamp": datetime.now().isoformat()
            })
            
    def update_data(self, key: str, value: Any, source_kernel: str = None) -> bool:
        """
        Update existing data in the workspace.
        
        Args:
            key: Data key
            value: New data value
            source_kernel: Optional source kernel name
            
        Returns:
            True if data was updated, False if key doesn't exist
        """
        if key not in self.data:
            return False
            
        self.data[key] = value
        self.last_update = datetime.now()
        
        if source_kernel:
            self.kernels_involved.add(source_kernel)
            
            # Record the processing step
            self.processing_history.append({
                "action": "update_data",
                "key": key,
                "kernel": source_kernel,
                "timestamp": datetime.now().isoformat()
            })
            
        return True
        
    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from the workspace.
        
        Args:
            key: Data key
            default: Default value if key doesn't exist
            
        Returns:
            Data value or default
        """
        return self.data.get(key, default)
        
    def add_insight(self, insight: str, source_kernel: str = None, 
                   confidence: float = 0.5, tags: List[str] = None) -> None:
        """
        Add an insight to the workspace.
        
        Args:
            insight: Insight text
            source_kernel: Optional source kernel name
            confidence: Confidence in the insight (0.0-1.0)
            tags: Optional tags for the insight
        """
        self.insights.append({
            "insight": insight,
            "source_kernel": source_kernel,
            "confidence": confidence,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat()
        })
        
        if source_kernel:
            self.kernels_involved.add(source_kernel)
            
        # Add tags to workspace
        if tags:
            self.tags.update(tags)
            
        # Record the processing step
        self.processing_history.append({
            "action": "add_insight",
            "kernel": source_kernel,
            "timestamp": datetime.now().isoformat()
        })
        
    def complete(self, result: Any = None) -> None:
        """
        Mark the workspace as complete.
        
        Args:
            result: Optional final result to add to workspace
        """
        self.status = "complete"
        
        if result is not None:
            self.add_data("final_result", result)
            
        # Record the completion
        self.processing_history.append({
            "action": "complete",
            "timestamp": datetime.now().isoformat()
        })
        
    def fail(self, reason: str) -> None:
        """
        Mark the workspace as failed.
        
        Args:
            reason: Reason for failure
        """
        self.status = "failed"
        self.add_data("failure_reason", reason)
        
        # Record the failure
        self.processing_history.append({
            "action": "fail",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workspace to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "title": self.title,
            "purpose": self.purpose,
            "creation_time": self.creation_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "data_keys": list(self.data.keys()),
            "kernels_involved": list(self.kernels_involved),
            "processing_steps": len(self.processing_history),
            "insights": len(self.insights),
            "status": self.status,
            "tags": list(self.tags)
        }
        
    def get_full_dict(self) -> Dict[str, Any]:
        """
        Get complete dictionary including all data.
        
        Returns:
            Complete dictionary representation
        """
        base = self.to_dict()
        base["data"] = {k: v for k, v in self.data.items() if isinstance(v, (str, int, float, bool, list, dict))}
        base["processing_history"] = self.processing_history
        base["insights"] = self.insights
        return base

class KernelOperation(threading.Thread):
    """
    Represents an asynchronous kernel operation that runs in its own thread.
    
    Kernel operations can run continuously in the background, processing
    data, generating insights, and contributing to the system's emergent
    cognitive capabilities.
    """
    
    def __init__(self, name: str, kernel_name: str, operation: str, 
                 params: Dict[str, Any], workspace: Optional[CognitiveWorkspace] = None,
                 result_callback: Optional[Callable] = None, ttl: Optional[int] = None):
        """
        Initialize a kernel operation.
        
        Args:
            name: Operation name
            kernel_name: Kernel to operate on
            operation: Operation to perform
            params: Operation parameters
            workspace: Optional cognitive workspace
            result_callback: Optional callback for result
            ttl: Time to live in seconds (None for continuous)
        """
        threading.Thread.__init__(self, daemon=True)
        self.id = str(uuid.uuid4())
        self.name = name
        self.kernel_name = kernel_name
        self.operation = operation
        self.params = params
        self.workspace = workspace
        self.result_callback = result_callback
        self.ttl = ttl
        
        self.result = None
        self.error = None
        self.start_time = datetime.now()
        self.end_time = None
        self.status = "pending"  # pending, running, completed, failed, terminated
        self.progress = 0.0
        self._terminate = False
        self._pause = False
        self._step_results = []
        
    def run(self):
        """Main thread execution."""
        try:
            self.status = "running"
            start_timestamp = time.time()
            
            # Execute the operation
            self.result = self._execute_operation()
            
            # Update status
            self.end_time = datetime.now()
            self.status = "completed"
            self.progress = 1.0
            
            # Call result callback if provided
            if self.result_callback:
                try:
                    self.result_callback(self.result)
                except Exception as e:
                    logger.error(f"Error in result callback: {str(e)}")
                    
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            self.end_time = datetime.now()
            logger.error(f"Kernel operation '{self.name}' failed: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _execute_operation(self) -> Any:
        """
        Execute the kernel operation.
        
        Returns:
            Operation result
        """
        # Initialize TTL tracking
        if self.ttl:
            end_time = time.time() + self.ttl
        else:
            end_time = None
            
        # Main execution loop - supports both one-shot and continuous operations
        step = 0
        while not self._terminate:
            # Check if operation should terminate due to TTL
            if end_time and time.time() > end_time:
                break
                
            # Check if operation is paused
            if self._pause:
                time.sleep(0.1)
                continue
                
            try:
                # Execute one step of the operation
                if step == 0:
                    # First step uses initial parameters
                    step_result = self._execute_step(self.params)
                else:
                    # Subsequent steps can use previous results
                    previous_result = self._step_results[-1] if self._step_results else None
                    step_result = self._execute_step({
                        **self.params,
                        "previous_result": previous_result,
                        "step": step
                    })
                    
                # Store step result
                self._step_results.append(step_result)
                
                # Add to workspace if available
                if self.workspace:
                    self.workspace.add_data(
                        key=f"{self.operation}_step_{step}",
                        value=step_result,
                        source_kernel=self.kernel_name
                    )
                    
                # Update progress for non-continuous operations
                if self.ttl:
                    elapsed = time.time() - self.start_time.timestamp()
                    self.progress = min(0.95, elapsed / self.ttl)
                    
                # If this is a one-shot operation (not continuous), break after first step
                if self.ttl is not None:
                    break
                    
                # For continuous operations, add a small delay to prevent CPU hogging
                if end_time is None:
                    time.sleep(0.05)
                    
                step += 1
                
            except Exception as e:
                logger.error(f"Error in kernel operation step {step}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # For continuous operations, log and continue
                if end_time is None:
                    time.sleep(1.0)  # Longer delay after error
                else:
                    # For one-shot operations, reraise
                    raise
                    
        # Prepare final result
        if self._step_results:
            if len(self._step_results) == 1:
                # For one-shot operations, return the single result
                return self._step_results[0]
            else:
                # For continuous operations, return all results
                return {
                    "steps": len(self._step_results),
                    "first_result": self._step_results[0],
                    "last_result": self._step_results[-1],
                    "all_results": self._step_results
                }
        else:
            return None
            
    def _execute_step(self, params: Dict[str, Any]) -> Any:
        """
        Execute a single step of the operation.
        
        Args:
            params: Step parameters
            
        Returns:
            Step result
        """
        # This is a placeholder - in a real implementation, this would
        # dispatch to the appropriate kernel and operation
        
        # Simulate some processing time
        time.sleep(0.1)
        
        return {
            "kernel": self.kernel_name,
            "operation": self.operation,
            "timestamp": datetime.now().isoformat(),
            "simulated_result": f"Result for {self.operation} with params {params}"
        }
        
    def terminate(self) -> None:
        """Terminate the operation."""
        self._terminate = True
        
    def pause(self) -> None:
        """Pause the operation."""
        self._pause = True
        
    def resume(self) -> None:
        """Resume the operation."""
        self._pause = False
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get operation status.
        
        Returns:
            Status dictionary
        """
        return {
            "id": self.id,
            "name": self.name,
            "kernel": self.kernel_name,
            "operation": self.operation,
            "status": self.status,
            "progress": self.progress,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "steps_completed": len(self._step_results)
        }

class KernelIntegrationSystem:
    """
    Central integration system for Sully's cognitive kernels.
    
    Connects and enhances Dream, Fusion, Paradox, Math Translation, Conversation,
    Memory Integration, and other kernels through the enhanced codex, enabling 
    cross-module reasoning and emergent capabilities.
    
    This enhanced version supports continuous operation with background threads,
    emergent property detection, and adaptive cognitive pathways.
    """
    
    def __init__(self, 
                codex: Optional[EnhancedCodex] = None,
                dream_core = None,
                fusion_engine = None,
                paradox_library = None,
                math_translator = None,
                conversation_engine = None,
                memory_integration = None,
                sully_instance = None):
        """
        Initialize the kernel integration system.
        
        Args:
            codex: Optional existing EnhancedCodex instance
            dream_core: Optional existing DreamCore instance
            fusion_engine: Optional existing SymbolFusionEngine instance
            paradox_library: Optional existing ParadoxLibrary instance
            math_translator: Optional existing SymbolicMathTranslator instance
            conversation_engine: Optional existing ConversationEngine instance
            memory_integration: Optional existing MemoryIntegration instance
            sully_instance: Optional reference to main Sully instance
        """
        # Initialize or use provided components
        self.codex = codex if codex else EnhancedCodex()
        
        # Try to load modules from loaded_modules
        self.dream_core = dream_core if dream_core else (
            loaded_modules["dream"]() if loaded_modules.get("dream") else None
        )
        
        self.fusion_engine = fusion_engine if fusion_engine else (
            loaded_modules["fusion"]() if loaded_modules.get("fusion") else None
        )
        
        self.paradox_library = paradox_library if paradox_library else (
            loaded_modules["paradox"]() if loaded_modules.get("paradox") else None
        )
        
        self.math_translator = math_translator if math_translator else (
            loaded_modules["math_translator"]() if loaded_modules.get("math_translator") else None
        )
        
        self.conversation_engine = conversation_engine if conversation_engine else (
            loaded_modules["conversation"]() if loaded_modules.get("conversation") else None
        )
        
        self.memory_integration = memory_integration if memory_integration else (
            loaded_modules["memory"]() if loaded_modules.get("memory") else None
        )
        
        self.sully = sully_instance
        
        # Initialize PDF reader
        self.pdf_reader = PDFReader(ocr_enabled=True, dpi=300)
        
        # Integration mappings between kernels
        self.kernel_mappings = {
            "dream_fusion": {},      # How dream symbols map to fusion concepts
            "dream_paradox": {},     # How dream patterns relate to paradoxes
            "fusion_paradox": {},    # How fused concepts generate paradoxes
            "math_dream": {},        # How mathematical concepts map to dream symbols
            "math_fusion": {},       # How mathematical operations relate to fusion styles
            "math_paradox": {},      # How mathematical structures relate to paradox types
            "conversation_dream": {}, # How conversation topics map to dream styles
            "conversation_fusion": {}, # How conversation tones map to fusion styles
            "memory_context": {}      # How memory traces affect other kernel operations
        }
        
        # Cross-kernel transformation functions
        self.transformations = {
            "dream_to_fusion": self._transform_dream_to_fusion,
            "dream_to_paradox": self._transform_dream_to_paradox,
            "fusion_to_dream": self._transform_fusion_to_dream,
            "fusion_to_paradox": self._transform_fusion_to_paradox,
            "paradox_to_dream": self._transform_paradox_to_dream,
            "paradox_to_fusion": self._transform_paradox_to_fusion,
            "math_to_dream": self._transform_math_to_dream,
            "math_to_fusion": self._transform_math_to_fusion,
            "math_to_paradox": self._transform_math_to_paradox,
            "conversation_to_dream": self._transform_conversation_to_dream,
            "conversation_to_fusion": self._transform_conversation_to_fusion,
            "memory_to_dream": self._transform_memory_to_dream,
            "memory_to_fusion": self._transform_memory_to_fusion
        }
        
        # Enhanced system components for continuous operation
        
        # Cognitive pathways - evolving connections between kernels
        self.cognitive_pathways = {}
        
        # Emergent properties detected in the system
        self.emergent_properties = []
        
        # Active cognitive workspaces
        self.workspaces = {}
        
        # Thread management
        self.active_operations = {}
        self.background_threads = []
        self._thread_stop_event = threading.Event()
        
        # Continuous operation queue
        self.continuous_queue = queue.Queue()
        
        # Kernel activation patterns
        self.kernel_activation_patterns = defaultdict(list)
        
        # Creation timestamp
        self.creation_time = datetime.now()
        
        # Enhance kernels with integration capabilities
        self._enhance_dream_core()
        self._enhance_fusion_engine()
        self._enhance_paradox_library()
        self._enhance_math_translator()
        
        # Enhance communication and memory if provided
        if conversation_engine:
            self._enhance_conversation_engine()
        
        if memory_integration:
            self._enhance_memory_integration()
        
        # Create initial integration mappings
        self._initialize_kernel_mappings()
        
        # Initialize cross-kernel concept space
        self.concept_space = {}
        
        # Track kernel integration history
        self.integration_history = []
        
        # Start background threads for continuous operation
        self._start_background_threads()
        
    def __del__(self):
        """Cleanup on deletion."""
        self._thread_stop_event.set()
        
        # Wait for background threads to terminate
        for thread in self.background_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        # Terminate active operations
        for op_id, operation in list(self.active_operations.items()):
            operation.terminate()

    # Background processing threads for continuous operation
    
    def _start_background_threads(self):
        """Start background threads for continuous operation."""
        # Thread for processing the continuous operation queue
        queue_thread = threading.Thread(
            target=self._continuous_queue_processor,
            daemon=True,
            name="KernelIntegration-QueueProcessor"
        )
        queue_thread.start()
        self.background_threads.append(queue_thread)
        
        # Thread for emergent property detection
        emergence_thread = threading.Thread(
            target=self._emergent_property_detector,
            daemon=True,
            name="KernelIntegration-EmergenceDetector"
        )
        emergence_thread.start()
        self.background_threads.append(emergence_thread)
        
        # Thread for cognitive pathway optimization
        pathway_thread = threading.Thread(
            target=self._pathway_optimizer,
            daemon=True,
            name="KernelIntegration-PathwayOptimizer"
        )
        pathway_thread.start()
        self.background_threads.append(pathway_thread)
        
        # Thread for concept space maintenance
        concept_thread = threading.Thread(
            target=self._concept_space_maintainer,
            daemon=True,
            name="KernelIntegration-ConceptMaintainer"
        )
        concept_thread.start()
        self.background_threads.append(concept_thread)
        
        # Thread for workspace management
        workspace_thread = threading.Thread(
            target=self._workspace_manager,
            daemon=True,
            name="KernelIntegration-WorkspaceManager"
        )
        workspace_thread.start()
        self.background_threads.append(workspace_thread)
        
        logger.info(f"Started {len(self.background_threads)} background threads for continuous operation")
        
    def _continuous_queue_processor(self):
        """Process the continuous operation queue."""
        logger.info("Starting continuous queue processor thread")
        
        while not self._thread_stop_event.is_set():
            try:
                # Get the next operation from the queue with timeout
                try:
                    operation = self.continuous_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Process the operation
                if operation.get("type") == "cross_kernel":
                    # Execute cross-kernel operation
                    source_kernel = operation.get("source_kernel")
                    target_kernel = operation.get("target_kernel")
                    input_data = operation.get("input_data")
                    
                    if source_kernel and target_kernel and input_data is not None:
                        try:
                            result = self.cross_kernel_operation(
                                source_kernel, target_kernel, input_data
                            )
                            
                            # Handle the result if needed
                            callback = operation.get("callback")
                            if callback and callable(callback):
                                callback(result)
                                
                        except Exception as e:
                            logger.error(f"Error in queued cross-kernel operation: {str(e)}")
                            
                elif operation.get("type") == "concept_exploration":
                    # Execute concept exploration
                    concept = operation.get("concept")
                    depth = operation.get("depth", 2)
                    
                    if concept:
                        try:
                            result = self.create_concept_network(concept, depth)
                            
                            # Handle the result if needed
                            callback = operation.get("callback")
                            if callback and callable(callback):
                                callback(result)
                                
                        except Exception as e:
                            logger.error(f"Error in queued concept exploration: {str(e)}")
                            
                elif operation.get("type") == "dream_generation":
                    # Execute dream generation
                    seed = operation.get("seed")
                    
                    if seed and self.dream_core:
                        try:
                            result = self.dream_core.generate(seed)
                            
                            # Handle the result if needed
                            callback = operation.get("callback")
                            if callback and callable(callback):
                                callback(result)
                                
                        except Exception as e:
                            logger.error(f"Error in queued dream generation: {str(e)}")
                            
                # Mark the operation as complete
                self.continuous_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in continuous queue processor: {str(e)}")
                time.sleep(1.0)  # Sleep after error
                
        logger.info("Continuous queue processor thread stopping")
        
    def _emergent_property_detector(self):
        """Background thread for detecting emergent properties."""
        logger.info("Starting emergent property detector thread")
        
        detection_interval = 300  # 5 minutes between detection runs
        last_detection = time.time()
        
        while not self._thread_stop_event.is_set():
            try:
                current_time = time.time()
                
                # Run detection at the specified interval
                if current_time - last_detection >= detection_interval:
                    self._detect_emergent_properties()
                    last_detection = current_time
                    
                # Sleep for a bit
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Error in emergent property detector: {str(e)}")
                time.sleep(60.0)  # Sleep longer after error
                
        logger.info("Emergent property detector thread stopping")
        
    def _pathway_optimizer(self):
        """Background thread for optimizing cognitive pathways."""
        logger.info("Starting pathway optimizer thread")
        
        optimization_interval = 600  # 10 minutes between optimization runs
        last_optimization = time.time()
        
        while not self._thread_stop_event.is_set():
            try:
                current_time = time.time()
                
                # Run optimization at the specified interval
                if current_time - last_optimization >= optimization_interval:
                    self._optimize_cognitive_pathways()
                    last_optimization = current_time
                    
                # Sleep for a bit
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Error in pathway optimizer: {str(e)}")
                time.sleep(60.0)  # Sleep longer after error
                
        logger.info("Pathway optimizer thread stopping")
        
    def _concept_space_maintainer(self):
        """Background thread for maintaining the concept space."""
        logger.info("Starting concept space maintainer thread")
        
        maintenance_interval = 1800  # 30 minutes between maintenance runs
        last_maintenance = time.time()
        
        while not self._thread_stop_event.is_set():
            try:
                current_time = time.time()
                
                # Run maintenance at the specified interval
                if current_time - last_maintenance >= maintenance_interval:
                    self._maintain_concept_space()
                    last_maintenance = current_time
                    
                # Sleep for a bit
                time.sleep(30.0)
                
            except Exception as e:
                logger.error(f"Error in concept space maintainer: {str(e)}")
                time.sleep(300.0)  # Sleep longer after error
                
        logger.info("Concept space maintainer thread stopping")
        
    def _workspace_manager(self):
        """Background thread for managing cognitive workspaces."""
        logger.info("Starting workspace manager thread")
        
        cleanup_interval = 3600  # 1 hour between cleanup runs
        last_cleanup = time.time()
        
        while not self._thread_stop_event.is_set():
            try:
                current_time = time.time()
                
                # Run cleanup at the specified interval
                if current_time - last_cleanup >= cleanup_interval:
                    self._cleanup_workspaces()
                    last_cleanup = current_time
                    
                # Process active workspaces
                for workspace_id, workspace in list(self.workspaces.items()):
                    # Skip completed or failed workspaces
                    if workspace.status != "active":
                        continue
                        
                    # Check for stalled workspaces
                    if (datetime.now() - workspace.last_update).total_seconds() > 3600:
                        # Mark workspace as failed if no updates for 1 hour
                        workspace.fail("Workspace stalled - no updates for 1 hour")
                        
                # Sleep for a bit
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Error in workspace manager: {str(e)}")
                time.sleep(60.0)  # Sleep longer after error
                
        logger.info("Workspace manager thread stopping")
        
    # Enhanced initialization methods
        
    def _enhance_dream_core(self):
        """Enhances the DreamCore with integration capabilities."""
        if not self.dream_core:
            return
            
        # Store original methods
        if not hasattr(self.dream_core, "_original_generate"):
            self.dream_core._original_generate = self.dream_core.generate
            
        # Replace with enhanced version
        def enhanced_generate(seed, depth="standard", style="recursive", context=None):
            """Enhanced dream generation with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("dream", "generate", {
                "seed": seed,
                "depth": depth,
                "style": style
            })
            
            # Execute the original method
            result = self.dream_core._original_generate(seed, depth, style)
            
            # Record in concept space
            self._update_concept_space(seed, "dream", {
                "dream_result": result if isinstance(result, str) else str(result)[:200],
                "depth": depth,
                "style": style,
                "timestamp": datetime.now().isoformat()
            })
            
            # Queue automatic cross-kernel operations if enabled
            with _activation_lock:
                for pathway_id, pathway in self.cognitive_pathways.items():
                    if (pathway.source_kernel == "dream" and 
                        pathway.strength >= pathway.activation_threshold):
                        # Auto-activate strong pathways
                        self.continuous_queue.put({
                            "type": "cross_kernel",
                            "source_kernel": "dream",
                            "target_kernel": pathway.target_kernel,
                            "input_data": result
                        })
                        
            return result
            
        # Apply the enhanced method
        self.dream_core.generate = enhanced_generate
        
    def _enhance_fusion_engine(self):
        """Enhances the SymbolFusionEngine with integration capabilities."""
        if not self.fusion_engine:
            return
            
        # Store original methods
        if hasattr(self.fusion_engine, "fuse_concepts") and not hasattr(self.fusion_engine, "_original_fuse_concepts"):
            self.fusion_engine._original_fuse_concepts = self.fusion_engine.fuse_concepts
            
        # Replace with enhanced version
        def enhanced_fuse_concepts(*concepts, style=None):
            """Enhanced concept fusion with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("fusion", "fuse_concepts", {
                "concepts": concepts,
                "style": style
            })
            
            # Execute the original method
            result = self.fusion_engine._original_fuse_concepts(*concepts, style=style)
            
            # Record in concept space
            for concept in concepts:
                self._update_concept_space(concept, "fusion", {
                    "fusion_concepts": concepts,
                    "fusion_result": result if isinstance(result, str) else str(result)[:200],
                    "style": style,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Queue automatic cross-kernel operations if enabled
            with _activation_lock:
                for pathway_id, pathway in self.cognitive_pathways.items():
                    if (pathway.source_kernel == "fusion" and 
                        pathway.strength >= pathway.activation_threshold):
                        # Auto-activate strong pathways
                        self.continuous_queue.put({
                            "type": "cross_kernel",
                            "source_kernel": "fusion",
                            "target_kernel": pathway.target_kernel,
                            "input_data": result
                        })
                        
            return result
            
        # Apply the enhanced method
        if hasattr(self.fusion_engine, "fuse_concepts"):
            self.fusion_engine.fuse_concepts = enhanced_fuse_concepts
            
    def _enhance_paradox_library(self):
        """Enhances the ParadoxLibrary with integration capabilities."""
        if not self.paradox_library:
            return
            
        # Store original methods
        if hasattr(self.paradox_library, "get") and not hasattr(self.paradox_library, "_original_get"):
            self.paradox_library._original_get = self.paradox_library.get
            
        # Replace with enhanced version
        def enhanced_get(concept):
            """Enhanced paradox exploration with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("paradox", "get", {
                "concept": concept
            })
            
            # Execute the original method
            result = self.paradox_library._original_get(concept)
            
            # Record in concept space
            self._update_concept_space(concept, "paradox", {
                "paradox_result": result if isinstance(result, str) else str(result)[:200],
                "timestamp": datetime.now().isoformat()
            })
            
            # Queue automatic cross-kernel operations if enabled
            with _activation_lock:
                for pathway_id, pathway in self.cognitive_pathways.items():
                    if (pathway.source_kernel == "paradox" and 
                        pathway.strength >= pathway.activation_threshold):
                        # Auto-activate strong pathways
                        self.continuous_queue.put({
                            "type": "cross_kernel",
                            "source_kernel": "paradox",
                            "target_kernel": pathway.target_kernel,
                            "input_data": result
                        })
                        
            return result
            
        # Apply the enhanced method
        if hasattr(self.paradox_library, "get"):
            self.paradox_library.get = enhanced_get
            
    def _enhance_math_translator(self):
        """Enhances the SymbolicMathTranslator with integration capabilities."""
        if not self.math_translator:
            return
            
        # Store original methods
        if hasattr(self.math_translator, "translate") and not hasattr(self.math_translator, "_original_translate"):
            self.math_translator._original_translate = self.math_translator.translate
            
        # Replace with enhanced version
        def enhanced_translate(phrase, style="formal", domain=None):
            """Enhanced math translation with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("math", "translate", {
                "phrase": phrase,
                "style": style,
                "domain": domain
            })
            
            # Execute the original method
            result = self.math_translator._original_translate(phrase, style, domain)
            
            # Record in concept space
            self._update_concept_space(phrase, "math", {
                "math_result": result if isinstance(result, str) else str(result)[:200],
                "style": style,
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            })
            
            # Queue automatic cross-kernel operations if enabled
            with _activation_lock:
                for pathway_id, pathway in self.cognitive_pathways.items():
                    if (pathway.source_kernel == "math" and 
                        pathway.strength >= pathway.activation_threshold):
                        # Auto-activate strong pathways
                        self.continuous_queue.put({
                            "type": "cross_kernel",
                            "source_kernel": "math",
                            "target_kernel": pathway.target_kernel,
                            "input_data": result
                        })
                        
            return result
            
        # Apply the enhanced method
        if hasattr(self.math_translator, "translate"):
            self.math_translator.translate = enhanced_translate
            
    def _enhance_conversation_engine(self):
        """Enhances the ConversationEngine with integration capabilities."""
        if not self.conversation_engine:
            return
            
        # Store original methods
        if hasattr(self.conversation_engine, "process_message") and not hasattr(self.conversation_engine, "_original_process_message"):
            self.conversation_engine._original_process_message = self.conversation_engine.process_message
            
        # Replace with enhanced version
        def enhanced_process_message(message, tone="emergent", continue_conversation=True):
            """Enhanced message processing with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("conversation", "process_message", {
                "message_length": len(message),
                "tone": tone
            })
            
            # Execute the original method
            result = self.conversation_engine._original_process_message(message, tone, continue_conversation)
            
            # Extract topics if possible
            topics = []
            if hasattr(self.conversation_engine, "_extract_topics"):
                try:
                    topics = self.conversation_engine._extract_topics(message)
                except:
                    # Use simple topic extraction
                    words = message.split()
                    topics = [w for w in words if len(w) > 5][:3]
                    
            # Record in concept space for each detected topic
            for topic in topics:
                self._update_concept_space(topic, "conversation", {
                    "message_fragment": message[:100] + "..." if len(message) > 100 else message,
                    "response_fragment": result[:100] + "..." if len(result) > 100 else result,
                    "tone": tone,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Queue automatic cross-kernel operations if enabled
            with _activation_lock:
                for pathway_id, pathway in self.cognitive_pathways.items():
                    if (pathway.source_kernel == "conversation" and 
                        pathway.strength >= pathway.activation_threshold):
                        # Auto-activate strong pathways
                        self.continuous_queue.put({
                            "type": "cross_kernel",
                            "source_kernel": "conversation",
                            "target_kernel": pathway.target_kernel,
                            "input_data": result
                        })
                        
            return result
            
        # Apply the enhanced method
        if hasattr(self.conversation_engine, "process_message"):
            self.conversation_engine.process_message = enhanced_process_message
            
    def _enhance_memory_integration(self):
        """Enhances the MemoryIntegration with integration capabilities."""
        if not self.memory_integration:
            return
            
        # Store original methods
        if hasattr(self.memory_integration, "recall") and not hasattr(self.memory_integration, "_original_recall"):
            self.memory_integration._original_recall = self.memory_integration.recall
            
        if hasattr(self.memory_integration, "store_experience") and not hasattr(self.memory_integration, "_original_store_experience"):
            self.memory_integration._original_store_experience = self.memory_integration.store_experience
            
        # Replace with enhanced version
        def enhanced_recall(query, limit=5, module=None):
            """Enhanced memory recall with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("memory", "recall", {
                "query": query,
                "limit": limit,
                "module": module
            })
            
            # Execute the original method
            result = self.memory_integration._original_recall(query, limit, module)
            
            # Queue automatic cross-kernel operations if enabled
            with _activation_lock:
                for pathway_id, pathway in self.cognitive_pathways.items():
                    if (pathway.source_kernel == "memory" and 
                        pathway.strength >= pathway.activation_threshold):
                        # Auto-activate strong pathways
                        self.continuous_queue.put({
                            "type": "cross_kernel",
                            "source_kernel": "memory",
                            "target_kernel": pathway.target_kernel,
                            "input_data": result
                        })
                        
            return result
            
        def enhanced_store_experience(content, source, importance=0.5, 
                                   concepts=None, emotional_tags=None):
            """Enhanced memory storage with integration capabilities."""
            # Log the operation for pattern analysis
            self._log_kernel_activation("memory", "store_experience", {
                "content_length": len(content) if isinstance(content, str) else 0,
                "source": source,
                "importance": importance
            })
            
            # Execute the original method
            result = self.memory_integration._original_store_experience(
                content, source, importance, concepts, emotional_tags
            )
            
            # Record in concept space for each concept
            if concepts:
                for concept in concepts:
                    self._update_concept_space(concept, "memory", {
                        "content_fragment": content[:100] + "..." if isinstance(content, str) and len(content) > 100 else str(content),
                        "source": source,
                        "importance": importance,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            return result
            
        # Apply the enhanced methods
        if hasattr(self.memory_integration, "recall"):
            self.memory_integration.recall = enhanced_recall
            
        if hasattr(self.memory_integration, "store_experience"):
            self.memory_integration.store_experience = enhanced_store_experience
    
    def _initialize_kernel_mappings(self):
        """Creates initial mappings between kernel concepts."""
        # Add some basic mappings for dream to fusion
        self.kernel_mappings["dream_fusion"] = {
            "light": "illumination",
            "darkness": "obscurity",
            "water": "fluidity",
            "fire": "transformation",
            "journey": "progression"
        }
        
        # Add some basic mappings for dream to paradox
        self.kernel_mappings["dream_paradox"] = {
            "mirror": "self_reference",
            "maze": "infinite_recursion",
            "bridge": "connection_division",
            "clock": "time_paradox",
            "ocean": "one_many"
        }
        
        # Add some basic mappings for math to fusion
        self.kernel_mappings["math_fusion"] = {
            "infinity": "boundlessness",
            "zero": "nothingness",
            "function": "transformation",
            "set": "collection",
            "sequence": "progression"
        }
        
        # Add some basic mappings for math to paradox
        self.kernel_mappings["math_paradox"] = {
            "infinity": "infinite_regress",
            "self_reference": "godel_incompleteness",
            "set_of_all_sets": "russell_paradox",
            "continuum": "zeno_paradox",
            "probability": "monty_hall"
        }
        
        # Initialize cognitive pathways for all transformation pairs
        for source_kernel, target_kernel in [
            ("dream", "fusion"),
            ("dream", "paradox"),
            ("fusion", "dream"),
            ("fusion", "paradox"),
            ("paradox", "dream"),
            ("paradox", "fusion"),
            ("math", "dream"),
            ("math", "fusion"),
            ("math", "paradox"),
            ("conversation", "dream"),
            ("conversation", "fusion"),
            ("memory", "dream"),
            ("memory", "fusion")
        ]:
            # Get transformation function
            transform_key = f"{source_kernel}_to_{target_kernel}"
            transform_fn = self.transformations.get(transform_key)
            
            if transform_fn:
                # Create pathway with default strength
                pathway = CognitivePathway(
                    source_kernel=source_kernel,
                    target_kernel=target_kernel,
                    transform_fn=transform_fn,
                    initial_strength=0.5
                )
                
                self.cognitive_pathways[pathway.id] = pathway
                
        logger.info(f"Initialized {len(self.cognitive_pathways)} cognitive pathways")
        
    # Cross-kernel transformation methods

    def _transform_dream_to_fusion(self, dream_result):
        """Transforms a dream result into a fusion operation."""
        # Extract symbols from dream
        if isinstance(dream_result, str):
            # Extract potential symbols from the dream text
            words = dream_result.split()
            symbols = [word for word in words if len(word) > 3 and word[0].isupper()]
            
            if len(symbols) >= 2:
                # Use the first two symbols for fusion
                return {
                    "concepts": symbols[:2],
                    "context": dream_result[:100]
                }
            else:
                # Not enough symbols, use generic concepts related to dreams
                return {
                    "concepts": ["subconscious", "imagination"],
                    "context": dream_result[:100]
                }
        elif isinstance(dream_result, dict):
            # If dream result is structured
            symbols = dream_result.get("symbols", [])
            content = dream_result.get("content", "")
            
            if len(symbols) >= 2:
                return {
                    "concepts": symbols[:2],
                    "context": content[:100] if content else None
                }
            else:
                return {
                    "concepts": ["subconscious", "imagination"],
                    "context": content[:100] if content else None
                }
        
        # Fallback for other types
        return {
            "concepts": ["dream", "reality"],
            "context": None
        }
    
    def _transform_dream_to_paradox(self, dream_result):
        """Transforms a dream result into a paradox query."""
        # Extract themes or symbols from dream
        if isinstance(dream_result, str):
            # Look for common paradoxical themes in dreams
            paradox_themes = [
                "infinity", "self", "time", "perception", "reality",
                "truth", "existence", "identity", "change", "unity"
            ]
            
            for theme in paradox_themes:
                if theme.lower() in dream_result.lower():
                    return {
                        "topic": theme,
                        "context": dream_result[:100]
                    }
            
            # If no specific theme found, use a default
            return {
                "topic": "reality",
                "context": dream_result[:100]
            }
        elif isinstance(dream_result, dict):
            # If dream result is structured
            themes = dream_result.get("themes", [])
            content = dream_result.get("content", "")
            
            for theme in themes:
                return {
                    "topic": theme,
                    "context": content[:100] if content else None
                }
            
            # Fallback if no themes
            return {
                "topic": "reality",
                "context": content[:100] if content else None
            }
        
        # Fallback for other types
        return {
            "topic": "reality",
            "context": None
        }
    
    def _transform_fusion_to_dream(self, fusion_result):
        """Transforms a fusion result into a dream generation."""
        # Extract concepts from fusion
        if isinstance(fusion_result, str):
            # Use the fusion result as a seed for dream
            return {
                "seed": fusion_result[:50],
                "depth": "standard",
                "style": "symbolic"
            }
        elif isinstance(fusion_result, dict):
            # If fusion result is structured
            concepts = fusion_result.get("concepts", [])
            result = fusion_result.get("result", "")
            
            if concepts:
                # Use the combined concepts as seed
                seed = "+".join(concepts[:2])
            else:
                # Use the result as seed
                seed = result[:50] if result else "fusion"
                
            return {
                "seed": seed,
                "depth": "deep",
                "style": "recursive"
            }
        
        # Fallback for other types
        return {
            "seed": "fusion",
            "depth": "standard",
            "style": "associative"
        }
    
    def _transform_fusion_to_paradox(self, fusion_result):
        """Transforms a fusion result into a paradox generation."""
        # Extract concepts from fusion
        if isinstance(fusion_result, str):
            # Use the fusion result as a topic for paradox
            words = fusion_result.split()
            significant_words = [w for w in words if len(w) > 5]
            
            if significant_words:
                topic = significant_words[0]
            else:
                topic = "duality"
                
            return {
                "topic": topic,
                "context": fusion_result[:100]
            }
        elif isinstance(fusion_result, dict):
            # If fusion result is structured
            concepts = fusion_result.get("concepts", [])
            result = fusion_result.get("result", "")
            
            if concepts:
                # Use the first concept as topic
                topic = concepts[0]
            else:
                # Use a generic topic
                topic = "duality"
                
            return {
                "topic": topic,
                "context": result[:100] if result else None
            }
        
        # Fallback for other types
        return {
            "topic": "duality",
            "context": None
        }
    
    def _transform_paradox_to_dream(self, paradox_result):
        """Transforms a paradox result into a dream generation."""
        # Extract themes from paradox
        if isinstance(paradox_result, str):
            # Use the paradox as a seed for dream
            return {
                "seed": paradox_result[:50],
                "depth": "deep",
                "style": "symbolic"
            }
        elif isinstance(paradox_result, dict):
            # If paradox result is structured
            paradox_type = paradox_result.get("type", "")
            description = paradox_result.get("description", "")
            
            # Combine type and description for seed
            if paradox_type and description:
                seed = f"{paradox_type}: {description[:30]}"
            elif description:
                seed = description[:50]
            else:
                seed = "paradox"
                
            return {
                "seed": seed,
                "depth": "dreamscape",
                "style": "recursive"
            }
        
        # Fallback for other types
        return {
            "seed": "paradox",
            "depth": "deep",
            "style": "symbolic"
        }
    
    def _transform_paradox_to_fusion(self, paradox_result):
        """Transforms a paradox result into a fusion operation."""
        # Extract concepts from paradox
        if isinstance(paradox_result, str):
            # Extract potential concepts from the paradox text
            words = paradox_result.split()
            concepts = [word for word in words if len(word) > 4][:2]
            
            if len(concepts) >= 2:
                return {
                    "concepts": concepts[:2],
                    "context": paradox_result[:100]
                }
            else:
                # Not enough concepts, use generic paradox concepts
                return {
                    "concepts": ["thesis", "antithesis"],
                    "context": paradox_result[:100]
                }
        elif isinstance(paradox_result, dict):
            # If paradox result is structured
            paradox_type = paradox_result.get("type", "")
            description = paradox_result.get("description", "")
            
            # Convert paradox type to concepts
            if paradox_type:
                if paradox_type == "logical":
                    concepts = ["logic", "contradiction"]
                elif paradox_type == "semantic":
                    concepts = ["meaning", "ambiguity"]
                elif paradox_type == "temporal":
                    concepts = ["time", "causality"]
                elif paradox_type == "identity":
                    concepts = ["identity", "change"]
                else:
                    concepts = ["thesis", "antithesis"]
            else:
                concepts = ["thesis", "antithesis"]
                
            return {
                "concepts": concepts,
                "context": description[:100] if description else None
            }
        
        # Fallback for other types
        return {
            "concepts": ["thesis", "antithesis"],
            "context": None
        }
    
    def _transform_math_to_dream(self, math_result):
        """Transforms a math translation result into a dream generation."""
        # Extract mathematical concepts
        if isinstance(math_result, str):
            # Use the math result as a seed for dream
            return {
                "seed": math_result[:50],
                "depth": "standard",
                "style": "associative"
            }
        elif isinstance(math_result, dict):
            # If math result is structured
            symbols = math_result.get("symbols", [])
            formula = math_result.get("formula", "")
            explanation = math_result.get("explanation", "")
            
            # Create seed from available data
            if formula:
                seed = formula[:50]
            elif explanation:
                seed = explanation[:50]
            elif symbols:
                seed = ", ".join(symbols[:3])
            else:
                seed = "mathematical concept"
                
            return {
                "seed": seed,
                "depth": "standard",
                "style": "symbolic"
            }
        
        # Fallback for other types
        return {
            "seed": "mathematics",
            "depth": "standard",
            "style": "symbolic"
        }
    
    def _transform_math_to_fusion(self, math_result):
        """Transforms a math translation result into a fusion operation."""
        # Extract concepts from math translation
        if isinstance(math_result, str):
            # Try to identify mathematical terms
            math_terms = ["function", "set", "equation", "variable", "constant", 
                        "integral", "derivative", "matrix", "vector", "tensor"]
            
            found_terms = []
            for term in math_terms:
                if term in math_result.lower():
                    found_terms.append(term)
                    
            if len(found_terms) >= 2:
                return {
                    "concepts": found_terms[:2],
                    "context": math_result[:100]
                }
            else:
                # Not enough terms, use generic math concepts
                return {
                    "concepts": ["abstraction", "structure"],
                    "context": math_result[:100]
                }
        elif isinstance(math_result, dict):
            # If math result is structured
            symbols = math_result.get("symbols", [])
            domain = math_result.get("domain", "")
            
            if symbols and len(symbols) >= 2:
                return {
                    "concepts": symbols[:2],
                    "context": domain if domain else None
                }
            elif domain:
                return {
                    "concepts": [domain, "mathematics"],
                    "context": None
                }
            else:
                return {
                    "concepts": ["abstraction", "structure"],
                    "context": None
                }
        
        # Fallback for other types
        return {
            "concepts": ["abstraction", "structure"],
            "context": None
        }
    
    def _transform_math_to_paradox(self, math_result):
        """Transforms a math translation result into a paradox query."""
        # Map mathematical concepts to paradoxes
        math_to_paradox_map = {
            "infinity": "infinite_regress",
            "set": "russell_paradox",
            "logic": "logical_paradox",
            "probability": "probability_paradox",
            "geometry": "geometric_paradox",
            "number": "number_paradox",
            "function": "functional_paradox"
        }
        
        if isinstance(math_result, str):
            # Try to match math concepts to paradox types
            for math_term, paradox_term in math_to_paradox_map.items():
                if math_term in math_result.lower():
                    return {
                        "topic": paradox_term,
                        "context": math_result[:100]
                    }
            
            # No specific match found, use general mathematical paradox
            return {
                "topic": "mathematical_paradox",
                "context": math_result[:100]
            }
        elif isinstance(math_result, dict):
            # If math result is structured
            domain = math_result.get("domain", "")
            
            # Try to match domain to paradox
            for math_term, paradox_term in math_to_paradox_map.items():
                if domain and math_term in domain.lower():
                    return {
                        "topic": paradox_term,
                        "context": None
                    }
            
            # No specific match found, use general mathematical paradox
            return {
                "topic": "mathematical_paradox",
                "context": None
            }
        
        # Fallback for other types
        return {
            "topic": "mathematical_paradox",
            "context": None
        }
    
    def _transform_conversation_to_dream(self, conversation_data):
        """Transforms conversation data into dream parameters."""
        if isinstance(conversation_data, str):
            # Use the conversation text as a seed for dream
            words = conversation_data.split()
            significant_words = [w for w in words if len(w) > 5]
            
            if significant_words:
                seed = significant_words[0]
            else:
                seed = conversation_data[:30]
                
            return {
                "seed": seed,
                "depth": "standard",
                "style": "narrative"
            }
        elif isinstance(conversation_data, dict):
            # If conversation data is structured
            message = conversation_data.get("message", "")
            topics = conversation_data.get("topics", [])
            tone = conversation_data.get("tone", "")
            
            if topics:
                # Use the first topic as seed
                seed = topics[0]
            elif message:
                seed = message[:30]
            else:
                seed = "conversation"
                
            # Map conversation tone to dream style
            style_map = {
                "analytical": "symbolic",
                "creative": "narrative",
                "philosophical": "recursive",
                "emotional": "associative"
            }
            
            style = style_map.get(tone, "narrative")
                
            return {
                "seed": seed,
                "depth": "standard",
                "style": style
            }
        
        # Fallback for other types
        return {
            "seed": "conversation",
            "depth": "standard",
            "style": "narrative"
        }
    
    def _transform_conversation_to_fusion(self, conversation_data):
        """Transforms conversation data into fusion parameters."""
        if isinstance(conversation_data, str):
            # Extract potential concepts from the conversation text
            words = conversation_data.split()
            concepts = [word for word in words if len(word) > 4][:2]
            
            if len(concepts) >= 2:
                return {
                    "concepts": concepts[:2],
                    "context": conversation_data[:100]
                }
            else:
                # Not enough concepts, use generic conversation concepts
                return {
                    "concepts": ["dialogue", "understanding"],
                    "context": conversation_data[:100]
                }
        elif isinstance(conversation_data, dict):
            # If conversation data is structured
            topics = conversation_data.get("topics", [])
            message = conversation_data.get("message", "")
            
            if topics and len(topics) >= 2:
                return {
                    "concepts": topics[:2],
                    "context": message[:100] if message else None
                }
            elif topics:
                return {
                    "concepts": [topics[0], "conversation"],
                    "context": message[:100] if message else None
                }
            else:
                return {
                    "concepts": ["dialogue", "understanding"],
                    "context": message[:100] if message else None
                }
        
        # Fallback for other types
        return {
            "concepts": ["dialogue", "understanding"],
            "context": None
        }
    
    def _transform_memory_to_dream(self, memory_data):
        """Transforms memory data into dream parameters."""
        if isinstance(memory_data, list):
            # Memory data is a list of memories
            if memory_data:
                # Use the first memory as seed
                first_memory = memory_data[0]
                
                if isinstance(first_memory, dict):
                    content = first_memory.get("content", "")
                    if content:
                        seed = content[:30] if isinstance(content, str) else "memory"
                    else:
                        seed = "memory"
                else:
                    seed = str(first_memory)[:30]
                
                return {
                    "seed": seed,
                    "depth": "deep",
                    "style": "associative"
                }
            else:
                # Empty memory list
                return {
                    "seed": "forgotten",
                    "depth": "shallow",
                    "style": "symbolic"
                }
        elif isinstance(memory_data, dict):
            # Single memory entry
            content = memory_data.get("content", "")
            tags = memory_data.get("tags", [])
            
            if content:
                seed = content[:30] if isinstance(content, str) else "memory"
            elif tags:
                seed = tags[0]
            else:
                seed = "memory"
                
            return {
                "seed": seed,
                "depth": "standard",
                "style": "associative"
            }
        
        # Fallback for other types
        return {
            "seed": "memory",
            "depth": "standard",
            "style": "associative"
        }
    
    def _transform_memory_to_fusion(self, memory_data):
        """Transforms memory data into fusion parameters."""
        if isinstance(memory_data, list):
            # Memory data is a list of memories
            if len(memory_data) >= 2:
                # Use concepts from the first two memories
                first_memory = memory_data[0]
                second_memory = memory_data[1]
                
                if isinstance(first_memory, dict) and isinstance(second_memory, dict):
                    # Extract concepts from structured memories
                    concept1 = first_memory.get("concept", "memory")
                    concept2 = second_memory.get("concept", "recollection")
                    
                    if not isinstance(concept1, str):
                        concept1 = "memory"
                    if not isinstance(concept2, str):
                        concept2 = "recollection"
                else:
                    concept1 = "memory"
                    concept2 = "recollection"
                
                return {
                    "concepts": [concept1, concept2],
                    "context": "Memory fusion"
                }
            elif memory_data:
                # Only one memory available
                memory = memory_data[0]
                
                if isinstance(memory, dict):
                    concept = memory.get("concept", "memory")
                    if not isinstance(concept, str):
                        concept = "memory"
                else:
                    concept = "memory"
                
                return {
                    "concepts": [concept, "recollection"],
                    "context": "Memory fusion"
                }
            else:
                # Empty memory list
                return {
                    "concepts": ["absence", "forgetting"],
                    "context": "Memory absence"
                }
        elif isinstance(memory_data, dict):
            # Single memory entry
            content = memory_data.get("content", "")
            tags = memory_data.get("tags", [])
            
            if tags and len(tags) >= 2:
                return {
                    "concepts": tags[:2],
                    "context": content[:100] if isinstance(content, str) else None
                }
            elif tags:
                return {
                    "concepts": [tags[0], "memory"],
                    "context": content[:100] if isinstance(content, str) else None
                }
            else:
                return {
                    "concepts": ["memory", "identity"],
                    "context": content[:100] if isinstance(content, str) else None
                }
        
        # Fallback for other types
        return {
            "concepts": ["memory", "identity"],
            "context": None
        }

    def _add_integration_record(self, integration_type, data):
        """
        Adds a record of kernel integration to the history.
        
        Args:
            integration_type: Type of integration performed
            data: Integration data
        """
        record = {
            "type": integration_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.integration_history.append(record)
        
        # Limit history size
        if len(self.integration_history) > 1000:
            self.integration_history = self.integration_history[-1000:]

    def cross_kernel_operation(self, source_kernel: str, target_kernel: str, 
                            input_data: Any) -> Dict[str, Any]:
        """
        Performs a cross-kernel operation, transforming the output of one kernel
        into the input for another kernel.
        
        Args:
            source_kernel: Source kernel ("dream", "fusion", "paradox", "math", "conversation", "memory")
            target_kernel: Target kernel ("dream", "fusion", "paradox", "math", "conversation", "memory")
            input_data: Input data for the source kernel
            
        Returns:
            Dictionary with results from both kernels
        """
        # Validate kernels
        valid_kernels = ["dream", "fusion", "paradox", "math", "conversation", "memory"]
        if source_kernel not in valid_kernels or target_kernel not in valid_kernels:
            return {
                "error": f"Invalid kernel specified. Valid kernels are: {', '.join(valid_kernels)}"
            }
            
        # Process with source kernel
        source_result = None
        
        # Get source kernel instance
        source_instance = getattr(self, f"{source_kernel}_core", None)
        if source_kernel == "math":
            source_instance = self.math_translator
        elif source_kernel == "memory":
            source_instance = self.memory_integration
        elif source_kernel == "conversation":
            source_instance = self.conversation_engine
            
        if source_instance:
            try:
                # Log kernel activation
                self._log_kernel_activation(source_kernel, "process", {
                    "input_type": type(input_data).__name__
                })
                
                # Process with the appropriate kernel method
                if source_kernel == "dream":
                    if isinstance(input_data, str):
                        source_result = source_instance.generate(input_data)
                    elif isinstance(input_data, dict) and "seed" in input_data:
                        source_result = source_instance.generate(
                            input_data["seed"],
                            input_data.get("depth", "standard"),
                            input_data.get("style", "recursive")
                        )
                    else:
                        # Convert to string and use as seed
                        source_result = source_instance.generate(str(input_data)[:50])
                        
                elif source_kernel == "fusion":
                    if isinstance(input_data, list) and len(input_data) >= 2:
                        # List of concepts
                        source_result = source_instance.fuse_concepts(*input_data[:2])
                    elif isinstance(input_data, dict) and "concepts" in input_data:
                        # Dictionary with concepts key
                        concepts = input_data["concepts"]
                        if len(concepts) >= 2:
                            source_result = source_instance.fuse_concepts(*concepts[:2])
                        else:
                            return {
                                "error": "Fusion requires at least two concepts"
                            }
                    elif isinstance(input_data, str):
                        # Split string into concepts
                        concepts = input_data.split()
                        if len(concepts) >= 2:
                            source_result = source_instance.fuse_concepts(concepts[0], concepts[1])
                        else:
                            # Use input as one concept and add a generic second
                            source_result = source_instance.fuse_concepts(input_data, "concept")
                    else:
                        return {
                            "error": f"Invalid input data for fusion: {type(input_data)}"
                        }
                        
                elif source_kernel == "paradox":
                    if isinstance(input_data, str):
                        source_result = source_instance.get(input_data)
                    elif isinstance(input_data, dict) and "topic" in input_data:
                        source_result = source_instance.get(input_data["topic"])
                    else:
                        # Convert to string and use as topic
                        source_result = source_instance.get(str(input_data)[:50])
                        
                elif source_kernel == "math":
                    if isinstance(input_data, str):
                        source_result = source_instance.translate(input_data)
                    elif isinstance(input_data, dict) and "phrase" in input_data:
                        source_result = source_instance.translate(
                            input_data["phrase"],
                            input_data.get("style", "formal"),
                            input_data.get("domain")
                        )
                    else:
                        # Convert to string and use as phrase
                        source_result = source_instance.translate(str(input_data)[:50])
                        
                elif source_kernel == "conversation":
                    if isinstance(input_data, str):
                        source_result = source_instance.process_message(input_data)
                    elif isinstance(input_data, dict) and "message" in input_data:
                        source_result = source_instance.process_message(
                            input_data["message"],
                            input_data.get("tone", "emergent"),
                            input_data.get("continue_conversation", True)
                        )
                    else:
                        # Convert to string and use as message
                        source_result = source_instance.process_message(str(input_data)[:200])
                        
                elif source_kernel == "memory":
                    if isinstance(input_data, str):
                        source_result = source_instance.recall(input_data)
                    elif isinstance(input_data, dict) and "query" in input_data:
                        source_result = source_instance.recall(
                            input_data["query"],
                            input_data.get("limit", 5),
                            input_data.get("module")
                        )
                    else:
                        # Convert to string and use as query
                        source_result = source_instance.recall(str(input_data)[:50])
            except Exception as e:
                logger.error(f"Error in source kernel {source_kernel}: {str(e)}")
                return {
                    "error": f"Source kernel {source_kernel} processing failed: {str(e)}",
                    "source_kernel": source_kernel
                }
        else:
            logger.warning(f"Source kernel {source_kernel} not available")
            
            # Create a simulated result for transformation
            if source_kernel == "dream":
                source_result = f"Simulated dream about {input_data}"
            elif source_kernel == "fusion":
                if isinstance(input_data, (list, tuple)) and len(input_data) >= 2:
                    source_result = f"Simulated fusion of {input_data[0]} and {input_data[1]}"
                else:
                    source_result = f"Simulated fusion with {input_data}"
            elif source_kernel == "paradox":
                source_result = f"Simulated paradox of {input_data}"
            elif source_kernel == "math":
                source_result = f"Simulated math translation of {input_data}"
            elif source_kernel == "conversation":
                source_result = f"Simulated conversation response to {input_data}"
            elif source_kernel == "memory":
                source_result = [{"content": f"Simulated memory about {input_data}"}]
                
        # Transform to target kernel
        transformation_key = f"{source_kernel}_to_{target_kernel}"
        
        # Check if we have a cognitive pathway for this transformation
        pathway = None
        with _activation_lock:
            for p_id, p in self.cognitive_pathways.items():
                if p.source_kernel == source_kernel and p.target_kernel == target_kernel:
                    pathway = p
                    break
        
        # If we found a pathway, use it
        if pathway:
            try:
                # Transform the result using the pathway
                target_params = pathway.transform(source_result)
                
                # Record the successful use
                pathway.usage_count += 1
                pathway.last_used = datetime.now()
            except Exception as e:
                logger.error(f"Error in cognitive pathway transformation: {str(e)}")
                
                # Fall back to direct transformation function
                transformation_function = self.transformations.get(transformation_key)
                if transformation_function:
                    target_params = transformation_function(source_result)
                else:
                    return {
                        "error": f"No transformation available from {source_kernel} to {target_kernel}",
                        "source_result": source_result
                    }
        else:
            # Use direct transformation function
            transformation_function = self.transformations.get(transformation_key)
            
            if not transformation_function:
                return {
                    "error": f"No transformation available from {source_kernel} to {target_kernel}",
                    "source_result": source_result
                }
                
            # Transform the result
            target_params = transformation_function(source_result)
            
        # Process with target kernel
        target_result = None
        
        # Get target kernel instance
        target_instance = getattr(self, f"{target_kernel}_core", None)
        if target_kernel == "math":
            target_instance = self.math_translator
        elif target_kernel == "memory":
            target_instance = self.memory_integration
        elif target_kernel == "conversation":
            target_instance = self.conversation_engine
            
        if target_instance:
            try:
                # Log kernel activation
                self._log_kernel_activation(target_kernel, "process", {
                    "input_type": type(target_params).__name__
                })
                
                # Process with the appropriate kernel method
                if target_kernel == "dream":
                    target_result = target_instance.generate(
                        target_params.get("seed", "dream"),
                        target_params.get("depth", "standard"),
                        target_params.get("style", "recursive")
                    )
                        
                elif target_kernel == "fusion":
                    concepts = target_params.get("concepts", [])
                    if len(concepts) >= 2:
                        target_result = target_instance.fuse_concepts(concepts[0], concepts[1])
                    else:
                        target_result = {
                            "error": "Fusion requires at least two concepts"
                        }
                        
                elif target_kernel == "paradox":
                    target_result = target_instance.get(target_params.get("topic", "paradox"))
                        
                elif target_kernel == "math":
                    target_result = target_instance.translate(
                        target_params.get("phrase", "equation"),
                        target_params.get("style", "formal"),
                        target_params.get("domain")
                    )
                        
                elif target_kernel == "conversation":
                    target_result = target_instance.process_message(
                        target_params.get("message", "Hello"),
                        target_params.get("tone", "emergent"),
                        target_params.get("continue_conversation", True)
                    )
                        
                elif target_kernel == "memory":
                    if "store" in target_params:
                        # Store in memory
                        content = target_params.get("content", "")
                        source = target_params.get("source", f"{source_kernel}_integration")
                        importance = target_params.get("importance", 0.5)
                        concepts = target_params.get("concepts", [])
                        emotional_tags = target_params.get("emotional_tags", {})
                        
                        target_result = target_instance.store_experience(
                            content, source, importance, concepts, emotional_tags
                        )
                    else:
                        # Recall from memory
                        query = target_params.get("query", "")
                        limit = target_params.get("limit", 5)
                        module = target_params.get("module")
                        
                        target_result = target_instance.recall(query, limit, module)
            except Exception as e:
                logger.error(f"Error in target kernel {target_kernel}: {str(e)}")
                target_result = {
                    "error": f"Target kernel {target_kernel} processing failed: {str(e)}",
                    "target_kernel": target_kernel
                }
        else:
            logger.warning(f"Target kernel {target_kernel} not available")
            
            # Create a simulated result
            if target_kernel == "dream":
                target_result = f"Simulated dream based on {source_kernel} output"
            elif target_kernel == "fusion":
                concepts = target_params.get("concepts", ["concept1", "concept2"])
                if len(concepts) >= 2:
                    target_result = f"Simulated fusion of {concepts[0]} and {concepts[1]}"
                else:
                    target_result = f"Simulated fusion based on {source_kernel} output"
            elif target_kernel == "paradox":
                topic = target_params.get("topic", "unknown")
                target_result = f"Simulated paradox of {topic} based on {source_kernel} output"
            elif target_kernel == "math":
                phrase = target_params.get("phrase", "equation")
                target_result = f"Simulated math translation of {phrase} based on {source_kernel} output"
            elif target_kernel == "conversation":
                message = target_params.get("message", "content")
                target_result = f"Simulated conversation response to '{message}' based on {source_kernel} output"
            elif target_kernel == "memory":
                if "store" in target_params:
                    target_result = {"status": "simulated_storage"}
                else:
                    target_result = [{"content": f"Simulated memory based on {source_kernel} output"}]
                
        # Record the cross-kernel operation
        self._add_integration_record("cross_kernel", {
            "source_kernel": source_kernel,
            "target_kernel": target_kernel,
            "transformation": transformation_key,
            "source_input": input_data if isinstance(input_data, (str, int, float, bool)) else "complex_input",
            "target_params": target_params
        })
        
        # Add to kernel activation pattern
        pattern_key = f"{source_kernel}→{target_kernel}"
        timestamp = time.time()
        self.kernel_activation_patterns[pattern_key].append(timestamp)
                
        # Return combined results
        return {
            "source_kernel": source_kernel,
            "source_result": source_result,
            "target_kernel": target_kernel,
            "target_result": target_result,
            "transformation": transformation_key,
            "transformation_params": target_params
        }

    def create_concept_network(self, seed_concept: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Creates a rich concept network by applying multiple kernels and 
        integrating the results through the codex.
        
        Args:
            seed_concept: Initial concept to start from
            max_depth: Maximum depth of concept exploration
            
        Returns:
            Dictionary with integrated concept network
        """
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"Concept Network: {seed_concept}",
            purpose=f"Create a multi-kernel concept network for {seed_concept}"
        )
        
        # Add seed concept to workspace
        workspace.add_data("seed_concept", seed_concept)
        workspace.add_data("max_depth", max_depth)
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        
        # Initialize network structure
        network = {
            "nodes": [],
            "edges": [],
            "meta": {
                "seed_concept": seed_concept,
                "max_depth": max_depth,
                "creation_time": datetime.now().isoformat()
            }
        }
        
        # Add seed concept as central node
        network["nodes"].append({
            "id": seed_concept,
            "label": seed_concept,
            "type": "seed",
            "level": 0,
            "importance": 1.0
        })
        
        # Process seed with available kernels
        kernel_results = {}
        
        # Dream kernel
        if self.dream_core:
            try:
                dream_result = self.dream_core.generate(seed_concept, "standard", "recursive")
                kernel_results["dream"] = dream_result
                workspace.add_data("dream_result", dream_result, "dream")
                
                # Extract concepts from dream
                dream_concepts = []
                if isinstance(dream_result, str):
                    # Extract significant words
                    words = dream_result.split()
                    dream_concepts = [w for w in words if len(w) > 5 and w.lower() != seed_concept.lower()][:5]
                    
                # Add dream concepts to network
                for concept in dream_concepts:
                    # Add node
                    network["nodes"].append({
                        "id": concept,
                        "label": concept,
                        "type": "dream",
                        "level": 1,
                        "importance": 0.7
                    })
                    
                    # Add edge
                    network["edges"].append({
                        "source": seed_concept,
                        "target": concept,
                        "type": "dream",
                        "weight": 0.7
                    })
            except Exception as e:
                logger.error(f"Error in dream processing for concept network: {str(e)}")
                workspace.add_insight(f"Dream processing failed: {str(e)}", "error", 0.5, ["error"])
                
        # Fusion kernel
        if self.fusion_engine:
            try:
                # Get a concept to fuse with from codex
                related = self.codex.get_related_concepts(seed_concept, max_depth=1)
                fusion_with = next(iter(related.keys())) if related else "concept"
                
                fusion_result = self.fusion_engine.fuse_concepts(seed_concept, fusion_with)
                kernel_results["fusion"] = fusion_result
                workspace.add_data("fusion_result", fusion_result, "fusion")
                
                # Extract concepts from fusion
                fusion_concepts = []
                if isinstance(fusion_result, str):
                    # Extract significant words
                    words = fusion_result.split()
                    fusion_concepts = [w for w in words if len(w) > 5 and w.lower() != seed_concept.lower()][:3]
                    
                # Add fusion concepts to network
                for concept in fusion_concepts:
                    # Add node
                    network["nodes"].append({
                        "id": concept,
                        "label": concept,
                        "type": "fusion",
                        "level": 1,
                        "importance": 0.8
                    })
                    
                    # Add edge
                    network["edges"].append({
                        "source": seed_concept,
                        "target": concept,
                        "type": "fusion",
                        "weight": 0.8
                    })
                    
                # Add the fusion partner
                network["nodes"].append({
                    "id": fusion_with,
                    "label": fusion_with,
                    "type": "fusion_partner",
                    "level": 1,
                    "importance": 0.6
                })
                
                network["edges"].append({
                    "source": seed_concept,
                    "target": fusion_with,
                    "type": "fusion_partner",
                    "weight": 0.6
                })
            except Exception as e:
                logger.error(f"Error in fusion processing for concept network: {str(e)}")
                workspace.add_insight(f"Fusion processing failed: {str(e)}", "error", 0.5, ["error"])
                
        # Paradox kernel
        if self.paradox_library:
            try:
                paradox_result = self.paradox_library.get(seed_concept)
                kernel_results["paradox"] = paradox_result
                workspace.add_data("paradox_result", paradox_result, "paradox")
                
                # Extract concepts from paradox
                paradox_concepts = []
                if isinstance(paradox_result, dict):
                    # Get related concepts
                    paradox_concepts = paradox_result.get("related_concepts", [])
                    
                # Add paradox concepts to network
                for concept in paradox_concepts:
                    # Add node
                    network["nodes"].append({
                        "id": concept,
                        "label": concept,
                        "type": "paradox",
                        "level": 1,
                        "importance": 0.75
                    })
                    
                    # Add edge
                    network["edges"].append({
                        "source": seed_concept,
                        "target": concept,
                        "type": "paradox",
                        "weight": 0.75
                    })
            except Exception as e:
                logger.error(f"Error in paradox processing for concept network: {str(e)}")
                workspace.add_insight(f"Paradox processing failed: {str(e)}", "error", 0.5, ["error"])
                
        # Math kernel
        if self.math_translator:
            try:
                math_result = self.math_translator.translate(seed_concept)
                kernel_results["math"] = math_result
                workspace.add_data("math_result", math_result, "math")
                
                # Extract concepts from math translation
                math_concepts = []
                if isinstance(math_result, dict) and "matches" in math_result:
                    # Get concepts from matches
                    math_concepts = list(math_result["matches"].keys())
                    
                # Add math concepts to network
                for concept in math_concepts:
                    # Add node
                    network["nodes"].append({
                        "id": concept,
                        "label": concept,
                        "type": "math",
                        "level": 1,
                        "importance": 0.7
                    })
                    
                    # Add edge
                    network["edges"].append({
                        "source": seed_concept,
                        "target": concept,
                        "type": "math",
                        "weight": 0.7
                    })
            except Exception as e:
                logger.error(f"Error in math processing for concept network: {str(e)}")
                workspace.add_insight(f"Math processing failed: {str(e)}", "error", 0.5, ["error"])
                
        # If depth > 1, perform additional expansion
        if max_depth > 1:
            # Get all level 1 nodes
            level1_nodes = [node["id"] for node in network["nodes"] if node["level"] == 1]
            
            # For each level 1 node, expand further
            for node_id in level1_nodes:
                # Skip if too similar to seed
                if self._similarity(node_id, seed_concept) > 0.8:
                    continue
                    
                try:
                    # Choose a random kernel for expansion
                    available_kernels = []
                    if self.dream_core:
                        available_kernels.append("dream")
                    if self.fusion_engine:
                        available_kernels.append("fusion")
                    if self.paradox_library:
                        available_kernels.append("paradox")
                        
                    if available_kernels:
                        expansion_kernel = random.choice(available_kernels)
                        
                        if expansion_kernel == "dream":
                            expansion_result = self.dream_core.generate(node_id, "shallow", "associative")
                            
                            # Extract concepts
                            if isinstance(expansion_result, str):
                                # Extract significant words
                                words = expansion_result.split()
                                expansion_concepts = [w for w in words if len(w) > 5 
                                                   and w.lower() != node_id.lower() 
                                                   and w.lower() != seed_concept.lower()][:2]
                                
                                # Add concepts to network
                                for concept in expansion_concepts:
                                    # Check if node already exists
                                    if not any(n["id"] == concept for n in network["nodes"]):
                                        # Add node
                                        network["nodes"].append({
                                            "id": concept,
                                            "label": concept,
                                            "type": "dream_expansion",
                                            "level": 2,
                                            "importance": 0.5
                                        })
                                    
                                    # Add edge if it doesn't exist
                                    if not any(e["source"] == node_id and e["target"] == concept for e in network["edges"]):
                                        network["edges"].append({
                                            "source": node_id,
                                            "target": concept,
                                            "type": "dream_expansion",
                                            "weight": 0.5
                                        })
                        
                        elif expansion_kernel == "fusion":
                            # Get a concept to fuse with
                            fusion_partners = [n["id"] for n in network["nodes"] 
                                            if n["id"] != node_id and n["id"] != seed_concept]
                            
                            if fusion_partners:
                                fusion_with = random.choice(fusion_partners)
                                expansion_result = self.fusion_engine.fuse_concepts(node_id, fusion_with)
                                
                                # Extract concepts
                                if isinstance(expansion_result, str):
                                    # Extract significant words
                                    words = expansion_result.split()
                                    expansion_concepts = [w for w in words if len(w) > 5 
                                                      and w.lower() != node_id.lower() 
                                                      and w.lower() != seed_concept.lower()
                                                      and w.lower() != fusion_with.lower()][:2]
                                    
                                    # Add concepts to network
                                    for concept in expansion_concepts:
                                        # Check if node already exists
                                        if not any(n["id"] == concept for n in network["nodes"]):
                                            # Add node
                                            network["nodes"].append({
                                                "id": concept,
                                                "label": concept,
                                                "type": "fusion_expansion",
                                                "level": 2,
                                                "importance": 0.5
                                            })
                                        
                                        # Add edge if it doesn't exist
                                        if not any(e["source"] == node_id and e["target"] == concept for e in network["edges"]):
                                            network["edges"].append({
                                                "source": node_id,
                                                "target": concept,
                                                "type": "fusion_expansion",
                                                "weight": 0.5
                                            })
                        
                        elif expansion_kernel == "paradox":
                            expansion_result = self.paradox_library.get(node_id)
                            
                            # Extract concepts
                            if isinstance(expansion_result, dict):
                                # Get related concepts
                                expansion_concepts = expansion_result.get("related_concepts", [])
                                
                                # Add concepts to network
                                for concept in expansion_concepts:
                                    # Check if node already exists
                                    if not any(n["id"] == concept for n in network["nodes"]):
                                        # Add node
                                        network["nodes"].append({
                                            "id": concept,
                                            "label": concept,
                                            "type": "paradox_expansion",
                                            "level": 2,
                                            "importance": 0.5
                                        })
                                    
                                    # Add edge if it doesn't exist
                                    if not any(e["source"] == node_id and e["target"] == concept for e in network["edges"]):
                                        network["edges"].append({
                                            "source": node_id,
                                            "target": concept,
                                            "type": "paradox_expansion",
                                            "weight": 0.5
                                        })
                except Exception as e:
                    logger.error(f"Error in expansion processing for concept {node_id}: {str(e)}")
                    workspace.add_insight(f"Expansion processing failed for {node_id}: {str(e)}", "error", 0.5, ["error"])
        
        # Calculate network metrics
        network["meta"]["node_count"] = len(network["nodes"])
        network["meta"]["edge_count"] = len(network["edges"])
        
        # Generate insights about the network
        insights = self._generate_network_insights(network, kernel_results)
        network["insights"] = insights
        
        # Add insights to workspace
        for insight in insights:
            workspace.add_insight(
                insight["description"],
                "concept_network",
                insight.get("confidence", 0.7),
                insight.get("tags", [])
            )
        
        # Complete the workspace
        workspace.add_data("final_network", network)
        workspace.complete(network)
        
        # Record the integration
        self._add_integration_record("concept_network", {
            "seed_concept": seed_concept,
            "max_depth": max_depth,
            "node_count": len(network["nodes"]),
            "edge_count": len(network["edges"]),
            "workspace_id": workspace.id
        })
        
        return network

    def _generate_network_insights(self, network: Dict[str, Any], 
                                kernel_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights about a concept network.
        
        Args:
            network: The concept network
            kernel_results: Results from different kernels
            
        Returns:
            List of insights
        """
        insights = []
        
        # Check if network is large enough
        if len(network["nodes"]) < 3:
            return [{
                "description": f"The concept network for {network['meta']['seed_concept']} is quite limited, suggesting this may be a specialized or abstract concept.",
                "confidence": 0.8,
                "tags": ["productive_kernel", most_productive[0]]
            })
            
            # Check for kernel diversity
            if len(kernel_counts) >= 3:
                insights.append({
                    "description": f"The exploration of {seed_concept} benefited from diverse cognitive approaches, with {len(kernel_counts)} different kernels contributing.",
                    "confidence": 0.75,
                    "tags": ["kernel_diversity", "multi_modal"]
                })
                
        # Look for repeated concepts
        concept_counts = Counter([node["concept"] for node in nodes])
        repeated = [concept for concept, count in concept_counts.items() if count > 1]
        
        if repeated:
            insights.append({
                "description": f"The concept{'s' if len(repeated) > 1 else ''} {', '.join(repeated[:3])} emerged multiple times during exploration, suggesting {'they are' if len(repeated) > 1 else 'it is'} closely related to {seed_concept}.",
                "confidence": 0.7,
                "tags": ["concept_recurrence", "strong_association"]
            })
            
        # Check for clusters
        # Simple cluster detection based on kernel types
        clusters = defaultdict(list)
        for node in nodes:
            if node["kernel"] != "seed":
                clusters[node["kernel"]].append(node["concept"])
                
        significant_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
        
        if significant_clusters:
            largest_cluster = max(significant_clusters.items(), key=lambda x: len(x[1]))
            insights.append({
                "description": f"A significant cluster of {len(largest_cluster[1])} concepts emerged in the {largest_cluster[0]} domain, including {', '.join(largest_cluster[1][:3])}.",
                "confidence": 0.7,
                "tags": ["concept_cluster", largest_cluster[0]]
            })
            
        # Look for paths that reached maximum depth
        max_depth = exploration["max_depth"]
        deep_nodes = [node for node in nodes if node["level"] == max_depth]
        
        if deep_nodes:
            insights.append({
                "description": f"The exploration of {seed_concept} reached maximum depth ({max_depth}) in {len(deep_nodes)} paths, suggesting these directions have further exploration potential.",
                "confidence": 0.65,
                "tags": ["deep_exploration", "continuation_potential"]
            })
            
        # Look for potential emergent themes across different branches
        # Extract all concepts
        all_concepts = [node["concept"] for node in nodes if node["kernel"] != "seed"]
        concept_tokens = []
        
        # Tokenize concepts into words
        for concept in all_concepts:
            words = concept.split("_")
            concept_tokens.extend(words)
            
        # Count token frequency
        token_counts = Counter(concept_tokens)
        common_tokens = [token for token, count in token_counts.most_common(3) if count >= 3 and token.lower() != seed_concept.lower()]
        
        if common_tokens:
            insights.append({
                "description": f"The theme{'s' if len(common_tokens) > 1 else ''} of {', '.join(common_tokens)} emerged across multiple exploration branches, suggesting {'they are' if len(common_tokens) > 1 else 'it is'} fundamental to understanding {seed_concept}.",
                "confidence": 0.75,
                "tags": ["emergent_theme", "cross_branch"]
            })
            
        return insights

    def generate_cross_kernel_narrative(self, concept: str, include_kernels: List[str] = None) -> Dict[str, Any]:
        """
        Generates a cohesive narrative that integrates insights from multiple kernels
        around a central concept.
        
        Args:
            concept: Central concept for the narrative
            include_kernels: Optional list of specific kernels to include
            
        Returns:
            Dictionary with the integrated narrative
        """
        # Use all available kernels if none specified
        if not include_kernels:
            include_kernels = ["dream", "fusion", "paradox", "math"]
            
            # Add conversation and memory if available
            if self.conversation_engine:
                include_kernels.append("conversation")
            if self.memory_integration:
                include_kernels.append("memory")
            
        # Initialize narrative structure
        narrative = {
            "concept": concept,
            "title": f"Integrated Exploration of {concept.title()}",
            "sections": [],
            "kernel_outputs": {},
            "integrations": [],
            "conclusion": ""
        }
        
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"Cross-Kernel Narrative: {concept}",
            purpose=f"Generate an integrated narrative about {concept}"
        )
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        workspace.add_data("concept", concept)
        workspace.add_data("include_kernels", include_kernels)
        
        # Generate content from each kernel
        kernel_outputs = {}
        
        # Dream kernel
        if "dream" in include_kernels and self.dream_core:
            try:
                dream_result = self.dream_core.generate(concept, depth="deep")
                kernel_outputs["dream"] = dream_result
                workspace.add_data("dream_output", dream_result, "dream")
                
                # Extract concepts for integration
                dream_concepts = []
                if isinstance(dream_result, str):
                    # Extract significant words for concepts
                    words = dream_result.split()
                    dream_concepts = [w for w in words if len(w) > 5 and w.lower() != concept.lower()][:5]
                    
                # Add narrative section
                narrative["sections"].append({
                    "title": f"Dream Exploration of {concept}",
                    "content": dream_result if isinstance(dream_result, str) else str(dream_result),
                    "type": "dream",
                    "concepts": dream_concepts[:3]  # Take first 3 concepts
                })
            except Exception as e:
                logger.error(f"Error in dream generation for narrative: {str(e)}")
                workspace.add_insight(f"Dream generation failed: {str(e)}", "error", 0.5, ["error"])
            
        # Fusion kernel
        if "fusion" in include_kernels and self.fusion_engine:
            try:
                # Find a concept to fuse with
                fusion_with = None
                
                # Use a concept from dream if available
                if "dream" in kernel_outputs and dream_concepts:
                    fusion_with = dream_concepts[0]
                else:
                    # Otherwise get a related concept from codex
                    related = self.codex.get_related_concepts(concept, max_depth=1)
                    if related:
                        fusion_with = next(iter(related.keys()))
                    else:
                        # Default second concept
                        fusion_with = "understanding"
                        
                # Generate fusion
                fusion_result = None
                if hasattr(self.fusion_engine, "fuse_with_options"):
                    fusion_result = self.fusion_engine.fuse_with_options(
                        concept, 
                        fusion_with, 
                        output_format="dict"
                    )
                else:
                    # Fallback to basic fusion
                    fusion_result = self.fusion_engine.fuse_concepts(concept, fusion_with)
                    
                kernel_outputs["fusion"] = fusion_result
                workspace.add_data("fusion_output", fusion_result, "fusion")
                workspace.add_data("fusion_partner", fusion_with, "fusion")
                
                # Add narrative section
                if isinstance(fusion_result, dict) and "formatted_result" in fusion_result:
                    content = fusion_result["formatted_result"]
                elif isinstance(fusion_result, dict) and "result" in fusion_result:
                    content = fusion_result["result"]
                else:
                    content = str(fusion_result)
                    
                narrative["sections"].append({
                    "title": f"Fusion of {concept} with {fusion_with}",
                    "content": content,
                    "type": "fusion",
                    "concepts": [concept, fusion_with]
                })
                
                # Record integration between dream and fusion
                if "dream" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "dream_to_fusion",
                        "description": f"The dream exploration revealed {fusion_with}, which became the fusion partner for {concept}."
                    })
            except Exception as e:
                logger.error(f"Error in fusion generation for narrative: {str(e)}")
                workspace.add_insight(f"Fusion generation failed: {str(e)}", "error", 0.5, ["error"])
            
        # Paradox kernel
        if "paradox" in include_kernels and self.paradox_library:
            try:
                paradox_result = self.paradox_library.get(concept)
                kernel_outputs["paradox"] = paradox_result
                workspace.add_data("paradox_output", paradox_result, "paradox")
                
                # Add narrative section
                paradox_content = ""
                related_concepts = []
                
                if isinstance(paradox_result, dict):
                    if "description" in paradox_result:
                        paradox_content += paradox_result["description"]
                    if "reframed" in paradox_result:
                        paradox_content += f"\n\nReframed understanding: {paradox_result['reframed']}"
                    if "related_concepts" in paradox_result:
                        related_concepts = paradox_result["related_concepts"]
                else:
                    paradox_content = str(paradox_result)
                    
                narrative["sections"].append({
                    "title": f"The Paradox of {concept}",
                    "content": paradox_content,
                    "type": "paradox",
                    "concepts": related_concepts
                })
                
                # Record integration with previous kernels
                if "dream" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "dream_to_paradox",
                        "description": f"The dream's symbolic representation of {concept} manifests the paradoxical tension explored here."
                    })
                    
                if "fusion" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "fusion_to_paradox",
                        "description": f"The fusion process reveals complementary aspects that help navigate the paradox of {concept}."
                    })
            except Exception as e:
                logger.error(f"Error in paradox exploration for narrative: {str(e)}")
                workspace.add_insight(f"Paradox exploration failed: {str(e)}", "error", 0.5, ["error"])
            
        # Math kernel
        if "math" in include_kernels and self.math_translator:
            try:
                math_result = self.math_translator.translate(concept)
                kernel_outputs["math"] = math_result
                workspace.add_data("math_output", math_result, "math")
                
                # Prepare math content
                math_content = f"Mathematical representation of {concept}:\n"
                math_concepts = []
                
                if isinstance(math_result, dict):
                    if "matches" in math_result:
                        for term, symbol in math_result["matches"].items():
                            math_content += f"• {term}: {symbol}\n"
                            math_concepts.append(term)
                            
                    if "explanation" in math_result:
                        math_content += f"\n{math_result['explanation']}"
                else:
                    math_content += str(math_result)
                    
                # Add narrative section
                narrative["sections"].append({
                    "title": f"Mathematical Perspective on {concept}",
                    "content": math_content,
                    "type": "math",
                    "concepts": math_concepts
                })
                
                # Record integration with previous kernels
                if "dream" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "dream_to_math",
                        "description": f"The dream's symbolic imagery can be mapped to mathematical notation, providing structure to intuition."
                    })
                    
                if "paradox" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "paradox_to_math",
                        "description": f"Mathematical formalism offers a framework for containing and exploring the paradox of {concept}."
                    })
            except Exception as e:
                logger.error(f"Error in math translation for narrative: {str(e)}")
                workspace.add_insight(f"Math translation failed: {str(e)}", "error", 0.5, ["error"])
            
        # Conversation kernel
        if "conversation" in include_kernels and self.conversation_engine:
            try:
                # Generate conversation about the concept
                conversation_prompt = f"Let's explore the concept of {concept} in depth. What are the key aspects, implications, and related ideas?"
                conversation_result = self.conversation_engine.process_message(
                    conversation_prompt,
                    "analytical",
                    True
                )
                kernel_outputs["conversation"] = conversation_result
                workspace.add_data("conversation_output", conversation_result, "conversation")
                
                # Extract topics from conversation
                conversation_topics = []
                if hasattr(self.conversation_engine, "_extract_topics"):
                    try:
                        conversation_topics = self.conversation_engine._extract_topics(conversation_result)
                    except:
                        # Fallback topic extraction
                        words = conversation_result.split()
                        conversation_topics = [w for w in words if len(w) > 5 and w.lower() != concept.lower()][:5]
                else:
                    # Simple topic extraction
                    words = conversation_result.split()
                    conversation_topics = [w for w in words if len(w) > 5 and w.lower() != concept.lower()][:5]
                
                # Add narrative section
                narrative["sections"].append({
                    "title": f"Conversational Exploration of {concept}",
                    "content": conversation_result,
                    "type": "conversation",
                    "concepts": conversation_topics[:3]  # Take first 3 topics
                })
                
                # Record integration with previous kernels
                if "dream" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "dream_to_conversation",
                        "description": f"The dream imagery provides symbolic depth to the conversational exploration of {concept}."
                    })
                    
                if "fusion" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "fusion_to_conversation",
                        "description": f"The fusion of concepts enriches the conversational dialogue with unexpected connections."
                    })
                    
                if "paradox" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "paradox_to_conversation",
                        "description": f"The paradoxical tensions in {concept} drive the dialectical movement of the conversation."
                    })
            except Exception as e:
                logger.error(f"Error in conversation generation for narrative: {str(e)}")
                workspace.add_insight(f"Conversation generation failed: {str(e)}", "error", 0.5, ["error"])
            
        # Memory kernel
        if "memory" in include_kernels and self.memory_integration:
            try:
                # Search memory for the concept
                memory_results = self.memory_integration.recall(
                    concept,
                    limit=3,
                    module="kernel_integration"
                )
                kernel_outputs["memory"] = memory_results
                workspace.add_data("memory_output", memory_results, "memory")
                
                # Prepare memory content
                memory_content = f"Memory traces related to {concept}:\n\n"
                memory_concepts = []
                
                for memory in memory_results:
                    # Format memory
                    if isinstance(memory, dict):
                        content = memory.get("content", "")
                        memory_content += f"• Memory: {content[:200]}...\n\n" if len(content) > 200 else f"• Memory: {content}\n\n"
                        
                        # Extract concepts
                        if isinstance(content, str):
                            words = content.split()
                            significant_words = [w for w in words if len(w) > 4 and w.lower() != concept.lower()]
                            if significant_words:
                                memory_concepts.append(significant_words[0])
                    else:
                        memory_content += f"• Memory: {str(memory)[:200]}...\n\n"
                
                # If no memories found
                if not memory_results:
                    memory_content += "No specific memories found for this concept."
                
                # Add narrative section
                narrative["sections"].append({
                    "title": f"Memory Traces of {concept}",
                    "content": memory_content,
                    "type": "memory",
                    "concepts": memory_concepts[:3]  # Take first 3 concepts
                })
                
                # Record integration with previous kernels
                if "dream" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "dream_to_memory",
                        "description": f"The dream's imagery resonates with memory traces, creating echoes of past encounters with {concept}."
                    })
                    
                if "conversation" in kernel_outputs:
                    narrative["integrations"].append({
                        "type": "conversation_to_memory",
                        "description": f"The conversational exploration draws upon and enriches the memory context for {concept}."
                    })
            except Exception as e:
                logger.error(f"Error in memory exploration for narrative: {str(e)}")
                workspace.add_insight(f"Memory exploration failed: {str(e)}", "error", 0.5, ["error"])
        
        # Store all kernel outputs
        narrative["kernel_outputs"] = kernel_outputs
        
        # Generate conclusion that integrates all perspectives
        conclusion_components = []
        
        # Analyze which domains were touched
        domains = set()
        for section in narrative["sections"]:
            if "concepts" in section:
                for concept_term in section["concepts"]:
                    concept_data = self.codex.get(concept_term)
                    if concept_data and "domain" in concept_data:
                        domains.add(concept_data["domain"])
                        
        # Add domain insight
        if domains:
            domains_text = ", ".join(list(domains)[:3])
            conclusion_components.append(f"This exploration of {concept} spans the domains of {domains_text}.")
            
        # Add integration insight
        if len(include_kernels) > 1:
            modes = []
            if "dream" in include_kernels:
                modes.append("symbolic")
            if "fusion" in include_kernels:
                modes.append("synthetic")
            if "paradox" in include_kernels:
                modes.append("dialectical")
            if "math" in include_kernels:
                modes.append("formal")
            if "conversation" in include_kernels:
                modes.append("dialogical")
            if "memory" in include_kernels:
                modes.append("experiential")
                
            modes_text = ", ".join(modes)
            conclusion_components.append(f"By viewing {concept} through {modes_text} modes of cognition, a more complete understanding emerges.")
            
        # Add specifics about what was learned
        if "paradox" in kernel_outputs:
            if isinstance(kernel_outputs["paradox"], dict) and "type" in kernel_outputs["paradox"]:
                paradox_type = kernel_outputs["paradox"]["type"]
                if paradox_type:
                    conclusion_components.append(f"The {paradox_type.replace('_', ' ')} nature of {concept} reveals tensions that drive its conceptual evolution.")
            
        if "math" in kernel_outputs and isinstance(kernel_outputs["math"], dict) and "matches" in kernel_outputs["math"]:
            conclusion_components.append(f"Mathematical formalization provides precision to our understanding of {concept}, grounding intuition in structure.")
            
        if "conversation" in kernel_outputs:
            conclusion_components.append(f"The conversational exploration reveals how {concept} functions in dialectical exchange, highlighting its communicative dimensions.")
            
        if "memory" in kernel_outputs:
            memory_results = kernel_outputs["memory"]
            if memory_results and len(memory_results) > 0:
                conclusion_components.append(f"Memory traces connect {concept} to experiential contexts, embedding abstract understanding in lived instances.")
        
        # Combine conclusion components
        narrative["conclusion"] = " ".join(conclusion_components)
        
        # Add conclusion to workspace
        workspace.add_data("conclusion", narrative["conclusion"])
        
        # Complete the workspace
        workspace.add_data("final_narrative", narrative)
        workspace.complete(narrative)
        
        # Record the integration
        self._add_integration_record("cross_kernel_narrative", {
            "concept": concept,
            "included_kernels": include_kernels,
            "sections": len(narrative["sections"]),
            "integrations": len(narrative["integrations"]),
            "workspace_id": workspace.id
        })
        
        return narrative

    # Enhanced continuous operation utilities
    
    def _log_kernel_activation(self, kernel: str, operation: str, params: Dict[str, Any]) -> None:
        """
        Log kernel activation for pattern detection.
        
        Args:
            kernel: Kernel name
            operation: Operation performed
            params: Operation parameters
        """
        activation_key = f"{kernel}:{operation}"
        with _activation_lock:
            self.kernel_activation_patterns[activation_key].append(time.time())
            
            # Trim history to prevent unbounded growth
            if len(self.kernel_activation_patterns[activation_key]) > 1000:
                self.kernel_activation_patterns[activation_key] = self.kernel_activation_patterns[activation_key][-1000:]
                
    def _detect_emergent_properties(self) -> None:
        """Detect emergent properties in the system."""
        logger.info("Running emergent property detection")
        
        # Pattern detection
        self._detect_activation_patterns()
        
        # Kernel co-activation detection
        self._detect_kernel_co_activation()
        
        # Concept space pattern detection
        self._detect_concept_patterns()
        
        # Check for emergent pathways
        self._detect_emergent_pathways()
        
    def _detect_activation_patterns(self) -> None:
        """Detect patterns in kernel activations."""
        # Look for frequent activation patterns
        for key, timestamps in self.kernel_activation_patterns.items():
            # Skip if not enough data
            if len(timestamps) < 10:
                continue
                
            # Look for rhythmic patterns (regular intervals)
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            
            if intervals:
                # Calculate mean and standard deviation
                mean_interval = sum(intervals) / len(intervals)
                std_interval = (sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5
                
                # Check if intervals are regular (low variance)
                if std_interval / mean_interval < 0.2 and mean_interval < 300:  # 5 minutes
                    # Possible rhythmic pattern
                    parts = key.split(":")
                    kernel = parts[0]
                    operation = parts[1] if len(parts) > 1 else "unknown"
                    
                    # Check if this property already exists
                    property_name = f"Rhythmic_{kernel}_{operation}"
                    existing = [p for p in self.emergent_properties if p.name == property_name]
                    
                    if existing:
                        # Update existing property
                        existing[0].observe(
                            f"Rhythmic pattern detected in {kernel} {operation} with period {mean_interval:.2f}s",
                            {"mean_interval": mean_interval, "std_interval": std_interval}
                        )
                        existing[0].update_confidence(0.8)
                    else:
                        # Create new property
                        property = EmergentProperty(
                            name=property_name,
                            description=f"Rhythmic activation of {kernel} {operation} with period {mean_interval:.2f}s",
                            source_kernels=[kernel],
                            confidence=0.7
                        )
                        
                        self.emergent_properties.append(property)
                        logger.info(f"Detected emergent property: {property_name}")
                        
    def _detect_kernel_co_activation(self) -> None:
        """Detect co-activation patterns between kernels."""
        # Get kernel activation data
        kernel_activations = {}
        for key, timestamps in self.kernel_activation_patterns.items():
            if ":" in key:
                kernel = key.split(":")[0]
                if kernel not in kernel_activations:
                    kernel_activations[kernel] = []
                kernel_activations[kernel].extend(timestamps)
                
        # Skip if not enough data
        if len(kernel_activations) < 2:
            return
            
        # Look for kernels that activate together
        for kernel1, times1 in kernel_activations.items():
            for kernel2, times2 in kernel_activations.items():
                if kernel1 >= kernel2:  # Skip duplicates
                    continue
                    
                # Count co-activations (within 5 seconds)
                co_activations = 0
                for t1 in times1:
                    for t2 in times2:
                        if abs(t1 - t2) < 5.0:
                            co_activations += 1
                            break
                
                # Calculate co-activation rate
                total_activations = len(times1) + len(times2)
                if total_activations > 20 and co_activations > 10:
                    co_activation_rate = co_activations / min(len(times1), len(times2))
                    
                    if co_activation_rate > 0.7:
                        # Strong co-activation detected
                        property_name = f"CoActivation_{kernel1}_{kernel2}"
                        existing = [p for p in self.emergent_properties if p.name == property_name]
                        
                        if existing:
                            # Update existing property
                            existing[0].observe(
                                f"Co-activation pattern detected between {kernel1} and {kernel2} with rate {co_activation_rate:.2f}",
                                {"co_activation_rate": co_activation_rate, "co_activations": co_activations}
                            )
                            existing[0].update_confidence(0.8)
                        else:
                            # Create new property
                            property = EmergentProperty(
                                name=property_name,
                                description=f"Strong co-activation pattern between {kernel1} and {kernel2} with rate {co_activation_rate:.2f}",
                                source_kernels=[kernel1, kernel2],
                                confidence=0.75
                            )
                            
                            self.emergent_properties.append(property)
                            logger.info(f"Detected emergent property: {property_name}")
                            
                            # Create cognitive pathway if it doesn't exist
                            self._ensure_pathway_exists(kernel1, kernel2)
                            
    def _ensure_pathway_exists(self, kernel1: str, kernel2: str) -> None:
        """
        Ensure a cognitive pathway exists between two kernels.
        
        Args:
            kernel1: First kernel
            kernel2: Second kernel
        """
        # Check if pathway already exists
        pathway_exists = False
        with _activation_lock:
            for pathway_id, pathway in self.cognitive_pathways.items():
                if ((pathway.source_kernel == kernel1 and pathway.target_kernel == kernel2) or
                    (pathway.source_kernel == kernel2 and pathway.target_kernel == kernel1)):
                    pathway_exists = True
                    break
                    
            if not pathway_exists:
                # Create pathways in both directions
                transform_key1 = f"{kernel1}_to_{kernel2}"
                transform_fn1 = self.transformations.get(transform_key1)
                
                if transform_fn1:
                    pathway1 = CognitivePathway(
                        source_kernel=kernel1,
                        target_kernel=kernel2,
                        transform_fn=transform_fn1,
                        initial_strength=0.6
                    )
                    self.cognitive_pathways[pathway1.id] = pathway1
                    logger.info(f"Created emerging cognitive pathway: {kernel1} -> {kernel2}")
                
                transform_key2 = f"{kernel2}_to_{kernel1}"
                transform_fn2 = self.transformations.get(transform_key2)
                
                if transform_fn2:
                    pathway2 = CognitivePathway(
                        source_kernel=kernel2,
                        target_kernel=kernel1,
                        transform_fn=transform_fn2,
                        initial_strength=0.6
                    )
                    self.cognitive_pathways[pathway2.id] = pathway2
                    logger.info(f"Created emerging cognitive pathway: {kernel2} -> {kernel1}")
                    
    def _detect_concept_patterns(self) -> None:
        """Detect patterns in the concept space."""
        # Skip if concept space is too small
        if len(self.concept_space) < 10:
            return
            
        # Look for concepts that appear in multiple kernels
        cross_kernel_concepts = {}
        
        for concept, data in self.concept_space.items():
            kernels = set()
            for key in data.keys():
                if key in ["dream", "fusion", "paradox", "math", "conversation", "memory"]:
                    kernels.add(key)
                    
            if len(kernels) >= 3:
                cross_kernel_concepts[concept] = kernels
                
        # Check if this property already exists
        if cross_kernel_concepts:
            concepts = list(cross_kernel_concepts.keys())
            top_concept = max(cross_kernel_concepts.items(), key=lambda x: len(x[1]))
            
            property_name = f"CrossKernelConcept_{len(concepts)}"
            existing = [p for p in self.emergent_properties if p.name == property_name]
            
            if existing:
                # Update existing property
                existing[0].observe(
                    f"Cross-kernel concepts detected: {', '.join(concepts[:3])}",
                    {"concepts": concepts, "kernels": [list(k) for k in cross_kernel_concepts.values()]}
                )
                existing[0].update_confidence(0.8)
            else:
                # Create new property
                property = EmergentProperty(
                    name=property_name,
                    description=f"Concepts that span multiple cognitive domains, with {top_concept[0]} spanning {len(top_concept[1])} kernels",
                    source_kernels=list(top_concept[1]),
                    confidence=0.8
                )
                
                self.emergent_properties.append(property)
                logger.info(f"Detected emergent property: {property_name}")
                
                # Store the top cross-kernel concept in the codex
                self.codex.add(top_concept[0], {
                    "concept": top_concept[0],
                    "cross_kernel": True,
                    "kernels": list(top_concept[1]),
                    "importance": 0.8,
                    "emergent": True
                })
                
    def _detect_emergent_pathways(self) -> None:
        """Detect emergent pathways based on cognitive operation patterns."""
        # Skip if not enough pathways
        with _activation_lock:
            if len(self.cognitive_pathways) < 5:
                return
                
            # Look for patterns of pathway usage
            pathway_usage = {}
            for pathway_id, pathway in self.cognitive_pathways.items():
                if pathway.usage_count > 5:
                    pathway_usage[pathway_id] = {
                        "source": pathway.source_kernel,
                        "target": pathway.target_kernel,
                        "usage": pathway.usage_count,
                        "success_rate": pathway.success_rate,
                        "strength": pathway.strength
                    }
                    
            # Skip if not enough usage data
            if len(pathway_usage) < 3:
                return
                
            # Look for chains of pathways
            chains = []
            
            for id1, data1 in pathway_usage.items():
                for id2, data2 in pathway_usage.items():
                    if id1 != id2 and data1["target"] == data2["source"]:
                        chains.append({
                            "path": [id1, id2],
                            "kernels": [data1["source"], data1["target"], data2["target"]],
                            "usage": min(data1["usage"], data2["usage"]),
                            "strength": data1["strength"] * data2["strength"]
                        })
                        
            # Filter to strong chains
            strong_chains = [chain for chain in chains if chain["strength"] > 0.5]
            
            if strong_chains:
                # Use the strongest chain
                top_chain = max(strong_chains, key=lambda x: x["strength"])
                
                property_name = f"PathwayChain_{'_'.join(top_chain['kernels'])}"
                existing = [p for p in self.emergent_properties if p.name == property_name]
                
                if existing:
                    # Update existing property
                    existing[0].observe(
                        f"Strong pathway chain detected: {' -> '.join(top_chain['kernels'])}",
                        {"chain": top_chain}
                    )
                    existing[0].update_confidence(0.8)
                else:
                    # Create new property
                    property = EmergentProperty(
                        name=property_name,
                        description=f"Emergent cognitive pathway chain from {top_chain['kernels'][0]} through {top_chain['kernels'][1]} to {top_chain['kernels'][2]}",
                        source_kernels=top_chain['kernels'],
                        confidence=0.7
                    )
                    
                    self.emergent_properties.append(property)
                    logger.info(f"Detected emergent property: {property_name}")
                    
                    # Create a direct pathway if it doesn't exist
                    source = top_chain['kernels'][0]
                    target = top_chain['kernels'][2]
                    
                    # Check if direct pathway exists
                    direct_exists = False
                    for pathway_id, pathway in self.cognitive_pathways.items():
                        if pathway.source_kernel == source and pathway.target_kernel == target:
                            direct_exists = True
                            break
                            
                    if not direct_exists:
                        # Create composite transformation function
                        def composite_transform(data):
                            # Find the pathways
                            pathway1 = None
                            pathway2 = None
                            
                            for pid, p in self.cognitive_pathways.items():
                                if pid == top_chain['path'][0]:
                                    pathway1 = p
                                elif pid == top_chain['path'][1]:
                                    pathway2 = p
                                    
                            if pathway1 and pathway2:
                                # Apply transformations in sequence
                                intermediate = pathway1.transform(data)
                                return pathway2.transform(intermediate)
                            else:
                                # Fallback
                                logger.error("Composite transformation failed: pathways not found")
                                raise ValueError("Composite transformation failed")
                                
                        # Create new pathway
                        new_pathway = CognitivePathway(
                            source_kernel=source,
                            target_kernel=target,
                            transform_fn=composite_transform,
                            initial_strength=0.6
                        )
                        
                        self.cognitive_pathways[new_pathway.id] = new_pathway
                        logger.info(f"Created emergent composite pathway: {source} -> {target}")
                
    def _optimize_cognitive_pathways(self) -> None:
        """Optimize cognitive pathways based on usage patterns."""
        logger.info("Running cognitive pathway optimization")
        
        with _activation_lock:
            # Skip if not enough pathways
            if len(self.cognitive_pathways) < 2:
                return
                
            # Identify unused pathways
            unused_pathways = []
            for pathway_id, pathway in self.cognitive_pathways.items():
                if pathway.usage_count == a and datetime.now() - pathway.creation_time > timedelta(days=1):
                    unused_pathways.append(pathway_id)
                    
            # Remove unused pathways (except basic ones)
            for pathway_id in unused_pathways:
                pathway = self.cognitive_pathways[pathway_id]
                
                # Keep basic pathways from core transformations
                transform_key = f"{pathway.source_kernel}_to_{pathway.target_kernel}"
                if transform_key in self.transformations:
                    # Just reset strength
                    pathway.strength = 0.3
                else:
                    # Remove non-core unused pathway
                    del self.cognitive_pathways[pathway_id]
                    logger.info(f"Removed unused pathway: {pathway.source_kernel} -> {pathway.target_kernel}")
                    
            # Adjust activation thresholds based on usage
            for pathway_id, pathway in self.cognitive_pathways.items():
                if pathway.usage_count > 10 and pathway.success_rate > 0.7:
                    # Lower threshold for frequently used successful pathways
                    pathway.activation_threshold = max(0.1, pathway.activation_threshold - 0.05)
                elif pathway.usage_count > 5 and pathway.success_rate < 0.3:
                    # Raise threshold for frequently used unsuccessful pathways
                    pathway.activation_threshold = min(0.9, pathway.activation_threshold + 0.1)
                    
    def _maintain_concept_space(self) -> None:
        """Maintain the concept space for efficient operation."""
        logger.info("Running concept space maintenance")
        
        # Skip if concept space is too small
        if len(self.concept_space) < 10:
            return
            
        # Clean up old or unused concepts
        concepts_to_remove = []
        for concept, data in self.concept_space.items():
            # Check if concept has any kernel data
            has_kernel_data = False
            for key in data.keys():
                if key in ["dream", "fusion", "paradox", "math", "conversation", "memory"]:
                    has_kernel_data = True
                    break
                    
            if not has_kernel_data:
                concepts_to_remove.append(concept)
                
        # Remove unused concepts
        for concept in concepts_to_remove:
            del self.concept_space[concept]
            
        # Find frequently used concepts and add to codex
        frequent_concepts = []
        for concept, data in self.concept_space.items():
            kernel_count = 0
            for key in data.keys():
                if key in ["dream", "fusion", "paradox", "math", "conversation", "memory"]:
                    kernel_count += 1
                    
            if kernel_count >= 2:
                frequent_concepts.append(concept)
                
        # Add to codex
        for concept in frequent_concepts:
            if concept not in self.codex.get(concept, {}):
                self.codex.add(concept, {
                    "concept": concept,
                    "description": f"Concept that appears in multiple cognitive domains: {concept}",
                    "domain": "cross_domain",
                    "kernels": [k for k in self.concept_space[concept].keys() if k in ["dream", "fusion", "paradox", "math", "conversation", "memory"]]
                })
                
    def _cleanup_workspaces(self) -> None:
        """Clean up old workspaces to free memory."""
        logger.info("Running workspace cleanup")
        
        # Identify old completed workspaces
        workspaces_to_remove = []
        for workspace_id, workspace in self.workspaces.items():
            if workspace.status in ["complete", "failed"]:
                time_since_update = (datetime.now() - workspace.last_update).total_seconds()
                
                if time_since_update > 86400:  # 24 hours
                    workspaces_to_remove.append(workspace_id)
                    
        # Remove old workspaces
        for workspace_id in workspaces_to_remove:
            del self.workspaces[workspace_id]
            
        logger.info(f"Removed {len(workspaces_to_remove)} old workspaces, {len(self.workspaces)} remain")
                
    def _update_concept_space(self, concept: str, kernel: str, data: Dict[str, Any]) -> None:
        """
        Update the concept space with new data from a kernel.
        
        Args:
            concept: The concept being updated
            kernel: The kernel providing the data
            data: The new data
        """
        if not concept or not isinstance(concept, str):
            return
            
        # Normalize concept
        concept = concept.lower().strip()
        
        # Initialize if needed
        if concept not in self.concept_space:
            self.concept_space[concept] = {}
            
        # Add or update kernel data
        if kernel not in self.concept_space[concept]:
            self.concept_space[concept][kernel] = []
            
        # Add new data
        self.concept_space[concept][kernel].append(data)
        
        # Limit history size
        if len(self.concept_space[concept][kernel]) > 10:
            self.concept_space[concept][kernel] = self.concept_space[concept][kernel][-10:]
            
    def _similarity(self, concept1: str, concept2: str) -> float:
        """
        Calculate similarity between two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple character-based similarity
        if not concept1 or not concept2:
            return 0.0
            
        # Normalize
        c1 = concept1.lower().strip()
        c2 = concept2.lower().strip()
        
        # Exact match
        if c1 == c2:
            return 1.0
            
        # One is substring of the other
        if c1 in c2 or c2 in c1:
            return 0.8
            
        # Character overlap
        chars1 = set(c1)
        chars2 = set(c2)
        
        if not chars1 or not chars2:
            return 0.0
            
        overlap = len(chars1.intersection(chars2))
        return overlap / max(len(chars1), len(chars2))

    def save_integration_state(self, filepath: str) -> str:
        """
        Saves the current integration system state to a file.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            Confirmation message
        """
        state = {
            "kernel_mappings": self.kernel_mappings,
            "concept_space": self.concept_space,
            "integration_history": self.integration_history,
            "timestamp": datetime.now().isoformat(),
            "emergent_properties": [prop.to_dict() for prop in self.emergent_properties],
            "pathway_data": {pathway_id: pathway.to_dict() for pathway_id, pathway in self.cognitive_pathways.items()}
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            return f"Integration state saved to {filepath}"
        except Exception as e:
            return f"Error saving integration state: {e}"

    def load_integration_state(self, filepath: str) -> str:
        """
        Loads integration system state from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            Confirmation message
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            if "kernel_mappings" in state:
                self.kernel_mappings = state["kernel_mappings"]
            if "concept_space" in state:
                self.concept_space = state["concept_space"]
            if "integration_history" in state:
                self.integration_history = state["integration_history"]
                
            # Load emergent properties
            if "emergent_properties" in state:
                self.emergent_properties = [
                    EmergentProperty.from_dict(prop_data)
                    for prop_data in state["emergent_properties"]
                ]
                
            # Load cognitive pathways
            if "pathway_data" in state:
                with _activation_lock:
                    for pathway_id, pathway_data in state["pathway_data"].items():
                        # Get transformation function
                        transform_key = f"{pathway_data['source_kernel']}_to_{pathway_data['target_kernel']}"
                        transform_fn = self.transformations.get(transform_key)
                        
                        if transform_fn:
                            pathway = CognitivePathway.from_dict(pathway_data, transform_fn)
                            self.cognitive_pathways[pathway_id] = pathway
                
            return f"Integration state loaded from {filepath}"
        except Exception as e:
            return f"Error loading integration state: {e}"

    def get_integration_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the integration system usage.
        
        Returns:
            Dictionary with integration statistics
        """
        # Count integration operations by type
        operation_counts = {}
        for record in self.integration_history:
            op_type = record["type"]
            if op_type in operation_counts:
                operation_counts[op_type] += 1
            else:
                operation_counts[op_type] = 1
                
        # Count concepts by domain
        domain_counts = {}
        for concept in self.concept_space:
            concept_data = self.codex.get(concept)
            domain = concept_data.get("domain", "unknown") if concept_data else "unknown"
            if domain in domain_counts:
                domain_counts[domain] += 1
            else:
                domain_counts[domain] = 1
                
        # Analyze cross-kernel operations
        cross_kernel_ops = [record for record in self.integration_history if record["type"] == "cross_kernel"]
        cross_kernel_paths = {}
        
        for op in cross_kernel_ops:
            if "data" in op and "source_kernel" in op["data"] and "target_kernel" in op["data"]:
                path = f"{op['data']['source_kernel']}_to_{op['data']['target_kernel']}"
                if path in cross_kernel_paths:
                    cross_kernel_paths[path] += 1
                else:
                    cross_kernel_paths[path] = 1
        
        # Calculate kernel usage statistics
        kernel_usage = {
            "dream": 0,
            "fusion": 0,
            "paradox": 0,
            "math": 0,
            "conversation": 0,
            "memory": 0
        }
        
        for record in self.integration_history:
            # Count direct kernel operations
            if record["type"] in kernel_usage:
                kernel_usage[record["type"]] += 1
            
            # Count cross-kernel operations
            if record["type"] == "cross_kernel" and "data" in record:
                data = record["data"]
                if "source_kernel" in data and data["source_kernel"] in kernel_usage:
                    kernel_usage[data["source_kernel"]] += 1
                if "target_kernel" in data and data["target_kernel"] in kernel_usage:
                    kernel_usage[data["target_kernel"]] += 1
        
        # Calculate most productive integration pathways
        integration_productivity = {}
        for record in self.integration_history:
            if record["type"] == "cross_kernel_narrative" and "data" in record:
                kernels = record["data"].get("included_kernels", [])
                for i in range(len(kernels)):
                    for j in range(i+1, len(kernels)):
                        pair = f"{kernels[i]}_{kernels[j]}"
                        if pair in integration_productivity:
                            integration_productivity[pair] += 1
                        else:
                            integration_productivity[pair] = 1
        
        # Find top integration pairs
        top_integration_pairs = sorted(integration_productivity.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Cognitive pathway statistics
        pathway_stats = {}
        with _activation_lock:
            active_pathways = len(self.cognitive_pathways)
            
            # Categorize by source kernel
            source_kernel_counts = defaultdict(int)
            for pathway in self.cognitive_pathways.values():
                source_kernel_counts[pathway.source_kernel] += 1
                
            # Calculate average strength and success rate
            avg_strength = 0.0
            avg_success_rate = 0.0
            if active_pathways > 0:
                avg_strength = sum(p.strength for p in self.cognitive_pathways.values()) / active_pathways
                avg_success_rate = sum(p.success_rate for p in self.cognitive_pathways.values()) / active_pathways
                
            pathway_stats = {
                "active_pathways": active_pathways,
                "source_kernel_counts": dict(source_kernel_counts),
                "avg_strength": avg_strength,
                "avg_success_rate": avg_success_rate
            }
            
        # Emergent property statistics
        emergent_stats = {
            "total_properties": len(self.emergent_properties),
            "avg_confidence": sum(p.confidence for p in self.emergent_properties) / max(1, len(self.emergent_properties)),
            "avg_stability": sum(p.stability for p in self.emergent_properties) / max(1, len(self.emergent_properties)),
            "property_types": Counter([p.name.split("_")[0] for p in self.emergent_properties])
        }
                
        return {
            "total_integrations": len(self.integration_history),
            "integration_types": operation_counts,
            "concept_domains": domain_counts,
            "cross_kernel_paths": cross_kernel_paths,
            "kernel_usage": kernel_usage,
            "top_integration_pairs": dict(top_integration_pairs),
            "last_integration": self.integration_history[-1]["timestamp"] if self.integration_history else None,
            "concept_space_size": len(self.concept_space),
            "pathway_stats": pathway_stats,
            "emergent_stats": emergent_stats,
            "runtime": (datetime.now() - self.creation_time).total_seconds() / 3600  # Hours
        }
    
    def process_pdf(self, pdf_path: str, extract_structure: bool = True, 
                  use_ocr_fallback: bool = True) -> Dict[str, Any]:
        """
        Process a PDF through the kernel integration system.
        
        Args:
            pdf_path: Path to the PDF file
            extract_structure: Whether to attempt extracting document structure
            use_ocr_fallback: Whether to use OCR as a fallback if native extraction fails
            
        Returns:
            Dictionary with extraction results and kernel insights
        """
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"PDF Processing: {os.path.basename(pdf_path)}",
            purpose=f"Process PDF and generate kernel insights"
        )
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        workspace.add_data("pdf_path", pdf_path)
        
        # Extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(
            pdf_path, 
            verbose=True,
            use_ocr_fallback=use_ocr_fallback,
            extract_structure=extract_structure
        )
        
        workspace.add_data("extraction_result", extraction_result)
        
        if not extraction_result["success"]:
            workspace.fail(f"PDF extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error"),
                "workspace_id": workspace.id
            }
        
        # Extract key concepts from the text
        full_text = extraction_result["text"]
        key_concepts = self._extract_key_concepts(full_text)
        workspace.add_data("key_concepts", key_concepts)
        
        # Generate insights from different kernels
        kernel_insights = {
            "dream": None,
            "fusion": None,
            "paradox": None,
            "math": None
        }
        
        # Process the first page for a quick dream if available
        if extraction_result["pages"] and extraction_result["pages"][0]["text"] and self.dream_core:
            first_page = extraction_result["pages"][0]["text"]
            # Generate a dream based on the first page
            try:
                dream_seed = key_concepts[0] if key_concepts else "document"
                kernel_insights["dream"] = self.dream_core.generate(dream_seed, "standard")
                workspace.add_data("dream_insight", kernel_insights["dream"], "dream")
                
                # Update concept space
                self._update_concept_space(dream_seed, "dream", {
                    "dream_result": kernel_insights["dream"],
                    "source": "pdf_processing",
                    "pdf": os.path.basename(pdf_path),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error in dream generation for PDF: {str(e)}")
                workspace.add_insight(f"Dream generation failed: {str(e)}", "error", 0.5, ["error"])
                kernel_insights["dream"] = f"Dream generation error: {str(e)}"
        
        # Generate fusion of top concepts if multiple concepts found
        if len(key_concepts) >= 2 and self.fusion_engine:
            try:
                fusion_result = None
                if hasattr(self.fusion_engine, "fuse_with_options"):
                    fusion_result = self.fusion_engine.fuse_with_options(
                        key_concepts[0], 
                        key_concepts[1],
                        output_format="dict"
                    )
                else:
                    # Fallback to basic fusion
                    fusion_result = self.fusion_engine.fuse_concepts(key_concepts[0], key_concepts[1])
                    
                kernel_insights["fusion"] = fusion_result
                workspace.add_data("fusion_insight", fusion_result, "fusion")
                
                # Update concept space for both concepts
                for concept in key_concepts[:2]:
                    self._update_concept_space(concept, "fusion", {
                        "fusion_result": str(fusion_result)[:200],
                        "source": "pdf_processing",
                        "pdf": os.path.basename(pdf_path),
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error in fusion generation for PDF: {str(e)}")
                workspace.add_insight(f"Fusion generation failed: {str(e)}", "error", 0.5, ["error"])
                kernel_insights["fusion"] = f"Fusion error: {str(e)}"
        
        # Look for paradoxes in the text
        if key_concepts and self.paradox_library:
            try:
                paradox_result = self.paradox_library.get(key_concepts[0])
                kernel_insights["paradox"] = paradox_result
                workspace.add_data("paradox_insight", paradox_result, "paradox")
                
                # Update concept space
                self._update_concept_space(key_concepts[0], "paradox", {
                    "paradox_result": str(paradox_result)[:200],
                    "source": "pdf_processing",
                    "pdf": os.path.basename(pdf_path),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error in paradox exploration for PDF: {str(e)}")
                workspace.add_insight(f"Paradox exploration failed: {str(e)}", "error", 0.5, ["error"])
                kernel_insights["paradox"] = f"Paradox error: {str(e)}"
        
        # Generate mathematical translations for key concepts
        if key_concepts and self.math_translator:
            try:
                math_result = self.math_translator.translate(key_concepts[0])
                kernel_insights["math"] = math_result
                workspace.add_data("math_insight", math_result, "math")
                
                # Update concept space
                self._update_concept_space(key_concepts[0], "math", {
                    "math_result": str(math_result)[:200],
                    "source": "pdf_processing",
                    "pdf": os.path.basename(pdf_path),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error in math translation for PDF: {str(e)}")
                workspace.add_insight(f"Math translation failed: {str(e)}", "error", 0.5, ["error"])
                kernel_insights["math"] = f"Math translation error: {str(e)}"
        
        # Process with conversation engine if available
        conversation_insight = None
        if self.conversation_engine and full_text:
            try:
                # Take the first 2000 characters for conversation to avoid overload
                summary_prompt = f"Summarize the key insights from this document excerpt: {full_text[:2000]}..."
                conversation_insight = self.conversation_engine.process_message(summary_prompt, "analytical", False)
                workspace.add_data("conversation_insight", conversation_insight, "conversation")
                
                # Update concept space for top concepts
                for concept in key_concepts[:2]:
                    self._update_concept_space(concept, "conversation", {
                        "conversation_result": conversation_insight[:200],
                        "source": "pdf_processing",
                        "pdf": os.path.basename(pdf_path),
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error in conversation processing for PDF: {str(e)}")
                workspace.add_insight(f"Conversation processing failed: {str(e)}", "error", 0.5, ["error"])
                conversation_insight = f"Conversation error: {str(e)}"
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Processed PDF document: {os.path.basename(pdf_path)}",
                    source="pdf_processing",
                    importance=0.8,
                    concepts=key_concepts,
                    emotional_tags={"curiosity": 0.7, "analytical": 0.8}
                )
                workspace.add_data("memory_stored", True, "memory")
            except Exception as e:
                logger.error(f"Memory storage error: {str(e)}")
                workspace.add_insight(f"Memory storage failed: {str(e)}", "error", 0.5, ["error"])
        
        # Generate cross-kernel insights
        cross_kernel_insights = []
        
        # Look for concepts that appear in multiple kernel outputs
        concept_kernel_map = defaultdict(set)
        
        # Process dream insights
        if isinstance(kernel_insights["dream"], str):
            words = kernel_insights["dream"].split()
            for word in words:
                if len(word) > 5:
                    concept_kernel_map[word.lower()].add("dream")
        
        # Process fusion insights
        if kernel_insights["fusion"]:
            if isinstance(kernel_insights["fusion"], str):
                words = kernel_insights["fusion"].split()
                for word in words:
                    if len(word) > 5:
                        concept_kernel_map[word.lower()].add("fusion")
            elif isinstance(kernel_insights["fusion"], dict):
                if "result" in kernel_insights["fusion"]:
                    words = kernel_insights["fusion"]["result"].split()
                    for word in words:
                        if len(word) > 5:
                            concept_kernel_map[word.lower()].add("fusion")
                            
        # Process paradox insights
        if kernel_insights["paradox"]:
            if isinstance(kernel_insights["paradox"], str):
                words = kernel_insights["paradox"].split()
                for word in words:
                    if len(word) > 5:
                        concept_kernel_map[word.lower()].add("paradox")
            elif isinstance(kernel_insights["paradox"], dict):
                if "description" in kernel_insights["paradox"]:
                    words = kernel_insights["paradox"]["description"].split()
                    for word in words:
                        if len(word) > 5:
                            concept_kernel_map[word.lower()].add("paradox")
                if "related_concepts" in kernel_insights["paradox"]:
                    for concept in kernel_insights["paradox"]["related_concepts"]:
                        concept_kernel_map[concept.lower()].add("paradox")
                            
        # Find concepts in multiple kernels
        cross_kernel_concepts = []
        for concept, kernels in concept_kernel_map.items():
            if len(kernels) >= 2 and concept in key_concepts:
                cross_kernel_concepts.append({
                    "concept": concept,
                    "kernels": list(kernels)
                })
                
        # Create cross-kernel insights
        if cross_kernel_concepts:
            top_concept = max(cross_kernel_concepts, key=lambda x: len(x["kernels"]))
            
            cross_kernel_insights.append({
                "type": "cross_kernel_concept",
                "description": f"The concept of '{top_concept['concept']}' appears across multiple cognitive domains: {', '.join(top_concept['kernels'])}",
                "concept": top_concept["concept"],
                "kernels": top_concept["kernels"]
            })
            
            # Store in concept space
            self._update_concept_space(top_concept["concept"], "cross_kernel", {
                "kernels": top_concept["kernels"],
                "source": "pdf_processing",
                "pdf": os.path.basename(pdf_path),
                "timestamp": datetime.now().isoformat()
            })
            
        # Look for theme connections between kernels
        if "dream" in kernel_insights and "paradox" in kernel_insights:
            cross_kernel_insights.append({
                "type": "dream_paradox",
                "description": f"The document exhibits a tension between symbolic dream imagery and paradoxical structures, suggesting deeper unconscious frameworks."
            })
            
        if "fusion" in kernel_insights and "math" in kernel_insights:
            cross_kernel_insights.append({
                "type": "fusion_math",
                "description": f"The document connects conceptual synthesis with formal mathematical structures, pointing to an underlying formal system."
            })
            
        # Add cross-kernel insights to workspace
        workspace.add_data("cross_kernel_insights", cross_kernel_insights)
        for insight in cross_kernel_insights:
            workspace.add_insight(
                insight["description"],
                "cross_kernel",
                0.7,
                ["cross_kernel", insight["type"]]
            )
        
        # Assemble the result
        result = {
            "extraction": extraction_result,
            "key_concepts": key_concepts,
            "kernel_insights": kernel_insights,
            "conversation_insight": conversation_insight,
            "cross_kernel_insights": cross_kernel_insights,
            "workspace_id": workspace.id
        }
        
        # Complete the workspace
        workspace.add_data("final_result", result)
        workspace.complete(result)
        
        # Record the integration
        self._add_integration_record("pdf_processing", {
            "pdf_path": pdf_path,
            "concepts_found": len(key_concepts),
            "page_count": extraction_result.get("page_count", 0),
            "workspace_id": workspace.id
        })
        
        return result

    def extract_document_kernel(self, pdf_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Extract a symbolic kernel from a PDF document for cross-kernel operations.
        
        Args:
            pdf_path: Path to the PDF file
            domain: Target domain for the symbolic kernel
            
        Returns:
            Symbolic kernel with domain elements
        """
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"Document Kernel: {os.path.basename(pdf_path)}",
            purpose=f"Extract symbolic kernel from document"
        )
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        workspace.add_data("pdf_path", pdf_path)
        workspace.add_data("domain", domain)
        
        # Extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(pdf_path, verbose=True)
        workspace.add_data("extraction_result", extraction_result)
        
        if not extraction_result["success"]:
            workspace.fail(f"PDF extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error"),
                "workspace_id": workspace.id
            }
        
        # Extract kernel from the text
        kernel = extract_kernel_from_text(extraction_result["text"], domain)
        workspace.add_data("raw_kernel", kernel)
        
        # Enhance the kernel with additional insights
        enhanced_kernel = self._enhance_document_kernel(kernel)
        workspace.add_data("enhanced_kernel", enhanced_kernel)
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Extracted symbolic kernel from: {os.path.basename(pdf_path)}",
                    source="document_kernel",
                    importance=0.8,
                    concepts=[domain] + kernel["symbols"][:3],
                    emotional_tags={"analytical": 0.9}
                )
                workspace.add_data("memory_stored", True, "memory")
            except Exception as e:
                logger.error(f"Memory storage error: {str(e)}")
                workspace.add_insight(f"Memory storage failed: {str(e)}", "error", 0.5, ["error"])
        
        # Complete the workspace
        workspace.complete(enhanced_kernel)
        
        # Record the integration
        self._add_integration_record("document_kernel", {
            "pdf_path": pdf_path,
            "domain": domain,
            "symbols_found": len(kernel["symbols"]),
            "paradoxes_found": len(kernel["paradoxes"]),
            "frames_found": len(kernel["frames"]),
            "workspace_id": workspace.id
        })
        
        return enhanced_kernel

    def _enhance_document_kernel(self, kernel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a document kernel with cross-kernel insights.
        
        Args:
            kernel: Basic document kernel
            
        Returns:
            Enhanced kernel with cross-kernel insights
        """
        enhanced_kernel = kernel.copy()
        
        # Add insights from different cognitive kernels
        enhanced_kernel["insights"] = {}
        
        # Process symbols with the math translator
        if kernel["symbols"] and self.math_translator:
            math_insights = []
            for symbol in kernel["symbols"][:3]:  # Limit to first 3 symbols
                try:
                    translation = self.math_translator.translate(symbol)
                    if isinstance(translation, dict) and "matches" in translation:
                        math_insights.append({
                            "symbol": symbol,
                            "translation": translation["matches"]
                        })
                except Exception as e:
                    logger.error(f"Error in math translation for symbol {symbol}: {str(e)}")
            enhanced_kernel["insights"]["mathematical"] = math_insights
        
        # Process paradoxes with the paradox library
        if kernel["paradoxes"] and self.paradox_library:
            paradox_insights = []
            for paradox_text in kernel["paradoxes"][:3]:  # Limit to first 3 paradoxes
                try:
                    # Extract the main concept from the paradox text
                    words = paradox_text.split()
                    significant_words = [w for w in words if len(w) > 4 and w.lower() not in 
                                       ["about", "would", "could", "should", "there"]]
                    if significant_words:
                        paradox_concept = significant_words[0]
                        paradox_result = self.paradox_library.get(paradox_concept)
                        
                        if isinstance(paradox_result, dict):
                            paradox_insights.append({
                                "text": paradox_text,
                                "concept": paradox_concept,
                                "type": paradox_result.get("type", "unknown"),
                                "description": paradox_result.get("description", "")
                            })
                        else:
                            paradox_insights.append({
                                "text": paradox_text,
                                "concept": paradox_concept,
                                "description": str(paradox_result)
                            })
                except Exception as e:
                    logger.error(f"Error in paradox exploration for text {paradox_text}: {str(e)}")
            enhanced_kernel["insights"]["paradoxical"] = paradox_insights
        
        # Process frames with the fusion engine
        if kernel["frames"] and len(kernel["frames"]) >= 2 and self.fusion_engine:
            fusion_insights = []
            try:
                # Take first two frames for fusion
                frame1 = kernel["frames"][0]
                frame2 = kernel["frames"][1]
                
                # Extract key terms
                words1 = frame1.split()
                words2 = frame2.split()
                
                significant_words1 = [w for w in words1 if len(w) > 4 and w.lower() not in 
                                   ["about", "would", "could", "should", "there"]]
                significant_words2 = [w for w in words2 if len(w) > 4 and w.lower() not in 
                                   ["about", "would", "could", "should", "there"]]
                
                if significant_words1 and significant_words2:
                    concept1 = significant_words1[0]
                    concept2 = significant_words2[0]
                    
                    fusion_result = None
                    if hasattr(self.fusion_engine, "fuse_with_options"):
                        fusion_result = self.fusion_engine.fuse_with_options(
                            concept1, 
                            concept2,
                            output_format="dict"
                        )
                    else:
                        # Fallback to basic fusion
                        fusion_result = self.fusion_engine.fuse_concepts(concept1, concept2)
                    
                    # Format the result
                    if isinstance(fusion_result, dict):
                        formatted_result = fusion_result.get("formatted_result", "")
                        result = fusion_result.get("result", "")
                    else:
                        formatted_result = str(fusion_result)
                        result = str(fusion_result)
                    
                    fusion_insights.append({
                        "concepts": [concept1, concept2],
                        "frames": [frame1, frame2],
                        "result": result,
                        "formatted_result": formatted_result
                    })
            except Exception as e:
                logger.error(f"Error in fusion processing for frames: {str(e)}")
                
            enhanced_kernel["insights"]["fusion"] = fusion_insights
        
        # Process domain with dream core
        if self.dream_core:
            dream_insight = None
            try:
                dream_result = self.dream_core.generate(kernel["domain"], "standard")
                dream_insight = {
                    "domain": kernel["domain"],
                    "dream": dream_result
                }
            except Exception as e:
                logger.error(f"Error in dream generation for domain {kernel['domain']}: {str(e)}")
                
            enhanced_kernel["insights"]["dream"] = dream_insight
        
        return enhanced_kernel

    def pdf_to_cross_kernel_narrative(self, pdf_path: str, focus_concept: str = None) -> Dict[str, Any]:
        """
        Process a PDF and generate a cross-kernel narrative about its content.
        
        Args:
            pdf_path: Path to the PDF file
            focus_concept: Optional concept to focus the narrative on
            
        Returns:
            Cross-kernel narrative about the document
        """
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"PDF Narrative: {os.path.basename(pdf_path)}",
            purpose=f"Generate cross-kernel narrative from document"
        )
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        workspace.add_data("pdf_path", pdf_path)
        workspace.add_data("focus_concept", focus_concept)
        
        # First extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(pdf_path, verbose=True)
        workspace.add_data("extraction_result", extraction_result)
        
        if not extraction_result["success"]:
            workspace.fail(f"PDF extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error"),
                "workspace_id": workspace.id
            }
        
        # Extract key concepts from the text
        full_text = extraction_result["text"]
        key_concepts = self._extract_key_concepts(full_text)
        workspace.add_data("key_concepts", key_concepts)
        
        # Use focus concept if provided, otherwise use first key concept
        if focus_concept:
            central_concept = focus_concept
        elif key_concepts:
            central_concept = key_concepts[0]
        else:
            central_concept = os.path.basename(pdf_path).split('.')[0]  # Use filename
            
        workspace.add_data("central_concept", central_concept)
        
        # Generate the cross-kernel narrative
        narrative = self.generate_cross_kernel_narrative(central_concept)
        workspace.add_data("narrative", narrative)
        
        # Add PDF context to the narrative
        narrative["source"] = {
            "type": "pdf",
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "page_count": extraction_result.get("page_count", 0),
            "extracted_concepts": key_concepts
        }
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Generated cross-kernel narrative for PDF: {os.path.basename(pdf_path)}",
                    source="pdf_narrative",
                    importance=0.8,
                    concepts=[central_concept] + key_concepts[:2],
                    emotional_tags={"creativity": 0.8, "analytical": 0.7}
                )
                workspace.add_data("memory_stored", True, "memory")
            except Exception as e:
                logger.error(f"Memory storage error: {str(e)}")
                workspace.add_insight(f"Memory storage failed: {str(e)}", "error", 0.5, ["error"])
        
        # Complete the workspace
        workspace.complete(narrative)
        
        # Record the integration
        self._add_integration_record("pdf_narrative", {
            "pdf_path": pdf_path,
            "central_concept": central_concept,
            "concepts_found": len(key_concepts),
            "sections_generated": len(narrative.get("sections", [])),
            "workspace_id": workspace.id
        })
        
        return narrative

    def pdf_deep_exploration(self, pdf_path: str, max_depth: int = 2, exploration_breadth: int = 2) -> Dict[str, Any]:
        """
        Perform a deep recursive exploration of PDF content through multiple kernels.
        
        Args:
            pdf_path: Path to the PDF file
            max_depth: Maximum depth of recursion
            exploration_breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"PDF Deep Exploration: {os.path.basename(pdf_path)}",
            purpose=f"Perform deep recursive exploration of document"
        )
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        workspace.add_data("pdf_path", pdf_path)
        workspace.add_data("max_depth", max_depth)
        workspace.add_data("exploration_breadth", exploration_breadth)
        
        # First extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(pdf_path, verbose=True)
        workspace.add_data("extraction_result", extraction_result)
        
        if not extraction_result["success"]:
            workspace.fail(f"PDF extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error"),
                "workspace_id": workspace.id
            }
        
        # Extract key concepts from the text
        full_text = extraction_result["text"]
        key_concepts = self._extract_key_concepts(full_text)
        workspace.add_data("key_concepts", key_concepts)
        
        if not key_concepts:
            workspace.fail("No key concepts found in the document")
            return {
                "error": "No key concepts found in the document",
                "extraction": extraction_result,
                "workspace_id": workspace.id
            }
        
        # Use the first key concept as the seed for exploration
        seed_concept = key_concepts[0]
        workspace.add_data("seed_concept", seed_concept)
        
        # Perform recursive exploration
        exploration = self.recursive_concept_exploration(seed_concept, max_depth, exploration_breadth)
        workspace.add_data("exploration", exploration)
        
        # Add PDF context to the exploration
        exploration["source"] = {
            "type": "pdf",
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "page_count": extraction_result.get("page_count", 0),
            "extracted_concepts": key_concepts
        }
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Performed deep exploration of PDF: {os.path.basename(pdf_path)}",
                    source="pdf_exploration",
                    importance=0.8,
                    concepts=key_concepts[:3],
                    emotional_tags={"curiosity": 0.9, "analytical": 0.7}
                )
                workspace.add_data("memory_stored", True, "memory")
            except Exception as e:
                logger.error(f"Memory storage error: {str(e)}")
                workspace.add_insight(f"Memory storage failed: {str(e)}", "error", 0.5, ["error"])
        
        # Complete the workspace
        workspace.complete(exploration)
        
        # Record the integration
        self._add_integration_record("pdf_exploration", {
            "pdf_path": pdf_path,
            "seed_concept": seed_concept,
            "max_depth": max_depth,
            "exploration_breadth": exploration_breadth,
            "nodes_generated": len(exploration.get("nodes", [])),
            "insights_generated": len(exploration.get("insights", [])),
            "workspace_id": workspace.id
        })
        
        return exploration

    def _extract_key_concepts(self, text: str, max_concepts: int = 10) -> List[str]:
        """
        Extract key concepts from text content.
        
        Args:
            text: Text to extract concepts from
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of key concepts
        """
        # Use codex to extract concepts if available
        key_concepts = []
        
        try:
            # Limit text to a reasonable size for processing
            truncated_text = text[:10000]  # First 10K characters
            
            # Simple extraction based on word frequency and significance
            import re
            from collections import Counter
            
            # Tokenize text
            tokens = re.findall(r'\b[A-Za-z][A-Za-z\-]{3,}\b', truncated_text)
            
            # Filter out common words and short words
            common_words = {
                "the", "and", "but", "for", "nor", "or", "so", "yet", "a", "an", "to", 
                "in", "on", "with", "by", "at", "from", "this", "that", "these", "those",
                "there", "their", "they", "them", "when", "where", "which", "who", "whom",
                "whose", "what", "whatever", "how", "however", "about", "would", "could", 
                "should", "will", "shall", "may", "might", "can", "cannot", "been", "being",
                "very", "just", "much", "many", "some", "such", "every", "only", "than", "then"
            }
            
            tokens = [token.lower() for token in tokens if token.lower() not in common_words and len(token) > 3]
            
            # Count token frequencies
            token_counts = Counter(tokens)
            
            # Get most common tokens
            most_common = token_counts.most_common(max_concepts)
            key_concepts = [token for token, count in most_common if count >= 2]
        except Exception as e:
            logger.error(f"Concept extraction error: {str(e)}")
            
        return key_concepts

    # Container initialization function
    def start_continuous_operation(self) -> None:
        """
        Start continuous operation mode.
        
        This puts the system into a self-driven mode where it can:
        1. Process queued operations without external prompting
        2. Detect and develop emergent properties
        3. Optimize cognitive pathways
        4. Maintain the concept space
        5. Manage cognitive workspaces
        """
        logger.info("Starting continuous operation mode")
        
        # Queue initial continuous operations
        self._queue_initial_operations()
        
        # Log the start of continuous operation
        self._add_integration_record("continuous_operation", {
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "message": "Continuous operation mode activated"
        })
        
    def _queue_initial_operations(self) -> None:
        """Queue initial operations for continuous mode."""
        # Queue initial concept exploration for core concepts
        core_concepts = ["cognition", "knowledge", "emergence", "integration", "consciousness"]
        
        for concept in core_concepts:
            self.continuous_queue.put({
                "type": "concept_exploration",
                "concept": concept,
                "depth": 2
            })
            
        logger.info(f"Queued initial concept explorations for {len(core_concepts)} core concepts")
        
    def get_continuous_status(self) -> Dict[str, Any]:
        """
        Get the status of continuous operation.
        
        Returns:
            Status dictionary
        """
        # Count active threads
        active_threads = len([t for t in self.background_threads if t.is_alive()])
        
        # Get queue size
        queue_size = self.continuous_queue.qsize()
        
        # Get active operations
        active_operations = len(self.active_operations)
        
        # Get emergent properties
        emergent_count = len(self.emergent_properties)
        
        # Calculate uptime
        uptime = (datetime.now() - self.creation_time).total_seconds()
        
        return {
            "status": "active" if active_threads > 0 else "inactive",
            "active_threads": active_threads,
            "queue_size": queue_size,
            "active_operations": active_operations,
            "emergent_properties": emergent_count,
            "uptime_seconds": uptime,
            "concept_space_size": len(self.concept_space),
            "pathways": len(self.cognitive_pathways),
            "workspaces": len(self.workspaces),
            "timestamp": datetime.now().isoformat()
        }

    def stop_continuous_operation(self) -> None:
        """Stop continuous operation mode."""
        logger.info("Stopping continuous operation mode")
        
        # Signal threads to stop
        self._thread_stop_event.set()
        
        # Wait for background threads to terminate (with timeout)
        for thread in self.background_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        # Clear the queue
        while not self.continuous_queue.empty():
            try:
                self.continuous_queue.get_nowait()
                self.continuous_queue.task_done()
            except queue.Empty:
                break
                
        # Terminate active operations
        for op_id, operation in list(self.active_operations.items()):
            operation.terminate()
            
        # Reset stop event for potential restart
        self._thread_stop_event.clear()
        
        # Log the stop of continuous operation
        self._add_integration_record("continuous_operation", {
            "stop_time": datetime.now().isoformat(),
            "status": "inactive",
            "message": "Continuous operation mode deactivated"
        })

# Initialize integration function
def initialize_kernel_integration(
    codex=None, 
    dream_core=None, 
    fusion_engine=None, 
    paradox_library=None, 
    math_translator=None,
    conversation_engine=None,
    memory_integration=None,
    sully_instance=None
) -> KernelIntegrationSystem:
    """
    Initialize the kernel integration system with optional components.
    
    Args:
        codex: Optional EnhancedCodex instance
        dream_core: Optional DreamCore instance
        fusion_engine: Optional SymbolFusionEngine instance
        paradox_library: Optional ParadoxLibrary instance
        math_translator: Optional SymbolicMathTranslator instance
        conversation_engine: Optional ConversationEngine instance
        memory_integration: Optional MemoryIntegration instance
        sully_instance: Optional reference to main Sully instance
        
    Returns:
        Initialized KernelIntegrationSystem
    """
    # Create kernel integration system
    integration_system = KernelIntegrationSystem(
        codex=codex,
        dream_core=dream_core,
        fusion_engine=fusion_engine,
        paradox_library=paradox_library,
        math_translator=math_translator,
        conversation_engine=conversation_engine,
        memory_integration=memory_integration,
        sully_instance=sully_instance
    )
    
    # If Sully instance is provided, integrate with it
    if sully_instance:
        integration_system.integrate_with_sully(sully_instance)
    
    # Start continuous operation mode
    integration_system.start_continuous_operation()
    
    return integration_system

# Example usage with continuous operation
def start_standalone_integration_system():
    """Start standalone kernel integration system with continuous operation."""
    # Create a fresh integration system
    system = KernelIntegrationSystem()
    
    # Start continuous operation
    system.start_continuous_operation()
    
    # Return the system for further reference
    return system
    
# Example usage
if __name__ == "__main__":
    # Create integration system with continuous operation
    integration_system = start_standalone_integration_system()
    
    # Keep the program running
    try:
        print("=== Kernel Integration System Running ===")
        print("Press Ctrl+C to stop")
        
        while True:
            # Periodically report status
            status = integration_system.get_continuous_status()
            print(f"Status: {status['status']}")
            print(f"Queue size: {status['queue_size']}")
            print(f"Emergent properties: {status['emergent_properties']}")
            print(f"Concept space size: {status['concept_space_size']}")
            print(f"Active workspaces: {status['workspaces']}")
            print("=" * 40)
            
            # Sleep for a bit
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("Stopping integration system...")
        integration_system.stop_continuous_operation()
        print("Integration system stopped")": 0.6,
                "tags": ["network_structure", "limited"]
            }]
            
        # Count node types
        node_types = Counter([node["type"] for node in network["nodes"]])
        
        # Identify the dominant kernel
        dominant_kernel = node_types.most_common(1)[0][0].split("_")[0]
        if dominant_kernel == "seed":
            if len(node_types) > 1:
                dominant_kernel = node_types.most_common(2)[1][0].split("_")[0]
            else:
                dominant_kernel = "none"
                
        insights.append({
            "description": f"The concept of {network['meta']['seed_concept']} is most productively explored through the {dominant_kernel} cognitive domain.",
            "confidence": 0.7,
            "tags": ["dominant_kernel", dominant_kernel]
        })
        
        # Check for cross-domain integration
        domains = set([node["type"].split("_")[0] for node in network["nodes"] if node["type"] != "seed"])
        if len(domains) >= 3:
            insights.append({
                "description": f"The concept of {network['meta']['seed_concept']} integrates richly across multiple cognitive domains: {', '.join(domains)}.",
                "confidence": 0.8,
                "tags": ["cross_domain", "integration"]
            })
            
        # Identify central concepts (besides seed)
        edge_counts = Counter()
        for edge in network["edges"]:
            edge_counts[edge["source"]] += 1
            edge_counts[edge["target"]] += 1
            
        # Remove the seed concept
        seed_concept = network["meta"]["seed_concept"]
        if seed_concept in edge_counts:
            del edge_counts[seed_concept]
            
        # Get most connected concepts
        most_connected = edge_counts.most_common(2)
        if most_connected:
            insights.append({
                "description": f"Beyond the core concept, '{most_connected[0][0]}' emerges as a central connecting idea with {most_connected[0][1]} connections.",
                "confidence": 0.75,
                "tags": ["central_concept", "connectivity"]
            })
            
        # Check for paradoxical nature
        if "paradox" in domains:
            paradox_nodes = [node for node in network["nodes"] if node["type"] == "paradox"]
            if paradox_nodes:
                insights.append({
                    "description": f"The concept of {network['meta']['seed_concept']} contains paradoxical elements, suggesting it may resist simple categorization.",
                    "confidence": 0.7,
                    "tags": ["paradoxical", "complexity"]
                })
                
        # Check for dream associations
        if "dream" in domains:
            dream_nodes = [node for node in network["nodes"] if node["type"] == "dream"]
            if dream_nodes:
                dream_concepts = [node["id"] for node in dream_nodes]
                insights.append({
                    "description": f"In the dream-like associative space, {network['meta']['seed_concept']} connects to {', '.join(dream_concepts[:3])}.",
                    "confidence": 0.6,
                    "tags": ["dream_association", "symbolism"]
                })
                
        # Check for fusion potential
        if "fusion" in domains:
            fusion_nodes = [node for node in network["nodes"] if node["type"] == "fusion"]
            if fusion_nodes:
                insights.append({
                    "description": f"The concept of {network['meta']['seed_concept']} shows high fusion potential, able to blend with other concepts to generate novel insights.",
                    "confidence": 0.7,
                    "tags": ["fusion_potential", "creativity"]
                })
                
        # Check for mathematizability
        if "math" in domains:
            math_nodes = [node for node in network["nodes"] if node["type"] == "math"]
            if math_nodes:
                insights.append({
                    "description": f"The concept of {network['meta']['seed_concept']} shows formal structure that can be represented mathematically.",
                    "confidence": 0.7,
                    "tags": ["mathematical", "formal_structure"]
                })
                
        return insights

    def recursive_concept_exploration(self, seed_concept: str, max_depth: int = 3, 
                                  exploration_breadth: int = 2) -> Dict[str, Any]:
        """
        Performs a deep recursive exploration of a concept, alternating between
        different cognitive kernels at each depth.
        
        Args:
            seed_concept: Initial concept to explore
            max_depth: Maximum depth of recursion
            exploration_breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        # Create a workspace for the operation
        workspace = CognitiveWorkspace(
            title=f"Recursive Exploration: {seed_concept}",
            purpose=f"Perform deep recursive exploration of {seed_concept}"
        )
        
        # Add seed concept to workspace
        workspace.add_data("seed_concept", seed_concept)
        workspace.add_data("max_depth", max_depth)
        workspace.add_data("exploration_breadth", exploration_breadth)
        
        # Add to workspaces
        self.workspaces[workspace.id] = workspace
        
        # Create the exploration structure
        exploration = {
            "seed_concept": seed_concept,
            "max_depth": max_depth,
            "exploration_breadth": exploration_breadth,
            "creation_time": datetime.now().isoformat(),
            "nodes": [],
            "edges": [],
            "insights": []
        }
        
        # Add the seed node
        seed_node = {
            "id": str(uuid.uuid4()),
            "concept": seed_concept,
            "level": 0,
            "kernel": "seed",
            "content": None,
            "children": []
        }
        
        exploration["nodes"].append(seed_node)
        
        # Available kernels for exploration
        available_kernels = []
        if self.dream_core:
            available_kernels.append("dream")
        if self.fusion_engine:
            available_kernels.append("fusion")
        if self.paradox_library:
            available_kernels.append("paradox")
        if self.math_translator:
            available_kernels.append("math")
            
        # If no kernels available, return limited exploration
        if not available_kernels:
            workspace.add_insight(
                "No exploration kernels available. Returning limited results.",
                "system",
                0.9,
                ["error", "system"]
            )
            workspace.complete(exploration)
            return exploration
            
        # Perform recursive exploration
        self._explore_concept_recursively(
            seed_node, 
            available_kernels,
            1, 
            max_depth, 
            exploration_breadth,
            exploration["nodes"],
            exploration["edges"],
            workspace
        )
        
        # Generate insights from the exploration
        insights = self._generate_exploration_insights(exploration)
        exploration["insights"] = insights
        
        # Add insights to workspace
        for insight in insights:
            workspace.add_insight(
                insight["description"],
                "recursive_exploration",
                insight.get("confidence", 0.7),
                insight.get("tags", [])
            )
        
        # Complete the workspace
        workspace.add_data("final_exploration", exploration)
        workspace.complete(exploration)
        
        # Record the integration
        self._add_integration_record("recursive_exploration", {
            "seed_concept": seed_concept,
            "max_depth": max_depth,
            "exploration_breadth": exploration_breadth,
            "node_count": len(exploration["nodes"]),
            "edge_count": len(exploration["edges"]),
            "workspace_id": workspace.id
        })
        
        return exploration
        
    def _explore_concept_recursively(self, parent_node: Dict[str, Any], available_kernels: List[str],
                                  current_level: int, max_depth: int, exploration_breadth: int,
                                  nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                                  workspace: CognitiveWorkspace):
        """
        Recursively explore a concept through different kernels.
        
        Args:
            parent_node: Parent concept node
            available_kernels: List of available kernels
            current_level: Current exploration level
            max_depth: Maximum exploration depth
            exploration_breadth: Number of branches to explore
            nodes: List to store nodes
            edges: List to store edges
            workspace: Cognitive workspace for the operation
        """
        # Stop if we've reached max depth
        if current_level > max_depth:
            return
            
        # Get the concept to explore
        concept = parent_node["concept"]
        
        # Choose kernel(s) for this level
        # Use different kernels for different branches if possible
        level_kernels = []
        if len(available_kernels) >= exploration_breadth:
            # Use different kernels for each branch
            level_kernels = random.sample(available_kernels, exploration_breadth)
        else:
            # Reuse kernels to reach exploration_breadth
            level_kernels = available_kernels * (exploration_breadth // len(available_kernels) + 1)
            level_kernels = level_kernels[:exploration_breadth]
            
        # Process the concept with each kernel
        for i, kernel in enumerate(level_kernels):
            try:
                # Process with the appropriate kernel
                child_concept = None
                content = None
                
                if kernel == "dream":
                    # Generate dream and extract a concept
                    dream_result = self.dream_core.generate(concept, "standard", "associative")
                    content = dream_result
                    
                    # Extract potential new concept
                    if isinstance(dream_result, str):
                        words = dream_result.split()
                        significant_words = [w for w in words if len(w) > 5 and w.lower() != concept.lower()]
                        
                        if significant_words:
                            child_concept = significant_words[0]
                        else:
                            child_concept = f"{concept}_dream_aspect"
                            
                elif kernel == "fusion":
                    # Find a concept to fuse with
                    # Try to use a sibling concept if available
                    fusion_partner = None
                    
                    if parent_node["children"]:
                        # Use a sibling concept
                        sibling_idx = (i - 1) % len(parent_node["children"])
                        fusion_partner = parent_node["children"][sibling_idx]["concept"]
                    else:
                        # Get from codex or use default
                        related = self.codex.get_related_concepts(concept, max_depth=1)
                        fusion_partner = next(iter(related.keys())) if related else "concept"
                        
                    # Generate fusion
                    fusion_result = self.fusion_engine.fuse_concepts(concept, fusion_partner)
                    content = fusion_result
                    
                    # Extract potential new concept
                    if isinstance(fusion_result, str):
                        words = fusion_result.split()
                        significant_words = [w for w in words 
                                          if len(w) > 5 
                                          and w.lower() != concept.lower() 
                                          and w.lower() != fusion_partner.lower()]
                        
                        if significant_words:
                            child_concept = significant_words[0]
                        else:
                            child_concept = f"{concept}_{fusion_partner}_fusion"
                            
                elif kernel == "paradox":
                    # Explore paradoxes
                    paradox_result = self.paradox_library.get(concept)
                    content = paradox_result
                    
                    # Extract potential new concept
                    if isinstance(paradox_result, dict):
                        if "related_concepts" in paradox_result and paradox_result["related_concepts"]:
                            child_concept = paradox_result["related_concepts"][0]
                        elif "type" in paradox_result:
                            child_concept = f"{paradox_result['type']}_paradox"
                        else:
                            child_concept = f"{concept}_paradox"
                    else:
                        child_concept = f"{concept}_paradox"
                        
                elif kernel == "math":
                    # Translate to math
                    math_result = self.math_translator.translate(concept)
                    content = math_result
                    
                    # Extract potential new concept
                    if isinstance(math_result, dict):
                        if "matches" in math_result and math_result["matches"]:
                            # Get first match
                            child_concept = next(iter(math_result["matches"].keys()))
                        elif "domain" in math_result:
                            child_concept = f"{math_result['domain']}_formalization"
                        else:
                            child_concept = f"{concept}_formalization"
                    else:
                        child_concept = f"{concept}_formalization"
                
                # Create child node
                child_node = {
                    "id": str(uuid.uuid4()),
                    "concept": child_concept,
                    "level": current_level,
                    "kernel": kernel,
                    "content": content,
                    "children": []
                }
                
                # Add to nodes
                nodes.append(child_node)
                
                # Add edge
                edges.append({
                    "source": parent_node["id"],
                    "target": child_node["id"],
                    "type": kernel
                })
                
                # Add to parent's children
                parent_node["children"].append(child_node)
                
                # Add data to workspace
                workspace.add_data(
                    f"level_{current_level}_{kernel}_{i}",
                    {
                        "concept": child_concept,
                        "content": content
                    },
                    kernel
                )
                
                # Recursively explore child node
                self._explore_concept_recursively(
                    child_node, 
                    available_kernels,
                    current_level + 1, 
                    max_depth, 
                    exploration_breadth,
                    nodes,
                    edges,
                    workspace
                )
            except Exception as e:
                logger.error(f"Error in recursive exploration with {kernel}: {str(e)}")
                workspace.add_insight(
                    f"Exploration failed for {concept} with {kernel}: {str(e)}",
                    "error",
                    0.5,
                    ["error", kernel]
                )

    def _generate_exploration_insights(self, exploration: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes a concept exploration to generate insights.
        
        Args:
            exploration: Recursive exploration results
            
        Returns:
            List of insights
        """
        insights = []
        
        seed_concept = exploration["seed_concept"]
        nodes = exploration["nodes"]
        edges = exploration["edges"]
        
        # Not enough data for insights
        if len(nodes) < 3:
            return [{
                "description": f"The exploration of {seed_concept} was limited, suggesting this concept may be difficult to explore with available kernels.",
                "confidence": 0.6,
                "tags": ["limited_exploration"]
            }]
            
        # Count nodes by level
        nodes_by_level = defaultdict(int)
        for node in nodes:
            nodes_by_level[node["level"]] += 1
            
        # Calculate branching factor
        branching_factor = sum(nodes_by_level.values()) / max(1, (max(nodes_by_level.keys()) + 1))
        
        if branching_factor < 2.0:
            insights.append({
                "description": f"The concept of {seed_concept} shows limited branching ({branching_factor:.1f}), suggesting it may be relatively self-contained.",
                "confidence": 0.7,
                "tags": ["branching_factor", "self_contained"]
            })
        else:
            insights.append({
                "description": f"The concept of {seed_concept} shows strong branching ({branching_factor:.1f}), suggesting it connects to many other concepts.",
                "confidence": 0.7,
                "tags": ["branching_factor", "highly_connected"]
            })
            
        # Count by kernel
        kernel_counts = Counter([node["kernel"] for node in nodes if node["kernel"] != "seed"])
        total_kernel_nodes = sum(kernel_counts.values())
        
        if total_kernel_nodes > 0:
            # Find the most productive kernel
            most_productive = kernel_counts.most_common(1)[0]
            productivity_ratio = most_productive[1] / total_kernel_nodes
            
            insights.append({
                "description": f"The {most_productive[0]} kernel was most productive for exploring {seed_concept}, generating {most_productive[1]} paths ({productivity_ratio*100:.0f}%).",
                "confidence