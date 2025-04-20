# sully_engine/memory_integration.py
# 🔄 Memory System Integration Module

from typing import Dict, List, Any, Optional, Union, Tuple, cast
import os
import json
import time
import random  # Added missing import for random module
from datetime import datetime

from sully_engine.memory import SullySearchMemory, MemoryTrace, EpisodicContext, EmotionalState

class MemoryIntegration:
    """
    Integration module for Sully's memory system.
    Handles cross-module communication and ensures coordinated memory operations.
    """
    
    def __init__(self, memory_path: Optional[str] = None):
        """
        Initialize the memory integration module.
        
        Args:
            memory_path: Path to memory storage file
        """
        # Core memory system
        self.memory = SullySearchMemory(memory_path)
        
        # Module interaction tracking
        self.module_interactions: Dict[str, Dict[str, Any]] = {}
        
        # Cache frequently accessed memories for performance
        self.memory_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_expiry: Dict[str, float] = {}  # For cache invalidation
        self.cache_lifetime: int = 300  # 5 minutes
        
        # Module connection registry
        self.connected_modules: set = set()
        
        # Memory operation statistics
        self.operation_stats: Dict[str, Any] = {
            "total_queries": 0,
            "total_stores": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def register_module(self, module_name: str) -> None:
        """
        Register a module with the memory system.
        
        Args:
            module_name: Name of the module to register
        """
        self.connected_modules.add(module_name)
        self.module_interactions[module_name] = {
            "queries": 0,
            "stores": 0,
            "last_interaction": datetime.now().isoformat()
        }
    
    def store_interaction(self, message: str, response: str, 
                        module: str = "general", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an interaction between user and Sully.
        
        Args:
            message: User's message
            response: Sully's response
            module: Source module
            metadata: Additional metadata
            
        Returns:
            ID of stored memory
        """
        # Update stats
        self.operation_stats["total_stores"] += 1
        if module in self.module_interactions:
            self.module_interactions[module]["stores"] += 1
            self.module_interactions[module]["last_interaction"] = datetime.now().isoformat()
        
        # Create combined content
        content = {
            "user_message": message,
            "sully_response": response
        }
        
        # Create context with module info
        context = {
            "timestamp": datetime.now().isoformat(),
            "module": module
        }
        
        # Add additional metadata if provided
        if metadata:
            context.update(metadata)
        
        # Store in memory
        memory_id = self.memory.store_query(message, response, context)
        
        # Invalidate any affected cache entries
        self._invalidate_related_cache(message)
        
        return memory_id
    
    def store_experience(self, content: str, source: str, 
                       importance: float = 0.6, emotional_tags: Optional[Dict[str, float]] = None,
                       concepts: Optional[List[str]] = None, is_episodic: bool = False) -> str:
        """
        Store a general experience or knowledge.
        
        Args:
            content: Content to store
            source: Source of the content
            importance: Importance of the content
            emotional_tags: Emotional tags for the content
            concepts: Relevant concepts
            is_episodic: Whether this should be part of an episodic memory
            
        Returns:
            ID of stored memory
        """
        # Update stats
        self.operation_stats["total_stores"] += 1
        if source in self.module_interactions:
            self.module_interactions[source]["stores"] += 1
            self.module_interactions[source]["last_interaction"] = datetime.now().isoformat()
        
        # Store in memory
        memory_id = self.memory.store_experience(
            content=content,
            source=source,
            importance=importance,
            emotional_tags=emotional_tags if emotional_tags else {},
            concepts=concepts if concepts else [],
            episodic=is_episodic
        )
        
        # Invalidate related cache entries
        if concepts:
            for concept in concepts:
                self._invalidate_related_cache(concept)
        
        return memory_id
    
    def recall(self, query: str, limit: int = 5, module: str = "general", 
             include_emotional: bool = True) -> List[Dict[str, Any]]:
        """
        Recall relevant memories based on a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            module: Requesting module
            include_emotional: Whether to include emotional associations
            
        Returns:
            List of relevant memories
        """
        # Update stats
        self.operation_stats["total_queries"] += 1
        if module in self.module_interactions:
            self.module_interactions[module]["queries"] += 1
            self.module_interactions[module]["last_interaction"] = datetime.now().isoformat()
        
        # Check cache first
        cache_key = f"{query}_{limit}_{include_emotional}"
        if cache_key in self.memory_cache and time.time() < self.cache_expiry.get(cache_key, 0):
            self.operation_stats["cache_hits"] += 1
            return self.memory_cache[cache_key]
        
        self.operation_stats["cache_misses"] += 1
        
        # If including emotional content, find emotional content in query
        if include_emotional:
            # Simple emotional keyword detection
            emotional_keywords = {
                "joy": ["happy", "joy", "delighted", "pleased", "glad"],
                "sadness": ["sad", "unhappy", "depressed", "down", "miserable"],
                "anger": ["angry", "upset", "furious", "outraged", "annoyed"],
                "fear": ["afraid", "scared", "frightened", "terrified", "anxious"],
                "surprise": ["surprised", "amazed", "astonished", "shocked"],
                "trust": ["trust", "believe", "confidence", "faith"]
            }
            
            # Check for emotional content in query
            query_lower = query.lower()
            emotion_filter = None
            
            for emotion, keywords in emotional_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    emotion_filter = emotion
                    break
            
            # Perform search with emotion filter if found
            if emotion_filter:
                memory_dict = self.memory.search(
                    keyword=query,
                    limit=limit,
                    emotional_filter=emotion_filter
                )
                results = list(memory_dict.values())
            else:
                # Standard search
                memory_dict = self.memory.search(keyword=query, limit=limit)
                results = list(memory_dict.values())
        else:
            # Standard search without emotional consideration
            memory_dict = self.memory.search(keyword=query, limit=limit)
            results = list(memory_dict.values())
        
        # Cache the results
        self.memory_cache[cache_key] = results
        self.cache_expiry[cache_key] = time.time() + self.cache_lifetime
        
        return results
    
    def get_emotional_context(self) -> Dict[str, Any]:
        """
        Get the current emotional context.
        
        Returns:
            Emotional context information
        """
        return self.memory.get_emotional_summary()
    
    def update_emotional_state(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """
        Update the emotional state.
        
        Args:
            emotions: Dictionary mapping emotions to intensities
            
        Returns:
            Updated emotional state
        """
        return self.memory.update_emotional_state(emotions)
    
    def begin_episode(self, description: str, module: str = "general") -> str:
        """
        Begin a new episodic memory context.
        
        Args:
            description: Description of the episode
            module: Source module
            
        Returns:
            ID of the created episode
        """
        # Create spatial markers with module info
        spatial_markers = {"module": module}
        
        # Begin episodic context
        episode_id = self.memory.begin_episodic_context(
            situation=description,
            spatial_markers=spatial_markers,
            entities=[module]
        )
        
        return episode_id
    
    def end_episode(self, summary: str) -> Dict[str, Any]:
        """
        End the current episodic memory context.
        
        Args:
            summary: Summary of the episode
            
        Returns:
            Episode information
        """
        return self.memory.close_episodic_context(summary)
    
    def get_recent_memories(self, days: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent memories from the specified time period.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memories
        """
        return self.memory.get_temporal_context(datetime.now(), window_days=days, limit=limit)
    
    def find_connections(self, concept1: str, concept2: str) -> List[Dict[str, Any]]:
        """
        Find connections between two concepts in memory.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            List of connections
        """
        return self.memory.find_connections(concept1, concept2)
    
    def get_self_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get self-referential memories for introspection.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of self-referential memories
        """
        return self.memory.get_self_referential_memories(limit)
    
    def strengthen_memory(self, memory_id: str, boost: float = 0.1) -> Dict[str, Any]:
        """
        Strengthen a specific memory.
        
        Args:
            memory_id: ID of the memory to strengthen
            boost: Amount to boost importance
            
        Returns:
            Updated memory information
        """
        return self.memory.reinforce_memory(memory_id, importance_adjustment=boost)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Memory statistics
        """
        # Get core memory stats
        core_stats = self.memory.get_memory_statistics()
        
        # Combine with integration stats
        integrated_stats = {
            **core_stats,
            "operation_stats": self.operation_stats,
            "module_interactions": self.module_interactions,
            "connected_modules": list(self.connected_modules),
            "cache_size": len(self.memory_cache)
        }
        
        return integrated_stats
    
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in memory usage and content.
        
        Returns:
            Analysis of memory patterns
        """
        # Get all memories
        memories = []
        try:
            memories = list(self.memory.storage.values()) if hasattr(self.memory, 'storage') else []
        except AttributeError:
            return {"error": "Memory storage not accessible"}
        
        if not memories:
            return {"error": "No memories available for analysis"}
        
        # Analyze memory types
        memory_types: Dict[str, int] = {}
        for memory in memories:
            memory_type = getattr(memory, 'memory_type', 'unknown')
            if memory_type not in memory_types:
                memory_types[memory_type] = 0
            memory_types[memory_type] += 1
        
        # Analyze emotional patterns
        emotion_counts: Dict[str, Dict[str, Union[int, float]]] = {}
        for memory in memories:
            emotional_valence = getattr(memory, 'emotional_valence', {})
            for emotion, strength in emotional_valence.items():
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = {"count": 0, "avg_strength": 0.0, "total_strength": 0.0}
                emotion_counts[emotion]["count"] += 1
                emotion_counts[emotion]["total_strength"] += strength
        
        # Calculate average strengths
        for emotion in emotion_counts:
            if emotion_counts[emotion]["count"] > 0:
                emotion_counts[emotion]["avg_strength"] = (
                    cast(float, emotion_counts[emotion]["total_strength"]) / 
                    cast(int, emotion_counts[emotion]["count"])
                )
        
        # Analyze access patterns
        access_distribution: Dict[str, int] = {}
        for memory in memories:
            access_count = getattr(memory, 'access_count', 0)
            access_level = "never" if access_count == 0 else \
                          "low" if access_count < 3 else \
                          "medium" if access_count < 10 else \
                          "high"
            if access_level not in access_distribution:
                access_distribution[access_level] = 0
            access_distribution[access_level] += 1
        
        # Identify frequently co-occurring concepts (simplified)
        concept_pairs: Dict[Tuple[str, str], int] = {}
        for memory in memories:
            content = getattr(memory, 'content', '')
            if not isinstance(content, str):
                continue
                
            words = content.lower().split()
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:  # Skip short words
                    pair = (words[i], words[i+1])
                    if pair not in concept_pairs:
                        concept_pairs[pair] = 0
                    concept_pairs[pair] += 1
        
        # Get top co-occurring concepts
        top_pairs = sorted(concept_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_pairs_formatted = [{"pair": f"{p[0]} + {p[1]}", "count": c} for (p[0], p[1]), c in top_pairs]
        
        return {
            "memory_type_distribution": memory_types,
            "emotional_patterns": emotion_counts,
            "access_patterns": access_distribution,
            "co_occurring_concepts": top_pairs_formatted,
            "total_memories_analyzed": len(memories)
        }
    
    def clear_cache(self) -> None:
        """Clear the memory cache."""
        self.memory_cache = {}
        self.cache_expiry = {}
    
    def _invalidate_related_cache(self, query: str) -> None:
        """
        Invalidate cache entries related to a query.
        
        Args:
            query: Query to invalidate related cache for
        """
        query_lower = query.lower()
        keys_to_remove = []
        
        # Find cache keys that might be affected by this new content
        for key in self.memory_cache:
            # If the query is a substring of the cache key, invalidate it
            if query_lower in key.lower():
                keys_to_remove.append(key)
        
        # Remove affected cache entries
        for key in keys_to_remove:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]
    
    def save_memory_state(self) -> bool:
        """
        Save memory state to disk.
        
        Returns:
            Success indicator
        """
        try:
            # Export full memory system
            return True if hasattr(self.memory, '_save_to_file') and self.memory._save_to_file() else False
        except Exception as e:
            print(f"Error saving memory state: {str(e)}")
            return False
    
    def search_by_emotion(self, emotion: str, strength_threshold: float = 0.5, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for memories with a specific emotional component.
        
        Args:
            emotion: Emotion to search for
            strength_threshold: Minimum emotional strength
            limit: Maximum number of results
            
        Returns:
            List of memories with the specified emotion
        """
        return self.memory.get_emotional_memories(emotion, strength_threshold, limit)
    
    def get_episode(self, episode_id: str) -> Dict[str, Any]:
        """
        Get a complete episodic memory.
        
        Args:
            episode_id: ID of the episode
            
        Returns:
            Episodic memory data
        """
        return self.memory.get_episodic_memory(episode_id)

# Integration with reasoning module
def integrate_with_reasoning(memory_integration: MemoryIntegration, reasoning_node) -> None:
    """
    Integrate memory with the reasoning system.
    
    Args:
        memory_integration: Memory integration module
        reasoning_node: Reasoning node to integrate with
    """
    # Register the reasoning module
    memory_integration.register_module("reasoning")
    
    # Patch reasoning module's memory access for performance
    if hasattr(reasoning_node, "memory"):
        original_memory = reasoning_node.memory
        reasoning_node.memory = memory_integration
    else:
        # Create memory attribute if it doesn't exist
        original_memory = None
        reasoning_node.memory = memory_integration
    
    # Add integration methods
    def reason_with_memory(self, message: str, tone: str = "emergent") -> Dict[str, Any]:
        """Reason with memory integration."""
        # Get relevant memories
        memories = memory_integration.recall(message, limit=3, module="reasoning")
        
        # Include memory context in reasoning
        context = f"{message}"
        if memories:
            context += f"\n\nRelevant context from memory:"
            for i, memory in enumerate(memories):
                if isinstance(memory.get("content"), dict):
                    # For query memories
                    if "query" in memory["content"] and "result" in memory["content"]:
                        context += f"\n- Previous query: {memory['content']['query']}"
                        result = memory['content']['result']
                        if isinstance(result, dict) and "response" in result:
                            context += f"\n  Response: {result['response'][:100]}..."
                        else:
                            context += f"\n  Response: {str(result)[:100]}..."
                else:
                    # For experience memories
                    context += f"\n- Relevant memory: {str(memory.get('content', ''))[:100]}..."
        
        # Process with reasoning
        if original_memory and hasattr(original_memory, "reason"):
            result = original_memory.reason(context, tone)
        else:
            # Fallback if original memory or reason method doesn't exist
            result = {"response": f"Processed with tone: {tone}", "confidence": 0.8}
        
        # Store reasoning result in memory
        memory_integration.store_interaction(message, str(result), "reasoning", {
            "tone": tone,
            "with_memory_context": len(memories) > 0
        })
        
        return result
    
    # Add the method to the reasoning node
    reasoning_node.reason_with_memory = reason_with_memory.__get__(reasoning_node)

# Integration with conversation engine
def integrate_with_conversation(memory_integration: MemoryIntegration, conversation_engine) -> None:
    """
    Integrate memory with the conversation engine.
    
    Args:
        memory_integration: Memory integration module
        conversation_engine: Conversation engine to integrate with
    """
    # Register the conversation module
    memory_integration.register_module("conversation")
    
    # Patch conversation engine's memory access for performance
    if hasattr(conversation_engine, "memory"):
        original_memory = conversation_engine.memory
        conversation_engine.memory = memory_integration
    else:
        original_memory = None
        conversation_engine.memory = memory_integration
    
    # Add helper method for emotional tone detection if it doesn't exist
    if not hasattr(conversation_engine, "_detect_emotional_tone"):
        def _detect_emotional_tone(self, message: str) -> Dict[str, float]:
            """Simple detection of emotional tone in message."""
            emotional_tone = {
                "joy": 0.0, "sadness": 0.0, "anger": 0.0, 
                "fear": 0.0, "surprise": 0.0, "trust": 0.0
            }
            
            # Simple keyword matching for emotional tone
            joy_words = ["happy", "joy", "delighted", "pleased", "glad"]
            sadness_words = ["sad", "unhappy", "depressed", "down", "miserable"]
            anger_words = ["angry", "upset", "furious", "outraged", "annoyed"]
            fear_words = ["afraid", "scared", "frightened", "terrified", "anxious"]
            surprise_words = ["surprised", "amazed", "astonished", "shocked"]
            trust_words = ["trust", "believe", "confidence", "faith"]
            
            message_lower = message.lower()
            
            # Check for emotional keywords
            for word in joy_words:
                if word in message_lower:
                    emotional_tone["joy"] += 0.2
            
            for word in sadness_words:
                if word in message_lower:
                    emotional_tone["sadness"] += 0.2
            
            for word in anger_words:
                if word in message_lower:
                    emotional_tone["anger"] += 0.2
            
            for word in fear_words:
                if word in message_lower:
                    emotional_tone["fear"] += 0.2
            
            for word in surprise_words:
                if word in message_lower:
                    emotional_tone["surprise"] += 0.2
            
            for word in trust_words:
                if word in message_lower:
                    emotional_tone["trust"] += 0.2
            
            # Cap values at 1.0
            for emotion in emotional_tone:
                emotional_tone[emotion] = min(emotional_tone[emotion], 1.0)
            
            return emotional_tone
        
        conversation_engine._detect_emotional_tone = _detect_emotional_tone.__get__(conversation_engine)
    
    # Add integration methods
    def process_with_memory(self, message: str, tone: str = "emergent", 
                          continue_conversation: bool = True) -> str:
        """Process message with memory integration."""
        # Get emotional context
        emotional_context = memory_integration.get_emotional_context()
        
        # Adjust tone based on emotional context if needed
        adjusted_tone = tone
        if "dominant_emotion" in emotional_context:
            dominant_emotion = emotional_context["dominant_emotion"]
            emotion_intensity = emotional_context.get("intensity", 0)
            
            # If there's a strong emotion, adjust tone accordingly
            if emotion_intensity > 0.7:
                if dominant_emotion == "joy":
                    adjusted_tone = "creative"
                elif dominant_emotion == "sadness":
                    adjusted_tone = "ethereal"
                elif dominant_emotion == "anger":
                    adjusted_tone = "critical"
                elif dominant_emotion == "fear":
                    adjusted_tone = "analytical"
        
        # Get memory context
        memories = memory_integration.recall(message, limit=3, module="conversation")
        
        # Process message with original method if it exists
        if hasattr(self, "process_message"):
            response = self.process_message(message, adjusted_tone, continue_conversation)
        else:
            # Fallback if process_message doesn't exist
            response = f"Processed message with tone: {adjusted_tone}"
        
        # Store the interaction
        memory_integration.store_interaction(message, response, "conversation", {
            "tone": adjusted_tone,
            "continue_conversation": continue_conversation,
            "emotional_context": emotional_context
        })
        
        # Update emotional state based on detected emotion in message
        emotional_tone = self._detect_emotional_tone(message)
        memory_integration.update_emotional_state(emotional_tone)
        
        # If we have memories and should continue conversation, potentially reference them
        if memories and continue_conversation and random.random() < 0.3:  # 30% chance
            memory_reference = self._generate_memory_reference(memories)
            if memory_reference:
                response += f"\n\n{memory_reference}"
        
        return response
    
    def _generate_memory_reference(self, memories: List[Dict[str, Any]]) -> str:
        """Generate a reference to previous memory."""
        if not memories:
            return ""
            
        memory = memories[0]  # Take the first/most relevant memory
        
        if isinstance(memory.get("content"), dict):
            # For query memories
            if "query" in memory["content"]:
                query = memory["content"]["query"]
                return f"This reminds me of when you previously asked about {query}."
        else:
            # For experience memories
            content = str(memory.get("content", ""))[:50]
            return f"I recall we discussed something similar earlier: {content}..."
        
        return ""
    
    # Add the methods to the conversation engine
    conversation_engine.process_with_memory = process_with_memory.__get__(conversation_engine)
    conversation_engine._generate_memory_reference = _generate_memory_reference.__get__(conversation_engine)

# Full integration with Sully
def integrate_with_sully(sully, memory_path: Optional[str] = None) -> MemoryIntegration:
    """
    Integrate memory with the full Sully system.
    
    Args:
        sully: The main Sully instance
        memory_path: Optional path to memory storage
        
    Returns:
        Memory integration module
    """
    # Create memory integration
    memory_integration = MemoryIntegration(memory_path)
    
    # Register all modules
    for module_name in [
        "reasoning", "conversation", "codex", "judgment", "dream", 
        "translator", "fusion", "paradox", "continuous_learning", 
        "autonomous_goals", "visual_cognition", "emergence", "virtue", "intuition"
    ]:
        if hasattr(sully, module_name):
            memory_integration.register_module(module_name)
            
            # Set memory for the module
            module = getattr(sully, module_name)
            if hasattr(module, "memory"):
                original_memory = getattr(module, "memory")
                setattr(module, "memory", memory_integration)
    
    # Integrate with reasoning
    if hasattr(sully, "reasoning_node"):
        integrate_with_reasoning(memory_integration, sully.reasoning_node)
    
    # Integrate with conversation
    if hasattr(sully, "conversation"):
        integrate_with_conversation(memory_integration, sully.conversation)
    
    # Add the memory integration to Sully
    sully.memory_integration = memory_integration
    
    # Add helper method for emotional analysis if it doesn't exist
    if not hasattr(sully, "_analyze_emotional_content"):
        def _analyze_emotional_content(self, text: str) -> Dict[str, float]:
            """Analyze emotional content in text."""
            emotional_valence = {
                "joy": 0.0, "sadness": 0.0, "anger": 0.0, 
                "fear": 0.0, "surprise": 0.0, "trust": 0.0
            }
            
            # Simple keyword matching for emotional valence
            joy_words = ["happy", "joy", "delighted", "pleased", "glad"]
            sadness_words = ["sad", "unhappy", "depressed", "down", "miserable"]
            anger_words = ["angry", "upset", "furious", "outraged", "annoyed"]
            fear_words = ["afraid", "scared", "frightened", "terrified", "anxious"]
            surprise_words = ["surprised", "amazed", "astonished", "shocked"]
            trust_words = ["trust", "believe", "confidence", "faith"]
            
            text_lower = text.lower()
            
            # Check for emotional keywords
            for word in joy_words:
                if word in text_lower:
                    emotional_valence["joy"] += 0.2
            
            for word in sadness_words:
                if word in text_lower:
                    emotional_valence["sadness"] += 0.2
            
            for word in anger_words:
                if word in text_lower:
                    emotional_valence["anger"] += 0.2
            
            for word in fear_words:
                if word in text_lower:
                    emotional_valence["fear"] += 0.2
            
            for word in surprise_words:
                if word in text_lower:
                    emotional_valence["surprise"] += 0.2
            
            for word in trust_words:
                if word in text_lower:
                    emotional_valence["trust"] += 0.2
            
            # Cap values at 1.0
            for emotion in emotional_valence:
                emotional_valence[emotion] = min(emotional_valence[emotion], 1.0)
            
            return emotional_valence
        
        sully._analyze_emotional_content = _analyze_emotional_content.__get__(sully)
    
    # Add helper method for cognitive tone determination if it doesn't exist
    if not hasattr(sully, "_determine_cognitive_tone"):
        def _determine_cognitive_tone(self, text: str, emotional_valence: Dict[str, float]) -> str:
            """Determine cognitive tone based on text and emotional state."""
            # Default tone
            tone = "emergent"
            
            # Find dominant emotion
            dominant_emotion = max(emotional_valence.items(), key=lambda x: x[1])[0]
            max_intensity = max(emotional_valence.values())
            
            # If there's a strong emotion, set tone accordingly
            if max_intensity > 0.6:
                if dominant_emotion == "joy":
                    tone = "creative"
                elif dominant_emotion == "sadness":
                    tone = "ethereal"
                elif dominant_emotion == "anger":
                    tone = "critical"
                elif dominant_emotion == "fear":
                    tone = "analytical"
            
            # Check for keywords that might indicate specific tone needs
            text_lower = text.lower()
            
            # Keywords that might indicate need for specific tones
            analytical_keywords = ["analyze", "examine", "study", "evaluate", "assess"]
            critical_keywords = ["critique", "criticize", "review", "judge", "evaluate"]
            creative_keywords = ["creative", "imagine", "invent", "dream", "create"]
            ethereal_keywords = ["philosophy", "meaning", "purpose", "existence", "spiritual"]
            
            # Check for keywords
            if any(keyword in text_lower for keyword in analytical_keywords):
                tone = "analytical"
            elif any(keyword in text_lower for keyword in critical_keywords):
                tone = "critical"
            elif any(keyword in text_lower for keyword in creative_keywords):
                tone = "creative"
            elif any(keyword in text_lower for keyword in ethereal_keywords):
                tone = "ethereal"
            
            return tone
        
        sully._determine_cognitive_tone = _determine_cognitive_tone.__get__(sully)
    
    # Create new process method that uses memory
    def process_with_memory(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user input with memory integration.
        
        Args:
            user_input: User's message or query
            context: Optional contextual information
            
        Returns:
            Sully's response
        """
        if context is None:
            context = {}
            
        # Start episodic memory for this interaction
        episode_id = memory_integration.begin_episode(f"Processing input: {user_input[:50]}...", "sully")
        
        # Process emotional content
        emotional_valence = self._analyze_emotional_content(user_input)
        memory_integration.update_emotional_state(emotional_valence)
        
        # Get relevant memories
        memories = memory_integration.recall(user_input, limit=5, module="sully")
        
        # Determine if this is a recurring topic
        recurring_topic = False
        recurring_count = 0
        
        if len(memories) >= 2:
            # Check if there's a pattern of similar queries
            topic_words = set(user_input.lower().split())
            for memory in memories:
                if isinstance(memory.get("content"), dict) and "query" in memory["content"]:
                    memory_words = set(memory["content"]["query"].lower().split())
                    common_words = topic_words.intersection(memory_words)
                    if len(common_words) >= min(3, len(topic_words) // 2):
                        recurring_count += 1
            
            recurring_topic = recurring_count >= 2
        
        # Get cognitive tone based on input and memory context
        tone = self._determine_cognitive_tone(user_input, emotional_valence)
        
        # Modify tone if it's a recurring topic to add variety
        if recurring_topic and tone != "critical" and random.random() < 0.7:
            # Use more reflective or creative tones for recurring topics
            tone = random.choice(["creative", "ethereal", "critical"])
        
        # Process through appropriate engine based on memory context
        if recurring_topic and hasattr(self, "conversation") and hasattr(self.conversation, "process_with_memory"):
            # For recurring topics, use conversation engine with memory
            response = self.conversation.process_with_memory(user_input, tone)
        elif len(memories) > 0 and hasattr(self, "reasoning_node") and hasattr(self.reasoning_node, "reason_with_memory"):
            # For topics with relevant memories, use reasoning with memory
            result = self.reasoning_node.reason_with_memory(user_input, tone)
            if isinstance(result, dict) and "response" in result:
                response = result["response"]
            else:
                response = str(result)
        else:
            # Default to standard processing
            if hasattr(self, "process"):
                response = self.process(user_input, context)
            else:
                # Fallback if process method doesn't exist
                response = f"Processed input with tone: {tone}"
        
        # Store the interaction in memory as an episodic memory
        memory_integration.store_interaction(
            user_input, response, "sully",
            {
                "episodic_context": episode_id,
                "tone": tone,
                "recurring_topic": recurring_topic,
                "recurring_count": recurring_count
            }
        )
        
        # Close the episodic memory
        memory_integration.end_episode(f"Processed input: {user_input[:50]}... with tone: {tone}")
        
        # Save memory state
        memory_integration.save_memory_state()
        
        return response
    
    # Add the method to Sully
    sully.process_with_memory = process_with_memory.__get__(sully)
    
    return memory_integration

# Memory integration initialization helper
def initialize_memory_integration(config: Optional[Dict[str, Any]] = None) -> MemoryIntegration:
    """
    Initialize memory integration with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized memory integration
    """
    memory_path = None
    if config and "memory_file" in config:
        memory_path = config["memory_file"]
        
        # Create directory if needed
        directory = os.path.dirname(memory_path)
        if directory:  # Check if directory is not empty
            os.makedirs(directory, exist_ok=True)
    
    return MemoryIntegration(memory_path)