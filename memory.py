# sully_engine/memory.py
# 🧠 Sully's Experiential Memory System with Emotional Tagging & Self-Reference

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import os
import numpy as np
import re
import math
from collections import defaultdict, Counter

class MemoryTrace:
    """
    Represents a single memory trace with emotional tagging and episodic context.
    """
    def __init__(self, content: Any, memory_type: str = "semantic",
                emotional_valence: Dict[str, float] = None,
                source: str = "direct", importance: float = 0.5,
                context: Dict[str, Any] = None):
        """
        Initialize a memory trace with emotional and contextual information.
        
        Args:
            content: The primary content of the memory
            memory_type: Type of memory (semantic, episodic, procedural, self-referential)
            emotional_valence: Emotional tags and their strengths (0.0-1.0)
            source: Source of the memory (direct, derived, simulated)
            importance: Subjective importance of the memory (0.0-1.0)
            context: Contextual information about the memory formation
        """
        self.content = content
        self.memory_type = memory_type
        self.emotional_valence = emotional_valence or {}
        self.source = source
        self.importance = importance
        self.context = context or {}
        
        # Memory metadata
        self.creation_time = datetime.now()
        self.last_accessed = self.creation_time
        self.access_count = 0
        self.reinforcement_count = 0
        self.associations = set()
        self.memory_id = f"{memory_type}_{int(self.creation_time.timestamp())}"
        
        # Memory dynamics
        self.decay_rate = 0.01  # Base rate of memory decay per day
        self.stability = 0.5    # Initial memory stability (0.0-1.0)
        self.clarity = 0.9      # Initial memory clarity (0.0-1.0)
        
    def access(self) -> None:
        """
        Access this memory, updating access metadata and strengthening the memory.
        """
        self.last_accessed = datetime.now()
        self.access_count += 1
        
        # Strengthen memory through access (spacing effect)
        time_since_creation = (self.last_accessed - self.creation_time).total_seconds()
        
        # Memories accessed after a delay are strengthened more (spacing effect)
        if time_since_creation > 3600:  # More than an hour
            self.stability = min(1.0, self.stability + 0.05)
            
        # Very frequently accessed memories become more stable
        if self.access_count > 5:
            self.stability = min(1.0, self.stability + 0.02)
            
    def reinforce(self, emotional_update: Dict[str, float] = None) -> None:
        """
        Reinforce this memory with possible emotional updating.
        
        Args:
            emotional_update: Optional updated emotional valences
        """
        self.reinforcement_count += 1
        
        # Update stability based on reinforcement
        self.stability = min(1.0, self.stability + 0.1)
        
        # Update emotional valence if provided
        if emotional_update:
            for emotion, strength in emotional_update.items():
                if emotion in self.emotional_valence:
                    # Blend existing and new emotional association
                    self.emotional_valence[emotion] = (self.emotional_valence[emotion] + strength) / 2
                else:
                    self.emotional_valence[emotion] = strength
                    
    def decay(self, days_elapsed: float) -> None:
        """
        Apply time-based decay to the memory based on its properties.
        
        Args:
            days_elapsed: Number of days since last decay update
        """
        if days_elapsed <= 0:
            return
            
        # Calculate effective decay rate based on:
        # - Base decay rate
        # - Memory stability (more stable = slower decay)
        # - Emotional intensity (stronger emotions = slower decay)
        # - Importance (more important = slower decay)
        
        emotional_intensity = sum(self.emotional_valence.values()) / max(1, len(self.emotional_valence))
        effective_decay = self.decay_rate * (1 - self.stability * 0.5) * (1 - emotional_intensity * 0.3) * (1 - self.importance * 0.4)
        
        # Apply decay to clarity (memories become less clear over time)
        decay_amount = effective_decay * days_elapsed
        self.clarity = max(0.1, self.clarity - decay_amount)
        
    def add_association(self, memory_id: str) -> None:
        """
        Add an association to another memory.
        
        Args:
            memory_id: ID of the associated memory
        """
        self.associations.add(memory_id)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory trace to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "emotional_valence": self.emotional_valence,
            "source": self.source,
            "importance": self.importance,
            "context": self.context,
            "creation_time": self.creation_time.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "reinforcement_count": self.reinforcement_count,
            "associations": list(self.associations),
            "stability": self.stability,
            "clarity": self.clarity
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MemoryTrace':
        """
        Create memory trace from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created memory trace
        """
        memory = MemoryTrace(
            content=data["content"],
            memory_type=data["memory_type"],
            emotional_valence=data["emotional_valence"],
            source=data["source"],
            importance=data["importance"],
            context=data["context"]
        )
        
        # Set memory ID if provided
        if "memory_id" in data:
            memory.memory_id = data["memory_id"]
            
        # Set dates
        try:
            memory.creation_time = datetime.fromisoformat(data["creation_time"])
            memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        except:
            # Keep default dates if parsing fails
            pass
            
        # Set counters
        memory.access_count = data.get("access_count", 0)
        memory.reinforcement_count = data.get("reinforcement_count", 0)
        
        # Set associations
        memory.associations = set(data.get("associations", []))
        
        # Set dynamics
        memory.stability = data.get("stability", 0.5)
        memory.clarity = data.get("clarity", 0.9)
        
        return memory

class EpisodicContext:
    """
    Represents the context of an episodic memory with spatial, temporal, and situational details.
    """
    def __init__(self, situation: str, 
                 temporal_markers: Dict[str, Any] = None,
                 spatial_markers: Dict[str, Any] = None,
                 entities: List[str] = None,
                 preceding_context: str = None,
                 following_context: str = None):
        """
        Initialize an episodic context.
        
        Args:
            situation: Brief description of the situation
            temporal_markers: Temporal indicators (time, date, duration, etc.)
            spatial_markers: Spatial indicators (location, environment, etc.)
            entities: Other entities involved in the episode
            preceding_context: What happened before this episode
            following_context: What happened after this episode
        """
        self.situation = situation
        self.temporal_markers = temporal_markers or {"timestamp": datetime.now().isoformat()}
        self.spatial_markers = spatial_markers or {}
        self.entities = entities or []
        self.preceding_context = preceding_context
        self.following_context = following_context
        
        # Additional metadata
        self.creation_time = datetime.now()
        self.episodic_id = f"episode_{int(self.creation_time.timestamp())}"
        
    def update_following_context(self, context: str) -> None:
        """
        Update what happened after this episode.
        
        Args:
            context: Following context
        """
        self.following_context = context
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert episodic context to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "episodic_id": self.episodic_id,
            "situation": self.situation,
            "temporal_markers": self.temporal_markers,
            "spatial_markers": self.spatial_markers,
            "entities": self.entities,
            "preceding_context": self.preceding_context,
            "following_context": self.following_context,
            "creation_time": self.creation_time.isoformat()
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EpisodicContext':
        """
        Create episodic context from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created episodic context
        """
        context = EpisodicContext(
            situation=data["situation"],
            temporal_markers=data.get("temporal_markers"),
            spatial_markers=data.get("spatial_markers"),
            entities=data.get("entities"),
            preceding_context=data.get("preceding_context"),
            following_context=data.get("following_context")
        )
        
        # Set episode ID if provided
        if "episodic_id" in data:
            context.episodic_id = data["episodic_id"]
            
        # Set creation time
        try:
            context.creation_time = datetime.fromisoformat(data["creation_time"])
        except:
            # Keep default date if parsing fails
            pass
            
        return context

class EmotionalState:
    """
    Represents an emotional state with multiple dimensions and intensities.
    """
    
    # Core emotion dimensions
    CORE_EMOTIONS = {
        "joy": {"valence": 1.0, "arousal": 0.8},
        "trust": {"valence": 0.8, "arousal": 0.3},
        "fear": {"valence": -0.8, "arousal": 0.9},
        "surprise": {"valence": 0.2, "arousal": 0.9},
        "sadness": {"valence": -0.8, "arousal": -0.4},
        "disgust": {"valence": -0.6, "arousal": 0.2},
        "anger": {"valence": -0.7, "arousal": 0.8},
        "anticipation": {"valence": 0.4, "arousal": 0.5},
        "curiosity": {"valence": 0.7, "arousal": 0.6},
        "confusion": {"valence": -0.3, "arousal": 0.4},
        "awe": {"valence": 0.9, "arousal": 0.7},
        "contentment": {"valence": 0.8, "arousal": -0.2}
    }
    
    def __init__(self, emotional_values: Dict[str, float] = None):
        """
        Initialize an emotional state.
        
        Args:
            emotional_values: Dictionary mapping emotion names to intensities (0.0-1.0)
        """
        self.emotions = emotional_values or {}
        self.creation_time = datetime.now()
        
    def update(self, emotion: str, intensity: float) -> None:
        """
        Update an emotion's intensity.
        
        Args:
            emotion: Emotion name
            intensity: New intensity (0.0-1.0)
        """
        self.emotions[emotion] = max(0.0, min(1.0, intensity))
        
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """
        Get the dominant emotion in this state.
        
        Returns:
            Tuple of (emotion name, intensity)
        """
        if not self.emotions:
            return ("neutral", 0.0)
            
        return max(self.emotions.items(), key=lambda x: x[1])
        
    def get_valence_arousal(self) -> Tuple[float, float]:
        """
        Calculate the overall valence and arousal of this emotional state.
        
        Returns:
            Tuple of (valence, arousal) each in range -1.0 to 1.0
        """
        valence_sum = 0.0
        arousal_sum = 0.0
        weight_sum = 0.0
        
        for emotion, intensity in self.emotions.items():
            if emotion in self.CORE_EMOTIONS:
                valence_sum += self.CORE_EMOTIONS[emotion]["valence"] * intensity
                arousal_sum += self.CORE_EMOTIONS[emotion]["arousal"] * intensity
                weight_sum += intensity
                
        if weight_sum == 0:
            return (0.0, 0.0)  # Neutral state
            
        return (valence_sum / weight_sum, arousal_sum / weight_sum)
        
    def blend_with(self, other: 'EmotionalState', weight: float = 0.5) -> 'EmotionalState':
        """
        Blend this emotional state with another.
        
        Args:
            other: Other emotional state
            weight: Weight of this state in blend (0.0-1.0)
            
        Returns:
            Blended emotional state
        """
        blended = EmotionalState()
        
        # Blend emotions from both states
        for emotion in set(list(self.emotions.keys()) + list(other.emotions.keys())):
            self_intensity = self.emotions.get(emotion, 0.0)
            other_intensity = other.emotions.get(emotion, 0.0)
            
            blended_intensity = self_intensity * weight + other_intensity * (1 - weight)
            if blended_intensity > 0.05:  # Filter out very weak emotions
                blended.emotions[emotion] = blended_intensity
                
        return blended
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert emotional state to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "emotions": self.emotions,
            "creation_time": self.creation_time.isoformat()
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EmotionalState':
        """
        Create emotional state from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created emotional state
        """
        state = EmotionalState(data["emotions"])
        
        try:
            state.creation_time = datetime.fromisoformat(data["creation_time"])
        except:
            pass
            
        return state

class SullySearchMemory:
    """
    A sophisticated memory system for Sully that stores experiences, 
    queries, and knowledge with associative retrieval capabilities, emotional
    tagging, and episodic contexts.
    """
    
    def __init__(self, memory_file: Optional[str] = None):
        """
        Initialize the memory system with optional persistent storage.
        
        Args:
            memory_file: Optional file path for persisting memory
        """
        # Core memory storage
        self.storage = {}  # memory_id -> MemoryTrace
        self.episodes = {}  # episodic_id -> EpisodicContext
        
        # Memory indexing
        self.semantic_index = {}  # concept -> set(memory_ids)
        self.episodic_index = {}  # situation_key -> set(episodic_ids)
        self.emotional_index = {}  # emotion -> {memory_id: intensity}
        self.temporal_index = {}  # YYYY-MM-DD -> set(memory_ids)
        self.self_reference_index = set()  # set(memory_ids) for self-referential memories
        
        # Memory dynamics
        self.current_emotional_state = EmotionalState()
        self.emotional_history = []  # List of emotional states over time
        self.last_decay_update = datetime.now()
        
        # Experience integration
        self.current_episodic_context = None
        self.narrative_buffer = []  # For constructing coherent episodic narratives
        self.memory_consolidation_queue = []  # For scheduled memory consolidation
        
        # Constants
        self.MAX_HISTORY_LENGTH = 100
        self.MAX_NARRATIVE_BUFFER = 20
        self.CONSOLIDATION_THRESHOLD = 5
        
        # Persistence
        self.memory_file = memory_file
        
        # Load from file if provided and exists
        if memory_file and os.path.exists(memory_file):
            try:
                self._load_from_file()
            except Exception as e:
                print(f"Could not load memory file: {e}")

    def store_query(self, query: str, result: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores a symbolic query and its result in memory, with a timestamp and emotional context.
        
        Args:
            query: The input query or stimulus
            result: The response or output
            metadata: Additional contextual information
            
        Returns:
            ID of the stored memory
        """
        timestamp = datetime.now()
        
        # Evaluate emotional valence of query and result
        emotional_valence = self._evaluate_emotional_content(query, result)
        
        # Prepare context from current state
        context = {
            "timestamp": timestamp.isoformat(),
            "query_type": "direct",
            "emotional_state": self.current_emotional_state.to_dict() if self.current_emotional_state else None
        }
        
        # Add any additional metadata
        if metadata:
            context.update(metadata)
            
        # Determine importance based on emotional intensity and metadata
        importance = self._calculate_importance(emotional_valence, context)
        
        # Create memory trace
        memory_trace = MemoryTrace(
            content={
                "query": query,
                "result": result
            },
            memory_type="semantic",
            emotional_valence=emotional_valence,
            source="query",
            importance=importance,
            context=context
        )
        
        # Store in main memory
        memory_id = memory_trace.memory_id
        self.storage[memory_id] = memory_trace
        
        # Index by time period for temporal associations
        self._index_by_time(memory_id, timestamp)
        
        # Index by emotion for emotional associations
        self._index_by_emotion(memory_id, emotional_valence)
        
        # Extract and index key concepts
        self._index_concepts(memory_id, query, result)
        
        # Add to current episodic context if one exists
        if self.current_episodic_context:
            # Add memory to current episode
            self.current_episodic_context.entities.append(f"memory:{memory_id}")
            
            # Record episodic context in memory
            memory_trace.context["episodic_context"] = self.current_episodic_context.episodic_id
            
        # Update the narrative buffer
        self._update_narrative_buffer({
            "type": "query",
            "memory_id": memory_id,
            "timestamp": timestamp,
            "query": query,
            "summary": self._generate_summary(result)
        })
        
        # Save to persistent storage if configured
        if self.memory_file:
            self._save_to_file()
            
        return memory_id

    def store_experience(self, content: str, source: str, 
                        concepts: Optional[List[str]] = None,
                        emotional_tags: Optional[Dict[str, float]] = None,
                        importance: float = 0.5,
                        episodic: bool = False) -> str:
        """
        Stores a general experience or knowledge in memory with emotional and episodic context.
        
        Args:
            content: The main content to remember
            source: Where the experience/knowledge came from
            concepts: Key concepts related to this memory
            emotional_tags: Emotional associations with this memory
            importance: How important this memory is (0.0-1.0)
            episodic: Whether this should be part of an episodic memory
            
        Returns:
            ID of the stored memory
        """
        timestamp = datetime.now()
        
        # Evaluate emotional content if not provided
        if not emotional_tags:
            emotional_tags = self._evaluate_emotional_content(content)
            
        # Create context with current conditions
        context = {
            "timestamp": timestamp.isoformat(),
            "source_details": source,
            "emotional_state": self.current_emotional_state.to_dict() if self.current_emotional_state else None
        }
        
        # Determine memory type
        memory_type = "episodic" if episodic else "semantic"
        
        # Check for self-reference
        is_self_reference = self._check_self_reference(content)
        if is_self_reference:
            memory_type = "self_referential"
            
        # Create memory trace
        memory_trace = MemoryTrace(
            content=content,
            memory_type=memory_type,
            emotional_valence=emotional_tags,
            source=source,
            importance=importance,
            context=context
        )
        
        # Store in main memory
        memory_id = memory_trace.memory_id
        self.storage[memory_id] = memory_trace
        
        # Index by time period
        self._index_by_time(memory_id, timestamp)
        
        # Index by emotion
        self._index_by_emotion(memory_id, emotional_tags)
        
        # Index concepts
        if concepts:
            for concept in concepts:
                self._add_to_semantic_index(concept, memory_id)
        else:
            # Auto-extract concepts if none provided
            extracted_concepts = self._extract_key_concepts(content)
            for concept in extracted_concepts:
                self._add_to_semantic_index(concept, memory_id)
        
        # Handle episodic memory
        if episodic:
            # Create new episodic context if none exists
            if not self.current_episodic_context:
                self._begin_new_episode(f"Experience: {content[:50]}...")
                
            # Add to current episode
            self.current_episodic_context.entities.append(f"memory:{memory_id}")
            memory_trace.context["episodic_context"] = self.current_episodic_context.episodic_id
            
        # Handle self-referential memory
        if is_self_reference:
            self.self_reference_index.add(memory_id)
            
        # Update the narrative buffer
        self._update_narrative_buffer({
            "type": "experience",
            "memory_id": memory_id,
            "timestamp": timestamp,
            "summary": self._generate_summary(content)
        })
        
        # Check for memory consolidation
        if len(self.narrative_buffer) >= self.CONSOLIDATION_THRESHOLD:
            self._schedule_memory_consolidation()
            
        # Save to persistent storage if configured
        if self.memory_file:
            self._save_to_file()
            
        return memory_id

    def begin_episodic_context(self, situation: str, 
                              spatial_markers: Dict[str, Any] = None,
                              entities: List[str] = None) -> str:
        """
        Begin a new episodic context for subsequent memories.
        
        Args:
            situation: Description of the situation
            spatial_markers: Spatial context information
            entities: Entities initially involved
            
        Returns:
            ID of the created episodic context
        """
        # Close any existing episode
        if self.current_episodic_context:
            self._close_current_episode()
            
        # Create new episode
        episode = self._begin_new_episode(situation, spatial_markers, entities)
        
        return episode.episodic_id

    def close_episodic_context(self, summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Close the current episodic context and consolidate related memories.
        
        Args:
            summary: Optional summary of the episode
            
        Returns:
            Information about the closed episode
        """
        if not self.current_episodic_context:
            return {"status": "no_active_episode"}
            
        episode = self._close_current_episode(summary)
        
        # Extract memories associated with this episode
        memory_ids = []
        for memory_id, memory in self.storage.items():
            episode_id = memory.context.get("episodic_context")
            if episode_id == episode.episodic_id:
                memory_ids.append(memory_id)
                
        return {
            "episode_id": episode.episodic_id,
            "situation": episode.situation,
            "memory_count": len(memory_ids),
            "temporal_context": episode.temporal_markers,
            "spatial_context": episode.spatial_markers
        }

    def update_emotional_state(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """
        Update Sully's current emotional state.
        
        Args:
            emotions: Dictionary mapping emotion names to intensities (0.0-1.0)
            
        Returns:
            Updated emotional state information
        """
        # Create new emotional state
        new_state = EmotionalState(emotions)
        
        # Blend with current state for smooth transitions (80% new, 20% previous)
        if self.current_emotional_state and self.current_emotional_state.emotions:
            new_state = new_state.blend_with(self.current_emotional_state, 0.8)
            
        # Update current state
        self.current_emotional_state = new_state
        
        # Add to history
        self.emotional_history.append(new_state)
        
        # Limit history size
        if len(self.emotional_history) > self.MAX_HISTORY_LENGTH:
            self.emotional_history = self.emotional_history[-self.MAX_HISTORY_LENGTH:]
            
        # Return info about current state
        dominant_emotion, intensity = new_state.get_dominant_emotion()
        valence, arousal = new_state.get_valence_arousal()
        
        return {
            "dominant_emotion": dominant_emotion,
            "intensity": intensity,
            "valence": valence,
            "arousal": arousal,
            "emotions": new_state.emotions
        }

    def search(self, keyword: str, case_sensitive: bool = False, 
              limit: Optional[int] = None, include_associations: bool = True,
              emotional_filter: Optional[str] = None, 
              importance_threshold: float = 0.0,
              episodic_only: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Searches memory for entries containing the given keyword with advanced filtering.

        Args:
            keyword: Term to search within memory
            case_sensitive: Whether to respect case during match
            limit: Max number of results to return
            include_associations: Whether to include associated memories
            emotional_filter: Optional emotion to filter by
            importance_threshold: Minimum importance level for results
            episodic_only: Whether to only return episodic memories
            
        Returns:
            Dictionary of indexed matches from memory
        """
        # Apply memory decay before searching
        self._apply_memory_decay()
        
        direct_matches = {}
        
        # Direct search in storage
        for memory_id, memory in self.storage.items():
            # Skip if doesn't meet importance threshold
            if memory.importance < importance_threshold:
                continue
                
            # Skip if episodic filter doesn't match
            if episodic_only and memory.memory_type != "episodic":
                continue
                
            # Skip if emotional filter doesn't match
            if emotional_filter and emotional_filter not in memory.emotional_valence:
                continue
                
            # Check for match in content
            if self._memory_matches_keyword(memory, keyword, case_sensitive):
                direct_matches[memory_id] = self._prepare_memory_result(memory)
                
                # Stop if we've reached the limit
                if limit and len(direct_matches) >= limit:
                    break
                    
        # If we don't need to include associations or have reached the limit, return
        if not include_associations or (limit and len(direct_matches) >= limit):
            return direct_matches
            
        # Search for semantically associated memories
        semantically_associated = self._find_semantic_associations(keyword, direct_matches.keys())
        
        # Add semantic associations to results
        associated_count = 0
        for memory_id in semantically_associated:
            if memory_id not in direct_matches:
                # Check importance threshold
                memory = self.storage[memory_id]
                if memory.importance >= importance_threshold:
                    direct_matches[memory_id] = self._prepare_memory_result(memory, association_type="semantic")
                    associated_count += 1
                    
                    # Stop if we've reached the limit
                    remaining_slots = limit - len(direct_matches) if limit else float('inf')
                    if remaining_slots <= 0:
                        break
                        
        # Search for emotionally associated memories if emotional_filter provided
        if emotional_filter and remaining_slots > 0:
            emotionally_associated = self._find_emotional_associations(
                emotional_filter, set(direct_matches.keys()) | set(semantically_associated))
                
            # Add emotional associations to results
            for memory_id in emotionally_associated:
                if memory_id not in direct_matches:
                    memory = self.storage[memory_id]
                    if memory.importance >= importance_threshold:
                        direct_matches[memory_id] = self._prepare_memory_result(memory, association_type="emotional")
                        
                        # Stop if we've reached the limit
                        if limit and len(direct_matches) >= limit:
                            break
        
        return direct_matches

    def get_temporal_context(self, timestamp_or_date: Union[str, datetime],
                           window_days: int = 1, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get memories from around a specific time, providing temporal context.
        
        Args:
            timestamp_or_date: Timestamp or date string to get context for
            window_days: Number of days before and after to include
            limit: Maximum number of memories to return
            
        Returns:
            List of memories within the temporal window
        """
        # Apply memory decay before retrieving
        self._apply_memory_decay()
        
        # Parse timestamp if it's a string
        if isinstance(timestamp_or_date, str):
            try:
                target_date = datetime.fromisoformat(timestamp_or_date).date()
            except ValueError:
                # Try just the date portion if full ISO format fails
                try:
                    target_date = datetime.strptime(timestamp_or_date, "%Y-%m-%d").date()
                except ValueError:
                    return []  # Return empty list if unparseable
        else:
            target_date = timestamp_or_date.date()
        
        # Calculate window
        start_date = target_date - timedelta(days=window_days)
        end_date = target_date + timedelta(days=window_days)
        
        # Get all memory indices within window
        window_memories = []
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime("%Y-%m-%d")
            if date_key in self.temporal_index:
                window_memories.extend(self.temporal_index[date_key])
            current_date += timedelta(days=1)
        
        # Sort memories by timestamp
        dated_memories = []
        for memory_id in window_memories:
            memory = self.storage.get(memory_id)
            if memory:
                # Extract timestamp from context
                if isinstance(memory.context, dict) and "timestamp" in memory.context:
                    try:
                        timestamp = datetime.fromisoformat(memory.context["timestamp"])
                        dated_memories.append((timestamp, memory_id))
                    except:
                        # Use creation time if timestamp parsing fails
                        dated_memories.append((memory.creation_time, memory_id))
                else:
                    # Use creation time if no timestamp in context
                    dated_memories.append((memory.creation_time, memory_id))
                    
        # Sort by timestamp
        sorted_memories = sorted(dated_memories)
        
        # Return memories, with optional limit
        if limit:
            sorted_memories = sorted_memories[:limit]
        
        # Prepare results
        result_memories = []
        for _, memory_id in sorted_memories:
            memory = self.storage.get(memory_id)
            if memory:
                # Access the memory (updates access metadata)
                memory.access()
                result_memories.append(self._prepare_memory_result(memory))
                
        return result_memories

    def get_episodic_memory(self, episodic_id: str) -> Dict[str, Any]:
        """
        Get a complete episodic memory with all associated memory traces.
        
        Args:
            episodic_id: ID of the episodic context
            
        Returns:
            Complete episodic memory data
        """
        # Check if episode exists
        if episodic_id not in self.episodes:
            return {"error": "Episode not found"}
            
        # Get the episode
        episode = self.episodes[episodic_id]
        
        # Find all memory traces associated with this episode
        memory_traces = []
        for memory_id, memory in self.storage.items():
            if isinstance(memory.context, dict) and memory.context.get("episodic_context") == episodic_id:
                # Access the memory
                memory.access()
                memory_traces.append(self._prepare_memory_result(memory))
                
        # Sort memories by timestamp
        memory_traces.sort(key=lambda x: x.get("timestamp", ""))
        
        # Prepare episode data
        episode_data = {
            "episodic_id": episodic_id,
            "situation": episode.situation,
            "temporal_context": episode.temporal_markers,
            "spatial_context": episode.spatial_markers,
            "entities": episode.entities,
            "preceding_context": episode.preceding_context,
            "following_context": episode.following_context,
            "memories": memory_traces
        }
        
        return episode_data

    def get_emotional_memories(self, emotion: str, strength_threshold: float = 0.5,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get memories associated with a particular emotion above a threshold.
        
        Args:
            emotion: The emotion to find
            strength_threshold: Minimum emotional intensity
            limit: Maximum number of memories to return
            
        Returns:
            List of memories with the specified emotion
        """
        # Apply memory decay before retrieving
        self._apply_memory_decay()
        
        # Check if we have an index for this emotion
        if emotion not in self.emotional_index:
            return []
            
        # Get memories with this emotion above the threshold
        matching_memories = []
        for memory_id, strength in self.emotional_index[emotion].items():
            if strength >= strength_threshold and memory_id in self.storage:
                memory = self.storage[memory_id]
                # Access the memory
                memory.access()
                matching_memories.append((memory_id, strength, memory.creation_time))
                
        # Sort by emotional strength (descending) and recency (descending)
        matching_memories.sort(key=lambda x: (-x[1], -x[2].timestamp()))
        
        # Apply limit
        if limit:
            matching_memories = matching_memories[:limit]
            
        # Prepare results
        result_memories = []
        for memory_id, _, _ in matching_memories:
            memory = self.storage[memory_id]
            result_memories.append(self._prepare_memory_result(memory))
            
        return result_memories

    def get_self_referential_memories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get memories that reference Sully's own cognitive processes.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of self-referential memories
        """
        # Apply memory decay before retrieving
        self._apply_memory_decay()
        
        # Get self-referential memories
        self_memories = []
        for memory_id in self.self_reference_index:
            if memory_id in self.storage:
                memory = self.storage[memory_id]
                # Access the memory
                memory.access()
                self_memories.append((memory_id, memory.creation_time))
                
        # Sort by recency (most recent first)
        self_memories.sort(key=lambda x: -x[1].timestamp())
        
        # Apply limit
        if limit:
            self_memories = self_memories[:limit]
            
        # Prepare results
        result_memories = []
        for memory_id, _ in self_memories:
            memory = self.storage[memory_id]
            result_memories.append(self._prepare_memory_result(memory))
            
        return result_memories

    def reinforce_memory(self, memory_id: str, emotional_update: Optional[Dict[str, float]] = None,
                       importance_adjustment: float = 0.0) -> Dict[str, Any]:
        """
        Reinforce a memory, strengthening it and potentially updating its properties.
        
        Args:
            memory_id: ID of the memory to reinforce
            emotional_update: Optional updated emotional valences
            importance_adjustment: Adjustment to importance (-0.2 to 0.2)
            
        Returns:
            Updated memory information
        """
        # Check if memory exists
        if memory_id not in self.storage:
            return {"error": "Memory not found"}
            
        # Get the memory
        memory = self.storage[memory_id]
        
        # Reinforce the memory
        memory.reinforce(emotional_update)
        
        # Adjust importance if specified
        if importance_adjustment != 0.0:
            # Limit adjustment range
            adj = max(-0.2, min(0.2, importance_adjustment))
            memory.importance = max(0.0, min(1.0, memory.importance + adj))
            
        # Update emotional indices if emotional update provided
        if emotional_update:
            # Remove from old emotional indices
            for emotion in memory.emotional_valence:
                if emotion in self.emotional_index and memory_id in self.emotional_index[emotion]:
                    del self.emotional_index[emotion][memory_id]
                    
            # Add to new emotional indices
            self._index_by_emotion(memory_id, emotional_update)
            
        # Save to persistent storage if configured
        if self.memory_file:
            self._save_to_file()
            
        return self._prepare_memory_result(memory)

    def find_connections(self, concept1: str, concept2: str, max_path_length: int = 2) -> List[Dict[str, Any]]:
        """
        Find memory paths that connect two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            max_path_length: Maximum path length to consider
            
        Returns:
            List of memory paths connecting the concepts
        """
        # Normalize concepts
        concept1_lower = concept1.lower()
        concept2_lower = concept2.lower()
        
        # Get memories for each concept
        memories1 = self.semantic_index.get(concept1_lower, set())
        memories2 = self.semantic_index.get(concept2_lower, set())
        
        # Direct connections (memories that contain both concepts)
        direct_connections = memories1.intersection(memories2)
        
        paths = []
        
        # Add direct connections
        for memory_id in direct_connections:
            if memory_id in self.storage:
                memory = self.storage[memory_id]
                paths.append({
                    "path_length": 1,
                    "path_type": "direct",
                    "memories": [self._prepare_memory_result(memory)]
                })
                
        # If max_path_length > 1, find indirect connections
        if max_path_length > 1 and not direct_connections:
            # Build graph of memory associations
            memory_graph = {}
            
            # Add all memories of concept1 as starting points
            for memory_id in memories1:
                memory_graph[memory_id] = set()
                
                if memory_id in self.storage:
                    memory = self.storage[memory_id]
                    
                    # Add associations
                    memory_graph[memory_id].update(memory.associations)
                    
                    # Add memories with shared emotional valence
                    for emotion, strength in memory.emotional_valence.items():
                        if emotion in self.emotional_index and strength > 0.5:
                            for related_id in self.emotional_index[emotion]:
                                if related_id != memory_id and self.emotional_index[emotion][related_id] > 0.5:
                                    memory_graph[memory_id].add(related_id)
            
            # Add intermediate nodes to graph
            for memory_id in list(memory_graph.keys()):
                for related_id in memory_graph[memory_id]:
                    if related_id not in memory_graph and related_id in self.storage:
                        memory_graph[related_id] = set()
                        
                        # Add associations for intermediate node
                        memory = self.storage[related_id]
                        memory_graph[related_id].update(memory.associations)
                        
                        # Add emotional associations
                        for emotion, strength in memory.emotional_valence.items():
                            if emotion in self.emotional_index and strength > 0.5:
                                for emotion_related_id in self.emotional_index[emotion]:
                                    if emotion_related_id != related_id and self.emotional_index[emotion][emotion_related_id] > 0.5:
                                        memory_graph[related_id].add(emotion_related_id)
            
            # Find paths using breadth-first search
            for start_memory_id in memories1:
                # BFS data structures
                queue = [(start_memory_id, [start_memory_id])]  # (current_node, path_so_far)
                visited = {start_memory_id}
                
                while queue:
                    current, path = queue.pop(0)
                    
                    # Check if current node connects to concept2
                    if current in memories2:
                        # Path found
                        memory_path = []
                        for memory_id in path:
                            if memory_id in self.storage:
                                memory = self.storage[memory_id]
                                memory_path.append(self._prepare_memory_result(memory))
                        
                        paths.append({
                            "path_length": len(path),
                            "path_type": "indirect",
                            "memories": memory_path
                        })
                        break  # Found a path from this starting point
                        
                    # If path is already at max length, don't explore further
                    if len(path) >= max_path_length:
                        continue
                        
                    # Explore neighbors
                    for neighbor in memory_graph.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
        
        # Sort paths by length
        paths.sort(key=lambda x: x["path_length"])
        
        return paths[:5]  # Limit to top 5 paths

    def get_emotional_summary(self) -> Dict[str, Any]:
        """
        Get a summary of emotional patterns in memory.
        
        Returns:
            Summary of emotional patterns
        """
        # Apply memory decay before analyzing
        self._apply_memory_decay()
        
        # Get current emotional state
        current_emotion, current_intensity = self.current_emotional_state.get_dominant_emotion() if self.current_emotional_state else ("neutral", 0.0)
        
        # Analyze emotional history
        history_emotions = {}
        for state in self.emotional_history:
            dominant, intensity = state.get_dominant_emotion()
            if dominant not in history_emotions:
                history_emotions[dominant] = []
            history_emotions[dominant].append(intensity)
            
        avg_emotions = {}
        for emotion, intensities in history_emotions.items():
            avg_emotions[emotion] = sum(intensities) / len(intensities)
            
        # Find most common emotions
        if avg_emotions:
            most_common = sorted(avg_emotions.items(), key=lambda x: -x[1])[:3]
        else:
            most_common = []
            
        # Analyze emotional patterns in memories
        memory_emotions = {}
        for emotion, memories in self.emotional_index.items():
            total_strength = sum(memories.values())
            avg_strength = total_strength / len(memories) if memories else 0
            memory_emotions[emotion] = {
                "count": len(memories),
                "avg_strength": avg_strength,
                "total_strength": total_strength
            }
            
        # Find emotions with strongest memories
        strongest_emotions = sorted(memory_emotions.items(), key=lambda x: -x[1]["avg_strength"])[:3]
        
        return {
            "current_emotion": {
                "dominant": current_emotion,
                "intensity": current_intensity
            },
            "common_historical_emotions": most_common,
            "memory_emotion_stats": memory_emotions,
            "strongest_memory_emotions": strongest_emotions
        }

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Memory statistics
        """
        # Count memories by type
        memory_types = {}
        for memory in self.storage.values():
            if memory.memory_type not in memory_types:
                memory_types[memory.memory_type] = 0
            memory_types[memory.memory_type] += 1
            
        # Calculate average memory properties
        total_importance = 0
        total_clarity = 0
        total_stability = 0
        
        for memory in self.storage.values():
            total_importance += memory.importance
            total_clarity += memory.clarity
            total_stability += memory.stability
            
        avg_importance = total_importance / len(self.storage) if self.storage else 0
        avg_clarity = total_clarity / len(self.storage) if self.storage else 0
        avg_stability = total_stability / len(self.storage) if self.storage else 0
        
        # Get episode statistics
        episode_count = len(self.episodes)
        avg_memories_per_episode = 0
        
        if episode_count > 0:
            episode_memory_counts = []
            for episode_id in self.episodes:
                count = sum(1 for memory in self.storage.values() 
                          if isinstance(memory.context, dict) and memory.context.get("episodic_context") == episode_id)
                episode_memory_counts.append(count)
                
            avg_memories_per_episode = sum(episode_memory_counts) / len(episode_memory_counts)
            
        return {
            "total_memories": len(self.storage),
            "memory_types": memory_types,
            "episode_count": episode_count,
            "avg_memories_per_episode": avg_memories_per_episode,
            "avg_importance": avg_importance,
            "avg_clarity": avg_clarity,
            "avg_stability": avg_stability,
            "semantic_concepts": len(self.semantic_index),
            "emotional_categories": len(self.emotional_index)
        }

    def clear_memory(self) -> str:
        """
        Clears all stored memories and indices.
        
        Returns:
            Confirmation message
        """
        self.storage = {}
        self.episodes = {}
        self.semantic_index = {}
        self.episodic_index = {}
        self.emotional_index = {}
        self.temporal_index = {}
        self.self_reference_index = set()
        self.current_emotional_state = EmotionalState()
        self.emotional_history = []
        self.current_episodic_context = None
        self.narrative_buffer = []
        self.memory_consolidation_queue = []
        self.last_decay_update = datetime.now()
        
        # Clear persistent storage if configured
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception as e:
                print(f"Could not remove memory file: {e}")
        
        return "[Memory system cleared]"

    def export_memory(self) -> List[Dict[str, Any]]:
        """
        Returns the entire memory as a list of entries (for JSON export).
        
        Returns:
            List of all memory entries
        """
        return [memory.to_dict() for memory in self.storage.values()]

    def export_full_system(self) -> Dict[str, Any]:
        """
        Exports the complete memory system including all components.
        
        Returns:
            Dictionary containing all memory system components
        """
        # Convert storage dictionary (memory_id -> MemoryTrace)
        storage_dict = {memory_id: memory.to_dict() for memory_id, memory in self.storage.items()}
        
        # Convert episodes dictionary (episodic_id -> EpisodicContext)
        episodes_dict = {episode_id: episode.to_dict() for episode_id, episode in self.episodes.items()}
        
        # Convert emotional history
        emotional_history_list = [state.to_dict() for state in self.emotional_history]
        
        # Current emotional state
        current_emotional_dict = self.current_emotional_state.to_dict() if self.current_emotional_state else None
        
        # Current episodic context
        current_episodic_dict = self.current_episodic_context.to_dict() if self.current_episodic_context else None
        
        return {
            "storage": storage_dict,
            "episodes": episodes_dict,
            "semantic_index": {concept: list(memories) for concept, memories in self.semantic_index.items()},
            "episodic_index": {situation: list(episodes) for situation, episodes in self.episodic_index.items()},
            "emotional_index": self.emotional_index,
            "temporal_index": {date: list(memories) for date, memories in self.temporal_index.items()},
            "self_reference_index": list(self.self_reference_index),
            "current_emotional_state": current_emotional_dict,
            "emotional_history": emotional_history_list,
            "current_episodic_context": current_episodic_dict,
            "narrative_buffer": self.narrative_buffer,
            "last_decay_update": self.last_decay_update.isoformat(),
            "export_timestamp": datetime.now().isoformat()
        }

    def _extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract potential key concepts from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted concepts
        """
        import re
        
        # Tokenize text
        tokens = re.findall(r'\b[A-Za-z][A-Za-z\-]{3,}\b', text)
        
        # Filter out common words and short words
        common_words = {"the", "and", "but", "for", "nor", "or", "so", "yet", "a", "an", "to", "in", "on", "with", "by", "at", "from"}
        tokens = [token.lower() for token in tokens if token.lower() not in common_words and len(token) > 3]
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Extract multi-word concepts (phrases)
        phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        for phrase in phrases:
            token_counts[phrase.lower()] = 1
            
        # Get most significant concepts
        significant = [token for token, count in token_counts.most_common(10) if count >= 1]
        
        return significant

    def _evaluate_emotional_content(self, *texts) -> Dict[str, float]:
        """
        Evaluate the emotional content of one or more text items.
        
        Args:
            *texts: One or more text items to evaluate
            
        Returns:
            Dictionary mapping emotions to intensities
        """
        # Simplified emotion detection - would use more sophisticated NLP in a real implementation
        emotion_keywords = {
            "joy": ["happy", "joy", "delighted", "pleased", "glad", "excited", "wonderful"],
            "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "troubled", "sorrow"],
            "anger": ["angry", "upset", "furious", "outraged", "annoyed", "irritated", "frustrated"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "startled", "unexpected"],
            "disgust": ["disgusted", "repulsed", "revolted", "averse", "distaste", "unpleasant"],
            "trust": ["trust", "believe", "confidence", "faith", "reliable", "dependable", "sure"],
            "anticipation": ["anticipate", "expect", "look forward", "await", "hopeful", "excited"]
        }
        
        # Combine texts
        full_text = " ".join(str(text) for text in texts).lower()
        
        # Count emotion keywords
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in full_text)
            if count > 0:
                # Normalize score based on text length
                text_length = len(full_text.split())
                normalized_score = min(1.0, count / (text_length * 0.1))  # Adjust scaling factor as needed
                emotion_scores[emotion] = normalized_score
        
        return emotion_scores

    def _calculate_importance(self, emotional_valence: Dict[str, float], context: Dict[str, Any]) -> float:
        """
        Calculate the importance of a memory based on emotional content and context.
        
        Args:
            emotional_valence: Emotional associations with the memory
            context: Memory context
            
        Returns:
            Importance score (0.0-1.0)
        """
        # Base importance
        importance = 0.5
        
        # Adjust based on emotional intensity
        emotional_intensity = sum(emotional_valence.values()) / max(1, len(emotional_valence))
        importance += emotional_intensity * 0.3  # Emotions can contribute up to 0.3
        
        # Adjust based on context
        # Higher importance for memories related to current goals or episodic
        if context.get("goal_related", False):
            importance += 0.15
            
        if context.get("episodic_context"):
            importance += 0.1
            
        # Cap to valid range
        return max(0.0, min(1.0, importance))

    def _check_self_reference(self, text: str) -> bool:
        """
        Check if a text contains self-reference to Sully's cognitive processes.
        
        Args:
            text: Text to check
            
        Returns:
            Whether the text contains self-reference
        """
        self_reference_terms = [
            "sully", "my thinking", "my thought", "my reasoning", "my memory",
            "my knowledge", "my understanding", "my cognition", "my learning",
            "my mind", "my process", "my model", "my attention"
        ]
        
        text_lower = text.lower()
        
        return any(term in text_lower for term in self_reference_terms)

    def _index_by_time(self, memory_id: str, timestamp: datetime) -> None:
        """
        Index a memory by its timestamp.
        
        Args:
            memory_id: Memory ID
            timestamp: Memory timestamp
        """
        time_key = timestamp.strftime("%Y-%m-%d")
        if time_key not in self.temporal_index:
            self.temporal_index[time_key] = set()
        self.temporal_index[time_key].add(memory_id)

    def _index_by_emotion(self, memory_id: str, emotions: Dict[str, float]) -> None:
        """
        Index a memory by its emotional content.
        
        Args:
            memory_id: Memory ID
            emotions: Emotional valences
        """
        for emotion, strength in emotions.items():
            if emotion not in self.emotional_index:
                self.emotional_index[emotion] = {}
            self.emotional_index[emotion][memory_id] = strength

    def _add_to_semantic_index(self, concept: str, memory_id: str) -> None:
        """
        Add a memory to the semantic index for a concept.
        
        Args:
            concept: Concept to index by
            memory_id: Memory ID
        """
        concept_lower = concept.lower()
        if concept_lower not in self.semantic_index:
            self.semantic_index[concept_lower] = set()
        self.semantic_index[concept_lower].add(memory_id)

    def _add_to_episodic_index(self, situation: str, episodic_id: str) -> None:
        """
        Add an episode to the episodic index for a situation.
        
        Args:
            situation: Situation to index by
            episodic_id: Episode ID
        """
        # Create situation key by extracting keywords
        situation_keywords = self._extract_key_concepts(situation)
        for keyword in situation_keywords:
            if keyword not in self.episodic_index:
                self.episodic_index[keyword] = set()
            self.episodic_index[keyword].add(episodic_id)

    def _index_concepts(self, memory_id: str, query: str, result: Any) -> None:
        """
        Extract and index concepts from a query-result pair.
        
        Args:
            memory_id: Memory ID
            query: Query text
            result: Query result
        """
        # Extract concepts from query
        query_concepts = self._extract_key_concepts(query)
        
        # Extract concepts from result if it's a string
        result_concepts = []
        if isinstance(result, str):
            result_concepts = self._extract_key_concepts(result)
        elif isinstance(result, dict) and "response" in result:
            if isinstance(result["response"], str):
                result_concepts = self._extract_key_concepts(result["response"])
        
        # Combine unique concepts
        all_concepts = set(query_concepts + result_concepts)
        
        # Index all concepts
        for concept in all_concepts:
            self._add_to_semantic_index(concept, memory_id)

    def _memory_matches_keyword(self, memory: MemoryTrace, keyword: str, case_sensitive: bool) -> bool:
        """
        Check if a memory matches a keyword.
        
        Args:
            memory: Memory to check
            keyword: Keyword to search for
            case_sensitive: Whether to respect case during match
            
        Returns:
            Whether the memory matches the keyword
        """
        # Prepare content for search
        if isinstance(memory.content, str):
            content = memory.content
        elif isinstance(memory.content, dict):
            # For query-result memories, search both parts
            content_parts = []
            if "query" in memory.content:
                content_parts.append(str(memory.content["query"]))
            if "result" in memory.content:
                content_parts.append(str(memory.content["result"]))
            content = " ".join(content_parts)
        else:
            # Convert to string for other types
            content = str(memory.content)

        # Handle case sensitivity
        if not case_sensitive:
            content = content.lower()
            keyword = keyword.lower()
            
        # Check for match
        return keyword in content

    def _find_semantic_associations(self, keyword: str, exclude_ids: set) -> set:
        """
        Find semantically associated memories to a keyword.
        
        Args:
            keyword: Keyword to find associations for
            exclude_ids: Memory IDs to exclude
            
        Returns:
            Set of semantically associated memory IDs
        """
        # Extract concepts from the keyword
        concepts = self._extract_key_concepts(keyword)
        
        # Find memories associated with these concepts
        related_memories = set()
        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower in self.semantic_index:
                related_memories.update(self.semantic_index[concept_lower])
        
        # Remove excluded IDs
        return related_memories - set(exclude_ids)

    def _find_emotional_associations(self, emotion: str, exclude_ids: set) -> set:
        """
        Find emotionally associated memories.
        
        Args:
            emotion: Emotion to find associations for
            exclude_ids: Memory IDs to exclude
            
        Returns:
            Set of emotionally associated memory IDs
        """
        if emotion not in self.emotional_index:
            return set()
            
        # Get memories with this emotion (with strength > 0.3)
        related_memories = {memory_id for memory_id, strength 
                          in self.emotional_index[emotion].items() 
                          if strength > 0.3}
        
        # Remove excluded IDs
        return related_memories - set(exclude_ids)

    def _prepare_memory_result(self, memory: MemoryTrace, association_type: str = None) -> Dict[str, Any]:
        """
        Prepare a memory for inclusion in search results.
        
        Args:
            memory: Memory to prepare
            association_type: Optional type of association
            
        Returns:
            Dictionary with memory data
        """
        # Access the memory (updates access metadata)
        memory.access()
        
        # Base result data
        result = {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "memory_type": memory.memory_type,
            "emotional_valence": memory.emotional_valence,
            "importance": memory.importance,
            "clarity": memory.clarity,
            "creation_time": memory.creation_time.isoformat(),
            "last_accessed": memory.last_accessed.isoformat(),
            "access_count": memory.access_count
        }
        
        # Add context information
        if memory.context:
            if isinstance(memory.context, dict) and "timestamp" in memory.context:
                result["timestamp"] = memory.context["timestamp"]
            
            # Add episodic context if available
            if isinstance(memory.context, dict) and "episodic_context" in memory.context:
                result["episodic_context"] = memory.context["episodic_context"]
        
        # Add association type if provided
        if association_type:
            result["association_type"] = association_type
            
        return result

    def _generate_summary(self, content: Any) -> str:
        """
        Generate a concise summary of memory content.
        
        Args:
            content: Memory content to summarize
            
        Returns:
            Summary string
        """
        # Convert to string for processing
        if not isinstance(content, str):
            content_str = str(content)
        else:
            content_str = content
        
        # Simple truncation for now - would use more sophisticated summarization in a real implementation
        if len(content_str) > 100:
            return content_str[:97] + "..."
        return content_str

    def _update_narrative_buffer(self, entry: Dict[str, Any]) -> None:
        """
        Update the narrative buffer with a new entry.
        
        Args:
            entry: New buffer entry
        """
        # Add new entry to buffer
        self.narrative_buffer.append(entry)
        
        # Limit buffer size
        if len(self.narrative_buffer) > self.MAX_NARRATIVE_BUFFER:
            self.narrative_buffer = self.narrative_buffer[-self.MAX_NARRATIVE_BUFFER:]

    def _schedule_memory_consolidation(self) -> None:
        """
        Schedule memory consolidation based on narrative buffer contents.
        """
        # Extract buffers that need consolidation
        if len(self.narrative_buffer) >= self.CONSOLIDATION_THRESHOLD:
            # Add current buffer to consolidation queue
            self.memory_consolidation_queue.append(list(self.narrative_buffer))
            
            # Process consolidation immediately for simplicity
            # In a real implementation, this would be scheduled asynchronously
            self._consolidate_memories()

    def _consolidate_memories(self) -> None:
        """
        Consolidate memories in the consolidation queue.
        """
        if not self.memory_consolidation_queue:
            return
            
        # Process each buffer in the queue
        for buffer in self.memory_consolidation_queue:
            # Skip empty buffers
            if not buffer:
                continue
                
            # Prepare content and context for consolidated memory
            buffer_content = []
            related_memories = []
            timestamps = []
            emotional_valences = {}
            
            for entry in buffer:
                # Add entry summary to content
                buffer_content.append(entry.get("summary", ""))
                
                # Track related memory
                if "memory_id" in entry:
                    related_memories.append(entry["memory_id"])
                    
                # Track timestamp
                if "timestamp" in entry:
                    if isinstance(entry["timestamp"], datetime):
                        timestamps.append(entry["timestamp"])
                    elif isinstance(entry["timestamp"], str):
                        try:
                            timestamps.append(datetime.fromisoformat(entry["timestamp"]))
                        except ValueError:
                            pass
                
                # Extract emotions from related memories
                if "memory_id" in entry and entry["memory_id"] in self.storage:
                    memory = self.storage[entry["memory_id"]]
                    for emotion, strength in memory.emotional_valence.items():
                        if emotion in emotional_valences:
                            emotional_valences[emotion] = max(emotional_valences[emotion], strength)
                        else:
                            emotional_valences[emotion] = strength
            
            # Create consolidated memory content
            consolidated_content = "\n".join(buffer_content)
            
            # Determine time range
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
                time_context = {
                    "start_time": earliest.isoformat(),
                    "end_time": latest.isoformat()
                }
            else:
                time_context = {
                    "timestamp": datetime.now().isoformat()
                }
            
            # Create context
            context = {
                "consolidated_from": related_memories,
                "consolidation_type": "narrative",
                **time_context
            }
            
            # Add episodic context if current buffer is related to current episode
            if self.current_episodic_context:
                context["episodic_context"] = self.current_episodic_context.episodic_id
            
            # Calculate importance based on component memories
            avg_importance = 0.0
            count = 0
            for memory_id in related_memories:
                if memory_id in self.storage:
                    avg_importance += self.storage[memory_id].importance
                    count += 1
            
            if count > 0:
                avg_importance /= count
            else:
                avg_importance = 0.5  # Default importance
            
            # Create consolidated memory
            memory_trace = MemoryTrace(
                content=consolidated_content,
                memory_type="consolidated",
                emotional_valence=emotional_valences,
                source="consolidation",
                importance=avg_importance,
                context=context
            )
            
            # Store in main memory
            memory_id = memory_trace.memory_id
            self.storage[memory_id] = memory_trace
            
            # Create associations between consolidated memory and component memories
            for component_id in related_memories:
                if component_id in self.storage:
                    # Add bidirectional associations
                    memory_trace.add_association(component_id)
                    self.storage[component_id].add_association(memory_id)
            
            # Index by time
            self._index_by_time(memory_id, memory_trace.creation_time)
            
            # Index by emotion
            self._index_by_emotion(memory_id, emotional_valences)
            
            # Extract and index key concepts
            concepts = self._extract_key_concepts(consolidated_content)
            for concept in concepts:
                self._add_to_semantic_index(concept, memory_id)
        
        # Clear the consolidation queue
        self.memory_consolidation_queue = []

    def _begin_new_episode(self, situation: str, 
                          spatial_markers: Dict[str, Any] = None,
                          entities: List[str] = None) -> EpisodicContext:
        """
        Begin a new episodic context.
        
        Args:
            situation: Description of the situation
            spatial_markers: Spatial context information
            entities: Entities initially involved
            
        Returns:
            Created episodic context
        """
        # Close current episode if one exists
        if self.current_episodic_context:
            self._close_current_episode()
            
        # Create new episodic context
        episode = EpisodicContext(
            situation=situation,
            spatial_markers=spatial_markers,
            entities=entities or []
        )
        
        # Store in episodes
        self.episodes[episode.episodic_id] = episode
        
        # Set as current
        self.current_episodic_context = episode
        
        # Index episode
        self._add_to_episodic_index(situation, episode.episodic_id)
        
        return episode

    def _close_current_episode(self, summary: str = None) -> EpisodicContext:
        """
        Close the current episodic context.
        
        Args:
            summary: Optional summary of the episode
            
        Returns:
            Closed episodic context
        """
        if not self.current_episodic_context:
            return None
            
        episode = self.current_episodic_context
        
        # Update episode with summary if provided
        if summary:
            episode.following_context = summary
            
        # Check narrative buffer for the next context
        if self.narrative_buffer:
            next_entry = self.narrative_buffer[-1]
            if "summary" in next_entry:
                episode.following_context = next_entry["summary"]
        
        # Create a consolidated memory for this episode
        memory_ids = []
        for memory_id, memory in self.storage.items():
            if (isinstance(memory.context, dict) and 
                memory.context.get("episodic_context") == episode.episodic_id):
                memory_ids.append(memory_id)
        
        if memory_ids:
            # Create episode summary content
            content_parts = [f"Episode: {episode.situation}"]
            
            if episode.preceding_context:
                content_parts.append(f"Before: {episode.preceding_context}")
                
            # Add memory summaries
            for memory_id in memory_ids:
                memory = self.storage[memory_id]
                content_parts.append(self._generate_summary(memory.content))
                
            if episode.following_context:
                content_parts.append(f"After: {episode.following_context}")
                
            # Join content
            content = "\n".join(content_parts)
            
            # Create consolidated memory
            memory_trace = MemoryTrace(
                content=content,
                memory_type="episode_summary",
                emotional_valence={},  # Will be populated below
                source="episode",
                importance=0.7,  # Episodic memories are important
                context={
                    "episodic_context": episode.episodic_id,
                    "timestamp": datetime.now().isoformat(),
                    "memories": memory_ids
                }
            )
            
            # Calculate emotional valence from component memories
            for memory_id in memory_ids:
                memory = self.storage[memory_id]
                for emotion, strength in memory.emotional_valence.items():
                    if emotion in memory_trace.emotional_valence:
                        memory_trace.emotional_valence[emotion] = max(
                            memory_trace.emotional_valence[emotion], strength)
                    else:
                        memory_trace.emotional_valence[emotion] = strength
            
            # Store and index
            memory_id = memory_trace.memory_id
            self.storage[memory_id] = memory_trace
            
            # Create associations with component memories
            for component_id in memory_ids:
                memory_trace.add_association(component_id)
                self.storage[component_id].add_association(memory_id)
            
            # Index by time, emotion, concepts
            self._index_by_time(memory_id, memory_trace.creation_time)
            self._index_by_emotion(memory_id, memory_trace.emotional_valence)
            
            concepts = self._extract_key_concepts(content)
            for concept in concepts:
                self._add_to_semantic_index(concept, memory_id)
        
        # Clear current episodic context
        self.current_episodic_context = None
        
        return episode

    def _apply_memory_decay(self) -> None:
        """
        Apply time-based decay to all memories since last update.
        """
        now = datetime.now()
        days_elapsed = (now - self.last_decay_update).total_seconds() / 86400.0  # Convert to days
        
        if days_elapsed < 0.01:  # Don't apply decay for very small time intervals (< ~15 min)
            return
            
        # Apply decay to all memories
        for memory in self.storage.values():
            memory.decay(days_elapsed)
            
        # Update last decay time
        self.last_decay_update = now

    def _save_to_file(self) -> None:
        """
        Save memory system to persistent storage.
        """
        if not self.memory_file:
            return
            
        try:
            # Export full system
            data = self.export_full_system()
            
            # Save to file
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory to file: {e}")

    def _load_from_file(self) -> None:
        """
        Load memory system from persistent storage.
        """
        if not self.memory_file or not os.path.exists(self.memory_file):
            return
            
        try:
            # Load from file
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                
            # Load storage (memory_id -> MemoryTrace)
            self.storage = {}
            for memory_id, memory_data in data.get("storage", {}).items():
                self.storage[memory_id] = MemoryTrace.from_dict(memory_data)
                
            # Load episodes (episodic_id -> EpisodicContext)
            self.episodes = {}
            for episode_id, episode_data in data.get("episodes", {}).items():
                self.episodes[episode_id] = EpisodicContext.from_dict(episode_data)
                
            # Load indices
            self.semantic_index = {concept: set(memories) for concept, memories in data.get("semantic_index", {}).items()}
            self.episodic_index = {situation: set(episodes) for situation, episodes in data.get("episodic_index", {}).items()}
            self.emotional_index = data.get("emotional_index", {})
            self.temporal_index = {date: set(memories) for date, memories in data.get("temporal_index", {}).items()}
            self.self_reference_index = set(data.get("self_reference_index", []))
            
            # Load emotional state and history
            current_emotional_data = data.get("current_emotional_state")
            if current_emotional_data:
                self.current_emotional_state = EmotionalState.from_dict(current_emotional_data)
            else:
                self.current_emotional_state = EmotionalState()
                
            self.emotional_history = [EmotionalState.from_dict(state_data) 
                                     for state_data in data.get("emotional_history", [])]
                
            # Load current episodic context
            current_episodic_data = data.get("current_episodic_context")
            if current_episodic_data:
                self.current_episodic_context = EpisodicContext.from_dict(current_episodic_data)
            else:
                self.current_episodic_context = None
                
            # Load narrative buffer
            self.narrative_buffer = data.get("narrative_buffer", [])
            
            # Load last decay update time
            try:
                self.last_decay_update = datetime.fromisoformat(data.get("last_decay_update", datetime.now().isoformat()))
            except:
                self.last_decay_update = datetime.now()
                
        except Exception as e:
            print(f"Error loading memory from file: {e}")
            # Initialize empty memory if loading fails
            self.__init__(self.memory_file)
            
    def load_context(self, context: dict):
        """
        Load context dictionary into memory system.
        
        Args:
            context: Context dictionary to load
        """
        self.context = context
    
    def export_context(self) -> dict:
        """
        Export context dictionary from memory system.
        
        Returns:
            Context dictionary
        """
        return getattr(self, "context", {})