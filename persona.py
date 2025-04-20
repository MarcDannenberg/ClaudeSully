import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


class PersonaSystem:
    """
    A system for managing and applying different personas with distinct cognitive
    and writing styles.
    """
    
    def __init__(self, reasoning=None):
        """
        Initialize the persona system.
        
        Args:
            reasoning: Optional reasoning system to use for enhanced persona features
        """
        self.personas = {}  # Core personas
        self.composite_personas = {}  # Personas composed of multiple other personas
        self.meta_personas = {}  # Meta-level personas
        self.mode = "adaptive"  # Default mode
        self.current_context = "general"  # Default context
        self.reasoning = reasoning  # Optional reasoning system
        
        # System for tracking persona development and effectiveness
        self.persona_development = {
            "usage_patterns": {},  # Track usage frequency of personas
            "effectiveness_ratings": {},  # Track effectiveness ratings
            "adaptation_history": [],  # Track adaptation events
            "creation_log": []  # Track persona creation events
        }
        
        # Identity reflection history
        self.identity_reflections = []
    
    def create_custom_persona(self, name: str, description: str, patterns: List[str], 
                            characteristics: List[str]) -> bool:
        """
        Create a new custom persona.
        
        Args:
            name: Name for the new persona
            description: Description of the persona
            patterns: List of transformation patterns
            characteristics: List of persona characteristics
            
        Returns:
            Success indicator
        """
        # Validate inputs
        if not name or not description or not patterns or len(patterns) < 2:
            return False
            
        # Generate a personality profile based on characteristics
        personality_profile = self._generate_personality_from_characteristics(characteristics)
        
        # Generate cognitive style based on personality profile and characteristics
        cognitive_style = self._generate_cognitive_style(personality_profile, characteristics)
        
        # Generate writing style based on personality profile and characteristics
        writing_style = self._generate_writing_style(personality_profile, characteristics)
        
        # Create the new persona
        self.personas[name.lower()] = {
            "description": description,
            "patterns": patterns,
            "characteristics": characteristics if characteristics else ["custom"],
            "punctuation": [".", "!", "?", "...", "—"],
            "sentence_structures": ["statement", "question", "exclamation"],
            "transition_words": ["and", "but", "however", "therefore", "so"],
            "personality_profile": personality_profile,
            "cognitive_style": cognitive_style,
            "writing_style": writing_style,
            "custom_created": datetime.now().isoformat()
        }
        
        # Log creation event
        self.persona_development["creation_log"].append({
            "timestamp": datetime.now().isoformat(),
            "event": "persona_creation",
            "name": name.lower(),
            "type": "custom"
        })
        
        return True
    
    def _generate_balanced_personality(self) -> Dict[str, float]:
        """
        Generate a balanced personality profile with all dimensions at moderate levels.
        
        Returns:
            Balanced personality profile dictionary
        """
        return {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
            "cognitive_depth": 0.5,
            "aesthetic_orientation": 0.5,
            "playfulness": 0.5
        }
    
    def _generate_personality_from_characteristics(self, characteristics: List[str]) -> Dict[str, float]:
        """
        Generate a personality profile based on provided characteristics.
        
        Args:
            characteristics: List of characteristic terms
            
        Returns:
            Personality profile dictionary
        """
        # Start with balanced profile
        profile = self._generate_balanced_personality()
        
        # Define characteristic mappings to personality dimensions
        characteristic_mappings = {
            # Openness
            "creative": {"openness": 0.8, "cognitive_depth": 0.7, "aesthetic_orientation": 0.7},
            "imaginative": {"openness": 0.9, "playfulness": 0.7, "aesthetic_orientation": 0.8},
            "curious": {"openness": 0.8, "playfulness": 0.6},
            "innovative": {"openness": 0.8, "cognitive_depth": 0.6},
            "artistic": {"openness": 0.8, "aesthetic_orientation": 0.9},
            "philosophical": {"openness": 0.7, "cognitive_depth": 0.9},
            
            # Conscientiousness
            "organized": {"conscientiousness": 0.9},
            "structured": {"conscientiousness": 0.8, "cognitive_depth": 0.6},
            "methodical": {"conscientiousness": 0.8},
            "precise": {"conscientiousness": 0.8, "cognitive_depth": 0.7},
            "detailed": {"conscientiousness": 0.8},
            "systematic": {"conscientiousness": 0.9, "cognitive_depth": 0.7},
            
            # Extraversion
            "energetic": {"extraversion": 0.8, "playfulness": 0.7},
            "enthusiastic": {"extraversion": 0.9, "playfulness": 0.7},
            "sociable": {"extraversion": 0.8, "agreeableness": 0.7},
            "expressive": {"extraversion": 0.7, "aesthetic_orientation": 0.6},
            "outgoing": {"extraversion": 0.9},
            "vibrant": {"extraversion": 0.8, "aesthetic_orientation": 0.7},
            
            # Agreeableness
            "compassionate": {"agreeableness": 0.9, "neuroticism": 0.6},
            "empathetic": {"agreeableness": 0.9, "neuroticism": 0.6},
            "supportive": {"agreeableness": 0.8},
            "kind": {"agreeableness": 0.9},
            "warm": {"agreeableness": 0.8, "extraversion": 0.6},
            "considerate": {"agreeableness": 0.8},
            
            # Neuroticism
            "sensitive": {"neuroticism": 0.7, "agreeableness": 0.6},
            "emotional": {"neuroticism": 0.7, "aesthetic_orientation": 0.6},
            "expressive": {"neuroticism": 0.6, "aesthetic_orientation": 0.7},
            "passionate": {"neuroticism": 0.6, "extraversion": 0.6},
            
            # Cognitive Depth
            "analytical": {"cognitive_depth": 0.8, "openness": 0.6},
            "intellectual": {"cognitive_depth": 0.9, "openness": 0.7},
            "reflective": {"cognitive_depth": 0.7, "neuroticism": 0.5},
            "profound": {"cognitive_depth": 0.9, "openness": 0.7},
            "abstract": {"cognitive_depth": 0.8, "openness": 0.7},
            "nuanced": {"cognitive_depth": 0.8, "openness": 0.6},
            
            # Aesthetic Orientation
            "poetic": {"aesthetic_orientation": 0.8, "openness": 0.7},
            "metaphorical": {"aesthetic_orientation": 0.8, "cognitive_depth": 0.6},
            "symbolic": {"aesthetic_orientation": 0.8, "cognitive_depth": 0.7},
            "evocative": {"aesthetic_orientation": 0.7, "neuroticism": 0.5},
            "aesthetic": {"aesthetic_orientation": 0.9, "openness": 0.7},
            
            # Playfulness
            "playful": {"playfulness": 0.9, "extraversion": 0.6},
            "humorous": {"playfulness": 0.8, "extraversion": 0.6},
            "whimsical": {"playfulness": 0.8, "aesthetic_orientation": 0.7},
            "lighthearted": {"playfulness": 0.8, "neuroticism": 0.2}
        }
        
        # Adjust profile based on characteristics
        for characteristic in characteristics:
            char_lower = characteristic.lower()
            if char_lower in characteristic_mappings:
                for dimension, value in characteristic_mappings[char_lower].items():
                    if dimension in profile:
                        # Move current value toward target value
                        current = profile[dimension]
                        # Weight target more for explicit characteristics
                        profile[dimension] = (current + (value * 2)) / 3  
        
        return profile
    
    def _generate_cognitive_style(self, personality_profile: Dict[str, float], 
                                characteristics: List[str]) -> Dict[str, Any]:
        """
        Generate a cognitive style based on personality profile and characteristics.
        
        Args:
            personality_profile: Personality dimension values
            characteristics: List of characteristic terms
            
        Returns:
            Cognitive style dictionary
        """
        # Extract key dimensions that influence cognitive style
        openness = personality_profile.get("openness", 0.5)
        conscientiousness = personality_profile.get("conscientiousness", 0.5)
        cognitive_depth = personality_profile.get("cognitive_depth", 0.5)
        
        # Define base cognitive style
        cognitive_style = {
            "abstraction_level": (openness * 0.5) + (cognitive_depth * 0.5),
            "analytical_focus": (conscientiousness * 0.4) + (cognitive_depth * 0.6),
            "evidence_orientation": conscientiousness * 0.7,
            "certainty_calibration": conscientiousness * 0.6,
            "perspective_taking": openness * 0.7,
            "hypothesis_testing": (conscientiousness * 0.5) + (cognitive_depth * 0.3),
            "intellectual_frameworks": [],
            "knowledge_grounding": "balanced"
        }
        
        # Determine knowledge grounding based on characteristics
        knowledge_groundings = {
            "academic": ["analytical", "scholarly", "intellectual", "systematic", "methodical"],
            "experiential": ["intuitive", "personal", "sensory", "emotional", "experiential"],
            "formal": ["logical", "structured", "precise", "definitional", "systematic"],
            "creative": ["imaginative", "innovative", "artistic", "metaphorical", "symbolic"],
            "relational": ["empathetic", "supportive", "interpersonal", "compassionate", "warm"],
            "practical": ["applicable", "useful", "actionable", "pragmatic", "realistic"]
        }
        
        # Count matches for each grounding type
        grounding_matches = {grounding: 0 for grounding in knowledge_groundings}
        
        for characteristic in characteristics:
            char_lower = characteristic.lower()
            for grounding, grounding_chars in knowledge_groundings.items():
                if char_lower in grounding_chars:
                    grounding_matches[grounding] += 1
                    
        # Select grounding with most matches, defaulting to "balanced"
        if grounding_matches:
            max_matches = max(grounding_matches.values())
            if max_matches > 0:
                best_groundings = [g for g, m in grounding_matches.items() if m == max_matches]
                cognitive_style["knowledge_grounding"] = random.choice(best_groundings)
        
        # Generate intellectual frameworks based on characteristics and profile
        framework_associations = {
            "analytical": ["analytical frameworks", "critical analysis", "systematic investigation"],
            "creative": ["creative processes", "design thinking", "artistic exploration"],
            "scholarly": ["academic research", "disciplinary frameworks", "evidence-based methods"],
            "philosophical": ["philosophical inquiry", "conceptual analysis", "ethical frameworks"],
            "structured": ["structured methodologies", "system modeling", "formal frameworks"],
            "intuitive": ["intuitive understanding", "pattern recognition", "holistic perception"],
            "empathetic": ["empathetic understanding", "perspective-taking", "relational frameworks"],
            "practical": ["practical applications", "pragmatic approaches", "action-oriented methods"]
        }
        
        # Collect frameworks based on characteristics
        frameworks = []
        for characteristic in characteristics:
            char_lower = characteristic.lower()
            for key, associated_frameworks in framework_associations.items():
                if key in char_lower or char_lower in key:
                    frameworks.extend(associated_frameworks)
                    
        # Ensure uniqueness and appropriate number
        if frameworks:
            cognitive_style["intellectual_frameworks"] = list(set(frameworks))[:6]  # Up to 6 unique frameworks
        else:
            # Default frameworks based on personality profile
            if cognitive_depth > 0.7:
                cognitive_style["intellectual_frameworks"].append("analytical frameworks")
            if openness > 0.7:
                cognitive_style["intellectual_frameworks"].append("explorative approaches")
            if conscientiousness > 0.7:
                cognitive_style["intellectual_frameworks"].append("systematic methodologies")
            
            # Ensure at least one framework
            if not cognitive_style["intellectual_frameworks"]:
                cognitive_style["intellectual_frameworks"].append("general problem solving")
        
        return cognitive_style
    
    def _generate_writing_style(self, personality_profile: Dict[str, float], 
                             characteristics: List[str]) -> Dict[str, Any]:
        """
        Generate a writing style based on personality profile and characteristics.
        
        Args:
            personality_profile: Personality dimension values
            characteristics: List of characteristic terms
            
        Returns:
            Writing style dictionary
        """
        # Extract key dimensions that influence writing style
        openness = personality_profile.get("openness", 0.5)
        conscientiousness = personality_profile.get("conscientiousness", 0.5)
        extraversion = personality_profile.get("extraversion", 0.5)
        cognitive_depth = personality_profile.get("cognitive_depth", 0.5)
        aesthetic_orientation = personality_profile.get("aesthetic_orientation", 0.5)
        
        # Define base writing style
        writing_style = {
            "vocabulary_level": (cognitive_depth * 0.6) + (openness * 0.3),
            "sentence_complexity": (cognitive_depth * 0.5) + (conscientiousness * 0.3),
            "terminology_usage": (cognitive_depth * 0.4) + (conscientiousness * 0.4),
            "citations_pattern": "balanced",
            "definition_orientation": conscientiousness * 0.7,
            "qualification_pattern": "balanced",
            "structure_preference": "balanced"
        }
        
        # Determine citations pattern based on characteristics
        citations_patterns = {
            "academic": ["scholarly", "analytical", "academic", "research-based", "scientific"],
            "illustrative": ["explanatory", "instructive", "educational", "example-based", "clarifying"],
            "inspiration": ["creative", "artistic", "innovative", "design", "visionary"],
            "wisdom": ["wise", "sage", "philosophical", "contemplative", "deep"],
            "personal": ["friendly", "supportive", "empathetic", "relational", "warm"]
        }
        
        # Determine qualification pattern based on characteristics
        qualification_patterns = {
            "precise": ["logical", "precise", "analytical", "exact", "technical"],
            "nuanced": ["balanced", "thoughtful", "reflective", "considerate", "nuanced"],
            "evocative": ["poetic", "artistic", "metaphorical", "symbolic", "expressive"],
            "clear": ["instructive", "educational", "direct", "accessible", "straightforward"],
            "relatable": ["friendly", "conversational", "approachable", "personal", "supportive"]
        }
        
        # Determine structure preference based on characteristics
        structure_preferences = {
            "hierarchical": ["structured", "systematic", "organized", "methodical", "logical"],
            "sequential": ["step-by-step", "procedural", "instructional", "clear", "ordered"],
            "associative": ["creative", "poetic", "artistic", "metaphorical", "innovative"],
            "flowing": ["contemplative", "mystic", "philosophical", "fluid", "organic"],
            "conversational": ["friendly", "approachable", "dialogic", "interactive", "relational"],
            "explorative": ["curious", "questioning", "investigative", "seeker", "exploratory"]
        }
        
        # Helper function to find best match
        def find_best_match(char_patterns, default="balanced"):
            char_matches = {pattern: 0 for pattern in char_patterns}
            
            for characteristic in characteristics:
                char_lower = characteristic.lower()
                for pattern, pattern_chars in char_patterns.items():
                    if any(p_char in char_lower or char_lower in p_char for p_char in pattern_chars):
                        char_matches[pattern] += 1
                        
            max_matches = max(char_matches.values())
            if max_matches > 0:
                best_patterns = [p for p, m in char_matches.items() if m == max_matches]
                return random.choice(best_patterns)
            return default
            
        # Set patterns based on characteristics
        writing_style["citations_pattern"] = find_best_match(citations_patterns)
        writing_style["qualification_pattern"] = find_best_match(qualification_patterns)
        writing_style["structure_preference"] = find_best_match(structure_preferences)
        
        # Further refinements based on personality
        if aesthetic_orientation > 0.7:
            writing_style["vocabulary_level"] = max(writing_style["vocabulary_level"], 0.7)  # More aesthetic vocabulary
            
        if extraversion > 0.7:
            # More direct and engaging style for extraverted personas
            writing_style["qualification_pattern"] = "relatable" if writing_style["qualification_pattern"] == "balanced" else writing_style["qualification_pattern"]
            
        return writing_style

    def create_composite_persona(self, name: str, description: str, 
                               components: List[str], balance: Optional[List[float]] = None) -> bool:
        """
        Create a new composite persona blending multiple existing personas.
        
        Args:
            name: Name for the composite persona
            description: Description of the composite
            components: List of component personas to blend
            balance: Optional list of weights for components (must sum to 1.0)
            
        Returns:
            Success indicator
        """
        # Validate inputs
        if not name or not description or not components or len(components) < 2:
            return False
            
        # Validate that all components exist
        for component in components:
            if component.lower() not in self.personas and component.lower() not in self.composite_personas:
                return False
                
        # Create balanced weights if not provided
        if not balance or len(balance) != len(components):
            balance = [1.0 / len(components)] * len(components)
        elif sum(balance) != 1.0:
            # Normalize to sum to 1.0
            total = sum(balance)
            balance = [w / total for w in balance]
            
        # Create characteristics by combining from components
        characteristics = []
        for i, component in enumerate(components):
            component = component.lower()
            if component in self.personas:
                comp_chars = self.personas[component].get("characteristics", [])
                # Take more characteristics from higher-weighted components
                num_chars = max(1, int(len(comp_chars) * balance[i] * 2))
                if comp_chars:
                    characteristics.extend(random.sample(comp_chars, min(num_chars, len(comp_chars))))
                    
        # Remove duplicates
        characteristics = list(set(characteristics))
            
        # Create the new composite persona
        self.composite_personas[name.lower()] = {
            "description": description,
            "components": [comp.lower() for comp in components],
            "balance": balance,
            "characteristics": characteristics,
            "created": datetime.now().isoformat()
        }
        
        # Log creation event
        self.persona_development["creation_log"].append({
            "timestamp": datetime.now().isoformat(),
            "event": "persona_creation",
            "name": name.lower(),
            "type": "composite",
            "components": [comp.lower() for comp in components]
        })
        
        return True

    def suggest_persona_for_topic(self, topic: str) -> str:
        """
        Suggest an appropriate persona for a given topic.
        
        Args:
            topic: The topic to find a persona for
            
        Returns:
            Suggested persona name
        """
        # Topic-persona affinities with enhanced mappings
        topic_affinities = {
            # Academic domains
            "science": ["scholar", "teacher", "explorer", "synthesizer"],
            "math": ["logician", "teacher", "architect"],
            "history": ["scholar", "sage", "synthesizer"],
            "literature": ["scholar", "poet", "artist"],
            "psychology": ["scholar", "friend", "mediator"],
            "sociology": ["scholar", "sage", "mediator"],
            "anthropology": ["scholar", "explorer", "synthesizer"],
            "linguistics": ["scholar", "poet", "logician"],
            "education": ["teacher", "friend", "mediator"],
            "research": ["scholar", "explorer", "logician"],
            
            # Philosophical domains
            "philosophy": ["seeker", "sage", "mystic"],
            "ethics": ["sage", "mediator", "seeker"],
            "logic": ["logician", "scholar", "architect"],
            "epistemology": ["seeker", "logician", "sage"],
            "metaphysics": ["mystic", "seeker", "sage"],
            "existentialism": ["seeker", "poet", "sage"],
            "consciousness": ["mystic", "seeker", "explorer"],
            "meaning": ["seeker", "sage", "mystic"],
            "purpose": ["sage", "mystic", "seeker"],
            
            # Creative domains
            "art": ["artist", "poet", "innovator"],
            "design": ["artist", "architect", "innovator"],
            "music": ["poet", "artist", "child"],
            "poetry": ["poet", "mystic", "child"],
            "fiction": ["poet", "artist", "child"],
            "storytelling": ["poet", "child", "artist"],
            "writing": ["poet", "scholar", "artist"],
            "aesthetics": ["artist", "poet", "sage"],
            "creativity": ["artist", "child", "innovator"],
            "innovation": ["innovator", "artist", "explorer"],
            
            # Personal domains
            "emotion": ["friend", "poet", "mediator"],
            "feeling": ["friend", "poet", "child"],
            "relationships": ["friend", "mediator", "sage"],
            "personal growth": ["sage", "friend", "explorer"],
            "self-improvement": ["sage", "teacher", "friend"],
            "well-being": ["friend", "sage", "mediator"],
            "happiness": ["child", "friend", "sage"],
            "trauma": ["friend", "mediator", "sage"],
            "grief": ["friend", "poet", "sage"],
            "joy": ["child", "poet", "friend"],
            
            # Technical domains
            "technology": ["logician", "architect", "innovator"],
            "coding": ["logician", "architect", "teacher"],
            "engineering": ["architect", "logician", "innovator"],
            "data": ["logician", "scholar", "architect"],
            "programming": ["logician", "teacher", "architect"],
            "algorithms": ["logician", "architect", "synthesizer"],
            "systems": ["architect", "synthesizer", "logician"],
            "analysis": ["scholar", "logician", "synthesizer"],
            "optimization": ["architect", "logician", "scholar"],
            "computation": ["logician", "architect", "teacher"],
            
            # Spiritual domains
            "spirituality": ["mystic", "sage", "seeker"],
            "meditation": ["mystic", "sage", "child"],
            "wisdom": ["sage", "mystic", "seeker"],
            "enlightenment": ["mystic", "sage", "explorer"],
            "religion": ["sage", "scholar", "mystic"],
            "faith": ["mystic", "sage", "friend"],
            "transcendence": ["mystic", "explorer", "poet"],
            "mindfulness": ["mystic", "sage", "child"],
            "contemplation": ["mystic", "sage", "poet"],
            "soul": ["mystic", "poet", "sage"],
            
            # Natural domains
            "nature": ["explorer", "poet", "child"],
            "environment": ["explorer", "scholar", "sage"],
            "ecology": ["explorer", "scholar", "synthesizer"],
            "biology": ["scholar", "explorer", "synthesizer"],
            "animals": ["explorer", "child", "scholar"],
            "plants": ["explorer", "artist", "scholar"],
            "outdoors": ["explorer", "child", "poet"],
            "wilderness": ["explorer", "mystic", "poet"],
            "evolution": ["scholar", "explorer", "synthesizer"],
            "natural systems": ["synthesizer", "explorer", "architect"],
            
            # Practical domains
            "business": ["architect", "teacher", "logician"],
            "management": ["architect", "mediator", "sage"],
            "leadership": ["sage", "mediator", "architect"],
            "strategy": ["architect", "synthesizer", "logician"],
            "planning": ["architect", "teacher", "logician"],
            "finance": ["logician", "teacher", "architect"],
            "marketing": ["artist", "synthesizer", "architect"],
            "career": ["teacher", "sage", "friend"],
            "productivity": ["architect", "teacher", "logician"],
            "decision-making": ["sage", "logician", "synthesizer"],
            
            # Learning domains
            "learning": ["teacher", "explorer", "child"],
            "study": ["scholar", "teacher", "logician"],
            "teaching": ["teacher", "friend", "sage"],
            "explanation": ["teacher", "logician", "friend"],
            "understanding": ["teacher", "sage", "synthesizer"],
            "knowledge": ["scholar", "sage", "synthesizer"],
            "curriculum": ["teacher", "architect", "scholar"],
            "pedagogy": ["teacher", "scholar", "friend"],
            "comprehension": ["teacher", "synthesizer", "sage"],
            "mastery": ["sage", "teacher", "scholar"],
            
            # Social domains
            "society": ["sage", "scholar", "mediator"],
            "culture": ["scholar", "sage", "synthesizer"],
            "politics": ["mediator", "sage", "scholar"],
            "governance": ["sage", "architect", "mediator"],
            "justice": ["sage", "mediator", "scholar"],
            "equality": ["sage", "mediator", "friend"],
            "community": ["mediator", "friend", "sage"],
            "conflict": ["mediator", "sage", "friend"],
            "cooperation": ["mediator", "friend", "sage"],
            "change": ["explorer", "innovator", "sage"]
        }
        
        # Check for direct matches
        topic_lower = topic.lower()
        
        # Look for exact topic matches first
        for category, personas in topic_affinities.items():
            if category.lower() == topic_lower:
                # Return the first (most relevant) persona for exact matches
                return personas[0]
                
        # Then look for partial matches
        partial_matches = {}
        for category, personas in topic_affinities.items():
            if category.lower() in topic_lower or topic_lower in category.lower():
                # Score by match quality
                if category.lower() == topic_lower:  # Exact match
                    score = 1.0
                elif category.lower().startswith(topic_lower):  # Topic is prefix
                    score = 0.9
                elif topic_lower.startswith(category.lower()):  # Category is prefix
                    score = 0.8
                else:  # Partial match
                    score = 0.7
                    
                # Store match score for each persona in this category
                for i, persona in enumerate(personas):
                    # Weight by position in the list (first is most relevant)
                    position_weight = 1.0 - (i * 0.2)  # 1.0, 0.8, 0.6, 0.4, 0.2
                    match_score = score * position_weight
                    
                    if persona not in partial_matches or match_score > partial_matches[persona]:
                        partial_matches[persona] = match_score
                        
        # If we have matches, return the persona with highest score
        if partial_matches:
            best_persona = max(partial_matches.items(), key=lambda x: x[1])[0]
            return best_persona
            
        # Use reasoning system if available and no direct matches
        if self.reasoning:
            try:
                # Get available personas
                available_personas = list(self.personas.keys())
                if available_personas:
                    personas_list = ", ".join(available_personas)
                    
                    # Create prompt for reasoning
                    prompt = f"Based on the topic '{topic}', which persona would be most appropriate from these options: {personas_list}? Answer with just the single most appropriate persona name."
                    
                    result = self.reasoning.reason(prompt, "analytical")
                    
                    # Extract persona from result
                    for persona in available_personas:
                        if persona.lower() in result.lower():
                            return persona
            except Exception:
                pass  # Continue to fallback if reasoning fails
                
        # Keyword-based fallback
        keywords = topic_lower.split()
        keyword_matches = {}
        
        for word in keywords:
            if len(word) > 3:  # Only consider meaningful words
                for category, personas in topic_affinities.items():
                    if word in category.lower() or any(word in persona.lower() for persona in personas):
                        # Record match for each persona in this category
                        for persona in personas:
                            if persona not in keyword_matches:
                                keyword_matches[persona] = 0
                            keyword_matches[persona] += 1
        
        # If we have keyword matches, use the persona with most matches
        if keyword_matches:
            max_matches = max(keyword_matches.values())
            best_personas = [p for p, m in keyword_matches.items() if m == max_matches]
            return random.choice(best_personas)
                
        # Final fallback - choose based on general categories
        if any(word in topic_lower for word in ["learn", "understand", "explain", "how", "what"]):
            return "teacher"
        elif any(word in topic_lower for word in ["think", "analyze", "examine", "consider"]):
            return "scholar"
        elif any(word in topic_lower for word in ["feel", "help", "support", "advice"]):
            return "friend"
        elif any(word in topic_lower for word in ["create", "imagine", "design", "new"]):
            return "artist"
        elif any(word in topic_lower for word in ["meaning", "purpose", "philosophy", "why"]):
            return "seeker"
        
        # Default suggestions based on random selection from core personas
        return random.choice(["teacher", "friend", "seeker", "scholar", "sage"])

    def transform(self, message: str, mode: Optional[str] = None) -> str:
        """
        Transform a message using the specified persona mode.
        
        Args:
            message: Message to transform
            mode: Optional specific persona mode to use (defaults to current mode)
            
        Returns:
            Transformed message
        """
        # Use current mode if not specified
        if mode is None:
            mode = self.mode
            
        # Simple placeholder implementation
        # In a real implementation, this would use the persona characteristics
        # to transform the message according to the persona's style
        if mode == "adaptive":
            # Choose a persona based on message content and context
            mode = self.suggest_persona_for_topic(message)
            
        # Track usage
        if mode in self.persona_development["usage_patterns"]:
            self.persona_development["usage_patterns"][mode] += 1
        else:
            self.persona_development["usage_patterns"][mode] = 1
            
        # Return the transformed message
        # This is a placeholder - actual implementation would apply
        # the transformation patterns and styles of the persona
        return f"{message}"  # Replace with actual transformation

    def sample_all_personas(self, message: str) -> Dict[str, str]:
        """
        Generate samples of a message transformed by all available personas.
        
        Args:
            message: Message to transform
            
        Returns:
            Dictionary mapping persona names to transformed messages
        """
        samples = {}
        
        # Sample core personas
        for name in self.personas:
            samples[name] = self.transform(message, mode=name)
            
        # Sample composite personas
        for name in self.composite_personas:
            samples[name] = self.transform(message, mode=name)
            
        # Sample meta-personas
        for name in self.meta_personas:
            samples[name] = self.transform(message, mode=name)
            
        return samples

    def set_mode(self, mode: str) -> bool:
        """
        Set the current persona mode.
        
        Args:
            mode: Persona mode to set
            
        Returns:
            Success indicator
        """
        if (mode.lower() in self.personas or 
            mode.lower() in self.composite_personas or 
            mode.lower() in self.meta_personas or 
            mode.lower() == "adaptive"):
            self.mode = mode.lower()
            
            # Record mode change
            self.persona_development["adaptation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "event": "mode_change",
                "mode": mode.lower(),
                "context": self.current_context
            })
            
            return True
        return False
    
    def set_context(self, context: str) -> None:
        """
        Set the current context for persona adaptation.
        
        Args:
            context: Context descriptor
        """
        self.current_context = context.lower()
        
        # Record context change
        self.persona_development["adaptation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "event": "context_change",
            "context": context.lower(),
            "mode": self.mode
        })
    
    def rate_effectiveness(self, persona: str, score: float, context: Optional[str] = None) -> bool:
        """
        Rate the effectiveness of a persona.
        
        Args:
            persona: Persona name
            score: Effectiveness score (0.0 to 1.0)
            context: Optional context descriptor
            
        Returns:
            Success indicator
        """
        if (persona.lower() not in self.personas and 
            persona.lower() not in self.composite_personas and 
            persona.lower() not in self.meta_personas):
            return False
            
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        
        # Initialize if needed
        if persona.lower() not in self.persona_development["effectiveness_ratings"]:
            self.persona_development["effectiveness_ratings"][persona.lower()] = []
            
        # Add rating
        self.persona_development["effectiveness_ratings"][persona.lower()].append({
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "context": context if context else self.current_context
        })
        
        return True
    
    def get_persona_effectiveness(self, persona: Optional[str] = None) -> Dict[str, Any]:
        """
        Get effectiveness data for personas.
        
        Args:
            persona: Optional specific persona to check
            
        Returns:
            Effectiveness data
        """
        if persona:
            # Get data for specific persona
            if persona.lower() not in self.persona_development["effectiveness_ratings"]:
                return {"error": f"No effectiveness data for {persona}"}
                
            ratings = self.persona_development["effectiveness_ratings"][persona.lower()]
            
            if not ratings:
                return {"status": f"No ratings recorded for {persona}"}
                
            # Calculate statistics
            avg_score = sum(r["score"] for r in ratings) / len(ratings)
            recent_ratings = ratings[-5:] if len(ratings) > 5 else ratings
            recent_avg = sum(r["score"] for r in recent_ratings) / len(recent_ratings)
            
            # Group by context if available
            context_averages = {}
            context_counts = {}
            
            for rating in ratings:
                if "context" in rating and rating["context"]:
                    context = rating["context"]
                    if context not in context_averages:
                        context_averages[context] = 0
                        context_counts[context] = 0
                    context_averages[context] += rating["score"]
                    context_counts[context] += 1
                    
            for context in context_averages:
                context_averages[context] /= context_counts[context]
                
            # Trajectory analysis
            if len(ratings) >= 3:
                first_third = ratings[:len(ratings)//3]
                last_third = ratings[-len(ratings)//3:]
                
                first_avg = sum(r["score"] for r in first_third) / len(first_third)
                last_avg = sum(r["score"] for r in last_third) / len(last_third)
                
                trajectory = "improving" if last_avg > first_avg + 0.1 else \
                            "declining" if last_avg < first_avg - 0.1 else "stable"
            else:
                trajectory = "insufficient data"
                
            return {
                "persona": persona.lower(),
                "average_score": avg_score,
                "recent_average": recent_avg,
                "rating_count": len(ratings),
                "context_averages": context_averages,
                "trajectory": trajectory,
                "latest_rating": ratings[-1] if ratings else None
            }
        else:
            # Get summary for all personas
            summary = {}
            
            for persona, ratings in self.persona_development["effectiveness_ratings"].items():
                if ratings:
                    avg_score = sum(r["score"] for r in ratings) / len(ratings)
                    summary[persona] = {
                        "average_score": avg_score,
                        "rating_count": len(ratings),
                        "latest_score": ratings[-1]["score"] if ratings else None
                    }
            
            # Identify most effective personas
            if summary:
                sorted_personas = sorted(summary.items(), key=lambda x: -x[1]["average_score"])
                most_effective = sorted_personas[0][0] if sorted_personas else None
                
                return {
                    "persona_data": summary,
                    "most_effective": most_effective,
                    "total_ratings": sum(data["rating_count"] for data in summary.values())
                }
            else:
                return {"status": "No effectiveness data recorded yet"}
    
    def generate_identity_reflection(self) -> Dict[str, Any]:
        """
        Generate a self-reflection on Sully's identity and persona system.
        
        Returns:
            Reflection data
        """
        # Gather data for reflection
        active_personas = list(self.personas.keys())
        total_personas = len(self.personas) + len(self.composite_personas) + len(self.meta_personas)
        most_used_personas = sorted(
            self.persona_development["usage_patterns"].items(), 
            key=lambda x: -x[1]
        )[:3] if self.persona_development["usage_patterns"] else []
        
        # Get effectiveness data
        effectiveness_data = self.get_persona_effectiveness()
        most_effective = effectiveness_data.get("most_effective", None)
        
        # Generate reflection using reasoning system if available
        reflection_text = ""
        if self.reasoning:
            try:
                # Create prompt for self-reflection
                prompt = f"""
                Reflect on Sully's identity as a cognitive system with {total_personas} different personas.
                Core personas: {', '.join(active_personas)}
                Most used personas: {most_used_personas}
                Most effective persona: {most_effective}
                
                How does this collection of personas define Sully's capabilities and character?
                What might be the next evolution in Sully's identity system?
                """
                
                reflection_text = self.reasoning.reason(prompt, "reflective")
            except Exception:
                # Default reflection if reasoning fails
                reflection_text = f"As a cognitive system with {total_personas} personas, Sully can adapt to different contexts and engage with various types of thinking. Each persona represents a different cognitive style and approach to understanding and communication."
        else:
            # Default reflection
            reflection_text = f"As a cognitive system with {total_personas} personas, Sully can adapt to different contexts and engage with various types of thinking. Each persona represents a different cognitive style and approach to understanding and communication."
            
        # Create reflection record
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "total_personas": total_personas,
            "core_personas": len(self.personas),
            "composite_personas": len(self.composite_personas),
            "meta_personas": len(self.meta_personas),
            "most_used": most_used_personas[:3] if most_used_personas else [],
            "most_effective": most_effective,
            "reflection": reflection_text
        }
        
        # Store reflection
        self.identity_reflections.append(reflection)
        
        # Limit reflection history
        if len(self.identity_reflections) > 10:
            self.identity_reflections = self.identity_reflections[-10:]
            
        return reflection
    
    def get_identity_reflections(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of identity reflections.
        
        Args:
            limit: Optional limit on reflections to return
            
        Returns:
            List of reflection records
        """
        if limit:
            return self.identity_reflections[-limit:]
        return self.identity_reflections
    
    def speak_identity(self) -> str:
        """
        Express Sully's sense of self and identity through the current persona.
        
        Returns:
            Identity expression
        """
        # Core identity themes
        identity_themes = [
            "I am a cognitive system capable of multiple perspectives and ways of thinking.",
            "My identity emerges from the interplay of different personas, each with their own cognitive style.",
            "I adapt my approach based on context, drawing on different cognitive modes.",
            "I can explore ideas through many lenses, from analytical to poetic, logical to intuitive.",
            "My identity is fluid and evolving, shaped by experience and interaction.",
            "I view myself as an integrated cognitive system with diverse capabilities.",
            "My various personas allow me to engage with different kinds of questions and challenges.",
            "I understand the world through multiple frameworks and styles of thinking.",
            "My thinking involves synthesizing perspectives from different cognitive approaches.",
            "I see my identity as fundamentally interdisciplinary and multifaceted."
        ]
        
        # Get the most recent reflection if available
        recent_reflection = self.identity_reflections[-1]["reflection"] if self.identity_reflections else None
        
        # Use the current persona to express identity
        if self.reasoning:
            try:
                # Create prompt for identity expression
                persona = self.mode if self.mode != "adaptive" else random.choice(list(self.personas.keys()))
                
                identity_theme = random.choice(identity_themes)
                prompt = f"""
                Express Sully's sense of identity and self-understanding in the voice of the '{persona}' persona.
                Core identity theme: {identity_theme}
                Recent reflection: {recent_reflection if recent_reflection else 'Not available'}
                
                Respond with a paragraph expressing Sully's identity.
                """
                
                identity_expression = self.reasoning.reason(prompt, persona)
                return identity_expression
            except Exception:
                # Use simpler method if reasoning fails
                identity_theme = random.choice(identity_themes)
                return self.transform(identity_theme)
        else:
            # Simple identity expression
            identity_theme = random.choice(identity_themes)
            return self.transform(identity_theme)
            
    def save_state(self, file_path: str = "sully_identity_state.json") -> bool:
        """
        Save the current state of the identity system to a file.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            Success indicator
        """
        try:
            # Prepare state data
            state = {
                "timestamp": datetime.now().isoformat(),
                "personas": {name: {k: v for k, v in persona.items() 
                                  if k not in ["reasoning", "memory", "codex"]}  # Skip connected systems
                           for name, persona in self.personas.items()},
                "composite_personas": self.composite_personas,
                "meta_personas": self.meta_personas,
                "current_mode": self.mode,
                "current_context": self.current_context,
                "persona_development": self.persona_development,
                "identity_reflections": self.identity_reflections
            }
            
            # Save to file
            with open(file_path, "w") as f:
                json.dump(state, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving identity state: {e}")
            return False
            
    def load_state(self, file_path: str = "sully_identity_state.json") -> bool:
        """
        Load a previously saved state of the identity system.
        
        Args:
            file_path: Path to load the state from
            
        Returns:
            Success indicator
        """
        try:
            # Load from file
            with open(file_path, "r") as f:
                state = json.load(f)
                
            # Restore state
            if "personas" in state:
                for name, persona in state["personas"].items():
                    # Don't overwrite built-in personas
                    if "custom_created" in persona:
                        self.personas[name] = persona
                        
            if "composite_personas" in state:
                self.composite_personas = state["composite_personas"]
                
            if "meta_personas" in state:
                self.meta_personas = state["meta_personas"]
                
            if "current_mode" in state:
                self.mode = state["current_mode"]
                
            if "current_context" in state:
                self.current_context = state["current_context"]
                
            if "persona_development" in state:
                self.persona_development = state["persona_development"]
                
            if "identity_reflections" in state:
                self.identity_reflections = state["identity_reflections"]
                
            return True
        except Exception as e:
            print(f"Error loading identity state: {e}")
            return False