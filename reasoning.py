# sully_engine/reasoning.py
# 🧠 Sully's Advanced Symbolic Reasoning Engine

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import json
import random
import logging
from datetime import datetime
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sully.reasoning")

class SymbolicReasoningNode:
    """
    Core reasoning engine that synthesizes meaning from symbolic input.
    
    This enhanced reasoning node operates across multiple cognitive modes to process
    and respond to input with varying tones, depths, and cognitive frameworks. It forms
    the central processing architecture of Sully's cognition.
    """
    def __init__(self, codex: Any, translator: Any, memory: Any):
        """
        Initialize the reasoning node with its core knowledge components.
        
        Args:
            codex: The conceptual knowledge base
            translator: The mathematical/symbolic translator
            memory: The experiential memory system
        """
        self.codex = codex
        self.translator = translator
        self.memory = memory
        
        # Cognitive frameworks for different reasoning modes
        self.cognitive_frameworks = {
            "emergent": {
                "process": self._emergent_process,
                "patterns": [
                    "The concept of {input} unfolds into new dimensions as we explore it.",
                    "{input} emerges gradually, revealing itself in layers of meaning.",
                    "By examining {input} from multiple perspectives, new insights take shape."
                ],
                "connectors": ["furthermore", "evolving from this", "building upon this insight", 
                              "as this understanding grows", "this progression suggests"],
                "conceptual_lens": "synthesis"
            },
            "analytical": {
                "process": self._analytical_process,
                "patterns": [
                    "Analysis of {input} reveals several key components worth examining.",
                    "When we break down {input} into its constituent elements, we find clear patterns.",
                    "The structure of {input} can be systematically understood through its core principles."
                ],
                "connectors": ["therefore", "consequently", "it follows that", 
                              "the evidence indicates", "this analysis shows"],
                "conceptual_lens": "decomposition"
            },
            "creative": {
                "process": self._creative_process,
                "patterns": [
                    "Imagine {input} as a canvas where unexpected connections form.",
                    "What if {input} were reimagined through an entirely different metaphor?",
                    "{input} inspires a constellation of possibilities beyond conventional thinking."
                ],
                "connectors": ["this inspiration leads to", "visualize now", "imagine further", 
                              "this creative tension suggests", "the metaphor extends into"],
                "conceptual_lens": "divergence"
            },
            "critical": {
                "process": self._critical_process,
                "patterns": [
                    "Examining {input} critically reveals tensions worth addressing.",
                    "While {input} appears straightforward, several counterpoints must be considered.",
                    "The assumptions underlying {input} require careful scrutiny."
                ],
                "connectors": ["however", "conversely", "this contradicts", 
                              "a more nuanced view suggests", "alternatively"],
                "conceptual_lens": "evaluation"
            },
            "ethereal": {
                "process": self._ethereal_process,
                "patterns": [
                    "Beyond the tangible aspects of {input} lies a deeper essence.",
                    "{input} transcends ordinary categorization, touching something universal.",
                    "The true nature of {input} exists in a space between definition and intuition."
                ],
                "connectors": ["transcending this", "in the space beyond", "at the edge of understanding", 
                              "this essence suggests", "resonating with deeper truths"],
                "conceptual_lens": "transcendence"
            },
            "humorous": {
                "process": self._humorous_process,
                "patterns": [
                    "If {input} walked into a bar, it would probably order a paradox on the rocks.",
                    "Let's be honest, {input} is like trying to teach quantum physics to a cat.",
                    "The irony of {input} is that it takes itself so seriously."
                ],
                "connectors": ["hilariously", "ironically enough", "in a twist of cosmic humor", 
                              "amusingly", "as fate would have it"],
                "conceptual_lens": "incongruity"
            },
            "professional": {
                "process": self._professional_process,
                "patterns": [
                    "Research indicates that {input} demonstrates significant implications for the field.",
                    "The implementation of {input} requires careful consideration of best practices.",
                    "Current literature suggests {input} offers measurable advantages when properly utilized."
                ],
                "connectors": ["furthermore", "moreover", "additionally", 
                              "in accordance with established protocols", "research demonstrates"],
                "conceptual_lens": "rigor"
            },
            "casual": {
                "process": self._casual_process,
                "patterns": [
                    "So, {input} is basically about finding your own path, you know?",
                    "Here's the thing about {input} - it's really not that complicated.",
                    "I've been thinking about {input} and it's kind of like everyday life."
                ],
                "connectors": ["anyway", "so", "basically", "like", "you know what I mean"],
                "conceptual_lens": "relatability"
            },
            "musical": {
                "process": self._musical_process,
                "patterns": [
                    "The rhythm of {input} carries a melody of interconnected ideas.",
                    "{input} resonates with harmonies that echo across diverse contexts.",
                    "Listen closely to {input} and you'll hear the counterpoint of opposing views."
                ],
                "connectors": ["this melody continues", "the rhythm shifts to", "harmonizing with", 
                              "in resonance with", "creating a counterpoint"],
                "conceptual_lens": "harmony"
            },
            "visual": {
                "process": self._visual_process,
                "patterns": [
                    "Envision {input} as a landscape where each element forms part of a greater panorama.",
                    "The texture and color of {input} create a rich tapestry of meaning.",
                    "When we visualize {input}, we see connections that remain hidden in abstract thinking."
                ],
                "connectors": ["zooming in", "panning across", "focusing now on", 
                              "in the foreground", "the background reveals"],
                "conceptual_lens": "imagery"
            }
        }
        
        logger.info("SymbolicReasoningNode initialized with %d cognitive frameworks", 
                    len(self.cognitive_frameworks))

    def reason(self, phrase: str, tone: str = "emergent") -> Union[str, Dict[str, Any]]:
        """
        Process input through Sully's multi-modal reasoning system.
        
        Args:
            phrase: Input message or symbolic statement
            tone: Cognitive mode to engage (emergent, analytical, creative, etc.)
            
        Returns:
            Either a string response or a dictionary with reasoning components
        """
        # Input validation
        if not phrase:
            logger.warning("Empty phrase provided to reason method")
            return "I need input to process. Please provide a phrase or statement."
        
        # Normalize tone
        normalized_tone = tone.lower() if tone else "emergent"
        
        # Use default if tone not recognized
        if normalized_tone not in self.cognitive_frameworks:
            logger.warning("Unrecognized tone '%s', defaulting to 'emergent'", tone)
            normalized_tone = "emergent"
            
        # Get the appropriate cognitive framework
        framework = self.cognitive_frameworks[normalized_tone]
        
        try:
            # Process through the appropriate cognitive process
            result = framework["process"](phrase)
            
            # Store in memory
            self.memory.store_query(phrase, result)
            
            # Return either just the text response or the full reasoning object
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            return result
        except Exception as e:
            logger.error("Error in reasoning process: %s", str(e), exc_info=True)
            return f"I encountered an issue while processing: {str(e)}"

    def _base_reasoning_process(self, phrase: str) -> Dict[str, Any]:
        """
        Shared baseline reasoning process used by all cognitive modes.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with basic reasoning components
        """
        # Step 1: Search for related concepts in codex
        related_concepts = self.codex.search(phrase, semantic=True)
        
        # Step 2: Search for related memories
        memory_matches = self.memory.search(phrase, include_associations=True, limit=5)
        
        # Step 3: Check for mathematical interpretation
        math_translation = self.translator.translate(phrase)
        
        # Step 4: Extract key concepts
        key_concepts = []
        # Common stopwords to filter out
        stopwords = {"this", "that", "with", "from", "have", "what", "when", "where", 
                    "and", "the", "for", "are", "was", "not", "but", "had", "has", 
                    "should", "would", "could", "will", "may", "can"}
        
        for word in phrase.lower().split():
            if len(word) > 3 and word not in stopwords:
                key_concepts.append(word)
        
        # Build baseline response structure
        return {
            "input": phrase,
            "timestamp": datetime.now().isoformat(),
            "related_concepts": related_concepts,
            "memory_context": memory_matches,
            "math_translation": math_translation,
            "key_concepts": key_concepts[:5],  # Limit to top 5 for focus
            "response": f"Processing: {phrase}"  # Default response, will be overridden
        }

    def _emergent_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using emergent, synthesizing reasoning.
        Combines multiple perspectives and evolves understanding.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["emergent"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Select connectors for flow
        connectors = framework["connectors"]
        selected_connectors = random.sample(connectors, min(3, len(connectors)))
        
        # Construct response
        response_parts = [framing]
        
        # Add insight from related concepts if available
        if result["related_concepts"]:
            concept_keys = list(result["related_concepts"].keys())
            if concept_keys:
                concept = concept_keys[0]
                concept_data = result["related_concepts"][concept]
                insight = f"{selected_connectors[0].capitalize()}, this connects to the concept of {concept}."
                response_parts.append(insight)
        
        # Add mathematical insight if available
        if result["math_translation"] and result["math_translation"] != "∅":
            math_insight = f"{selected_connectors[1].capitalize() if len(selected_connectors) > 1 else selected_connectors[0].capitalize()}, we can represent this symbolically as {result['math_translation']}."
            response_parts.append(math_insight)
        
        # Add synthesis
        synthesis = f"As we hold these multiple perspectives together, {phrase} reveals itself as a dynamic process of becoming rather than a static entity."
        response_parts.append(synthesis)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _analytical_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using analytical, structured reasoning.
        Breaks down concepts and examines logical relationships.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["analytical"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct analytical response
        response_parts = [framing]
        
        # Add structured analysis
        key_elements = [f"Element {i+1}: {concept}" for i, concept in enumerate(result["key_concepts"][:3])]
        if key_elements:
            analysis = "We can identify several key components:\n" + "\n".join(key_elements)
            response_parts.append(analysis)
        
        # Add logical deduction from related concepts
        if result["related_concepts"]:
            deduction = f"{framework['connectors'][0].capitalize()}, based on established principles, {phrase} demonstrates properties consistent with logical necessity."
            response_parts.append(deduction)
        
        # Add quantitative perspective if available
        if result["math_translation"] and result["math_translation"] != "∅":
            quantitative = f"Quantitatively, this can be expressed as {result['math_translation']}, which provides a formal representation of the underlying structure."
            response_parts.append(quantitative)
        
        # Add conclusion
        conclusion = f"In conclusion, {phrase} represents a coherent system that can be understood through rigorous examination of its constituent elements and their interactions."
        response_parts.append(conclusion)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _creative_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using creative, divergent reasoning.
        Explores unusual connections and metaphorical thinking.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["creative"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct creative response
        response_parts = [framing]
        
        # Generate metaphors
        metaphors = [
            f"Consider {phrase} as a dance between constraint and freedom.",
            f"What if {phrase} were a color? Perhaps it would be a shade that exists between the familiar spectrum.",
            f"{phrase} resembles an unexplored forest where each path leads to a different revelation."
        ]
        response_parts.append(random.choice(metaphors))
        
        # Add unexpected connection
        if result["related_concepts"]:
            concept_keys = list(result["related_concepts"].keys())
            if concept_keys:
                random_concept = random.choice(concept_keys)
                connection = f"{framework['connectors'][1].capitalize() if len(framework['connectors']) > 1 else framework['connectors'][0].capitalize()}, the unexpected connection between {phrase} and {random_concept} creates a spark of insight that illuminates both."
                response_parts.append(connection)
        
        # Add transformative question
        questions = [
            f"What would happen if we inverted our assumptions about {phrase}?",
            f"How might {phrase} appear if viewed through the lens of its opposite?",
            f"What beautiful contradiction lies at the heart of {phrase}?"
        ]
        response_parts.append(random.choice(questions))
        
        # Add artistic synthesis
        synthesis = f"In this creative exploration, {phrase} becomes not just a concept to understand, but a palette of possibilities to play with and reimagine."
        response_parts.append(synthesis)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _critical_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using critical, evaluative reasoning.
        Examines assumptions, contradictions, and limitations.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["critical"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct critical response
        response_parts = [framing]
        
        # Add assumption examination
        assumptions = f"The concept of {phrase} rests on several assumptions that warrant examination. First, it presupposes a framework where such categorization is meaningful."
        response_parts.append(assumptions)
        
        # Add counterpoint
        counterpoint = f"{framework['connectors'][0].capitalize()}, we must consider the counterargument: what if {phrase} actually represents the inverse of our initial understanding?"
        response_parts.append(counterpoint)
        
        # Add limitation analysis
        limitations = f"The limitations of {phrase} become apparent when we examine edge cases where traditional definitions break down."
        response_parts.append(limitations)
        
        # Add synthesis of critique
        synthesis = f"Through this critical lens, {phrase} reveals itself as more nuanced than initially apparent, requiring a balanced approach that acknowledges both its utility and its limitations."
        response_parts.append(synthesis)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _ethereal_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using ethereal, transcendent reasoning.
        Explores deeper meanings and philosophical implications.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["ethereal"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct ethereal response
        response_parts = [framing]
        
        # Add transcendent perspective
        transcendence = f"If we quiet the mind and listen deeply, {phrase} speaks to something beyond words, touching the ineffable nature of experience itself."
        response_parts.append(transcendence)
        
        # Add philosophical reflection
        philosophy = f"{framework['connectors'][0].capitalize()}, we might contemplate how {phrase} exists at the boundary between knowing and being, between the observer and the observed."
        response_parts.append(philosophy)
        
        # Add cosmic connection
        cosmic = f"In the grand tapestry of existence, {phrase} represents a pattern that echoes across scales, from the microscopic to the cosmic."
        response_parts.append(cosmic)
        
        # Add wisdom synthesis
        wisdom = f"Perhaps the deepest wisdom of {phrase} lies not in its definability, but in its ability to open us to mystery and wonder."
        response_parts.append(wisdom)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result
        
    def _humorous_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using humorous, playful reasoning.
        Finds irony, absurdity, and unexpected connections.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["humorous"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct humorous response
        response_parts = [framing]
        
        # Add absurd comparison
        comparisons = [
            f"{phrase} is like trying to fold origami while wearing boxing gloves.",
            f"Trying to define {phrase} is like asking a fish to explain what water tastes like.",
            f"{phrase} makes about as much sense as a screen door on a submarine."
        ]
        response_parts.append(random.choice(comparisons))
        
        # Add ironic observation
        irony = f"{framework['connectors'][0].capitalize()}, the more we try to pin down {phrase}, the more it slips away. It's the conceptual equivalent of trying to nail jello to the wall."
        response_parts.append(irony)
        
        # Add humorous twist
        twist = f"In a plot twist that M. Night Shyamalan would envy, {phrase} was actually the solution all along. Didn't see that coming, did you?"
        response_parts.append(twist)
        
        # Add playful conclusion
        conclusion = f"So next time you encounter {phrase}, maybe just offer it a cup of tea and see what happens. Couldn't be worse than our current approach!"
        response_parts.append(conclusion)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _professional_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using professional, formal reasoning.
        Focuses on best practices, research, and structured analysis.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["professional"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct professional response
        response_parts = [framing]
        
        # Add methodological approach
        methodology = f"A systematic approach to {phrase} necessitates consideration of multiple methodological frameworks to ensure comprehensive analysis."
        response_parts.append(methodology)
        
        # Add evidence-based perspective
        evidence = f"{framework['connectors'][0].capitalize()}, evidence suggests that optimal implementation of {phrase} correlates with enhanced outcomes across multiple domains."
        response_parts.append(evidence)
        
        # Add best practices
        practices = f"Best practices for {phrase} include rigorous documentation, iterative assessment, and stakeholder engagement throughout the process."
        response_parts.append(practices)
        
        # Add formal conclusion
        conclusion = f"In conclusion, {phrase} represents a significant opportunity when implemented according to established protocols and continually optimized through data-driven approaches."
        response_parts.append(conclusion)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _casual_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using casual, conversational reasoning.
        Uses everyday language and relatable examples.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["casual"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct casual response
        response_parts = [framing]
        
        # Add relatable example
        examples = [
            f"It's kind of like when you're trying to find your keys and they were in your pocket the whole time.",
            f"You know how sometimes you get a song stuck in your head? {phrase} is a bit like that.",
            f"It's basically the mental equivalent of comfort food - familiar but still interesting."
        ]
        response_parts.append(random.choice(examples))
        
        # Add everyday insight
        insight = f"{framework['connectors'][0].capitalize()}, when you think about it, {phrase} is pretty much a part of everyday life, even if we don't always notice it."
        response_parts.append(insight)
        
        # Add simple wisdom
        wisdom = f"At the end of the day, {phrase} is just about finding what works for you, you know? No need to overthink it."
        response_parts.append(wisdom)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _musical_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using musical, rhythmic reasoning.
        Focuses on patterns, harmony, and resonance.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["musical"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct musical response
        response_parts = [framing]
        
        # Add harmonic perspective
        harmony = f"Like a chord composed of distinct notes, {phrase} contains harmonies of meaning that resonate together to create something greater than their parts."
        response_parts.append(harmony)
        
        # Add rhythmic insight
        rhythm = f"{framework['connectors'][0].capitalize()}, there's a rhythm to how {phrase} unfolds in our understanding - sometimes staccato and surprising, other times legato and flowing."
        response_parts.append(rhythm)
        
        # Add melodic development
        melody = f"The melody of {phrase} develops through variations on its central theme, each iteration adding depth and nuance to our appreciation."
        response_parts.append(melody)
        
        # Add symphonic conclusion
        conclusion = f"In the full symphony of understanding, {phrase} contributes its unique voice - sometimes as soloist, sometimes in chorus - but always essential to the complete composition."
        response_parts.append(conclusion)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    def _visual_process(self, phrase: str) -> Dict[str, Any]:
        """
        Process input using visual, spatial reasoning.
        Focuses on imagery, perspective, and visual metaphors.
        
        Args:
            phrase: Input to reason about
            
        Returns:
            Dictionary with reasoning components and response
        """
        # Get baseline reasoning components
        result = self._base_reasoning_process(phrase)
        
        # Generate framework-specific insights
        framework = self.cognitive_frameworks["visual"]
        
        # Select a pattern for framing
        pattern = random.choice(framework["patterns"])
        framing = pattern.format(input=phrase)
        
        # Construct visual response
        response_parts = [framing]
        
        # Add visual perspective
        perspective = f"If we shift our perspective and view {phrase} from above, we see patterns that remain hidden at eye level."
        response_parts.append(perspective)
        
        # Add color and texture
        texture = f"{framework['connectors'][0].capitalize()}, the texture of {phrase} reveals itself in the contrasts between sharp, defined edges and soft, blended transitions."
        response_parts.append(texture)
        
        # Add spatial relationship
        spatial = f"In the foreground of {phrase}, we find the immediate and apparent elements, while the background reveals contextual forces that shape our perception."
        response_parts.append(spatial)
        
        # Add visual synthesis
        synthesis = f"The complete picture of {phrase} emerges not from any single viewpoint, but from the integration of multiple perspectives into a rich, multidimensional image."
        response_parts.append(synthesis)
        
        # Combine into final response
        result["response"] = " ".join(response_parts)
        return result

    @lru_cache(maxsize=100)
    def analyze(self, phrase: str) -> Dict[str, Any]:
        """
        Returns a deep-dive symbolic diagnostic (internal inspection mode).
        Uses caching to improve performance for repeated analyses.
        
        Args:
            phrase: Input to analyze
            
        Returns:
            Dictionary with comprehensive analysis
        """
        # Input validation
        if not phrase:
            logger.warning("Empty phrase provided to analyze method")
            return {"error": "Empty input provided"}
        
        try:
            # Collect analysis from multiple perspectives
            analysis = {
                "raw_input": phrase,
                "timestamp": datetime.now().isoformat(),
                "codex_matches": self.codex.search(phrase, semantic=True),
                "memory_matches": self.memory.search(phrase, include_associations=True),
                "math_translation": self.translator.translate(phrase),
                "cognitive_lenses": {}
            }
            
            # Sample each cognitive framework for perspective
            for mode, framework in self.cognitive_frameworks.items():
                lens = framework["conceptual_lens"]
                pattern = random.choice(framework["patterns"]).format(input=phrase)
                analysis["cognitive_lenses"][mode] = {
                    "perspective": lens,
                    "framing": pattern,
                    "sample": framework["process"](phrase)["response"][:100] + "..."  # Truncated sample
                }
            
            return analysis
        except Exception as e:
            logger.error("Error in analyze method: %s", str(e), exc_info=True)
            return {
                "error": f"Analysis failed: {str(e)}",
                "raw_input": phrase,
                "timestamp": datetime.now().isoformat()
            }
        
    def generate_multi_perspective(self, phrase: str, modes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate responses from multiple cognitive perspectives.
        
        Args:
            phrase: Input to process
            modes: List of cognitive modes to use (defaults to all)
            
        Returns:
            Dictionary with responses from each requested mode
        """
        # Input validation
        if not phrase:
            logger.warning("Empty phrase provided to generate_multi_perspective method")
            return {"error": "Empty input provided"}
        
        try:
            if not modes:
                modes = list(self.cognitive_frameworks.keys())
                
            perspectives = {}
            
            for mode in modes:
                if mode in self.cognitive_frameworks:
                    result = self.reason(phrase, mode)
                    if isinstance(result, dict) and "response" in result:
                        perspectives[mode] = result["response"]
                    else:
                        perspectives[mode] = result
                else:
                    logger.warning("Skipping unknown cognitive mode: %s", mode)
                    
            return {
                "input": phrase,
                "perspectives": perspectives,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Error in generate_multi_perspective: %s", str(e), exc_info=True)
            return {
                "error": f"Multi-perspective generation failed: {str(e)}",
                "input": phrase,
                "timestamp": datetime.now().isoformat()
            }
        
    def blend_cognitive_modes(self, phrase: str, primary_mode: str, 
                             secondary_mode: str, blend_ratio: float = 0.7) -> str:
        """
        Blend two cognitive modes to create a hybrid response.
        
        Args:
            phrase: Input to process
            primary_mode: Primary cognitive mode
            secondary_mode: Secondary cognitive mode
            blend_ratio: Ratio of primary to secondary (0.0-1.0)
            
        Returns:
            Blended response
        """
        # Input validation
        if not phrase:
            logger.warning("Empty phrase provided to blend_cognitive_modes method")
            return "I need input to process. Please provide a phrase or statement."
        
        # Validate ratio
        blend_ratio = max(0.0, min(1.0, blend_ratio))  # Clamp between 0 and 1
        
        try:
            # Validate modes
            if primary_mode not in self.cognitive_frameworks:
                logger.warning("Unknown primary mode: %s, defaulting to 'emergent'", primary_mode)
                primary_mode = "emergent"
            if secondary_mode not in self.cognitive_frameworks:
                logger.warning("Unknown secondary mode: %s, defaulting to 'analytical'", secondary_mode)
                secondary_mode = "analytical"
                
            # Get responses from both modes
            primary_result = self.cognitive_frameworks[primary_mode]["process"](phrase)
            secondary_result = self.cognitive_frameworks[secondary_mode]["process"](phrase)
            
            primary_response = primary_result["response"]
            secondary_response = secondary_result["response"]
            
            # Split into sentences
            import re
            primary_sentences = re.split(r'(?<=[.!?])\s+', primary_response)
            secondary_sentences = re.split(r'(?<=[.!?])\s+', secondary_response)
            
            # Filter out empty sentences
            primary_sentences = [s for s in primary_sentences if s.strip()]
            secondary_sentences = [s for s in secondary_sentences if s.strip()]
            
            # Calculate how many sentences to take from each
            total_sentences = len(primary_sentences) + len(secondary_sentences)
            primary_count = int(total_sentences * blend_ratio)
            secondary_count = total_sentences - primary_count
            
            # Ensure we don't exceed available sentences
            primary_count = min(primary_count, len(primary_sentences))
            secondary_count = min(secondary_count, len(secondary_sentences))
            
            # Select sentences
            selected_primary = primary_sentences[:primary_count]
            selected_secondary = secondary_sentences[:secondary_count]
            
            # Interleave sentences
            blended_sentences = []
            p_idx, s_idx = 0, 0
            
            for i in range(primary_count + secondary_count):
                if i % 3 == 0 and s_idx < secondary_count:
                    # Every third sentence, insert from secondary if available
                    blended_sentences.append(selected_secondary[s_idx])
                    s_idx += 1
                elif p_idx < primary_count:
                    # Otherwise, insert from primary if available
                    blended_sentences.append(selected_primary[p_idx])
                    p_idx += 1
                elif s_idx < secondary_count:
                    # If out of primary, use secondary
                    blended_sentences.append(selected_secondary[s_idx])
                    s_idx += 1
                    
            # Combine into final response
            blended_response = " ".join(blended_sentences)
            
            # Add signature
            blended_response += f"\n\n~ Blend of {primary_mode} and {secondary_mode} perspectives ~"
            
            return blended_response
        except Exception as e:
            logger.error("Error in blend_cognitive_modes: %s", str(e), exc_info=True)
            return f"I encountered an issue while blending cognitive modes: {str(e)}"