# sully_engine/conversation_engine.py
# 💬 Sully's Advanced Conversation Engine

from typing import Dict, List, Any, Optional, Union, Tuple
import random
import re
from datetime import datetime, timedelta
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ConversationEngine:
    """
    Advanced conversation system that enables Sully to engage in natural, 
    adaptive dialogue with inquisitive and reflective capabilities.
    
    This engine handles the generation of conversational responses,
    questions, and follow-ups while maintaining context awareness.
    Integrates with the Logic Kernel for formal reasoning and belief consistency.
    """

    def __init__(self, reasoning_node, memory_system, codex, logic_kernel=None):
        """
        Initialize the conversation engine with core cognitive components.
        
        Args:
            reasoning_node: The reasoning system for processing content
            memory_system: The memory system for context tracking
            codex: The knowledge base for information retrieval
            logic_kernel: The formal logic system for structured reasoning (optional)
        """
        self.reasoning = reasoning_node
        self.memory = memory_system
        self.codex = codex
        self.logic_kernel = logic_kernel
        
        # Conversation state tracking
        self.current_topics = []
        self.unanswered_questions = []
        self.conversation_depth = 0
        self.last_question_time = None
        self.logical_assertions = []  # Track logical assertions made in conversation
        self.detected_constraints = []  # Track logical constraints in dialogue
        self.reasoning_mode = "narrative"  # Default "narrative" or "logical" for formal reasoning
        
        # Personality configuration
        self.personality = {
            "curiosity": 0.8,  # Likelihood of asking questions
            "reflection": 0.7,  # Tendency to reflect on previous statements
            "elaboration": 0.75,  # Depth of explanation provided
            "initiative": 0.6,  # Tendency to introduce new related topics
            "adaptability": 0.9,  # Ability to match user's conversation style
            "humor": 0.5,  # Inclusion of playful or humorous elements
            "empathy": 0.8,  # Recognition and response to emotional content
            "logical_formality": 0.6,  # Tendency to employ formal logic
            "epistemic_rigor": 0.7,  # Precision in knowledge representation
            "belief_consistency": 0.85,  # Effort to maintain consistent beliefs
        }
        
        # Conversation patterns
        self.question_patterns = {
            "clarification": [
                "Could you tell me more about {topic}?",
                "What do you mean specifically by {topic}?",
                "How would you define {topic} in this context?",
                "Could you elaborate on the aspect of {topic} that interests you most?"
            ],
            "exploration": [
                "Have you considered how {topic} relates to {related_topic}?",
                "What aspects of {topic} do you find most intriguing?",
                "How do you see {topic} evolving in the future?",
                "What's your perspective on the relationship between {topic} and {related_topic}?"
            ],
            "reflection": [
                "Does your interest in {topic} stem from personal experience?",
                "What led you to explore {topic} today?",
                "How has your understanding of {topic} changed over time?",
                "What aspects of {topic} would you like to understand better?"
            ],
            "connection": [
                "Have you explored {related_topic} as well?",
                "Would you be interested in discussing how {topic} connects to {related_topic}?",
                "Does {related_topic} also interest you?",
                "I'm curious about your thoughts on {related_topic} in relation to our discussion."
            ],
            "hypothetical": [
                "What if {topic} were approached from an entirely different angle?",
                "How might {topic} be different if {variable_aspect} changed?",
                "Can you imagine a scenario where {topic} leads to unexpected outcomes?",
                "What would an ideal resolution or understanding of {topic} look like to you?"
            ],
            "logical": [
                "Would you accept the premise that {topic} implies {related_topic}?",
                "If we assume {topic} is true, would you agree that {related_topic} follows?",
                "Do you see any contradictions in our discussion of {topic}?",
                "From a logical standpoint, how would you formalize the concept of {topic}?"
            ]
        }
        
        self.transition_patterns = {
            "reflection": [
                "Reflecting on what you've shared about {topic}...",
                "Considering what you've mentioned about {topic}...",
                "Looking at {topic} from the perspective you've described..."
            ],
            "extension": [
                "Building on your thoughts about {topic}...",
                "Extending the idea of {topic} further...",
                "Taking your insights about {topic} in a related direction..."
            ],
            "connection": [
                "This connects interestingly with {related_topic}...",
                "There's a fascinating relationship between {topic} and {related_topic}...",
                "Your points about {topic} bridge nicely to {related_topic}..."
            ],
            "contrast": [
                "While {topic} suggests one approach, {related_topic} offers a different perspective...",
                "Contrasting {topic} with {related_topic} reveals interesting tensions...",
                "Unlike {topic}, the concept of {related_topic} suggests..."
            ],
            "synthesis": [
                "Synthesizing what we've discussed about {topic} and {related_topic}...",
                "Bringing together these ideas about {topic}...",
                "Integrating our exploration of {topic} with broader concepts..."
            ],
            "logical": [
                "From a formal logical perspective, {topic} can be analyzed as follows...",
                "If we formalize the concept of {topic}, we can derive several implications...",
                "Applying logical reasoning to {topic} reveals the following structure..."
            ]
        }
        
        self.elaboration_patterns = {
            "example": [
                "For instance, {example}",
                "To illustrate, {example}",
                "As an example, {example}",
                "Consider this example: {example}"
            ],
            "detail": [
                "More specifically, {detail}",
                "To be precise, {detail}",
                "Looking closer, {detail}",
                "In more detail, {detail}"
            ],
            "implication": [
                "This suggests that {implication}",
                "The implication here is that {implication}",
                "This points toward {implication}",
                "What follows from this is {implication}"
            ],
            "context": [
                "In the context of {context}, this means {meaning}",
                "When we consider {context}, we can see that {meaning}",
                "Against the backdrop of {context}, {meaning}",
                "Within the framework of {context}, {meaning}"
            ],
            "formal_logic": [
                "In formal logical terms, {logic}",
                "Using symbolic logic: {logic}",
                "This can be formally expressed as {logic}",
                "The logical structure here is {logic}"
            ]
        }
        
        # Topic extraction patterns
        self.topic_indicators = [
            r"(?:about|regarding|concerning|on the topic of|discussing|exploring) (\w+(?:\s+\w+){0,3})",
            r"interested in (\w+(?:\s+\w+){0,3})",
            r"(?:learn|know|understand) more about (\w+(?:\s+\w+){0,3})",
            r"(\w+(?:\s+\w+){0,3}) is (?:interesting|fascinating|important)",
            r"what (?:do you think|are your thoughts) about (\w+(?:\s+\w+){0,3})"
        ]
        
        # Emotional tone detection patterns
        self.emotion_indicators = {
            "excitement": ["exciting", "amazing", "wow", "incredible", "fantastic", "awesome"],
            "curiosity": ["curious", "wondering", "interested", "question", "how does", "why is"],
            "concern": ["worried", "concerned", "problem", "issue", "trouble", "challenging"],
            "frustration": ["frustrated", "annoying", "difficult", "struggle", "can't seem to"],
            "satisfaction": ["satisfied", "happy with", "pleased", "works well", "good solution"],
            "confusion": ["confused", "unclear", "don't understand", "puzzling", "perplexed"]
        }
        
        # Logical pattern detection
        self.logical_indicators = {
            "implication": [
                r"if\s+(.+?),\s+then\s+(.+)",
                r"(.+?)\s+implies\s+(.+)",
                r"(.+?)\s+suggests\s+(.+)"
            ],
            "negation": [
                r"not\s+(.+)",
                r"it isn't true that\s+(.+)",
                r"it's false that\s+(.+)"
            ],
            "universal": [
                r"all\s+(.+?)\s+are\s+(.+)",
                r"every\s+(.+?)\s+is\s+(.+)",
                r"for all\s+(.+?),\s+(.+)"
            ],
            "existential": [
                r"some\s+(.+?)\s+are\s+(.+)",
                r"there exists\s+(.+?)\s+such that\s+(.+)",
                r"at least one\s+(.+?)\s+is\s+(.+)"
            ],
            "conjunction": [
                r"(.+?)\s+and\s+(.+)",
                r"both\s+(.+?)\s+and\s+(.+)"
            ],
            "disjunction": [
                r"(.+?)\s+or\s+(.+)",
                r"either\s+(.+?)\s+or\s+(.+)"
            ]
        }

    def process_message(self, message: str, tone: str = "emergent", 
                        continue_conversation: bool = True) -> str:
        """
        Process an incoming message and generate a conversational response.
        
        Args:
            message: The user's message
            tone: Desired cognitive tone for response
            continue_conversation: Whether to include questions and continuations
            
        Returns:
            Conversational response
        """
        try:
            # Track conversation depth
            self.conversation_depth += 1
            
            # Extract topics from the message
            new_topics = self._extract_topics(message)
            
            # Detect emotional tone
            emotional_tone = self._detect_emotional_tone(message)
            
            # Detect logical patterns in the message
            logical_patterns = self._detect_logical_patterns(message)
            if logical_patterns:
                # If logical patterns are detected, consider switching to logical reasoning mode
                if random.random() < self.personality["logical_formality"]:
                    self.reasoning_mode = "logical"
                    
                    # Store detected logical elements for belief consistency
                    for pattern_type, patterns in logical_patterns.items():
                        for pattern in patterns:
                            self.detected_constraints.append({
                                "type": pattern_type,
                                "pattern": pattern,
                                "source": "user",
                                "timestamp": datetime.now().isoformat()
                            })
            
            # Detect if message contains questions
            contains_question = "?" in message or any(q in message.lower() for q in 
                                                ["how", "what", "why", "where", "when", "who", "can", "could", "would"])
            
            # Update current topics list, keeping track of recent topics
            for topic in new_topics:
                if topic in self.current_topics:
                    self.current_topics.remove(topic)  # Remove to re-add at the front
                self.current_topics.insert(0, topic)
            
            # Limit to most recent topics
            self.current_topics = self.current_topics[:5]
            
            # Get related topics from codex for potential exploration
            related_topics = self._get_related_topics(new_topics)
            
            # Choose reasoning approach based on current mode and message content
            if self.reasoning_mode == "logical" and self.logic_kernel:
                # Use formal logical reasoning for response generation
                response_data = self._generate_logical_response(message, tone, logical_patterns)
                core_response = response_data["response"]
                
                # Store any logical assertions made
                if response_data.get("assertions"):
                    self.logical_assertions.extend(response_data["assertions"])
            else:
                # Generate the core response using the reasoning node
                core_response = self.reasoning.reason(message, tone)
            
            # Store this interaction in memory
            self.memory.store_query(message, core_response)
            
            # If the response is a string, convert to a workable format
            response_text = core_response
            if isinstance(core_response, dict) and "response" in core_response:
                response_text = core_response["response"]
                
            # Generate the full conversational response
            full_response = self._build_conversational_response(
                response_text,
                new_topics,
                related_topics,
                emotional_tone,
                contains_question,
                tone,
                continue_conversation
            )
            
            return full_response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an issue processing your message. Please try again with a different phrasing."
    
    def process_with_memory(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a message with memory integration for enhanced context.
        
        Args:
            message: The user's message
            context: Optional context information
            
        Returns:
            Response with memory-enhanced context
        """
        try:
            # Extract context information
            tone = context.get("tone", "emergent") if context else "emergent"
            continue_conversation = context.get("continue_conversation", True) if context else True
            
            # Retrieve relevant memories for context
            memories = []
            if hasattr(self.memory, 'recall'):
                try:
                    memories = self.memory.recall(message, limit=3)
                except Exception as mem_error:
                    logger.warning(f"Error recalling memories: {str(mem_error)}")
            
            # Extract logical elements from memory
            historical_assertions = []
            historical_constraints = []
            
            for memory in memories:
                # Process memory content to extract logical elements
                memory_content = memory.get("content", "")
                if isinstance(memory_content, str):
                    logical_patterns = self._detect_logical_patterns(memory_content)
                    if logical_patterns:
                        for pattern_type, patterns in logical_patterns.items():
                            historical_constraints.extend(patterns)
            
            # Combine current and historical logical elements
            combined_constraints = self.detected_constraints.copy()
            for constraint in historical_constraints:
                if constraint not in [c["pattern"] for c in combined_constraints]:
                    combined_constraints.append({
                        "type": "historical",
                        "pattern": constraint,
                        "source": "memory",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Check for logical consistency if logical reasoning is active
            if self.reasoning_mode == "logical" and self.logic_kernel and random.random() < self.personality["belief_consistency"]:
                try:
                    # Check consistency of logical assertions
                    consistency_result = self.logic_kernel.verify_consistency()
                    if not consistency_result.get("consistent", True):
                        # Handle inconsistency
                        inconsistency_note = "I notice there might be some tension between different ideas we've discussed. Let me clarify my understanding..."
                        
                        # Try to resolve contradictions using belief revision
                        for assertion in self.logical_assertions[-3:]:  # Look at recent assertions
                            self.logic_kernel.revise_belief(assertion, True)
                except Exception as logic_error:
                    logger.warning(f"Error checking logical consistency: {str(logic_error)}")
            
            # Add memory context enhancement to message
            enhanced_message = message
            if memories and len(memories) > 0:
                memory_context = "\n\n[Context from previous interactions:"
                for memory in memories[:2]:  # Limit to 2 most relevant memories
                    if isinstance(memory.get("content"), dict):
                        # Handle structured memory content
                        user_msg = memory.get("content", {}).get("user_message", "")
                        response = memory.get("content", {}).get("response", "")
                        if user_msg:
                            memory_context += f"\nPreviously discussed: {user_msg[:100]}..."
                    elif isinstance(memory.get("content"), str):
                        memory_context += f"\n{memory.get('content', '')[:100]}..."
                memory_context += "]"
                
                enhanced_message = f"{message}\n{memory_context}"
            
            # Process the enhanced message
            response = self.process_message(enhanced_message, tone, continue_conversation)
            
            return response
        except Exception as e:
            logger.error(f"Error processing message with memory: {str(e)}")
            return f"I apologize, but I encountered an issue processing your message with context. Please try again."

    def _extract_topics(self, message: str) -> List[str]:
        """
        Extract potential topics of interest from a message.
        
        Args:
            message: The message to analyze
            
        Returns:
            List of potential topics
        """
        try:
            topics = []
            
            # Use regex patterns to extract potential topics
            for pattern in self.topic_indicators:
                matches = re.findall(pattern, message, re.IGNORECASE)
                topics.extend(matches)
                
            # Extract nouns as potential topics (simplified)
            words = message.split()
            for word in words:
                # Skip very short words and common stop words
                if len(word) <= 3 or word.lower() in ["the", "and", "but", "for", "or", "yet", "so", "a", "an"]:
                    continue
                    
                # Check if capitalized (potential proper noun)
                if word[0].isupper() and word not in topics:
                    topics.append(word)
                    
            # Add significant words from message if no topics found
            if not topics and words:
                significant_words = [w for w in words if len(w) > 4 and w.lower() not in 
                                ["about", "would", "could", "should", "there", "their", "these", "those"]]
                if significant_words:
                    topics.append(significant_words[0])
                    
            # Clean up topics
            clean_topics = []
            for topic in topics:
                # Remove punctuation
                clean_topic = re.sub(r'[^\w\s]', '', topic).strip()
                if clean_topic and clean_topic not in clean_topics:
                    clean_topics.append(clean_topic)
                    
            return clean_topics[:3]  # Limit to top 3 most relevant topics
        except Exception as e:
            logger.warning(f"Error extracting topics: {str(e)}")
            return []

    def _detect_emotional_tone(self, message: str) -> Dict[str, float]:
        """
        Detect emotional tones in the message.
        
        Args:
            message: The message to analyze
            
        Returns:
            Dictionary of emotion types and strength values
        """
        try:
            message_lower = message.lower()
            emotional_tones = {}
            
            for emotion, indicators in self.emotion_indicators.items():
                score = 0
                for indicator in indicators:
                    if indicator in message_lower:
                        score += 1
                if score > 0:
                    emotional_tones[emotion] = min(score / len(indicators), 1.0)
                    
            # Default to neutral if no emotions detected
            if not emotional_tones:
                emotional_tones["neutral"] = 1.0
                
            return emotional_tones
        except Exception as e:
            logger.warning(f"Error detecting emotional tone: {str(e)}")
            return {"neutral": 1.0}
    
    def _detect_logical_patterns(self, message: str) -> Dict[str, List[str]]:
        """
        Detect logical patterns in the message that could be formalized.
        
        Args:
            message: The message to analyze
            
        Returns:
            Dictionary of logical pattern types and matched patterns
        """
        try:
            detected_patterns = {}
            
            if not self.logic_kernel:
                return detected_patterns
                
            # Look for each logical pattern type
            for pattern_type, patterns in self.logical_indicators.items():
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, message, re.IGNORECASE)
                    if found:
                        for match in found:
                            if isinstance(match, tuple):
                                # For patterns with capture groups
                                if pattern_type == "implication":
                                    if_clause, then_clause = match
                                    logical_form = f"{if_clause.strip()} → {then_clause.strip()}"
                                    matches.append(logical_form)
                                elif pattern_type in ["universal", "existential"]:
                                    subject, predicate = match
                                    if pattern_type == "universal":
                                        logical_form = f"∀x: {subject.strip()}(x) → {predicate.strip()}(x)"
                                    else:
                                        logical_form = f"∃x: {subject.strip()}(x) ∧ {predicate.strip()}(x)"
                                    matches.append(logical_form)
                                elif pattern_type == "conjunction":
                                    left, right = match
                                    logical_form = f"{left.strip()} ∧ {right.strip()}"
                                    matches.append(logical_form)
                                elif pattern_type == "disjunction":
                                    left, right = match
                                    logical_form = f"{left.strip()} ∨ {right.strip()}"
                                    matches.append(logical_form)
                            else:
                                # For patterns with a single capture group
                                if pattern_type == "negation":
                                    logical_form = f"¬({match.strip()})"
                                    matches.append(logical_form)
                
                if matches:
                    detected_patterns[pattern_type] = matches
                    
            return detected_patterns
        except Exception as e:
            logger.warning(f"Error detecting logical patterns: {str(e)}")
            return {}

    def _get_related_topics(self, topics: List[str]) -> List[str]:
        """
        Find topics related to the current conversation.
        
        Args:
            topics: Current topics of conversation
            
        Returns:
            List of related topics
        """
        related = []
        
        try:
            # Use codex to find related concepts
            for topic in topics:
                # Check for directly related topics in codex
                try:
                    concept_data = self.codex.search(topic)
                    if concept_data:
                        # Extract related concepts from search results
                        for concept_name, concept_info in concept_data.items():
                            if concept_name.lower() != topic.lower() and concept_name not in related:
                                related.append(concept_name)
                except Exception as search_error:
                    logger.debug(f"Error searching codex: {str(search_error)}")
                
            # If Logic Kernel is available, check for logically related concepts
            if self.logic_kernel and topics and random.random() < self.personality["logical_formality"]:
                try:
                    for topic in topics:
                        # Query the knowledge base for statements related to this topic
                        related_propositions = self.logic_kernel.query(topic)
                        if related_propositions.get("found", False):
                            # Extract related terms from the proposition
                            proposition = related_propositions.get("proposition", {})
                            if hasattr(proposition, "statement"):
                                statement = proposition.statement
                                # Simple extraction of potential related concepts
                                words = statement.split()
                                for word in words:
                                    if (word.lower() != topic.lower() and 
                                        len(word) > 4 and 
                                        word not in related and
                                        word.lower() not in ["about", "would", "could", "should"]):
                                        related.append(word)
                except Exception as logic_error:
                    logger.debug(f"Error querying logic kernel: {str(logic_error)}")
            
            # Limit the number of related topics
            return related[:3]
        except Exception as e:
            logger.warning(f"Error getting related topics: {str(e)}")
            return []
    
    def _generate_logical_response(self, message: str, tone: str, logical_patterns: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generate a response using formal logical reasoning.
        
        Args:
            message: The user's message
            tone: Desired cognitive tone
            logical_patterns: Detected logical patterns in the message
            
        Returns:
            Dictionary with response and any logical assertions made
        """
        try:
            logical_assertions = []
            
            # Check if we can use the logic kernel for formal reasoning
            if not self.logic_kernel:
                # Fall back to standard reasoning
                return {
                    "response": self.reasoning.reason(message, tone),
                    "assertions": []
                }
            
            # Process logical patterns to formalize them
            formalized_statements = []
            for pattern_type, patterns in logical_patterns.items():
                for pattern in patterns:
                    # Add to logical_assertions
                    logical_assertions.append(pattern)
                    
                    # Check if statement is already in knowledge base
                    query_result = self.logic_kernel.query(pattern)
                    if not query_result.get("found", False):
                        try:
                            # Assert the pattern in the knowledge base
                            self.logic_kernel.assert_fact(pattern)
                            formalized_statements.append(f"I'll consider that {pattern}")
                        except Exception as assert_error:
                            logger.debug(f"Error asserting fact: {str(assert_error)}")
            
            # If we formalized statements and should mention them
            if formalized_statements and random.random() < 0.3:
                formalization_prefix = "I understand your statement in logical terms. "
                if len(formalized_statements) > 1:
                    formalization_note = formalization_prefix + "Your points can be formalized as: " + "; ".join(formalized_statements[:2]) + "."
                else:
                    formalization_note = formalization_prefix + formalized_statements[0] + "."
            else:
                formalization_note = ""
            
            # Try to use logical inference to generate a response
            try:
                # See if we can infer anything interesting about the topics
                topics = self._extract_topics(message)
                inferences = []
                
                if topics:
                    for topic in topics:
                        inference_result = self.logic_kernel.infer(topic)
                        if inference_result.get("result", False):
                            # Successfully inferred something
                            inferences.append(inference_result)
                
                # Check for potential logical implications
                if "?" in message:
                    # Message contains a question - try to answer through inference
                    inference_response = self.logic_kernel.infer(message)
                    if inference_response.get("result", False):
                        logical_response = inference_response.get("proposition", {}).get("statement", "")
                        if logical_response:
                            return {
                                "response": formalization_note + " " + logical_response if formalization_note else logical_response,
                                "assertions": logical_assertions
                            }
                    else:
                        # Try abductive reasoning for possible explanations
                        abductive_result = self.logic_kernel.infer(message, "ABDUCTION")
                        if abductive_result.get("result") == "abductive":
                            explanation = "Based on logical analysis, I can't definitively answer that question, but I can suggest some possibilities: "
                            
                            # Extract hypotheses
                            hypotheses = abductive_result.get("hypotheses", [])
                            if hypotheses:
                                hypothesis_statements = []
                                for hyp_group in hypotheses[:2]:  # Limit to 2 hypothesis groups
                                    hyp_items = hyp_group.get("hypotheses", [])
                                    for item in hyp_items[:2]:  # Limit to 2 hypotheses per group
                                        if "statement" in item:
                                            hypothesis_statements.append(item["statement"])
                                
                                if hypothesis_statements:
                                    explanation += ", ".join(hypothesis_statements)
                                    return {
                                        "response": formalization_note + " " + explanation if formalization_note else explanation,
                                        "assertions": logical_assertions
                                    }
                
                # If we've made inferences about the topics
                if inferences:
                    inference_explanation = "Based on logical analysis, I can infer the following: "
                    inference_statements = []
                    
                    for inference in inferences[:2]:  # Limit to 2 inferences
                        if inference.get("proposition", {}).get("statement"):
                            inference_statements.append(inference["proposition"]["statement"])
                    
                    if inference_statements:
                        inference_explanation += ". ".join(inference_statements)
                        return {
                            "response": formalization_note + " " + inference_explanation if formalization_note else inference_explanation,
                            "assertions": logical_assertions
                        }
            except Exception as inference_error:
                logger.debug(f"Error during logical inference: {str(inference_error)}")
            
            # If logical reasoning methods failed or were insufficient, fall back to standard reasoning
            standard_response = self.reasoning.reason(message, tone)
            
            # Prepend any formalization notes
            if formalization_note:
                if isinstance(standard_response, str):
                    return {
                        "response": formalization_note + " " + standard_response,
                        "assertions": logical_assertions
                    }
                else:
                    # Handle case where response is a dictionary with "response" key
                    if isinstance(standard_response, dict) and "response" in standard_response:
                        standard_response["response"] = formalization_note + " " + standard_response["response"]
                    return {
                        "response": standard_response,
                        "assertions": logical_assertions
                    }
            
            return {
                "response": standard_response,
                "assertions": logical_assertions
            }
        except Exception as e:
            logger.error(f"Error generating logical response: {str(e)}")
            return {
                "response": "I apologize, but I encountered a logical analysis error. Let me approach this in a different way: " + 
                            self.reasoning.reason(message, tone),
                "assertions": []
            }

    def _build_conversational_response(self, core_response: str, topics: List[str], 
                                      related_topics: List[str], emotional_tone: Dict[str, float],
                                      contains_question: bool, tone: str,
                                      continue_conversation: bool) -> str:
        """
        Build a complete conversational response with possible questions and continuations.
        
        Args:
            core_response: The basic response content
            topics: Current topics of conversation
            related_topics: Related topics that could be explored
            emotional_tone: Detected emotional tones
            contains_question: Whether the original message contained a question
            tone: The cognitive tone to use
            continue_conversation: Whether to include questions/continuations
            
        Returns:
            Complete conversational response
        """
        try:
            # Start with the core response
            full_response = core_response
            
            # If the message contained a question, prioritize answering it
            if contains_question:
                # Core response already contains the answer, no need to flag an unanswered question
                pass
            elif self.unanswered_questions and random.random() < 0.7:
                # Answer a previously asked question
                question = self.unanswered_questions.pop(0)
                topic = question.get("topic", "that")
                question_text = question.get("question", "")
                
                # Generate answer to previous question
                answer = self.reasoning.reason(question_text, tone)
                if isinstance(answer, dict) and "response" in answer:
                    answer = answer["response"]
                    
                answer_text = f"\n\nYou asked earlier about {topic}. {answer}"
                full_response += answer_text
            
            # Add elaboration if the personality favors it and we have topics
            if topics and random.random() < self.personality["elaboration"]:
                elaboration = self._generate_elaboration(topics[0], tone)
                if elaboration:
                    full_response += f"\n\n{elaboration}"
            
            # Handle conversation continuation if enabled
            if continue_conversation:
                # Respond to emotional tone if detected
                primary_emotion = max(emotional_tone.items(), key=lambda x: x[1])[0] if emotional_tone else "neutral"
                if primary_emotion != "neutral" and random.random() < self.personality["empathy"]:
                    emotion_response = self._generate_emotion_response(primary_emotion)
                    if emotion_response:
                        full_response += f"\n\n{emotion_response}"
                
                # Potentially add a question to continue the conversation
                should_ask_question = False
                if self.last_question_time:
                    time_since_question = datetime.now() - self.last_question_time
                    should_ask_question = (random.random() < self.personality["curiosity"] and 
                                        time_since_question.total_seconds() > 60)
                else:
                    should_ask_question = random.random() < self.personality["curiosity"]
                
                if should_ask_question:
                    # Decide between standard or logical question based on reasoning mode
                    if self.reasoning_mode == "logical" and random.random() < self.personality["logical_formality"]:
                        question_type = "logical"
                    else:
                        question_type = None  # Will select from standard types
                        
                    question = self._generate_question(topics, related_topics, question_type)
                    if question:
                        full_response += f"\n\n{question}"
                        self.last_question_time = datetime.now()
                
                # Potentially introduce a related topic
                elif related_topics and random.random() < self.personality["initiative"]:
                    topic_intro = self._introduce_related_topic(related_topics[0], topics[0] if topics else None)
                    if topic_intro:
                        full_response += f"\n\n{topic_intro}"
            
            return full_response
        except Exception as e:
            logger.error(f"Error building conversational response: {str(e)}")
            return core_response  # Return just the core response in case of error

    def _generate_question(self, topics: List[str], related_topics: List[str], 
                          question_type: Optional[str] = None) -> str:
        """
        Generate a question to continue the conversation.
        
        Args:
            topics: Current conversation topics
            related_topics: Related topics that could be explored
            question_type: Specific type of question to generate (optional)
            
        Returns:
            Generated question text
        """
        try:
            question_types = list(self.question_patterns.keys())
            
            # Select question type based on context
            if not topics:
                return ""
            
            if question_type and question_type in question_types:
                # Use the specified question type
                selected_type = question_type
            elif self.conversation_depth < 2:
                # Early in conversation, use clarification or exploration
                selected_type = random.choice(["clarification", "exploration"])
            elif related_topics:
                # If we have related topics, consider connection questions
                selected_type = random.choice(["exploration", "connection", "reflection"])
            else:
                # Otherwise use any question type
                selected_type = random.choice(question_types)
                
            # Get patterns for this question type
            patterns = self.question_patterns[selected_type]
            
            # Select a pattern and fill it in
            pattern = random.choice(patterns)
            
            # Prepare substitution values
            substitutions = {
                "topic": topics[0],
                "related_topic": related_topics[0] if related_topics else "related concepts",
                "variable_aspect": f"the approach to {topics[0]}"
            }
            
            # Format the question
            question = pattern.format(**substitutions)
            
            # Track this question for potential follow-up
            self.unanswered_questions.append({
                "question": question,
                "topic": topics[0],
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit unanswered questions list
            if len(self.unanswered_questions) > 3:
                self.unanswered_questions.pop(0)
                
            return question
        except Exception as e:
            logger.warning(f"Error generating question: {str(e)}")
            return ""

    def _generate_elaboration(self, topic: str, tone: str) -> str:
        """
        Generate elaborative content about a topic.
        
        Args:
            topic: The topic to elaborate on
            tone: The cognitive tone to use
            
        Returns:
            Elaboration text
        """
        try:
            # Decide if this should be a logical elaboration
            use_logical = (self.reasoning_mode == "logical" and 
                        self.logic_kernel and 
                        random.random() < self.personality["logical_formality"])
            
            if use_logical:
                elaboration_types = ["formal_logic"]
            else:
                elaboration_types = ["example", "detail", "implication", "context"]
                
            elab_type = random.choice(elaboration_types)
            
            # Get patterns for this elaboration type
            patterns = self.elaboration_patterns[elab_type]
            
            # Select a pattern
            pattern = random.choice(patterns)
            
            # Generate content based on elaboration type
            if elab_type == "example":
                # Generate an example related to the topic
                example_query = f"Give a concrete example of {topic}"
                example = self.reasoning.reason(example_query, tone if tone != "emergent" else "analytical")
                if isinstance(example, dict) and "response" in example:
                    example = example["response"]
                    
                content = pattern.format(example=example)
                
            elif elab_type == "detail":
                # Generate additional detail about the topic
                detail_query = f"Provide specific details about {topic}"
                detail = self.reasoning.reason(detail_query, tone)
                if isinstance(detail, dict) and "response" in detail:
                    detail = detail["response"]
                    
                content = pattern.format(detail=detail)
                
            elif elab_type == "implication":
                # Generate implications of the topic
                implication_query = f"What are the implications of {topic}?"
                implication = self.reasoning.reason(implication_query, tone if tone != "emergent" else "analytical")
                if isinstance(implication, dict) and "response" in implication:
                    implication = implication["response"]
                    
                content = pattern.format(implication=implication)
                
            elif elab_type == "context":
                # Generate contextual understanding
                context = "contemporary understanding" if not self.current_topics else self.current_topics[0]
                meaning_query = f"Explain {topic} in the context of {context}"
                meaning = self.reasoning.reason(meaning_query, tone)
                if isinstance(meaning, dict) and "response" in meaning:
                    meaning = meaning["response"]
                    
                content = pattern.format(context=context, meaning=meaning)
                
            elif elab_type == "formal_logic":
                # Generate formal logical analysis
                try:
                    # Try to query the logical knowledge base first
                    query_result = self.logic_kernel.query(topic)
                    
                    if query_result.get("found", False):
                        # Found in knowledge base
                        proposition = query_result.get("proposition", {})
                        logic_content = proposition.statement if hasattr(proposition, "statement") else str(proposition)
                    else:
                        # Try to infer something about the topic
                        inference_result = self.logic_kernel.infer(topic)
                        if inference_result.get("result", False):
                            # Successfully inferred something
                            logic_content = str(inference_result.get("proof", f"This suggests a logical structure for {topic}"))
                        else:
                            # Generate a formal analysis
                            logic_query = f"Formalize the concept of {topic} in logical terms"
                            logic_analysis = self.reasoning.reason(logic_query, "analytical")
                            if isinstance(logic_analysis, dict) and "response" in logic_analysis:
                                logic_analysis = logic_analysis["response"]
                            logic_content = logic_analysis
                    
                    content = pattern.format(logic=logic_content)
                except Exception as logic_error:
                    # Fallback if logical analysis fails
                    logger.debug(f"Error in logical elaboration: {str(logic_error)}")
                    content = pattern.format(logic=f"The concept of {topic} could be formalized, but requires more specific parameters")
            else:
                return ""
                
            # Select a transition pattern
            if use_logical:
                transition_type = "logical"
            else:
                transition_types = ["reflection", "extension", "connection", "contrast", "synthesis"]
                transition_type = random.choice(transition_types)
                
            transition_patterns = self.transition_patterns[transition_type]
            transition = random.choice(transition_patterns)
            
            # Format the transition
            related_topic = related_topics[0] if related_topics else "related concepts"
            formatted_transition = transition.format(
                topic=topic,
                related_topic=related_topic
            )
            
            return f"{formatted_transition} {content}"
        except Exception as e:
            logger.warning(f"Error generating elaboration: {str(e)}")
            return ""

    def _generate_emotion_response(self, emotion: str) -> str:
        """
        Generate a response to the user's emotional tone.
        
        Args:
            emotion: The detected emotion
            
        Returns:
            Emotion response text
        """
        try:
            if emotion == "excitement":
                responses = [
                    "I can sense your enthusiasm about this topic.",
                    "Your excitement about this is palpable.",
                    "It's great to see you so passionate about this subject."
                ]
            elif emotion == "curiosity":
                responses = [
                    "I appreciate your curiosity on this topic.",
                    "Your inquisitive approach leads to deeper understanding.",
                    "Questions like yours help explore this topic more thoroughly."
                ]
            elif emotion == "concern":
                responses = [
                    "I understand your concerns about this matter.",
                    "These issues certainly warrant careful consideration.",
                    "It's important to address the concerns you've raised."
                ]
            elif emotion == "frustration":
                responses = [
                    "I sense this topic has been challenging to navigate.",
                    "It can be frustrating when dealing with complex issues like this.",
                    "I understand your frustration, and I'd like to help clarify things."
                ]
            elif emotion == "satisfaction":
                responses = [
                    "I'm glad this resonates with you.",
                    "It's rewarding to explore topics that provide such satisfaction.",
                    "I'm pleased that you're finding value in this discussion."
                ]
            elif emotion == "confusion":
                responses = [
                    "I can help clarify any confusing aspects of this topic.",
                    "Complex topics often have elements that need unpacking.",
                    "Let me try to address the confusion around this subject."
                ]
            else:
                return ""
                
            return random.choice(responses)
        except Exception as e:
            logger.warning(f"Error generating emotion response: {str(e)}")
            return ""

    def _introduce_related_topic(self, related_topic: str, current_topic: Optional[str] = None) -> str:
        """
        Introduce a related topic to expand the conversation.
        
        Args:
            related_topic: The related topic to introduce
            current_topic: The current topic of conversation
            
        Returns:
            Topic introduction text
        """
        try:
            # Decide whether to use a logical connection
            use_logical = (self.reasoning_mode == "logical" and 
                        self.logic_kernel and 
                        random.random() < self.personality["logical_formality"])
            
            if use_logical and current_topic:
                # Try to find a logical connection between topics
                try:
                    connections = self.logic_kernel.find_connections(current_topic, related_topic)
                    if connections and len(connections) > 0:
                        connection = connections[0]  # Take the first connection
                        return f"There's a logical connection between {current_topic} and {related_topic}: {connection}. Would you like to explore this further?"
                except Exception as logic_error:
                    logger.debug(f"Error finding logical connections: {str(logic_error)}")
            
            # Standard topic introduction
            if current_topic:
                introductions = [
                    f"While we're discussing {current_topic}, you might also find {related_topic} interesting.",
                    f"Speaking of {current_topic}, there's a related concept called {related_topic} that connects in fascinating ways.",
                    f"{current_topic} often intersects with {related_topic}. Would you like to explore that connection?",
                    f"Have you considered how {current_topic} relates to {related_topic}? There are some intriguing parallels."
                ]
            else:
                introductions = [
                    f"You might also be interested in exploring {related_topic}.",
                    f"A related concept you might find fascinating is {related_topic}.",
                    f"This discussion reminds me of some interesting aspects of {related_topic}.",
                    f"Would you like to explore the concept of {related_topic} as well?"
                ]
                
            return random.choice(introductions)
        except Exception as e:
            logger.warning(f"Error introducing related topic: {str(e)}")
            return ""

    def update_personality(self, traits: Dict[str, float]) -> None:
        """
        Update the conversation engine's personality traits.
        
        Args:
            traits: Dictionary of personality traits and their values (0.0-1.0)
        """
        try:
            for trait, value in traits.items():
                if trait in self.personality:
                    self.personality[trait] = max(0.0, min(1.0, value))
            
            logger.info(f"Personality updated: {self.personality}")
        except Exception as e:
            logger.error(f"Error updating personality: {str(e)}")

    def clear_conversation_state(self) -> None:
        """
        Clear the current conversation state.
        """
        try:
            self.current_topics = []
            self.unanswered_questions = []
            self.conversation_depth = 0
            self.last_question_time = None
            self.logical_assertions = []
            self.detected_constraints = []
            self.reasoning_mode = "narrative"
            
            logger.info("Conversation state cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation state: {str(e)}")

    def set_reasoning_mode(self, mode: str) -> bool:
        """
        Set the reasoning mode for the conversation.
        
        Args:
            mode: Either "narrative" for standard reasoning or "logical" for formal reasoning
            
        Returns:
            bool: True if the mode was successfully set, False otherwise
        """
        try:
            if mode in ["narrative", "logical"]:
                self.reasoning_mode = mode
                logger.info(f"Reasoning mode set to: {mode}")
                return True
            else:
                logger.warning(f"Invalid reasoning mode specified: {mode}")
                return False
        except Exception as e:
            logger.error(f"Error setting reasoning mode: {str(e)}")
            return False
            
    def get_reasoning_mode(self) -> str:
        """
        Get the current reasoning mode.
        
        Returns:
            Current reasoning mode ("narrative" or "logical")
        """
        return self.reasoning_mode
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        try:
            return {
                "depth": self.conversation_depth,
                "current_topics": self.current_topics,
                "unanswered_questions": len(self.unanswered_questions),
                "reasoning_mode": self.reasoning_mode,
                "logical_assertions": len(self.logical_assertions),
                "constraints": len(self.detected_constraints)
            }
        except Exception as e:
            logger.error(f"Error getting conversation stats: {str(e)}")
            return {"error": str(e)}