import spacy
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import softmax

class SemanticStructureAnalyzer:
    """
    Advanced semantic analysis system for improving Sully's understanding of natural language.
    
    This module provides grammatical structure analysis, concept identification,
    contextual relevance scoring, and semantic weighting to enhance conversational capabilities.
    """
    
    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initialize the semantic analyzer with NLP models and knowledge bases.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp = spacy.load(model_name)
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.tfidf_fitted = False
        
        # Corpus for training TF-IDF
        self.corpus = []
        
        # Concept importance cache
        self.concept_importance = {}
        
        # Module relevance weights
        self.module_weights = {
            "math_translation": 0.5,
            "dream": 0.6,
            "fusion": 0.7,
            "paradox": 0.5,
            "conversation": 0.9,
            "memory": 0.7,
            "logic": 0.6
        }
        
        # Concept domain mappings
        self.concept_domains = {
            "mathematics": ["equation", "formula", "theorem", "proof", "number", "calculation"],
            "philosophy": ["meaning", "existence", "consciousness", "ethics", "reality", "truth"],
            "science": ["experiment", "theory", "observation", "hypothesis", "evidence", "research"],
            "art": ["beauty", "creativity", "expression", "aesthetic", "design", "imagination"],
            "emotion": ["love", "fear", "joy", "sadness", "anger", "happiness", "feeling"],
            "technology": ["computer", "algorithm", "software", "hardware", "digital", "code", "programming"]
        }
        
        # Initialize response templates with variations
        self._initialize_response_templates()
        
        # Semantic field mappings
        self.semantic_fields = self._load_semantic_fields()

    def _initialize_response_templates(self) -> None:
        """Initialize diverse response templates for different contexts."""
        self.templates = {
            "statement": [
                "Looking at {concept} from multiple perspectives, I observe that {insight}.",
                "The concept of {concept} has several dimensions: {insight}.",
                "{concept} can be understood through various lenses, revealing that {insight}.",
                "When we examine {concept} carefully, we find that {insight}."
            ],
            "question": [
                "To address your question about {concept}, I'd suggest that {insight}.",
                "Regarding {concept}, I've considered several angles: {insight}.",
                "Your question about {concept} opens up interesting possibilities: {insight}.",
                "Exploring {concept} as you've asked leads me to think that {insight}."
            ],
            "elaboration": [
                "Building on what we've discussed about {concept}, {insight}.",
                "To elaborate further on {concept}, {insight}.",
                "Delving deeper into {concept}, we can see that {insight}.",
                "Let me expand on {concept}: {insight}."
            ],
            "bridge": [
                "This connects interestingly with {related_concept}, because {connection}.",
                "There's a fascinating relationship between {concept} and {related_concept}: {connection}.",
                "{concept} naturally leads us to consider {related_concept}, as {connection}.",
                "The link between {concept} and {related_concept} reveals that {connection}."
            ]
        }
        
        # Templates for mathematical translations (used only when relevant)
        self.math_templates = [
            "This can be represented formally as {formula}, where {explanation}.",
            "In mathematical terms, we might express this as {formula}, meaning {explanation}.",
            "A formal representation might be {formula}, which captures {explanation}."
        ]
    
    def _load_semantic_fields(self) -> Dict[str, List[str]]:
        """
        Load semantic field mappings.
        
        Returns:
            Dictionary mapping concepts to related concepts
        """
        # This could be expanded to load from a file or database
        return {
            "love": ["compassion", "care", "affection", "attachment", "devotion", "empathy"],
            "knowledge": ["understanding", "wisdom", "insight", "comprehension", "awareness", "cognition"],
            "meaning": ["purpose", "significance", "importance", "value", "implication", "sense"],
            "life": ["existence", "living", "being", "vitality", "experience", "journey"],
            "time": ["duration", "period", "interval", "moment", "instance", "era", "epoch"],
            "mind": ["intellect", "consciousness", "thought", "cognition", "awareness", "psyche"],
            "reality": ["existence", "actuality", "fact", "truth", "cosmos", "universe"],
            "truth": ["fact", "reality", "accuracy", "correctness", "validity", "veracity"]
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Add to corpus for TF-IDF training
        self.corpus.append(text)
        if len(self.corpus) > 100:
            self.corpus = self.corpus[-100:]  # Keep only most recent 100 entries
            self.tfidf_fitted = False  # Need to refit
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract grammatical structure
        grammatical_structure = self._extract_grammatical_structure(doc)
        
        # Extract and weight concepts
        weighted_concepts = self._extract_weighted_concepts(doc, text)
        
        # Determine text type
        text_type = self._determine_text_type(doc)
        
        # Determine relevant modules
        module_relevance = self._calculate_module_relevance(doc, weighted_concepts)
        
        # Find domain relevance
        domain_relevance = self._determine_domain_relevance(weighted_concepts)
        
        return {
            "structure": grammatical_structure,
            "weighted_concepts": weighted_concepts,
            "text_type": text_type,
            "module_relevance": module_relevance,
            "domain_relevance": domain_relevance,
            "doc": doc  # Include spaCy doc for further processing if needed
        }
    
    def _extract_grammatical_structure(self, doc) -> Dict[str, Any]:
        """
        Extract grammatical structure from parsed text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary with grammatical structure
        """
        # Extract subject-verb-object structure
        subjects = []
        verbs = []
        objects = []
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract syntactic dependencies
        for token in doc:
            if "subj" in token.dep_:
                subjects.append((token.text, token.i))
            elif token.pos_ == "VERB":
                verbs.append((token.text, token.i))
            elif "obj" in token.dep_:
                objects.append((token.text, token.i))
        
        # Extract noun chunks
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        return {
            "subjects": subjects,
            "verbs": verbs,
            "objects": objects,
            "entities": entities,
            "noun_chunks": noun_chunks,
            "root": doc.root.text if len(doc) > 0 else "",
            "sentence_structure": [(token.text, token.pos_, token.dep_) for token in doc]
        }
    
    def _extract_weighted_concepts(self, doc, text: str) -> List[Tuple[str, float]]:
        """
        Extract and weight concepts based on grammatical role and TF-IDF.
        
        Args:
            doc: spaCy Doc object
            text: Raw input text
            
        Returns:
            List of (concept, weight) tuples
        """
        weighted_concepts = []
        
        # Extract nouns, proper nouns, and named entities as potential concepts
        potential_concepts = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                # Assign initial weight based on grammatical role
                weight = 1.0
                if "subj" in token.dep_:
                    weight *= 1.5  # Subjects are more important
                if token.pos_ == "PROPN":
                    weight *= 1.2  # Proper nouns get extra weight
                
                potential_concepts.append((token.text.lower(), weight))
        
        # Add named entities with high weight
        for ent in doc.ents:
            potential_concepts.append((ent.text.lower(), 1.8))
        
        # Add noun chunks (may overlap with above, but that's OK)
        for chunk in doc.noun_chunks:
            potential_concepts.append((chunk.text.lower(), 1.3))
        
        # Apply TF-IDF weighting if we have enough corpus
        if len(self.corpus) > 5:
            if not self.tfidf_fitted:
                self.tfidf.fit(self.corpus)
                self.tfidf_fitted = True
            
            try:
                tfidf_vector = self.tfidf.transform([text])
                feature_names = self.tfidf.get_feature_names_out()
                
                # Get TF-IDF scores for words in the text
                tfidf_scores = {}
                for i, score in zip(tfidf_vector.indices, tfidf_vector.data):
                    word = feature_names[i]
                    tfidf_scores[word] = score
                
                # Adjust weights based on TF-IDF
                for i, (concept, weight) in enumerate(potential_concepts):
                    # Look for concept or parts of multi-word concepts in TF-IDF scores
                    concept_words = concept.split()
                    tfidf_weight = 0
                    for word in concept_words:
                        if word in tfidf_scores:
                            tfidf_weight += tfidf_scores[word]
                    
                    if len(concept_words) > 0:
                        tfidf_weight /= len(concept_words)
                        
                    # Combine initial weight with TF-IDF weight
                    potential_concepts[i] = (concept, weight * (1 + tfidf_weight))
            except:
                # If TF-IDF fails, just continue with initial weights
                pass
        
        # Remove duplicates and sort by weight
        seen_concepts = set()
        for concept, weight in sorted(potential_concepts, key=lambda x: x[1], reverse=True):
            if concept not in seen_concepts and len(concept) > 1:
                weighted_concepts.append((concept, weight))
                seen_concepts.add(concept)
        
        return weighted_concepts[:10]  # Return top 10 weighted concepts
    
    def _determine_text_type(self, doc) -> str:
        """
        Determine the type of text (question, statement, command, etc.).
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Text type as string
        """
        text = doc.text.strip()
        
        # Check for questions
        if text.endswith("?"):
            return "question"
        elif any(token.pos_ == "AUX" and token.i == 0 for token in doc):
            return "question"  # Starts with auxiliary verb
        elif any(token.lower_ in ("what", "who", "where", "when", "why", "how") and token.i == 0 for token in doc):
            return "question"  # Starts with WH-word
            
        # Check for commands
        if any(token.pos_ == "VERB" and token.i == 0 for token in doc):
            return "command"  # Starts with verb
            
        # Check for exclamations
        if text.endswith("!"):
            return "exclamation"
            
        # Default to statement
        return "statement"
    
    def _calculate_module_relevance(self, doc, weighted_concepts: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Calculate relevance scores for different cognitive modules.
        
        Args:
            doc: spaCy Doc object
            weighted_concepts: List of weighted concepts
            
        Returns:
            Dictionary with module relevance scores
        """
        module_scores = defaultdict(float)
        
        # Base weights for different modules
        for module, base_weight in self.module_weights.items():
            module_scores[module] = base_weight
        
        # Adjust based on text type
        text_type = self._determine_text_type(doc)
        if text_type == "question":
            module_scores["conversation"] += 0.2
        
        # Adjust based on concepts
        concepts = [c for c, _ in weighted_concepts]
        for concept in concepts:
            # Mathematical concepts increase math_translation relevance
            if any(math_term in concept for math_term in ["equation", "equal", "formula", "number", "calculate", "math"]):
                module_scores["math_translation"] += 0.3
                
            # Abstract concepts increase dream and fusion relevance
            if any(abstract_term in concept for abstract_term in ["meaning", "life", "consciousness", "existence", "truth"]):
                module_scores["dream"] += 0.3
                module_scores["fusion"] += 0.2
                
            # Logical concepts increase logic relevance
            if any(logic_term in concept for logic_term in ["logic", "reason", "valid", "argument", "conclude"]):
                module_scores["logic"] += 0.3
                
            # Paradoxical concepts increase paradox relevance
            if any(paradox_term in concept for paradox_term in ["paradox", "contradiction", "versus", "opposed"]):
                module_scores["paradox"] += 0.4
        
        # Check for specific lemmas that indicate module relevance
        lemmas = [token.lemma_ for token in doc]
        
        math_indicators = ["calculate", "compute", "equation", "formula", "solve", "math"]
        if any(indicator in lemmas for indicator in math_indicators):
            module_scores["math_translation"] += 0.3
            
        dream_indicators = ["dream", "imagine", "create", "envision", "symbolic"]
        if any(indicator in lemmas for indicator in dream_indicators):
            module_scores["dream"] += 0.3
            
        fusion_indicators = ["combine", "merge", "synthesize", "integrate", "connect"]
        if any(indicator in lemmas for indicator in fusion_indicators):
            module_scores["fusion"] += 0.3
            
        paradox_indicators = ["paradox", "contradiction", "opposite", "contrary", "versus"]
        if any(indicator in lemmas for indicator in paradox_indicators):
            module_scores["paradox"] += 0.3
        
        # Normalize scores with softmax to create a probability distribution
        scores_array = np.array(list(module_scores.values()))
        normalized_scores = softmax(scores_array)
        
        return {module: float(score) for module, score in zip(module_scores.keys(), normalized_scores)}
    
    def _determine_domain_relevance(self, weighted_concepts: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Determine relevance of different knowledge domains.
        
        Args:
            weighted_concepts: List of weighted concepts
            
        Returns:
            Dictionary with domain relevance scores
        """
        domain_scores = defaultdict(float)
        
        # Extract just the concepts
        concepts = [concept.lower() for concept, _ in weighted_concepts]
        
        # Score each domain based on concept overlap
        for domain, domain_concepts in self.concept_domains.items():
            for concept in concepts:
                # Check if any domain concept is in the user's concept
                for domain_concept in domain_concepts:
                    if domain_concept in concept:
                        domain_scores[domain] += 1.0
                        break
        
        # Normalize if we have scores
        if sum(domain_scores.values()) > 0:
            total = sum(domain_scores.values())
            domain_scores = {domain: score/total for domain, score in domain_scores.items()}
        
        return domain_scores
    
    def generate_response_template(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate appropriate response templates based on text analysis.
        
        Args:
            analysis: Output from analyze_text method
            
        Returns:
            Dictionary with template choices and parameters
        """
        # Extract key information
        text_type = analysis["text_type"]
        module_relevance = analysis["module_relevance"]
        weighted_concepts = analysis["weighted_concepts"]
        
        # Determine template type based on text type
        if text_type == "question":
            template_type = "question"
        else:
            template_type = "statement"
        
        # Select a template from the appropriate category
        templates = self.templates.get(template_type, self.templates["statement"])
        template = np.random.choice(templates)
        
        # Determine main concept from weighted concepts
        main_concept = weighted_concepts[0][0] if weighted_concepts else "this topic"
        
        # Determine if we should include a mathematical template
        include_math = module_relevance.get("math_translation", 0) > 0.3
        
        # Generate parameters for template
        parameters = {
            "concept": main_concept,
            "template_type": template_type,
            "include_math": include_math,
            "math_template": np.random.choice(self.math_templates) if include_math else None,
            "related_concepts": self._get_related_concepts(main_concept),
            "module_relevance": module_relevance
        }
        
        return {
            "template": template,
            "parameters": parameters
        }
    
    def _get_related_concepts(self, concept: str) -> List[str]:
        """
        Get concepts related to the given concept.
        
        Args:
            concept: Base concept
            
        Returns:
            List of related concepts
        """
        # Look for exact match in semantic fields
        concept_lower = concept.lower()
        if concept_lower in self.semantic_fields:
            return self.semantic_fields[concept_lower]
        
        # Look for partial matches
        for field_concept, related in self.semantic_fields.items():
            if field_concept in concept_lower or concept_lower in field_concept:
                return related
                
        # Use spaCy similarity for concepts not in our semantic fields
        related = []
        concept_doc = self.nlp(concept_lower)
        
        for field_concept in self.semantic_fields:
            field_doc = self.nlp(field_concept)
            similarity = concept_doc.similarity(field_doc)
            if similarity > 0.5:  # Threshold for similarity
                related.extend(self.semantic_fields[field_concept])
                
        return related if related else ["related topics", "similar ideas", "connected concepts"]
class ResponseGenerator:
    """
    Enhanced response generation system for Sully that leverages semantic analysis
    to produce more natural, contextually appropriate responses.
    """
    
    def __init__(self, semantic_analyzer, reasoning_engine=None, memory_system=None):
        """
        Initialize the response generator.
        
        Args:
            semantic_analyzer: SemanticStructureAnalyzer instance
            reasoning_engine: Optional reasoning engine for content generation
            memory_system: Optional memory system for context
        """
        self.analyzer = semantic_analyzer
        self.reasoning = reasoning_engine
        self.memory = memory_system
        
        # Balance parameters for different modules
        self.module_balance = {
            "math_translation": 0.3,  # Reduce from what appeared to be 0.9+
            "dream": 0.6,
            "fusion": 0.5,
            "paradox": 0.4,
            "conversation": 0.9,
            "memory": 0.6, 
            "logic": 0.5
        }
        
        # Response assembly configurations
        self.structural_variations = [
            # Standard pattern
            ["main_response", "bridge"],
            # With follow-up
            ["main_response", "bridge", "question"],
            # With elaboration
            ["main_response", "elaboration"],
            # Simple direct
            ["main_response"],
            # Complex with reference to previous
            ["reference_previous", "main_response", "bridge"]
        ]
    
    def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a natural, contextually appropriate response.
        
        Args:
            user_input: User's input text
            context: Optional conversation context
            
        Returns:
            Generated response
        """
        # Analyze input
        analysis = self.analyzer.analyze_text(user_input)
        
        # Get template recommendations
        template_info = self.analyzer.generate_response_template(analysis)
        
        # Get relevant concepts
        concepts = [c for c, _ in analysis["weighted_concepts"]]
        main_concept = concepts[0] if concepts else "this topic"
        
        # Determine which modules to use based on relevance
        module_choices = []
        for module, relevance in analysis["module_relevance"].items():
            # Apply balance factor
            adjusted_relevance = relevance * self.module_balance.get(module, 0.5)
            if adjusted_relevance > 0.2:  # Threshold for inclusion
                module_choices.append((module, adjusted_relevance))
                
        # Sort by adjusted relevance
        module_choices.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 2 modules max
        module_choices = module_choices[:2]
        
        # Select response structure pattern
        structure_pattern = np.random.choice(self.structural_variations)
        
        # Build response
        response_parts = []
        
        for part_type in structure_pattern:
            if part_type == "main_response":
                # Generate main response about the primary concept
                if self.reasoning:
                    # If we have a reasoning engine, use it
                    main_content = self._generate_with_reasoning(main_concept, module_choices, analysis)
                else:
                    # Otherwise use a template
                    template = template_info["template"]
                    params = template_info["parameters"]
                    main_content = template.format(
                        concept=main_concept,
                        insight="this concept has multiple aspects worth exploring"
                    )
                
                response_parts.append(main_content)
                
            elif part_type == "bridge":
                # Add a bridge to related concept
                related = self.analyzer._get_related_concepts(main_concept)
                if related:
                    related_concept = related[0]
                    bridge_templates = self.analyzer.templates["bridge"]
                    bridge = np.random.choice(bridge_templates).format(
                        concept=main_concept,
                        related_concept=related_concept,
                        connection="they share underlying principles"
                    )
                    response_parts.append(bridge)
                    
            elif part_type == "question":
                # Add a follow-up question
                question_templates = [
                    "What aspects of {concept} do you find most intriguing?",
                    "How do you see {concept} relating to your experiences?",
                    "Have you considered how {concept} might evolve in the future?",
                    "What led you to think about {concept} today?"
                ]
                question = np.random.choice(question_templates).format(concept=main_concept)
                response_parts.append(question)
                
            elif part_type == "elaboration":
                # Add elaboration on the main concept
                elaboration_templates = self.analyzer.templates["elaboration"]
                elaboration = np.random.choice(elaboration_templates).format(
                    concept=main_concept,
                    insight="there are multiple dimensions to consider beyond the obvious"
                )
                response_parts.append(elaboration)
                
            elif part_type == "reference_previous":
                # Reference previous topic if in context
                if context and "previous_topics" in context and context["previous_topics"]:
                    prev_topic = context["previous_topics"][0]
                    reference = f"You asked earlier about {prev_topic}. "
                    response_parts.append(reference)
        
        # Join the response parts
        full_response = " ".join(response_parts)
        
        return full_response
    
    def _generate_with_reasoning(self, concept: str, module_choices: List[Tuple[str, float]], 
                               analysis: Dict[str, Any]) -> str:
        """
        Generate response content using the reasoning engine.
        
        Args:
            concept: Main concept to address
            module_choices: List of (module, relevance) tuples
            analysis: Semantic analysis results
            
        Returns:
            Generated content
        """
        # If no reasoning engine, return template-based response
        if not self.reasoning:
            return f"The concept of {concept} has multiple interesting dimensions to explore."
            
        # Get the top module
        top_module = module_choices[0][0] if module_choices else "conversation"
        
        # Create a mapping of modules to cognitive tones
        module_to_tone = {
            "math_translation": "analytical",
            "dream": "creative",
            "fusion": "integrative",
            "paradox": "dialectical",
            "conversation": "conversational", 
            "memory": "reflective",
            "logic": "logical"
        }
        
        # Select appropriate tone based on top module
        tone = module_to_tone.get(top_module, "conversational")
        
        # Generate content with reasoning engine
        try:
            content = self.reasoning.reason(f"Discuss the concept of {concept}", tone)
            
            # If content is a dictionary with a 'response' key, extract it
            if isinstance(content, dict) and 'response' in content:
                content = content['response']
                
            return content
        except:
            # Fallback if reasoning fails
            return f"The concept of {concept} can be understood from multiple perspectives."
# Example usage:
def enhance_sully_language_capabilities(sully_instance):
    # Initialize the language processor
    language_processor = EnhancedLanguageProcessor()
    
    # Integrate with Sully
    language_processor.integrate_with_sully(sully_instance)
    
    return language_processor

# Then somewhere in your main code:
# enhanced_processor = enhance_sully_language_capabilities(sully)