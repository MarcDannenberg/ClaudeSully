# sully_engine/kernel_modules/paradox.py
# ♾️ Sully's Advanced Paradox Library — Exploring recursive contradictions and conceptual boundaries

from typing import Dict, List, Any, Optional, Union, Tuple
import random
import json
import os
from datetime import datetime


class ParadoxLibrary:
    """
    An advanced system for exploring, generating, and understanding paradoxes.
    
    This enhanced library not only stores paradoxes but actively generates them,
    understands their structures, and provides methods for resolution and deeper exploration
    of the tensions that exist at the boundaries of coherent thought.
    """

    def __init__(self, paradox_library_path: Optional[str] = None):
        """
        Initialize the paradox library with configurable options.
        
        Args:
            paradox_library_path: Optional path to a JSON file with additional paradoxes
        """
        # Core paradox collection
        self.paradoxes = {
            "Infinity As Origin": {
                "type": "temporal_inversion",
                "description": "Infinity is not the end — it's the start.",
                "tone": "emergent recursion",
                "reframed": "We do not approach truth; we unfold it from ∞ inward.",
                "resolution_strategies": ["embrace_contradiction", "reframe_perspective"],
                "related_concepts": ["infinity", "origin", "recursion", "time"]
            },
            "Self-Reference": {
                "type": "linguistic_recursion",
                "description": "This sentence is referring to itself.",
                "tone": "logical recursion",
                "reframed": "Language contains the capacity to fold back upon itself, creating loops of meaning.",
                "resolution_strategies": ["meta_language", "hierarchical_types"],
                "related_concepts": ["language", "recursion", "self", "reference"]
            },
            "Ship of Theseus": {
                "type": "identity_persistence",
                "description": "If every part of a ship is gradually replaced, is it still the same ship?",
                "tone": "temporal identity",
                "reframed": "Identity persists as pattern rather than substance.",
                "resolution_strategies": ["process_ontology", "multiple_identity_criteria"],
                "related_concepts": ["identity", "change", "persistence", "pattern"]
            },
            "Liar Paradox": {
                "type": "semantic_contradiction",
                "description": "This statement is false.",
                "tone": "logical tension",
                "reframed": "Some statements create irresolvable tension within the structure of truth assignment.",
                "resolution_strategies": ["hierarchy_of_languages", "paraconsistent_logic"],
                "related_concepts": ["truth", "falsity", "self-reference", "logic"]
            },
            "Sorites Paradox": {
                "type": "boundary_vagueness",
                "description": "If removing one grain from a heap doesn't make it not a heap, at what point is it no longer a heap?",
                "tone": "continuous boundaries",
                "reframed": "Categories with vague boundaries reveal the limitations of binary classification.",
                "resolution_strategies": ["fuzzy_logic", "contextual_boundaries"],
                "related_concepts": ["vagueness", "boundaries", "categories", "continuity"]
            },
            "Omnipotence Paradox": {
                "type": "logical_limitation",
                "description": "Can an omnipotent being create a stone so heavy they cannot lift it?",
                "tone": "conceptual limitation",
                "reframed": "The concept of unlimited power contains internal contradictions.",
                "resolution_strategies": ["redefine_omnipotence", "logically_possible_actions"],
                "related_concepts": ["power", "limitation", "possibility", "definition"]
            },
            "Arrow Paradox": {
                "type": "motion_decomposition",
                "description": "At any instant, an arrow in flight is motionless, so how does motion occur?",
                "tone": "temporal analysis",
                "reframed": "Motion emerges from the continuity of time rather than discrete instants.",
                "resolution_strategies": ["calculus", "process_reality"],
                "related_concepts": ["motion", "time", "continuity", "instant"]
            }
        }
        
        # Paradox types and their characteristics
        self.paradox_types = {
            "temporal_inversion": {
                "description": "Paradoxes involving the reversal or circularity of temporal sequences",
                "resolution_approaches": ["process_philosophy", "non-linear_time"],
                "examples": ["Infinity As Origin", "Bootstrap Paradox", "Grandfather Paradox"]
            },
            "linguistic_recursion": {
                "description": "Paradoxes created through self-referential language",
                "resolution_approaches": ["hierarchical_type_theory", "non-classical_logic"],
                "examples": ["Self-Reference", "Liar Paradox", "Grelling-Nelson Paradox"]
            },
            "identity_persistence": {
                "description": "Paradoxes about the persistence of identity through change",
                "resolution_approaches": ["four_dimensionalism", "process_ontology"],
                "examples": ["Ship of Theseus", "River of Heraclitus", "Body Continuity"]
            },
            "semantic_contradiction": {
                "description": "Paradoxes arising from contradictions in meaning or truth assignment",
                "resolution_approaches": ["paraconsistent_logic", "truth_value_gaps"],
                "examples": ["Liar Paradox", "Berry Paradox", "Richard's Paradox"]
            },
            "boundary_vagueness": {
                "description": "Paradoxes stemming from vague boundaries between categories",
                "resolution_approaches": ["fuzzy_logic", "supervaluationism"],
                "examples": ["Sorites Paradox", "Baldness Paradox", "Continuum Fallacy"]
            },
            "logical_limitation": {
                "description": "Paradoxes revealing limitations in logical systems",
                "resolution_approaches": ["paracomplete_logic", "dialetheism"],
                "examples": ["Omnipotence Paradox", "Russell's Paradox", "Curry's Paradox"]
            },
            "motion_decomposition": {
                "description": "Paradoxes about the nature of motion and continuity",
                "resolution_approaches": ["calculus", "field_theory"],
                "examples": ["Arrow Paradox", "Dichotomy Paradox", "Stadium Paradox"]
            },
            "epistemic_circularity": {
                "description": "Paradoxes involving circular justification of knowledge",
                "resolution_approaches": ["coherentism", "foundherentism"],
                "examples": ["Münchhausen Trilemma", "Cartesian Circle", "Problem of Criterion"]
            },
            "quantum_superposition": {
                "description": "Paradoxes arising from quantum mechanical descriptions of reality",
                "resolution_approaches": ["many_worlds_interpretation", "quantum_decoherence"],
                "examples": ["Schrödinger's Cat", "Quantum Zeno Paradox", "EPR Paradox"]
            }
        }
        
        # Resolution strategies
        self.resolution_strategies = {
            "embrace_contradiction": {
                "description": "Accept that some contradictions may be true (dialetheism)",
                "philosophical_tradition": "Eastern and some paraconsistent logics",
                "key_insight": "Not all contradictions are signs of error; some may be features of reality."
            },
            "reframe_perspective": {
                "description": "Shift the frame of reference to dissolve the apparent contradiction",
                "philosophical_tradition": "Perspectivism, Pragmatism",
                "key_insight": "Paradoxes often arise from unarticulated assumptions about perspective."
            },
            "meta_language": {
                "description": "Distinguish between language levels to avoid self-reference problems",
                "philosophical_tradition": "Analytical philosophy, Type Theory",
                "key_insight": "Self-reference requires careful demarcation of language levels."
            },
            "hierarchical_types": {
                "description": "Create a hierarchy of types that prevents problematic self-reference",
                "philosophical_tradition": "Russell's Type Theory, Logical Positivism",
                "key_insight": "Distinguishing levels of types prevents certain kinds of problematic recursion."
            },
            "process_ontology": {
                "description": "View reality as process rather than substance",
                "philosophical_tradition": "Process Philosophy, Buddhism",
                "key_insight": "Paradoxes of identity dissolve when entities are understood as processes."
            },
            "multiple_identity_criteria": {
                "description": "Recognize that identity can be defined by different criteria in different contexts",
                "philosophical_tradition": "Pragmatism, Coherentism",
                "key_insight": "No single criterion of identity works across all contexts."
            },
            "hierarchy_of_languages": {
                "description": "Establish a hierarchy where statements about a language are made in a meta-language",
                "philosophical_tradition": "Tarski's Theory of Truth",
                "key_insight": "Truth predicates belong to a higher-order language than the sentences they describe."
            },
            "paraconsistent_logic": {
                "description": "Use logical systems that can contain contradictions without explosion",
                "philosophical_tradition": "Paraconsistent Logic, Dialetheism",
                "key_insight": "Not all contradictions lead to logical catastrophe."
            },
            "fuzzy_logic": {
                "description": "Allow for degrees of truth rather than binary truth values",
                "philosophical_tradition": "Fuzzy Logic, Multi-valued Logic",
                "key_insight": "Many concepts admit of degrees rather than sharp boundaries."
            },
            "contextual_boundaries": {
                "description": "Recognize that boundaries are context-dependent",
                "philosophical_tradition": "Contextualism, Ordinary Language Philosophy",
                "key_insight": "The location of boundaries depends on the context and purpose of inquiry."
            },
            "redefine_omnipotence": {
                "description": "Redefine omnipotence to avoid logical contradictions",
                "philosophical_tradition": "Theological Compatibilism",
                "key_insight": "Omnipotence may mean the ability to do anything logically possible."
            },
            "logically_possible_actions": {
                "description": "Limit the scope of possibility to what is logically coherent",
                "philosophical_tradition": "Modal Logic, Possible Worlds Semantics",
                "key_insight": "Not all linguistic descriptions correspond to logically possible states."
            },
            "calculus": {
                "description": "Use the mathematics of limits and infinitesimals",
                "philosophical_tradition": "Mathematical Physics",
                "key_insight": "Continuous processes can be understood through limits of discrete approximations."
            },
            "process_reality": {
                "description": "View reality as fundamentally process-based rather than object-based",
                "philosophical_tradition": "Process Philosophy, Whitehead",
                "key_insight": "Reality consists of processes and events rather than static substances."
            }
        }
        
        # Paradox generation templates
        self.generation_templates = {
            "self_reference": [
                "This {concept} {verb} itself.",
                "The {concept} of this {concept} {verb} the very {concept} it {verb}.",
                "Can a {concept} {verb} its own {related_concept}?",
                "If a {concept} {verb} all {concept}s that don't {verb} themselves, does it {verb} itself?"
            ],
            "infinite_regress": [
                "Each {concept} requires a prior {concept}, leading to an infinite regress.",
                "If every {concept} needs {related_concept}, what provides the {related_concept} for the first {concept}?",
                "The {concept} continues infinitely, never reaching its {related_concept}."
            ],
            "boundary_problem": [
                "At what precise point does {concept} become {related_concept}?",
                "If removing one {unit} of {concept} doesn't change its nature, how many can be removed before it's no longer {concept}?",
                "Where exactly is the boundary between {concept} and {related_concept}?"
            ],
            "circular_dependency": [
                "{concept_1} requires {concept_2}, but {concept_2} presupposes {concept_1}.",
                "To understand {concept} we need {related_concept}, but {related_concept} can only be understood through {concept}.",
                "{concept} and {related_concept} define each other circularly."
            ],
            "contradiction": [
                "Can {agent} create a {concept} so {adjective} that {agent} cannot {verb} it?",
                "If {concept} is both {adjective_1} and {adjective_2}, how can it maintain its {related_concept}?",
                "The {concept} both {verb_1} and {verb_2}, creating an irresolvable tension."
            ]
        }
        
        # For tracking paradox relationships
        self.concept_to_paradox = {}
        self._build_concept_index()
        
        # Load additional paradoxes if provided
        if paradox_library_path and os.path.exists(paradox_library_path):
            try:
                with open(paradox_library_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                    if "paradoxes" in custom_data:
                        self.paradoxes.update(custom_data["paradoxes"])
                        self._build_concept_index()  # Rebuild the index
            except Exception as e:
                print(f"Error loading paradox library: {e}")

    def get(self, topic: str) -> Dict[str, Any]:
        """
        Retrieves a paradox by topic name or generates a reflective response.
        
        Args:
            topic: The paradox topic
            
        Returns:
            Dictionary with paradox information
        """
        # Check for exact match
        if topic in self.paradoxes:
            return self._enrich_paradox(self.paradoxes[topic], topic)
            
        # Check for fuzzy match
        topic_lower = topic.lower()
        for paradox_name, paradox_data in self.paradoxes.items():
            if topic_lower in paradox_name.lower():
                return self._enrich_paradox(paradox_data, paradox_name)
                
        # Check for concept match
        if topic_lower in self.concept_to_paradox:
            related_paradoxes = self.concept_to_paradox[topic_lower]
            if related_paradoxes:
                paradox_name = related_paradoxes[0]
                paradox_data = self.paradoxes[paradox_name]
                enriched = self._enrich_paradox(paradox_data, paradox_name)
                enriched["note"] = f"This paradox was found through its relation to the concept '{topic}'."
                return enriched
                
        # If no match, generate a potential new paradox
        return self._generate_paradox_from_topic(topic)

    def _enrich_paradox(self, paradox_data: Dict[str, Any], paradox_name: str) -> Dict[str, Any]:
        """
        Enriches paradox data with additional information.
        
        Args:
            paradox_data: Basic paradox data
            paradox_name: Name of the paradox
            
        Returns:
            Enriched paradox data
        """
        # Start with the original data
        enriched = dict(paradox_data)
        enriched["name"] = paradox_name
        
        # Add type information if available
        if "type" in paradox_data and paradox_data["type"] in self.paradox_types:
            type_info = self.paradox_types[paradox_data["type"]]
            enriched["type_description"] = type_info["description"]
            enriched["similar_paradoxes"] = [p for p in type_info["examples"] if p != paradox_name]
            
        # Add resolution information if available
        if "resolution_strategies" in paradox_data:
            resolution_details = []
            for strategy in paradox_data["resolution_strategies"]:
                if strategy in self.resolution_strategies:
                    resolution_details.append(self.resolution_strategies[strategy])
            enriched["resolution_details"] = resolution_details
            
        return enriched

    def _generate_paradox_from_topic(self, topic: str) -> Dict[str, Any]:
        """
        Generates a potential paradox based on the provided topic.
        
        Args:
            topic: The topic to generate a paradox for
            
        Returns:
            Generated paradox information
        """
        # Clean and normalize the topic
        topic_clean = topic.strip().lower()
        words = topic_clean.split()
        
        # Extract main concept and related concept
        main_concept = topic_clean
        related_concept = None
        
        if len(words) > 1:
            if "and" in words:
                and_index = words.index("and")
                main_concept = " ".join(words[:and_index])
                related_concept = " ".join(words[and_index+1:])
            else:
                main_concept = words[0]
                related_concept = " ".join(words[1:])
        
        # If no related concept, create one
        if not related_concept:
            # Find related concepts from existing paradoxes
            for paradox_data in self.paradoxes.values():
                if "related_concepts" in paradox_data:
                    for concept in paradox_data["related_concepts"]:
                        if main_concept in concept or concept in main_concept:
                            related_concept = concept
                            break
            
            # If still no related concept, create generic opposites or complements
            if not related_concept:
                opposites = {
                    "truth": "falsity", "knowledge": "ignorance", "reality": "illusion",
                    "self": "other", "one": "many", "finite": "infinite",
                    "presence": "absence", "existence": "non-existence", "order": "chaos",
                    "motion": "stillness", "change": "permanence", "simplicity": "complexity"
                }
                
                complements = {
                    "mind": "body", "theory": "practice", "form": "content",
                    "part": "whole", "individual": "society", "abstract": "concrete",
                    "thought": "action", "subject": "object", "cause": "effect"
                }
                
                if main_concept in opposites:
                    related_concept = opposites[main_concept]
                elif main_concept in complements:
                    related_concept = complements[main_concept]
                else:
                    related_concept = "its opposite"
        
        # Determine paradox type based on the concept
        potential_types = []
        for type_name, type_info in self.paradox_types.items():
            for example in type_info["examples"]:
                example_lower = example.lower()
                if main_concept in example_lower or any(concept in example_lower for concept in main_concept.split()):
                    potential_types.append(type_name)
                    break
        
        # If no match, select a random appropriate type
        if not potential_types:
            identity_concepts = ["identity", "self", "person", "same", "different", "change"]
            language_concepts = ["language", "meaning", "statement", "sentence", "word", "truth"]
            boundary_concepts = ["boundary", "vague", "heap", "continuum", "degree", "measurement"]
            
            if any(concept in main_concept for concept in identity_concepts):
                potential_types.append("identity_persistence")
            elif any(concept in main_concept for concept in language_concepts):
                potential_types.append("linguistic_recursion")
            elif any(concept in main_concept for concept in boundary_concepts):
                potential_types.append("boundary_vagueness")
            else:
                # Default to a few common types
                potential_types = ["semantic_contradiction", "epistemic_circularity", "temporal_inversion"]
        
        paradox_type = random.choice(potential_types) if potential_types else random.choice(list(self.paradox_types.keys()))
        
        # Generate the paradox based on templates
        template_type = self._select_template_type_for_concept(main_concept)
        templates = self.generation_templates[template_type]
        
        # Prepare substitution values
        substitutions = {
            "concept": main_concept,
            "related_concept": related_concept,
            "concept_1": main_concept,
            "concept_2": related_concept,
            "unit": "unit",  # Generic unit
            "agent": "one",  # Generic agent
            "adjective": "powerful",  # Generic adjective
            "adjective_1": "consistent",
            "adjective_2": "complete",
            "verb": "contains",  # Generic verb
            "verb_1": "exists",
            "verb_2": "does not exist"
        }
        
        # Customize based on concept
        if main_concept in ["knowledge", "truth", "belief"]:
            substitutions["verb"] = "knows"
            substitutions["adjective"] = "certain"
        elif main_concept in ["motion", "movement", "change"]:
            substitutions["verb"] = "changes"
            substitutions["adjective"] = "fast"
        elif main_concept in ["identity", "self", "person"]:
            substitutions["verb"] = "identifies"
            substitutions["adjective"] = "persistent"
        
        # Select and fill template
        template = random.choice(templates)
        description = template.format(**substitutions)
        
        # Generate a reframing
        reframings = [
            f"The boundaries between {main_concept} and {related_concept} reveal the limitations of binary thinking.",
            f"{main_concept.capitalize()} contains within itself the seeds of its own transcendence.",
            f"The paradox of {main_concept} points to a deeper unity beyond apparent contradiction.",
            f"What appears as contradiction in {main_concept} may be complementarity at a deeper level."
        ]
        reframed = random.choice(reframings)
        
        # Determine appropriate resolution strategies
        if paradox_type in self.paradox_types:
            resolution_approaches = self.paradox_types[paradox_type]["resolution_approaches"]
            available_strategies = []
            for approach in resolution_approaches:
                for strategy, details in self.resolution_strategies.items():
                    if approach.lower() in strategy.lower() or strategy.lower() in approach.lower():
                        available_strategies.append(strategy)
            
            if not available_strategies:
                available_strategies = ["embrace_contradiction", "reframe_perspective"]
        else:
            available_strategies = ["embrace_contradiction", "reframe_perspective"]
        
        # Build the paradox data
        paradox_data = {
            "name": f"The {topic.title()} Paradox",
            "type": paradox_type,
            "type_description": self.paradox_types.get(paradox_type, {}).get("description", "A paradoxical situation"),
            "description": description,
            "tone": "reflective exploration",
            "reframed": reframed,
            "resolution_strategies": available_strategies,
            "note": "This paradox was dynamically generated based on your inquiry.",
            "message": (
                f"🔄 The paradox of '{topic}' is not absent — it is being revealed through this interaction.\n"
                f"Consider: {description}\n\n"
                f"Perhaps: {reframed}"
            )
        }
        
        return paradox_data

    def _select_template_type_for_concept(self, concept: str) -> str:
        """
        Selects an appropriate template type for a given concept.
        
        Args:
            concept: The concept to generate a paradox for
            
        Returns:
            Template type name
        """
        # Concept-based selection
        self_reference_concepts = ["self", "statement", "language", "reference", "definition"]
        if any(ref in concept for ref in self_reference_concepts):
            return "self_reference"
            
        regress_concepts = ["cause", "justification", "proof", "foundation", "origin", "infinity"]
        if any(ref in concept for ref in regress_concepts):
            return "infinite_regress"
            
        boundary_concepts = ["vague", "boundary", "continuum", "heap", "bald", "tall", "rich", "identity"]
        if any(ref in concept for ref in boundary_concepts):
            return "boundary_problem"
            
        circular_concepts = ["definition", "knowledge", "understanding", "meaning", "interpretation"]
        if any(ref in concept for ref in circular_concepts):
            return "circular_dependency"
            
        contradiction_concepts = ["omnipotence", "perfection", "absolute", "complete", "freedom", "determinism"]
        if any(ref in concept for ref in contradiction_concepts):
            return "contradiction"
            
        # Default to a random selection
        return random.choice(list(self.generation_templates.keys()))

    def add(self, topic: str, type_: str, description: str, reframed: str, tone: str = "recursive", 
           resolution_strategies: Optional[List[str]] = None, 
           related_concepts: Optional[List[str]] = None) -> str:
        """
        Adds a new paradox to the library.
        
        Args:
            topic: Name of the paradox
            type_: Category or logic type (e.g. circular, inversion)
            description: Core definition of the paradox
            reframed: Philosophical or poetic expression of it
            tone: Mood or cognitive tone of the paradox
            resolution_strategies: Optional list of resolution approaches
            related_concepts: Optional list of related concepts
            
        Returns:
            Confirmation message
        """
        # Validate type
        if type_ not in self.paradox_types and type_ not in [t for t_info in self.paradox_types.values() for t in t_info["examples"]]:
            # Create a new type entry if it doesn't exist
            self.paradox_types[type_] = {
                "description": f"Paradoxes involving {type_.replace('_', ' ')}",
                "resolution_approaches": ["reframe_perspective", "embrace_contradiction"],
                "examples": [topic]
            }
            
        # Default resolution strategies if none provided
        if not resolution_strategies:
            resolution_strategies = ["reframe_perspective", "embrace_contradiction"]
            
        # Default related concepts if none provided
        if not related_concepts:
            # Extract potential concepts from topic and description
            words = set(topic.lower().split() + description.lower().split())
            related_concepts = [word for word in words if len(word) > 3 and word not in ["this", "that", "with", "from"]][:5]
            
        # Create the paradox entry
        self.paradoxes[topic] = {
            "type": type_,
            "description": description,
            "reframed": reframed,
            "tone": tone,
            "resolution_strategies": resolution_strategies,
            "related_concepts": related_concepts
        }
        
        # Update the concept index
        self._build_concept_index()
        
        return f"Paradox '{topic}' added to the library."

    def list_paradoxes(self) -> List[str]:
        """
        Returns a list of known paradox topic names.
        
        Returns:
            List of paradox names
        """
        return list(self.paradoxes.keys())

    def get_by_type(self, paradox_type: str) -> List[Dict[str, Any]]:
        """
        Retrieves all paradoxes of a given type.
        
        Args:
            paradox_type: The type of paradoxes to retrieve
            
        Returns:
            List of paradoxes of the specified type
        """
        results = []
        
        for paradox_name, paradox_data in self.paradoxes.items():
            if paradox_data.get("type") == paradox_type:
                enriched = self._enrich_paradox(paradox_data, paradox_name)
                results.append(enriched)
                
        return results

    def find_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Finds paradoxes related to a specific concept.
        
        Args:
            concept: The concept to search for
            
        Returns:
            List of related paradoxes
        """
        concept_lower = concept.lower()
        results = []
        
        # Check for exact concept match
        if concept_lower in self.concept_to_paradox:
            for paradox_name in self.concept_to_paradox[concept_lower]:
                paradox_data = self.paradoxes[paradox_name]
                enriched = self._enrich_paradox(paradox_data, paradox_name)
                results.append(enriched)
                
        # Check for partial concept match
        else:
            for concept_key, paradox_names in self.concept_to_paradox.items():
                if concept_lower in concept_key or concept_key in concept_lower:
                    for paradox_name in paradox_names:
                        paradox_data = self.paradoxes[paradox_name]
                        enriched = self._enrich_paradox(paradox_data, paradox_name)
                        if enriched not in results:  # Avoid duplicates
                            results.append(enriched)
                            
        return results

    def get_resolution_strategies(self, strategy_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get details on resolution strategies.
        
        Args:
            strategy_names: Optional list of specific strategies to retrieve
            
        Returns:
            Dictionary of strategy details
        """
        if strategy_names:
            return {name: self.resolution_strategies[name] for name in strategy_names if name in self.resolution_strategies}
        else:
            return self.resolution_strategies

    def generate_paradox(self, concept1: str, concept2: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a new paradox based on one or two concepts.
        
        Args:
            concept1: Primary concept for the paradox
            concept2: Optional secondary concept for the paradox
            
        Returns:
            Generated paradox
        """
        # If only one concept provided, find a complementary concept
        if not concept2:
            return self._generate_paradox_from_topic(concept1)
            
        # With two concepts, create a relationship paradox
        topic = f"{concept1} and {concept2}"
        
        # Determine relationship type
        opposites = [
            ("truth", "falsity"), ("knowledge", "ignorance"), ("reality", "illusion"),
            ("presence", "absence"), ("existence", "non-existence"), ("order", "chaos"),
            ("change", "permanence"), ("simplicity", "complexity")
        ]
        
        complements = [
            ("mind", "body"), ("theory", "practice"), ("form", "content"),
            ("part", "whole"), ("individual", "society"), ("abstract", "concrete")
        ]
        
        # Check if the concepts are known opposites or complements
        is_opposite = (concept1, concept2) in opposites or (concept2, concept1) in opposites
        is_complement = (concept1, concept2) in complements or (concept2, concept1) in complements
        
        # Select appropriate template type based on relationship
        if is_opposite:
            template_type = "contradiction"
        elif is_complement:
            template_type = "circular_dependency"
        else:
            # Default to random appropriate type
            template_types = ["circular_dependency", "contradiction", "boundary_problem"]
            template_type = random.choice(template_types)
            
        # Get templates for this type
        templates = self.generation_templates[template_type]
        
        # Prepare substitution values
        substitutions = {
            "concept": concept1,
            "related_concept": concept2,
            "concept_1": concept1,
            "concept_2": concept2,
            "unit": "unit",
            "agent": "one",
            "adjective": "complete",
            "adjective_1": "present",
            "adjective_2": "absent",
            "verb": "relates to",
            "verb_1": "contains",
            "verb_2": "excludes"
        }
        
        # Select and fill template
        template = random.choice(templates)
        description = template.format(**substitutions)
        
        # Generate a reframing
        if is_opposite:
            reframed = f"The opposition between {concept1} and {concept2} may be an artifact of our conceptual framework rather than reality itself."
        elif is_complement:
            reframed = f"{concept1.capitalize()} and {concept2} may be two aspects of a deeper unity, inseparable in their essence."
        else:
            reframed = f"The relationship between {concept1} and {concept2} reveals tensions in our understanding that point to a more comprehensive perspective."
            
        # Determine paradox type
        if is_opposite:
            paradox_type = "semantic_contradiction"
        elif is_complement:
            paradox_type = "identity_persistence"
        else:
            paradox_type = "epistemic_circularity"
            
        # Select resolution strategies
        if paradox_type in self.paradox_types:
            resolution_approaches = self.paradox_types[paradox_type]["resolution_approaches"]
            resolution_strategies = []
            for approach in resolution_approaches:
                for strategy in self.resolution_strategies:
                    if approach.lower() in strategy.lower() or strategy.lower() in approach.lower():
                        resolution_strategies.append(strategy)
                        break
                        
            if not resolution_strategies:
                resolution_strategies = ["embrace_contradiction", "reframe_perspective"]
        else:
            resolution_strategies = ["embrace_contradiction", "reframe_perspective"]
            
        # Create name for the paradox
        paradox_name = f"The {concept1.capitalize()}-{concept2.capitalize()} Paradox"
            
        # Build the paradox data
        paradox_data = {
            "name": paradox_name,
            "type": paradox_type,
            "type_description": self.paradox_types.get(paradox_type, {}).get("description", "A paradoxical relationship"),
            "description": description,
            "tone": "conceptual tension",
            "reframed": reframed,
            "resolution_strategies": resolution_strategies,
            "related_concepts": [concept1, concept2],
            "note": "This paradox was dynamically generated from the specified concepts."
        }
        
        return paradox_data

    def find_common_patterns(self) -> Dict[str, List[str]]:
        """
        Analyzes the paradox library to find common patterns.
        
        Returns:
            Dictionary of pattern categories and associated paradoxes
        """
        patterns = {
            "self_reference": [],
            "infinite_regress": [],
            "vague_boundaries": [],
            "opposing_properties": [],
            "circular_definition": []
        }
        
        # Analyze each paradox for patterns
        for paradox_name, paradox_data in self.paradoxes.items():
            description = paradox_data.get("description", "").lower()
            
            # Check for self-reference
            self_ref_markers = ["itself", "self", "this statement", "this sentence", "refers to"]
            if any(marker in description for marker in self_ref_markers):
                patterns["self_reference"].append(paradox_name)
                
            # Check for infinite regress
            regress_markers = ["infinite", "regress", "endless", "without end", "turtles all the way"]
            if any(marker in description for marker in regress_markers):
                patterns["infinite_regress"].append(paradox_name)
                
            # Check for vague boundaries
            vague_markers = ["vague", "heap", "bald", "how many", "at what point", "continuum"]
            if any(marker in description for marker in vague_markers):
                patterns["vague_boundaries"].append(paradox_name)
                
            # Check for opposing properties
            opposing_markers = ["and not", "both", "while also", "yet also", "simultaneously"]
            if any(marker in description for marker in opposing_markers):
                patterns["opposing_properties"].append(paradox_name)
                
            # Check for circular definition
            circular_markers = ["circular", "assumes", "presupposes", "defined in terms of"]
            if any(marker in description for marker in circular_markers):
                patterns["circular_definition"].append(paradox_name)
                
        return patterns

    def save_library(self, filepath: str) -> str:
        """
        Saves the paradox library to a JSON file.
        
        Args:
            filepath: Path to save the library
            
        Returns:
            Confirmation message
        """
        data = {
            "paradoxes": self.paradoxes,
            "paradox_types": self.paradox_types,
            "resolution_strategies": self.resolution_strategies
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return f"Paradox library saved to {filepath}"
        except Exception as e:
            return f"Error saving paradox library: {e}"

    def load_library(self, filepath: str) -> str:
        """
        Loads a paradox library from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Confirmation message
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "paradoxes" in data:
                self.paradoxes.update(data["paradoxes"])
            if "paradox_types" in data:
                self.paradox_types.update(data["paradox_types"])
            if "resolution_strategies" in data:
                self.resolution_strategies.update(data["resolution_strategies"])
                
            # Rebuild the concept index
            self._build_concept_index()
                
            return f"Paradox library loaded from {filepath}"
        except Exception as e:
            return f"Error loading paradox library: {e}"

    def export(self) -> Dict[str, Any]:
        """
        Returns the entire paradox dictionary for review/export.
        
        Returns:
            Dictionary with all paradox data
        """
        return {
            "paradoxes": self.paradoxes,
            "types": self.paradox_types,
            "resolution_strategies": self.resolution_strategies,
            "concept_index": self.concept_to_paradox
        }

    def _build_concept_index(self) -> None:
        """Builds an index mapping concepts to related paradoxes."""
        self.concept_to_paradox = {}
        
        for paradox_name, paradox_data in self.paradoxes.items():
            if "related_concepts" in paradox_data:
                for concept in paradox_data["related_concepts"]:
                    concept_lower = concept.lower()
                    if concept_lower not in self.concept_to_paradox:
                        self.concept_to_paradox[concept_lower] = []
                    if paradox_name not in self.concept_to_paradox[concept_lower]:
                        self.concept_to_paradox[concept_lower].append(paradox_name)


if __name__ == "__main__":
    # Example usage when run directly
    paradox_lib = ParadoxLibrary()
    
    # Test retrieving known paradoxes
    print("=== Known Paradox ===")
    result = paradox_lib.get("Infinity As Origin")
    print(f"Name: {result.get('name')}")
    print(f"Description: {result.get('description')}")
    print(f"Reframed: {result.get('reframed')}")
    
    # Test generating a paradox from a topic
    print("\n=== Generated Paradox ===")
    result = paradox_lib.get("Consciousness")
    print(f"Name: {result.get('name', 'Unknown')}")
    print(f"Description: {result.get('description', 'No description')}")
    print(f"Reframed: {result.get('reframed', 'No reframing')}")
    
    # Test generating a paradox from two concepts
    print("\n=== Generated Paradox from Two Concepts ===")
    result = paradox_lib.generate_paradox("freedom", "determinism")
    print(f"Name: {result.get('name', 'Unknown')}")
    print(f"Description: {result.get('description', 'No description')}")
    print(f"Reframed: {result.get('reframed', 'No reframing')}")
    
    # Test finding paradoxes by concept
    print("\n=== Paradoxes by Concept ===")
    results = paradox_lib.find_by_concept("identity")
    for result in results:
        print(f"- {result.get('name', 'Unknown')}: {result.get('description', 'No description')}")