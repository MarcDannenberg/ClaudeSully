# sully_engine/kernel_modules/fusion.py
# 🔗 Symbol Fusion Engine — Synthesize new meaning from multiple concepts

from typing import Dict, List, Any, Optional, Union, Tuple
import random
import json
import os
from datetime import datetime
import re

class SymbolFusionEngine:
    """
    Advanced concept fusion system that combines symbolic inputs into new emergent ideas.
    
    This enhanced engine supports multiple fusion strategies, conceptual blending, 
    network tracking, and different cognitive modes for concept integration.
    """

    def __init__(self, fusion_library_path: Optional[str] = None):
        """
        Initialize the fusion engine with configurable options.
        
        Args:
            fusion_library_path: Optional path to a JSON file with additional fusion patterns
        """
        # Default fusion style
        self.default_style = "entanglement"
        
        # Fusion styles with their characteristics
        self.fusion_styles = {
            "entanglement": {
                "description": "Concepts interweave, creating emergent properties beyond original components",
                "operator": "⨁",
                "connectors": ["becomes entangled with", "interweaves with", "resonates alongside"]
            },
            "synthesis": {
                "description": "Concepts combine to form a unified whole with new properties",
                "operator": "⊕",
                "connectors": ["synthesizes with", "combines with", "forms a whole with"]
            },
            "dialectic": {
                "description": "Thesis and antithesis resolve into synthesis through productive tension",
                "operator": "⇔",
                "connectors": ["resolves with", "dialectically engages", "finds resolution in"]
            },
            "emergence": {
                "description": "New properties emerge from the interaction of simpler components",
                "operator": "⇑",
                "connectors": ["gives rise to", "emerges with", "transcends through"]
            },
            "transformation": {
                "description": "Concepts transform each other into something different",
                "operator": "⟿",
                "connectors": ["transforms with", "alchemically combines with", "transmutes alongside"]
            },
            "network": {
                "description": "Concepts form nodes in a semantic network of relationships",
                "operator": "⇄",
                "connectors": ["connects to", "networks with", "forms pathways to"]
            },
            "fractal": {
                "description": "Concepts nest within each other at different scales",
                "operator": "⥮",
                "connectors": ["nests within", "fractally contains", "recursively joins"]
            },
            "quantum": {
                "description": "Concepts exist in superposition of multiple possibilities",
                "operator": "⨂",
                "connectors": ["superimposes with", "exists in quantum relation to", "entangles probabilistically with"]
            }
        }
        
        # Fusion patterns for different concept types
        self.fusion_patterns = {
            "abstract_abstract": [
                "{c1} and {c2} merge to create a conceptual space where {result}.",
                "When {c1} intersects with {c2}, a new paradigm emerges: {result}.",
                "The fusion of {c1} and {c2} transcends both to reveal {result}.",
                "{c1} {connector} {c2}, giving rise to {result}."
            ],
            "abstract_concrete": [
                "{c1} manifests through {c2}, revealing {result}.",
                "{c2} becomes a vessel for {c1}, expressing {result}.",
                "The abstract {c1} finds form in {c2}, demonstrating {result}.",
                "{c1} {connector} {c2}, materializing as {result}."
            ],
            "concrete_concrete": [
                "{c1} and {c2} combine their properties to create {result}.",
                "The physical interaction of {c1} with {c2} produces {result}.",
                "When {c1} meets {c2} in the material realm, {result} takes form.",
                "{c1} {connector} {c2}, forming {result}."
            ],
            "opposing_concepts": [
                "The tension between {c1} and {c2} resolves into {result}.",
                "{c1} and {c2}, seemingly contradictory, find harmony in {result}.",
                "The dialectic of {c1} versus {c2} synthesizes into {result}.",
                "{c1} {connector} {c2}, reconciling as {result}."
            ],
            "complementary_concepts": [
                "{c1} enhances {c2}, together creating {result}.",
                "The complementary nature of {c1} and {c2} amplifies into {result}.",
                "{c1} and {c2} mutually reinforce to generate {result}.",
                "{c1} {connector} {c2}, harmonizing into {result}."
            ]
        }
        
        # Emergent properties that can arise from fusion
        self.emergent_properties = {
            "synthesis": [
                "a unified framework",
                "an integrated perspective",
                "a holistic understanding",
                "a synthesized approach"
            ],
            "transcendence": [
                "something that transcends its components",
                "a higher-order concept",
                "an elevated understanding",
                "a transcendent insight"
            ],
            "paradox": [
                "a productive paradox",
                "a tension of opposites",
                "a dynamic contradiction",
                "an enigmatic duality"
            ],
            "novelty": [
                "an entirely new perspective",
                "an unexpected connection",
                "a novel framework",
                "an innovative approach"
            ],
            "depth": [
                "a deeper understanding",
                "a multi-layered concept",
                "a concept with newfound depth",
                "an insight with profound implications"
            ]
        }
        
        # Cognitive mode adaptations for fusion
        self.cognitive_modes = {
            "analytical": {
                "description": "Logical, structured fusion with precise outcomes",
                "patterns": [
                    "Through structured analysis, {c1} combines with {c2} to yield {result}.",
                    "When examined systematically, {c1} and {c2} produce {result}.",
                    "Logical integration of {c1} with {c2} generates {result}."
                ]
            },
            "creative": {
                "description": "Imaginative, unexpected combinations with novel outcomes",
                "patterns": [
                    "In a creative leap, {c1} dances with {c2} to birth {result}.",
                    "Imagining {c1} through the lens of {c2} reveals the surprising {result}.",
                    "The artistic blending of {c1} and {c2} inspires {result}."
                ]
            },
            "critical": {
                "description": "Evaluative fusion that examines tensions and contradictions",
                "patterns": [
                    "Critically examining the intersection of {c1} and {c2} exposes {result}.",
                    "The tension between {c1} and {c2}, when scrutinized, yields {result}.",
                    "Through careful evaluation, {c1} and {c2} resolve into {result}."
                ]
            },
            "ethereal": {
                "description": "Abstract, philosophical blending of transcendent concepts",
                "patterns": [
                    "Beyond ordinary understanding, {c1} and {c2} transcend into {result}.",
                    "In the realm of pure concept, {c1} merges with {c2} to reveal {result}.",
                    "The essence of {c1}, when unified with {c2}, illuminates {result}."
                ]
            }
        }
        
        # Concept categorization hints
        self.concept_categories = {
            "abstract": [
                "truth", "beauty", "justice", "freedom", "love", "time", "space", 
                "infinity", "knowledge", "wisdom", "ethics", "consciousness", 
                "existence", "meaning", "purpose", "reality", "possibility",
                "concept", "idea", "theory", "philosophy", "principle"
            ],
            "concrete": [
                "tree", "stone", "water", "fire", "earth", "animal", "human",
                "building", "machine", "book", "food", "tool", "vehicle", "art",
                "city", "mountain", "river", "ocean", "forest", "desert", "device"
            ],
            "opposing_pairs": [
                ("light", "darkness"), ("order", "chaos"), ("creation", "destruction"),
                ("simplicity", "complexity"), ("unity", "diversity"), ("finite", "infinite"),
                ("freedom", "constraint"), ("stability", "change"), ("certainty", "uncertainty")
            ],
            "complementary_pairs": [
                ("theory", "practice"), ("mind", "body"), ("analysis", "synthesis"),
                ("individual", "community"), ("part", "whole"), ("question", "answer"),
                ("cause", "effect"), ("form", "function"), ("structure", "process")
            ]
        }
        
        # For tracking concept networks and fusion history
        self.fusion_history = []
        self.concept_network = {}
        
        # Load additional fusion patterns if provided
        self.custom_patterns = {}
        if fusion_library_path and os.path.exists(fusion_library_path):
            try:
                with open(fusion_library_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                    if "patterns" in custom_data:
                        self.custom_patterns = custom_data["patterns"]
                    if "styles" in custom_data:
                        self.fusion_styles.update(custom_data["styles"])
            except Exception as e:
                print(f"Error loading fusion library: {e}")

    def fuse(self, *symbols: str) -> Union[Dict[str, Any], str]:
        """
        Combines symbols into a symbolic 'fusion' with emergent properties.
        
        Args:
            *symbols: Any number of string-based symbolic terms
            
        Returns:
            Either a dictionary with fusion details or a formatted fusion string
        """
        return self.fuse_with_options(*symbols)

    def fuse_with_options(self, *symbols: str, style: Optional[str] = None, 
                         cognitive_mode: Optional[str] = None,
                         output_format: str = "string") -> Union[Dict[str, Any], str]:
        """
        Advanced fusion with configurable options for style and cognitive approach.
        
        Args:
            *symbols: Input symbolic terms
            style: Fusion style (entanglement, synthesis, dialectic, etc.)
            cognitive_mode: Cognitive approach (analytical, creative, critical, ethereal)
            output_format: Return format ("string" or "dict")
            
        Returns:
            Either a dictionary with fusion details or a formatted fusion string
        """
        if not symbols or len(symbols) < 2:
            if output_format == "string":
                return "Fusion requires at least two symbols to create something new."
            return {
                "inputs": list(symbols),
                "result": "",
                "comment": "Fusion requires at least two symbols."
            }
            
        # Use provided style or default
        fusion_style = style.lower() if style else self.default_style
        if fusion_style not in self.fusion_styles:
            fusion_style = self.default_style
            
        # Get style information
        style_info = self.fusion_styles[fusion_style]
        style_operator = style_info["operator"]
        style_connectors = style_info["connectors"]
        
        # Categorize concepts for appropriate fusion patterns
        concept_types = self._categorize_concepts(symbols)
        pattern_key = self._determine_pattern_key(concept_types)
        
        # Select appropriate patterns
        if cognitive_mode and cognitive_mode in self.cognitive_modes:
            # Use cognitive mode-specific patterns
            patterns = self.cognitive_modes[cognitive_mode]["patterns"]
        else:
            # Use regular patterns based on concept types
            patterns = self.fusion_patterns.get(pattern_key, self.fusion_patterns["abstract_abstract"])
            
        # Generate fusion result
        fusion_result = self._generate_fusion_result(symbols, fusion_style)
        
        # Format the response
        connector = random.choice(style_connectors)
        pattern = random.choice(patterns)
        
        if len(symbols) == 2:
            # For two concepts, use direct substitution
            formatted_result = pattern.format(
                c1=symbols[0], 
                c2=symbols[1], 
                result=fusion_result,
                connector=connector
            )
        else:
            # For multiple concepts, handle more complex formatting
            c1 = symbols[0]
            c2 = ", ".join(symbols[1:-1]) + " and " + symbols[-1]
            formatted_result = pattern.format(
                c1=c1, 
                c2=c2, 
                result=fusion_result,
                connector=connector
            )
            
        # Create formal representation
        formal_fusion = f"{' ' + style_operator + ' '.join(symbols)}"
        
        # Add to fusion history and concept network
        fusion_record = {
            "timestamp": datetime.now().isoformat(),
            "inputs": list(symbols),
            "style": fusion_style,
            "cognitive_mode": cognitive_mode,
            "result": fusion_result,
            "formatted_result": formatted_result,
            "formal_representation": formal_fusion
        }
        self.fusion_history.append(fusion_record)
        
        # Update concept network
        self._update_concept_network(symbols, fusion_result, fusion_style)
        
        # Return appropriate format
        if output_format == "string":
            return formatted_result
            
        return {
            "inputs": list(symbols),
            "style": fusion_style,
            "formal_representation": formal_fusion,
            "result": fusion_result,
            "formatted_result": formatted_result,
            "comment": style_info["description"]
        }

    def fuse_with_style(self, *symbols: str, style: Optional[str] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Fuses symbols using a specified style.
        
        Args:
            *symbols: Input symbolic terms
            style: Optional style override
            
        Returns:
            Dictionary with fusion results
        """
        style = style or self.default_style
        result = self.fuse_with_options(*symbols, style=style, output_format="dict")
        
        # Convert to legacy format
        return {
            "style": style,
            "inputs": result["inputs"],
            "fusion": result["formal_representation"],
            "comment": result["comment"]
        }

    def _categorize_concepts(self, concepts: Tuple[str, ...]) -> List[str]:
        """
        Categorizes input concepts as abstract, concrete, etc.
        
        Args:
            concepts: The concepts to categorize
            
        Returns:
            List of category labels for each concept
        """
        categories = []
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Check for known categories
            if concept_lower in self.concept_categories["abstract"]:
                categories.append("abstract")
            elif concept_lower in self.concept_categories["concrete"]:
                categories.append("concrete")
            else:
                # Attempt to guess based on characteristics
                if any(concept_lower.endswith(suffix) for suffix in ["ness", "ity", "ion", "ism", "dom"]):
                    categories.append("abstract")
                elif len(concept_lower) > 3 and concept_lower[0].isupper():
                    categories.append("concrete")  # Proper nouns often concrete
                else:
                    # Default categorization heuristic
                    abstract_score = 0
                    if len(concept_lower) >= 6:
                        abstract_score += 1  # Longer words tend to be more abstract
                    if any(a in concept_lower for a in ["truth", "idea", "concept", "theory"]):
                        abstract_score += 2
                    
                    categories.append("abstract" if abstract_score >= 1 else "concrete")
        
        return categories

    def _determine_pattern_key(self, categories: List[str]) -> str:
        """
        Determines the appropriate pattern key based on concept categories.
        
        Args:
            categories: List of concept categories
            
        Returns:
            Pattern key for fusion
        """
        # Check for opposing concepts
        if len(categories) == 2:
            concept_pair = tuple(categories)
            if concept_pair in self.concept_categories["opposing_pairs"]:
                return "opposing_concepts"
            if concept_pair in self.concept_categories["complementary_pairs"]:
                return "complementary_concepts"
                
        # Check category combinations
        if all(cat == "abstract" for cat in categories):
            return "abstract_abstract"
        elif all(cat == "concrete" for cat in categories):
            return "concrete_concrete"
        else:
            return "abstract_concrete"

    def _generate_fusion_result(self, symbols: Tuple[str, ...], style: str) -> str:
        """
        Generates the result of fusing the given symbols.
        
        Args:
            symbols: The input symbols to fuse
            style: The fusion style
            
        Returns:
            Description of the fusion result
        """
        # Select random emergent property type
        property_type = random.choice(list(self.emergent_properties.keys()))
        property_description = random.choice(self.emergent_properties[property_type])
        
        # Generate different fusion results based on style
        if style == "entanglement":
            return f"{property_description} where neither can be understood apart from the other"
        elif style == "synthesis":
            return f"{property_description} that preserves and transcends the original elements"
        elif style == "dialectic":
            return f"{property_description} that resolves the tension between opposing forces"
        elif style == "emergence":
            return f"{property_description} that couldn't be predicted from the parts alone"
        elif style == "transformation":
            return f"{property_description} that represents a complete metamorphosis of the original concepts"
        elif style == "network":
            return f"{property_description} that forms nodes in a larger conceptual ecosystem"
        elif style == "fractal":
            return f"{property_description} that repeats its pattern at multiple scales of understanding"
        elif style == "quantum":
            return f"{property_description} that exists simultaneously in multiple states of meaning"
        else:
            return f"{property_description} born from the interaction of diverse elements"

    def _update_concept_network(self, symbols: Tuple[str, ...], result: str, style: str) -> None:
        """
        Updates the internal concept network with new relationships.
        
        Args:
            symbols: The concepts being fused
            result: The fusion result
            style: The fusion style
        """
        # Add all concepts to the network if not present
        for symbol in symbols:
            if symbol not in self.concept_network:
                self.concept_network[symbol] = {
                    "connections": {},
                    "fusion_count": 0,
                    "first_seen": datetime.now().isoformat()
                }
            
            # Increment fusion count
            self.concept_network[symbol]["fusion_count"] += 1
        
        # Add connections between all pairs
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Add relationship in both directions
                if symbol2 not in self.concept_network[symbol1]["connections"]:
                    self.concept_network[symbol1]["connections"][symbol2] = []
                if symbol1 not in self.concept_network[symbol2]["connections"]:
                    self.concept_network[symbol2]["connections"][symbol1] = []
                
                # Record this fusion
                fusion_info = {
                    "style": style,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                self.concept_network[symbol1]["connections"][symbol2].append(fusion_info)
                self.concept_network[symbol2]["connections"][symbol1].append(fusion_info)

    def get_fusion_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns the history of fusion operations.
        
        Args:
            limit: Optional limit on number of records to return
            
        Returns:
            List of fusion records
        """
        if limit:
            return self.fusion_history[-limit:]
        return self.fusion_history

    def get_concept_network(self) -> Dict[str, Any]:
        """
        Returns the current concept network.
        
        Returns:
            Dictionary representing the concept network
        """
        return self.concept_network

    def get_concept_connections(self, concept: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Gets all connections for a specific concept.
        
        Args:
            concept: The concept to find connections for
            
        Returns:
            Dictionary of connected concepts and their relationship details
        """
        if concept in self.concept_network:
            return self.concept_network[concept]["connections"]
        return {}

    def find_path_between_concepts(self, concept1: str, concept2: str, max_depth: int = 3) -> List[List[str]]:
        """
        Finds connection paths between two concepts in the network.
        
        Args:
            concept1: First concept
            concept2: Second concept
            max_depth: Maximum path length to search
            
        Returns:
            List of possible paths between the concepts
        """
        if concept1 not in self.concept_network or concept2 not in self.concept_network:
            return []
            
        # Simple breadth-first search
        paths = []
        visited = set()
        queue = [[concept1]]
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            # Skip if we've visited this node already
            if node in visited:
                continue
                
            # Mark as visited
            visited.add(node)
            
            # Check if we've reached the destination
            if node == concept2:
                paths.append(path)
                continue
                
            # Stop if we've reached max depth
            if len(path) >= max_depth:
                continue
                
            # Add all connected nodes to the queue
            if node in self.concept_network:
                for connected in self.concept_network[node]["connections"]:
                    if connected not in visited:
                        queue.append(path + [connected])
                        
        return paths

    def add_fusion_style(self, name: str, description: str, operator: str, 
                        connectors: List[str]) -> str:
        """
        Adds a new fusion style.
        
        Args:
            name: Style name
            description: Style description
            operator: Mathematical-like operator for the style
            connectors: List of verbal connectors for this style
            
        Returns:
            Confirmation message
        """
        name = name.lower()
        self.fusion_styles[name] = {
            "description": description,
            "operator": operator,
            "connectors": connectors
        }
        return f"Fusion style '{name}' added."

    def save_fusion_data(self, filepath: str) -> str:
        """
        Saves fusion history and concept network to a JSON file.
        
        Args:
            filepath: Path to save the data
            
        Returns:
            Confirmation message
        """
        data = {
            "fusion_history": self.fusion_history,
            "concept_network": self.concept_network,
            "fusion_styles": self.fusion_styles,
            "custom_patterns": self.custom_patterns
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return f"Fusion data saved to {filepath}"
        except Exception as e:
            return f"Error saving fusion data: {e}"

    def load_fusion_data(self, filepath: str) -> str:
        """
        Loads fusion history and concept network from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Confirmation message
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "fusion_history" in data:
                self.fusion_history = data["fusion_history"]
            if "concept_network" in data:
                self.concept_network = data["concept_network"]
            if "fusion_styles" in data:
                self.fusion_styles.update(data["fusion_styles"])
            if "custom_patterns" in data:
                self.custom_patterns.update(data["custom_patterns"])
                
            return f"Fusion data loaded from {filepath}"
        except Exception as e:
            return f"Error loading fusion data: {e}"


if __name__ == "__main__":
    # Example usage when run directly
    fusion_engine = SymbolFusionEngine()
    
    # Test different fusion styles
    test_pairs = [
        ("truth", "beauty"),
        ("chaos", "order"),
        ("technology", "nature"),
        ("mind", "body"),
        ("individual", "community")
    ]
    
    print("=== Fusion Examples ===")
    for pair in test_pairs:
        style = random.choice(list(fusion_engine.fusion_styles.keys()))
        result = fusion_engine.fuse_with_options(*pair, style=style)
        print(f"\nFusing '{pair[0]}' with '{pair[1]}' using {style} style:")
        print(result)
        
    # Test cognitive modes
    print("\n=== Cognitive Mode Fusion ===")
    for mode in fusion_engine.cognitive_modes:
        test_pair = random.choice(test_pairs)
        result = fusion_engine.fuse_with_options(*test_pair, cognitive_mode=mode)
        print(f"\nFusing using {mode} mode:")
        print(result)
        
    # Test concept network
    print("\n=== Concept Network ===")
    for pair in test_pairs:
        fusion_engine.fuse(*pair)
    
    # Get connections for a concept
    test_concept = test_pairs[0][0]
    connections = fusion_engine.get_concept_connections(test_concept)
    print(f"\nConnections for '{test_concept}':")
    for connected, details in connections.items():
        print(f"  - Connected to '{connected}' with {len(details)} fusions")