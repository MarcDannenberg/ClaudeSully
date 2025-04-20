# sully_engine/kernel_modules/continuous_learning.py
# 🧠 Sully's Continuous Learning System - Autonomous knowledge development

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import random
import re
from datetime import datetime, timedelta
import json
import os
import time
import numpy as np
from collections import Counter, defaultdict

class ConceptGraph:
    """
    Dynamic knowledge graph that represents concepts and their relationships.
    """
    
    def __init__(self):
        """Initialize the concept graph."""
        self.nodes = {}  # Concepts
        self.edges = defaultdict(list)  # Relationships between concepts
        self.node_strengths = {}  # Activation strength of concepts
        self.edge_weights = {}  # Relationship strengths
        
    def add_concept(self, concept: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new concept to the graph.
        
        Args:
            concept: Concept identifier
            metadata: Optional concept metadata
            
        Returns:
            Success indicator
        """
        if concept in self.nodes:
            # Update existing concept
            self.nodes[concept].update(metadata or {})
            return False
        else:
            # Add new concept
            self.nodes[concept] = metadata or {}
            self.node_strengths[concept] = 1.0  # Initial strength
            return True
            
    def add_relationship(self, concept1: str, concept2: str, relationship_type: str,
                         weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a relationship between concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            relationship_type: Type of relationship
            weight: Relationship strength
            metadata: Optional relationship metadata
            
        Returns:
            Success indicator
        """
        # Ensure both concepts exist
        for concept in [concept1, concept2]:
            if concept not in self.nodes:
                self.add_concept(concept)
                
        # Create the relationship
        edge_key = (concept1, concept2, relationship_type)
        
        # Check if relationship already exists
        exists = False
        for edge in self.edges[concept1]:
            if edge[0] == concept2 and edge[1] == relationship_type:
                exists = True
                break
                
        if not exists:
            # Add to edges list
            self.edges[concept1].append((concept2, relationship_type))
            
            # Set weight and metadata
            self.edge_weights[edge_key] = weight
            if edge_key not in self.edge_weights:
                self.edge_weights[edge_key] = {}
            if metadata:
                self.edge_weights[edge_key].update(metadata)
                
            return True
        else:
            # Update existing relationship
            self.edge_weights[edge_key] = weight
            if metadata:
                if not isinstance(self.edge_weights[edge_key], dict):
                    self.edge_weights[edge_key] = {"weight": self.edge_weights[edge_key]}
                self.edge_weights[edge_key].update(metadata)
                
            return False
            
    def get_related_concepts(self, concept: str, max_distance: int = 1) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept: The concept to find relations for
            max_distance: Maximum relationship distance
            
        Returns:
            Dictionary of related concepts with relationship info
        """
        if concept not in self.nodes:
            return {}
            
        # Start with direct relationships
        related = {}
        
        # Direct relationships (distance 1)
        for target, rel_type in self.edges[concept]:
            edge_key = (concept, target, rel_type)
            weight = self.edge_weights.get(edge_key, 1.0)
            
            if target not in related:
                related[target] = []
                
            related[target].append((rel_type, weight))
            
        # If max_distance > 1, find indirect relationships
        if max_distance > 1:
            visited = {concept}
            frontier = set(related.keys())
            
            for distance in range(2, max_distance + 1):
                new_frontier = set()
                
                for intermediate in frontier:
                    for target, rel_type in self.edges[intermediate]:
                        if target not in visited and target != concept:
                            edge_key = (intermediate, target, rel_type)
                            weight = self.edge_weights.get(edge_key, 1.0)
                            
                            # Discount weight by distance
                            effective_weight = weight / distance
                            
                            if target not in related:
                                related[target] = []
                                
                            related[target].append((f"{rel_type} (via {intermediate})", effective_weight))
                            new_frontier.add(target)
                            
                visited.update(frontier)
                frontier = new_frontier
                
        return related
        
    def strengthen_concept(self, concept: str, amount: float = 0.1) -> None:
        """
        Strengthen a concept through activation.
        
        Args:
            concept: Concept to strengthen
            amount: Strengthening amount
        """
        if concept in self.node_strengths:
            # Increase node strength
            self.node_strengths[concept] = min(2.0, self.node_strengths[concept] + amount)
            
            # Also strengthen directly connected concepts, but less
            for target, _ in self.edges[concept]:
                if target in self.node_strengths:
                    self.node_strengths[target] = min(2.0, self.node_strengths[target] + amount * 0.2)
    
    def decay_strengths(self, decay_rate: float = 0.01) -> None:
        """
        Apply time-based decay to concept and relationship strengths.
        
        Args:
            decay_rate: Rate of decay
        """
        # Decay node strengths
        for concept in self.node_strengths:
            self.node_strengths[concept] = max(0.1, self.node_strengths[concept] - decay_rate)
            
        # Decay edge weights
        for edge_key in self.edge_weights:
            if isinstance(self.edge_weights[edge_key], dict) and "weight" in self.edge_weights[edge_key]:
                self.edge_weights[edge_key]["weight"] = max(0.1, self.edge_weights[edge_key]["weight"] - decay_rate)
            elif isinstance(self.edge_weights[edge_key], (int, float)):
                self.edge_weights[edge_key] = max(0.1, self.edge_weights[edge_key] - decay_rate)
    
    def get_concept_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the concept graph.
        
        Returns:
            Dictionary of graph statistics
        """
        return {
            "num_concepts": len(self.nodes),
            "num_relationships": sum(len(rels) for rels in self.edges.values()),
            "avg_relationships_per_concept": sum(len(rels) for rels in self.edges.values()) / max(1, len(self.nodes)),
            "strongest_concepts": sorted([(c, s) for c, s in self.node_strengths.items()], key=lambda x: x[1], reverse=True)[:10],
            "most_connected_concepts": sorted([(c, len(self.edges[c])) for c in self.nodes], key=lambda x: x[1], reverse=True)[:10]
        }
    
    def export_to_json(self, filepath: str) -> bool:
        """
        Export the concept graph to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            
        Returns:
            Success indicator
        """
        try:
            graph_data = {
                "nodes": self.nodes,
                "edges": {source: [{"target": t, "type": ty} for t, ty in targets] 
                          for source, targets in self.edges.items()},
                "node_strengths": self.node_strengths,
                "edge_weights": {f"{s}|{t}|{ty}": w for (s, t, ty), w in self.edge_weights.items()}
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting concept graph: {str(e)}")
            return False
    
    def import_from_json(self, filepath: str) -> bool:
        """
        Import a concept graph from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Success indicator
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                
            # Import nodes
            self.nodes = graph_data.get("nodes", {})
            
            # Import edges
            self.edges = defaultdict(list)
            for source, targets in graph_data.get("edges", {}).items():
                for edge in targets:
                    self.edges[source].append((edge["target"], edge["type"]))
                    
            # Import node strengths
            self.node_strengths = graph_data.get("node_strengths", {})
            
            # Import edge weights
            self.edge_weights = {}
            for key, weight in graph_data.get("edge_weights", {}).items():
                try:
                    source, target, rel_type = key.split("|")
                    self.edge_weights[(source, target, rel_type)] = weight
                except:
                    pass
                    
            return True
        except Exception as e:
            print(f"Error importing concept graph: {str(e)}")
            return False


class ContinuousLearningSystem:
    """
    Advanced system for ongoing, unsupervised learning and knowledge development.
    Enables Sully to learn continuously from experiences, discovering patterns
    and building knowledge without explicit teaching.
    """

    def __init__(self, memory_system=None, codex=None, reasoning_node=None):
        """
        Initialize the continuous learning system.
        
        Args:
            memory_system: System for accessing memories
            codex: Knowledge base system
            reasoning_node: Reasoning engine for processing
        """
        self.memory = memory_system
        self.codex = codex
        self.reasoning = reasoning_node
        
        # Learning components
        self.concept_graph = ConceptGraph()
        self.experience_buffer = []
        self.learning_patterns = {}
        self.knowledge_gaps = set()
        self.contradictions = []
        
        # Learning configuration
        self.consolidation_schedule = {
            "last_run": datetime.now(),
            "interval": timedelta(hours=1),
            "running": False
        }
        
        # Learning statistics
        self.learning_stats = {
            "interactions_processed": 0,
            "concepts_learned": 0,
            "relationships_discovered": 0,
            "contradictions_resolved": 0,
            "knowledge_gaps_identified": 0,
            "knowledge_gaps_resolved": 0
        }
        
    def process_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new interaction for learning opportunities.
        
        Args:
            interaction_data: Data from the interaction
            
        Returns:
            Processing results
        """
        # Add to experience buffer for later consolidation
        self.experience_buffer.append({
            "timestamp": datetime.now().isoformat(),
            "data": interaction_data
        })
        
        # Immediate processing
        concepts_found = []
        
        # Extract content from interaction
        content = ""
        if isinstance(interaction_data, dict):
            # Extract from message/response structure
            if "message" in interaction_data:
                content += interaction_data["message"] + " "
            if "response" in interaction_data:
                content += interaction_data["response"] + " "
        elif isinstance(interaction_data, str):
            # Direct text content
            content = interaction_data
            
        # Extract concepts (basic approach)
        concepts_found = self._extract_concepts(content)
        
        # Identify relationships between concepts
        relationships = []
        for i, concept1 in enumerate(concepts_found):
            for concept2 in concepts_found[i+1:]:
                # Check for relationship indicators
                rel_type = self._identify_relationship_type(concept1, concept2, content)
                if rel_type:
                    relationships.append((concept1, concept2, rel_type))
                    
        # Update concept graph
        for concept in concepts_found:
            self.concept_graph.add_concept(concept, {
                "last_seen": datetime.now().isoformat(),
                "source": "interaction"
            })
            self.concept_graph.strengthen_concept(concept, 0.2)
            
        for concept1, concept2, rel_type in relationships:
            self.concept_graph.add_relationship(
                concept1, concept2, rel_type,
                metadata={"source": "interaction", "detected": datetime.now().isoformat()}
            )
            
        # Update learning statistics
        self.learning_stats["interactions_processed"] += 1
        self.learning_stats["concepts_learned"] += len(concepts_found)
        self.learning_stats["relationships_discovered"] += len(relationships)
        
        # Check if it's time for consolidation
        self._check_consolidation_schedule()
        
        return {
            "concepts_identified": concepts_found,
            "relationships_detected": relationships,
            "interaction_added": True
        }
        
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract concepts from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified concepts
        """
        # Tokenize text
        tokens = re.findall(r'\b[A-Za-z][A-Za-z\-]+\b', text)
        
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
        
        # If we have access to codex, use it to validate concepts
        validated_concepts = []
        if self.codex:
            try:
                for candidate in significant:
                    # Check if concept exists in codex
                    result = self.codex.search(candidate)
                    if result:
                        validated_concepts.append(candidate)
                if validated_concepts:
                    return validated_concepts
            except:
                pass
                
        return significant
        
    def _identify_relationship_type(self, concept1: str, concept2: str, text: str) -> Optional[str]:
        """
        Identify the relationship type between two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            text: Text containing the concepts
            
        Returns:
            Relationship type or None
        """
        # Pattern-based relationship extraction
        c1_idx = text.lower().find(concept1.lower())
        c2_idx = text.lower().find(concept2.lower())
        
        if c1_idx < 0 or c2_idx < 0:
            return None
            
        # Determine order
        if c1_idx < c2_idx:
            first, second = concept1, concept2
            between_text = text[c1_idx + len(concept1):c2_idx].lower()
        else:
            first, second = concept2, concept1
            between_text = text[c2_idx + len(concept2):c1_idx].lower()
            
        # Check for relationship indicators
        if "is a type of" in between_text or "is a kind of" in between_text:
            return "is_a"
        elif "is part of" in between_text or "belongs to" in between_text:
            return "part_of"
        elif "causes" in between_text or "leads to" in between_text:
            return "causes"
        elif "relates to" in between_text or "associated with" in between_text:
            return "related"
        elif "is similar to" in between_text or "is like" in between_text:
            return "similar"
        elif "contrasts with" in between_text or "opposite of" in between_text:
            return "opposite"
        else:
            return "mentioned_with"
            
    def consolidate_knowledge(self, force: bool = False) -> Dict[str, Any]:
        """
        Background process to consolidate experience into knowledge.
        
        Args:
            force: Whether to force consolidation regardless of schedule
            
        Returns:
            Consolidation results
        """
        # Check if we should run consolidation
        if not force:
            now = datetime.now()
            if (now - self.consolidation_schedule["last_run"]) < self.consolidation_schedule["interval"]:
                return {"status": "skipped", "reason": "Not scheduled yet"}
                
        # Check if already running
        if self.consolidation_schedule["running"]:
            return {"status": "skipped", "reason": "Already running"}
            
        # Mark as running
        self.consolidation_schedule["running"] = True
        
        try:
            # Process the experience buffer
            buffer_size = len(self.experience_buffer)
            
            # Extract recent experiences
            recent_experiences = self.experience_buffer[-min(buffer_size, 100):]
            
            # Identify patterns
            patterns_found = self._identify_patterns(recent_experiences)
            
            # Decay concept strengths to simulate forgetting
            self.concept_graph.decay_strengths(0.01)
            
            # Check for concept drift
            concept_drift = self._check_concept_drift()
            
            # Generate explorations for knowledge gaps
            explorations = self._generate_explorations()
            
            # Update learning statistics
            self.learning # Update learning statistics
            self.learning_stats["knowledge_gaps_identified"] += len(explorations)
            
            # Clean up buffer if it's getting too large
            if len(self.experience_buffer) > 1000:
                self.experience_buffer = self.experience_buffer[-1000:]
                
            # Update timestamp
            self.consolidation_schedule["last_run"] = datetime.now()
            
            # Consolidation complete
            result = {
                "status": "success", 
                "buffer_size": buffer_size,
                "patterns_found": patterns_found,
                "concept_drift": concept_drift,
                "explorations": explorations
            }
        except Exception as e:
            result = {"status": "error", "reason": str(e)}
        finally:
            # Mark as not running
            self.consolidation_schedule["running"] = False
            
        return result
        
    def _identify_patterns(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify patterns in recent experiences.
        
        Args:
            experiences: Recent experience data
            
        Returns:
            Identified patterns
        """
        patterns = []
        
        # Extract concepts from experiences
        all_concepts = []
        for exp in experiences:
            data = exp.get("data", "")
            if isinstance(data, dict) and "message" in data:
                concepts = self._extract_concepts(data["message"])
                all_concepts.extend(concepts)
                
        # Count concept frequencies
        concept_counts = Counter(all_concepts)
        
        # Identify recurring concepts
        recurring = [concept for concept, count in concept_counts.items() if count >= 3]
        
        for concept in recurring:
            pattern = {
                "type": "concept_recurrence",
                "concept": concept,
                "frequency": concept_counts[concept],
                "detected": datetime.now().isoformat()
            }
            patterns.append(pattern)
            
            # Strengthen recurring concepts
            self.concept_graph.strengthen_concept(concept, 0.3)
            
        # TODO: Add more sophisticated pattern detection
        
        return patterns
        
    def _check_concept_drift(self) -> List[Dict[str, Any]]:
        """
        Check for concepts that have evolved over time.
        
        Returns:
            List of concepts with drift
        """
        drift_detected = []
        
        # TODO: Implement concept drift detection
        
        return drift_detected
        
    def _generate_explorations(self) -> List[Dict[str, Any]]:
        """
        Generate exploration queries to fill knowledge gaps.
        
        Returns:
            List of exploration queries
        """
        explorations = []
        
        # Look for weak connections in the concept graph
        for concept, edges in self.concept_graph.edges.items():
            # Check concepts with few connections
            if 1 <= len(edges) <= 2:
                exploration = {
                    "type": "concept_expansion",
                    "concept": concept,
                    "query": f"How does {concept} relate to other concepts?",
                    "reason": "Concept has limited connections"
                }
                explorations.append(exploration)
                self.knowledge_gaps.add(concept)
                
        # TODO: Add more exploration generation methods
        
        return explorations[:5]  # Limit to 5 explorations at a time
        
    def _check_consolidation_schedule(self) -> None:
        """Check if it's time to run consolidation and schedule if needed."""
        now = datetime.now()
        if (now - self.consolidation_schedule["last_run"]) >= self.consolidation_schedule["interval"]:
            # Run in a separate thread or process in a real implementation
            # For now, we'll just run it directly
            if not self.consolidation_schedule["running"]:
                self.consolidate_knowledge()
                
    def apply_transfer_learning(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        """
        Apply knowledge from one domain to a different domain.
        
        Args:
            source_domain: Source domain for knowledge
            target_domain: Target domain to apply knowledge to
            
        Returns:
            Transfer learning results
        """
        # This requires reasoning capabilities
        if not self.reasoning:
            return {"success": False, "reason": "Reasoning engine required for transfer learning"}
            
        try:
            # Collect concepts from source domain
            source_concepts = []
            for concept in self.concept_graph.nodes:
                node_data = self.concept_graph.nodes[concept]
                if isinstance(node_data, dict) and node_data.get("domain") == source_domain:
                    source_concepts.append(concept)
                    
            # If no source concepts found, try searching in the source domain
            if not source_concepts and self.codex:
                search_results = self.codex.search(source_domain)
                if search_results:
                    source_concepts = list(search_results.keys())
                    
            if not source_concepts:
                return {"success": False, "reason": f"No concepts found in source domain: {source_domain}"}
                
            # Use reasoning to generate transfer hypotheses
            prompt = f"""
            Apply knowledge from the domain of {source_domain} to the domain of {target_domain}.
            
            Source domain concepts: {', '.join(source_concepts[:10])}
            
            Generate 3 hypotheses about {target_domain} based on principles from {source_domain}.
            For each hypothesis, explain:
            1. The principle from {source_domain}
            2. How it might apply to {target_domain}
            3. A specific prediction this transfer suggests
            """
            
            transfer_result = self.reasoning.reason(prompt, "analytical")
            
            if isinstance(transfer_result, dict) and "response" in transfer_result:
                transfer_text = transfer_result["response"]
            else:
                transfer_text = str(transfer_result)
                
            # Extract target domain concepts
            target_concepts = self._extract_concepts(transfer_text)
            
            # Create new relationships between domains
            bridge_relationships = []
            for source_concept in source_concepts[:3]:  # Limit to first 3 for simplicity
                for target_concept in target_concepts[:3]:
                    rel_type = "transfer_analogy"
                    self.concept_graph.add_relationship(
                        source_concept, target_concept, rel_type,
                        metadata={
                            "source_domain": source_domain,
                            "target_domain": target_domain,
                            "created": datetime.now().isoformat()
                        }
                    )
                    bridge_relationships.append((source_concept, target_concept, rel_type))
                    
            return {
                "success": True,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "source_concepts": source_concepts[:10],
                "target_concepts": target_concepts,
                "bridge_relationships": bridge_relationships,
                "transfer_hypotheses": transfer_text
            }
        except Exception as e:
            return {"success": False, "reason": str(e)}
            
    def generate_exploration_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Generate questions to actively explore knowledge boundaries.
        
        Args:
            limit: Maximum number of queries to generate
            
        Returns:
            List of exploration queries
        """
        queries = []
        
        # Check knowledge gaps
        for concept in list(self.knowledge_gaps)[:limit]:
            query = {
                "concept": concept,
                "query": f"What defines the concept of {concept} and how does it relate to other domains?",
                "type": "gap_exploration"
            }
            queries.append(query)
            
        # Look for concepts with high strength but few connections
        if len(queries) < limit:
            strong_concepts = sorted([(c, s) for c, s in self.concept_graph.node_strengths.items()], 
                                    key=lambda x: x[1], reverse=True)
            
            for concept, strength in strong_concepts:
                if len(queries) >= limit:
                    break
                    
                if concept in self.concept_graph.edges and len(self.concept_graph.edges[concept]) < 3:
                    query = {
                        "concept": concept,
                        "query": f"What are the most important relationships between {concept} and other concepts?",
                        "type": "connection_exploration"
                    }
                    queries.append(query)
                    
        # Generate transfer learning explorations
        if self.reasoning and len(queries) < limit:
            # Find domain pairs for potential transfer
            domains = set()
            for node_data in self.concept_graph.nodes.values():
                if isinstance(node_data, dict) and "domain" in node_data:
                    domains.add(node_data["domain"])
                    
            if len(domains) >= 2:
                domain_list = list(domains)
                source_domain = random.choice(domain_list)
                target_domain = random.choice([d for d in domain_list if d != source_domain])
                
                query = {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "query": f"What principles from {source_domain} might apply to {target_domain}?",
                    "type": "transfer_exploration"
                }
                queries.append(query)
                
        return queries
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the continuous learning process.
        
        Returns:
            Dictionary of learning statistics
        """
        # Combine system stats with concept graph stats
        graph_stats = self.concept_graph.get_concept_statistics()
        
        return {
            **self.learning_stats,
            "concept_graph": graph_stats,
            "experience_buffer_size": len(self.experience_buffer),
            "knowledge_gaps": len(self.knowledge_gaps),
            "contradictions": len(self.contradictions),
            "last_consolidation": self.consolidation_schedule["last_run"].isoformat()
        }
        
    def export_knowledge_state(self, directory: str = "knowledge_state") -> Dict[str, Any]:
        """
        Export the current knowledge state to files.
        
        Args:
            directory: Directory to save the knowledge state
            
        Returns:
            Export results
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Export concept graph
            graph_path = os.path.join(directory, "concept_graph.json")
            graph_success = self.concept_graph.export_to_json(graph_path)
            
            # Export learning statistics
            stats_path = os.path.join(directory, "learning_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.learning_stats, f, indent=2)
                
            # Export experience buffer sample (latest 100 entries)
            buffer_path = os.path.join(directory, "experience_sample.json")
            with open(buffer_path, 'w', encoding='utf-8') as f:
                json.dump(self.experience_buffer[-100:], f, indent=2)
                
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "files": {
                    "concept_graph": graph_path,
                    "learning_stats": stats_path,
                    "experience_buffer": buffer_path
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def import_knowledge_state(self, directory: str = "knowledge_state") -> Dict[str, Any]:
        """
        Import knowledge state from files.
        
        Args:
            directory: Directory to import from
            
        Returns:
            Import results
        """
        try:
            # Check if directory exists
            if not os.path.exists(directory):
                return {"success": False, "error": f"Directory not found: {directory}"}
                
            # Import concept graph
            graph_path = os.path.join(directory, "concept_graph.json")
            if os.path.exists(graph_path):
                graph_success = self.concept_graph.import_from_json(graph_path)
                
            # Import learning statistics
            stats_path = os.path.join(directory, "learning_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.learning_stats = json.load(f)
                    
            # Import experience buffer sample
            buffer_path = os.path.join(directory, "experience_sample.json")
            if os.path.exists(buffer_path):
                with open(buffer_path, 'r', encoding='utf-8') as f:
                    buffer_sample = json.load(f)
                    self.experience_buffer = buffer_sample
                    
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "imported": {
                    "concept_graph": os.path.exists(graph_path),
                    "learning_stats": os.path.exists(stats_path),
                    "experience_buffer": os.path.exists(buffer_path)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}