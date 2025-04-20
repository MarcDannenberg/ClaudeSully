# Check tags
            tag_relevance = 0
            for tag in concept.get("tags", []):
                tag_match = self._keyword_relevance(keywords, tag)
                tag_relevance = max(tag_relevance, tag_match)
            
            if tag_relevance > 0:
                relevance += tag_relevance * 1.2  # Weight tag matches a bit higher
                match_types.append("tags")
            
            # Check domains
            domain_relevance = 0
            for domain in concept.get("domains", []):
                domain_match = self._keyword_relevance(keywords, domain)
                domain_relevance = max(domain_relevance, domain_match)
            
            if domain_relevance > 0:
                relevance += domain_relevance
                match_types.append("domains")
            
            # Only include if there's some relevance
            if relevance > 0:
                results[concept_id] = {
                    "id": concept_id,
                    "topic": concept["topic"],
                    "type": concept["type"],
                    "relevance": min(1.0, relevance),  # Cap at 1.0
                    "description": self._get_concept_description(concept),
                    "match_type": ",".join(match_types)
                }
        
        return list(results.values())
    
    def _semantic_search(self, query: str, concept_types: List[str] = None,
                        domains: List[str] = None, tags: List[str] = None,
                        date_range: Tuple[float, float] = None,
                        min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform semantic vector-based search.
        
        Args:
            query: Search query
            concept_types: Optional list of concept types to filter by
            domains: Optional list of domains to filter by
            tags: Optional list of tags to filter by
            date_range: Optional (start, end) timestamps to filter by
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of search results
        """
        # Generate vector for query
        query_vector = self._generate_query_vector(query)
        if query_vector is None:
            # Fall back to keyword search if vector generation fails
            return self._keyword_search(query, concept_types, domains, tags, date_range, min_confidence)
        
        results = {}  # concept_id -> result dict
        
        # Calculate vector similarities
        for concept_id, concept in self.entries.items():
            if not self._passes_filters(concept, concept_types, domains, tags, date_range, min_confidence):
                continue
                
            if concept_id not in self.concept_vectors:
                continue
                
            # Calculate vector similarity
            similarity = self._vector_similarity(query_vector, self.concept_vectors[concept_id])
            
            # Only include if similarity is above threshold
            if similarity > 0.5:  # Threshold for semantic similarity
                results[concept_id] = {
                    "id": concept_id,
                    "topic": concept["topic"],
                    "type": concept["type"],
                    "relevance": similarity,
                    "description": self._get_concept_description(concept),
                    "match_type": "semantic"
                }
        
        return list(results.values())
    
    def _passes_filters(self, concept: Dict[str, Any], concept_types: List[str] = None,
                      domains: List[str] = None, tags: List[str] = None,
                      date_range: Tuple[float, float] = None,
                      min_confidence: float = 0.0) -> bool:
        """
        Check if a concept passes all the specified filters.
        
        Args:
            concept: The concept to check
            concept_types: Optional list of concept types to filter by
            domains: Optional list of domains to filter by
            tags: Optional list of tags to filter by
            date_range: Optional (start, end) timestamps to filter by
            min_confidence: Minimum confidence threshold
            
        Returns:
            Whether the concept passes all filters
        """
        # Check concept type filter
        if concept_types and concept["type"] not in concept_types:
            return False
        
        # Check domain filter
        if domains:
            concept_domains = [d.lower() for d in concept.get("domains", [])]
            if not any(d in concept_domains for d in domains):
                return False
        
        # Check tag filter
        if tags:
            concept_tags = [t.lower() for t in concept.get("tags", [])]
            if not any(t in concept_tags for t in tags):
                return False
        
        # Check date range filter
        if date_range:
            start_time, end_time = date_range
            modified_time = concept.get("modified")
            
            if modified_time < start_time or modified_time > end_time:
                return False
        
        # Check confidence threshold
        if min_confidence > 0:
            confidence = concept.get("confidence", {}).get("overall", 0)
            if confidence < min_confidence:
                return False
        
        return True
    
    def _keyword_relevance(self, keywords: List[str], text: str) -> float:
        """
        Calculate keyword relevance score for a text.
        
        Args:
            keywords: List of query keywords
            text: Text to check against
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        if not text_words:
            return 0.0
        
        # Count matching keywords
        matches = sum(1 for kw in keywords if kw in text_words)
        
        if matches == 0:
            # Check for partial matches
            for kw in keywords:
                if kw in text_lower:
                    matches += 0.5
        
        # Calculate relevance based on ratio of matching keywords
        relevance = matches / len(keywords)
        
        # Boost relevance for exact match
        if text_lower == " ".join(keywords):
            relevance = 1.0
        
        return relevance
    
    def _get_concept_description(self, concept: Dict[str, Any]) -> str:
        """
        Get a descriptive summary for a concept.
        
        Args:
            concept: The concept
            
        Returns:
            Description string
        """
        # Look for a description field in the data
        data = concept.get("data", {})
        
        if "description" in data:
            return data["description"]
        elif "definition" in data:
            return data["definition"]
        elif "summary" in data:
            return data["summary"]
        
        # If no dedicated description field, concatenate a few data fields
        description_parts = []
        
        priority_fields = ["overview", "abstract", "content", "details", "explanation"]
        
        for field in priority_fields:
            if field in data and isinstance(data[field], str):
                description_parts.append(data[field][:100])
                
                if len(description_parts) >= 2:
                    break
        
        if description_parts:
            return " ".join(description_parts)
        
        # Fallback to type-based generic description
        concept_type = concept.get("type", "concept")
        return f"A {concept_type} related to {concept['topic']}"
    
    def find_similar_concepts(self, concept_id: str, limit: int = 5, 
                            min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find concepts similar to the given concept.
        
        Args:
            concept_id: ID of the reference concept
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar concepts with similarity scores
        """
        if concept_id not in self.entries:
            return []
        
        if concept_id not in self.concept_vectors:
            # Generate vector if not yet created
            self.concept_vectors[concept_id] = self._generate_concept_vector(self.entries[concept_id])
        
        reference_vector = self.concept_vectors[concept_id]
        similar_concepts = []
        
        for other_id, other_vector in self.concept_vectors.items():
            # Skip the reference concept itself
            if other_id == concept_id:
                continue
                
            # Calculate similarity
            similarity = self._vector_similarity(reference_vector, other_vector)
            
            if similarity >= min_similarity:
                concept = self.entries[other_id]
                similar_concepts.append({
                    "id": other_id,
                    "topic": concept["topic"],
                    "type": concept["type"],
                    "similarity": similarity,
                    "description": self._get_concept_description(concept)
                })
        
        # Sort by similarity (descending)
        similar_concepts.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit results
        return similar_concepts[:limit]
    
    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Check for zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
            
        similarity = dot_product / norm_product
        
        # Normalize to 0-1 range (cosine similarity is between -1 and 1)
        return (similarity + 1) / 2
    
    def _generate_concept_vector(self, concept: Dict[str, Any]) -> np.ndarray:
        """
        Generate a vector representation for a concept.
        
        Args:
            concept: The concept
            
        Returns:
            Vector representation
        """
        # This is a simplified implementation
        # In a real system, you would use embeddings from a language model
        
        # Use a random but deterministic vector based on the concept's content
        content = (
            concept["topic"] + 
            " ".join(str(v) for v in concept["data"].values() if isinstance(v, (str, int, float))) +
            " ".join(concept.get("tags", [])) +
            " ".join(concept.get("domains", []))
        )
        
        # Hash the content for deterministic randomness
        hash_value = hash(content) % (2**32)
        np.random.seed(hash_value)
        
        # Generate a normalized random vector
        vector = np.random.normal(0, 1, self.vector_dimensions)
        return vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
    
    def _generate_query_vector(self, query: str) -> np.ndarray:
        """
        Generate a vector representation for a search query.
        
        Args:
            query: The search query
            
        Returns:
            Vector representation
        """
        # Simplified implementation
        # In a real system, you would use the same embedding model as for concepts
        
        # Use a random but deterministic vector based on the query
        hash_value = hash(query) % (2**32)
        np.random.seed(hash_value)
        
        # Generate a normalized random vector
        vector = np.random.normal(0, 1, self.vector_dimensions)
        return vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple Jaccard similarity of words
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _resolve_concept_identifier(self, identifier: str) -> Optional[str]:
        """
        Resolve a concept identifier (ID or topic) to a concept ID.
        
        Args:
            identifier: Concept ID or topic
            
        Returns:
            Resolved concept ID or None if not found
        """
        # Check if it's already a valid ID
        if identifier in self.entries:
            return identifier
            
        # Try to resolve as a topic
        normalized = identifier.lower()
        if normalized in self.terms:
            return self.terms[normalized]
            
        return None
    
    def create_cluster(self, name: str, concept_ids: List[str], 
                     description: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """
        Create a cluster of related concepts.
        
        Args:
            name: Name of the cluster
            concept_ids: List of concept IDs to include
            description: Optional description of the cluster
            metadata: Optional additional metadata
            
        Returns:
            ID of the created cluster
        """
        # Validate concept IDs
        valid_ids = [cid for cid in concept_ids if cid in self.entries]
        
        if not valid_ids:
            raise ValueError("No valid concept IDs provided")
            
        # Generate a unique ID for the cluster
        cluster_id = str(uuid.uuid4())
        
        # Create the cluster
        cluster = ConceptCluster(
            id=cluster_id,
            name=name,
            concepts=valid_ids,
            description=description,
            metadata=metadata or {}
        )
        
        # Store the cluster
        self.concept_clusters[cluster_id] = cluster
        
        # Update system metadata
        self.metadata["last_modified"] = datetime.datetime.now().timestamp()
        
        return cluster_id
    
    def add_to_cluster(self, cluster_id: str, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Add concepts to an existing cluster.
        
        Args:
            cluster_id: ID of the cluster
            concept_ids: List of concept IDs to add
            
        Returns:
            Updated cluster information
        """
        if cluster_id not in self.concept_clusters:
            raise ValueError(f"Cluster with ID '{cluster_id}' not found")
            
        cluster = self.concept_clusters[cluster_id]
        
        # Validate concept IDs
        valid_ids = [cid for cid in concept_ids if cid in self.entries and cid not in cluster.concepts]
        
        if not valid_ids:
            return {"id": cluster_id, "name": cluster.name, "concepts": cluster.concepts, "added": 0}
            
        # Add to cluster
        cluster.concepts.extend(valid_ids)
        
        # Update system metadata
        self.metadata["last_modified"] = datetime.datetime.now().timestamp()
        
        return {
            "id": cluster_id,
            "name": cluster.name,
            "concepts": cluster.concepts,
            "added": len(valid_ids)
        }
    
    def remove_from_cluster(self, cluster_id: str, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Remove concepts from a cluster.
        
        Args:
            cluster_id: ID of the cluster
            concept_ids: List of concept IDs to remove
            
        Returns:
            Updated cluster information
        """
        if cluster_id not in self.concept_clusters:
            raise ValueError(f"Cluster with ID '{cluster_id}' not found")
            
        cluster = self.concept_clusters[cluster_id]
        
        # Find concepts that are in the cluster
        to_remove = [cid for cid in concept_ids if cid in cluster.concepts]
        
        if not to_remove:
            return {"id": cluster_id, "name": cluster.name, "concepts": cluster.concepts, "removed": 0}
            
        # Remove from cluster
        cluster.concepts = [cid for cid in cluster.concepts if cid not in to_remove]
        
        # Update system metadata
        self.metadata["last_modified"] = datetime.datetime.now().timestamp()
        
        return {
            "id": cluster_id,
            "name": cluster.name,
            "concepts": cluster.concepts,
            "removed": len(to_remove)
        }
    
    def get_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get information about a cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Cluster information
        """
        if cluster_id not in self.concept_clusters:
            raise ValueError(f"Cluster with ID '{cluster_id}' not found")
            
        cluster = self.concept_clusters[cluster_id]
        
        # Get concepts in the cluster
        concepts = []
        for concept_id in cluster.concepts:
            if concept_id in self.entries:
                concept = self.entries[concept_id]
                concepts.append({
                    "id": concept_id,
                    "topic": concept["topic"],
                    "type": concept["type"]
                })
        
        return {
            "id": cluster_id,
            "name": cluster.name,
            "description": cluster.description,
            "created": cluster.created,
            "metadata": cluster.metadata,
            "concepts": concepts
        }
    
    def list_clusters(self) -> List[Dict[str, Any]]:
        """
        List all concept clusters.
        
        Returns:
            List of cluster information
        """
        result = []
        
        for cluster_id, cluster in self.concept_clusters.items():
            result.append({
                "id": cluster_id,
                "name": cluster.name,
                "description": cluster.description,
                "concepts_count": len(cluster.concepts)
            })
        
        # Sort by name
        result.sort(key=lambda x: x["name"])
        
        return result
    
    def delete_cluster(self, cluster_id: str) -> bool:
        """
        Delete a concept cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Whether the deletion was successful
        """
        if cluster_id not in self.concept_clusters:
            return False
            
        # Delete the cluster
        del self.concept_clusters[cluster_id]
        
        # Update system metadata
        self.metadata["last_modified"] = datetime.datetime.now().timestamp()
        
        return True
    
    def automatic_clustering(self, concept_ids: List[str] = None, 
                          threshold: float = 0.7,
                          max_clusters: int = 10) -> List[Dict[str, Any]]:
        """
        Automatically create clusters based on concept similarity.
        
        Args:
            concept_ids: List of concept IDs to cluster (all if None)
            threshold: Similarity threshold for clustering
            max_clusters: Maximum number of clusters to create
            
        Returns:
            List of created clusters
        """
        # Use all concepts if none specified
        if concept_ids is None:
            concept_ids = list(self.entries.keys())
        else:
            # Filter out invalid IDs
            concept_ids = [cid for cid in concept_ids if cid in self.entries]
        
        if not concept_ids:
            return []
        
        # Simplified clustering algorithm - for a real system, use a proper clustering algorithm
        clusters = []
        assigned = set()
        
        # Sort concepts by importance for better cluster seeds
        sorted_concepts = sorted(
            concept_ids, 
            key=lambda cid: self.access_stats.get(cid, AccessStatistics()).importance_score,
            reverse=True
        )
        
        for seed_id in sorted_concepts:
            # Skip if already assigned to a cluster
            if seed_id in assigned:
                continue
                
            # Skip if no vector
            if seed_id not in self.concept_vectors:
                continue
                
            # Create a new cluster
            cluster_members = [seed_id]
            assigned.add(seed_id)
            
            # Find similar concepts
            seed_vector = self.concept_vectors[seed_id]
            
            for other_id in concept_ids:
                # Skip if already assigned or no vector
                if other_id in assigned or other_id not in self.concept_vectors:
                    continue
                    
                # Calculate similarity
                similarity = self._vector_similarity(seed_vector, self.concept_vectors[other_id])
                
                if similarity >= threshold:
                    cluster_members.append(other_id)
                    assigned.add(other_id)
            
            # Only keep clusters with at least 2 members
            if len(cluster_members) > 1:
                # Determine a name for the cluster based on common domains or tags
                cluster_name = self._generate_cluster_name(cluster_members)
                
                # Create the cluster
                cluster_id = self.create_cluster(
                    name=cluster_name,
                    concept_ids=cluster_members,
                    description=f"Automatically generated cluster with similarity threshold {threshold}"
                )
                
                clusters.append({
                    "id": cluster_id,
                    "name": cluster_name,
                    "size": len(cluster_members),
                    "seed_concept": self.entries[seed_id]["topic"]
                })
                
                # Stop if we've reached the maximum number of clusters
                if len(clusters) >= max_clusters:
                    break
        
        return clusters
    
    def _generate_cluster_name(self, concept_ids: List[str]) -> str:
        """
        Generate a name for a cluster based on common domains or tags.
        
        Args:
            concept_ids: List of concept IDs in the cluster
            
        Returns:
            Generated cluster name
        """
        if not concept_ids:
            return "Empty Cluster"
            
        # Count domains and tags
        domain_counts = Counter()
        tag_counts = Counter()
        
        for concept_id in concept_ids:
            concept = self.entries[concept_id]
            
            for domain in concept.get("domains", []):
                domain_counts[domain] += 1
                
            for tag in concept.get("tags", []):
                tag_counts[tag] += 1
        
        # Try to use the most common domain first
        if domain_counts:
            most_common_domain = domain_counts.most_common(1)[0][0]
            return f"{most_common_domain} Concepts"
        
        # Fall back to most common tag
        if tag_counts:
            most_common_tag = tag_counts.most_common(1)[0][0]
            return f"{most_common_tag} Concepts"
        
        # Use the first concept's topic as a fallback
        seed_concept = self.entries[concept_ids[0]]
        return f"{seed_concept['topic']} Related Concepts"
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detect contradictions in the knowledge base.
        
        Returns:
            List of detected contradictions
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated contradiction detection
        
        contradictions = []
        
        # Look for concepts with opposite_of relations
        for concept_id, concept_relations in self.relations.items():
            for relation in concept_relations:
                if relation.relation_type == RelationType.OPPOSITE_OF.value:
                    target_id = relation.target_concept
                    
                    # Look for contradictory relations
                    if concept_id in self.relations and target_id in self.relations:
                        for source_rel in self.relations[concept_id]:
                            for target_rel in self.relations[target_id]:
                                # Check if they have similar relations to the same concept
                                if (source_rel.relation_type == target_rel.relation_type and
                                    source_rel.target_concept == target_rel.target_concept and
                                    source_rel.relation_type != RelationType.OPPOSITE_OF.value):
                                    
                                    contradictions.append({
                                        "type": "relational_contradiction",
                                        "concepts": [
                                            {
                                                "id": concept_id,
                                                "topic": self.entries[concept_id]["topic"]
                                            },
                                            {
                                                "id": target_id,
                                                "topic": self.entries[target_id]["topic"]
                                            }
                                        ],
                                        "relation": source_rel.relation_type,
                                        "common_target": {
                                            "id": source_rel.target_concept,
                                            "topic": self.entries[source_rel.target_concept]["topic"]
                                        }
                                    })
        
        # Store in system for later reference
        self.contradictions = contradictions
        
        return contradictions
    
    def _initialize_inference_rules(self) -> None:
        """Initialize default inference rules for relationship derivation."""
        # Transitive rules
        self.inference_rules.append({
            "name": "is_a_transitive",
            "source_rel": RelationType.IS_A.value,
            "target_rel": RelationType.IS_A.value,
            "inferred_rel": RelationType.IS_A.value,
            "confidence_factor": 0.9,
            "transitive": True
        })
        
        self.inference_rules.append({
            "name": "part_of_transitive",
            "source_rel": RelationType.PART_OF.value,
            "target_rel": RelationType.PART_OF.value,
            "inferred_rel": RelationType.PART_OF.value,
            "confidence_factor": 0.9,
            "transitive": True
        })
        
        # Inheritance rules
        self.inference_rules.append({
            "name": "property_inheritance",
            "source_rel": RelationType.IS_A.value,
            "target_rel": RelationType.HAS_PROPERTY.value,
            "inferred_rel": RelationType.HAS_PROPERTY.value,
            "confidence_factor": 0.8,
            "inheritance": True
        })
        
        # Combination rules
        self.inference_rules.append({
            "name": "causal_chain",
            "source_rel": RelationType.CAUSES.value,
            "target_rel": RelationType.CAUSES.value,
            "inferred_rel": RelationType.CAUSES.value,
            "confidence_factor": 0.7,
            "transitive": True
        })
        
        self.inference_rules.append({
            "name": "temporal_chain",
            "source_rel": RelationType.PRECEDES.value,
            "target_rel": RelationType.PRECEDES.value,
            "inferred_rel": RelationType.PRECEDES.value,
            "confidence_factor": 0.8,
            "transitive": True
        })
        
        # Implication rules
        self.inference_rules.append({
            "name": "logical_implication_chain",
            "source_rel": RelationType.IMPLIES.value,
            "target_rel": RelationType.IMPLIES.value,
            "inferred_rel": RelationType.IMPLIES.value,
            "confidence_factor": 0.7,
            "transitive": True
        })
        
        # Contradiction rules
        self.inference_rules.append({
            "name": "opposite_contradiction",
            "source_rel": RelationType.OPPOSITE_OF.value,
            "target_rel": RelationType.IS_A.value,
            "inferred_rel": RelationType.OPPOSITE_OF.value,
            "confidence_factor": 0.6,
            "special": "contradiction"
        })
    
    def _run_inference_for_concept(self, concept_id: str) -> None:
        """
        Run inference rules for a concept to derive new relationships.
        
        Args:
            concept_id: ID of the concept to run inference for
        """
        if concept_id not in self.relations:
            return
            
        # Get all direct relations for this concept
        direct_relations = self.relations[concept_id]
        
        # Apply inference rules
        for rule in self.inference_rules:
            source_rel_type = rule["source_rel"]
            
            # Find relations matching the source relation type
            for rel in direct_relations:
                if rel.relation_type != source_rel_type:
                    continue
                    
                # Get the target concept
                target_id = rel.target_concept
                if target_id not in self.relations:
                    continue
                    
                # For transitive rules
                if rule.get("transitive"):
                    self._apply_transitive_rule(concept_id, target_id, rule)
                
                # For inheritance rules
                elif rule.get("inheritance"):
                    self._apply_inheritance_rule(concept_id, target_id, rule)
                
                # For special rules
                elif rule.get("special") == "contradiction":
                    self._apply_contradiction_rule(concept_id, target_id, rule)
    
    def _run_inference_for_relation(self, source_id: str, target_id: str, relation_type: str) -> None:
        """
        Run inference rules specifically for a new relation.
        
        Args:
            source_id: ID of the source concept
            target_id: ID of the target concept
            relation_type: Type of the relation
        """
        # Find matching rules
        for rule in self.inference_rules:
            if rule["source_rel"] != relation_type:
                continue
                
            # For transitive rules
            if rule.get("transitive"):
                self._apply_transitive_rule(source_id, target_id, rule)
            
            # For inheritance rules
            elif rule.get("inheritance"):
                self._apply_inheritance_rule(source_id, target_id, rule)
            
            # For special rules
            elif rule.get("special") == "contradiction":
                self._apply_contradiction_rule(source_id, target_id, rule)
    
    def _apply_transitive_rule(self, source_id: str, middle_id: str, rule: Dict[str, Any]) -> None:
        """
        Apply a transitive inference rule.
        
        Args:
            source_id: ID of the source concept
            middle_id: ID of the middle concept
            rule: The inference rule to apply
        """
        target_rel_type = rule["target_rel"]
        inferred_rel_type = rule["inferred_rel"]
        confidence_factor = rule["confidence_factor"]
        
        # Find relations from the middle concept that match the target relation type
        if middle_id not in self.relations:
            return
            
        for middle_rel in self.relations[middle_id]:
            if middle_rel.relation_type != target_rel_type:
                continue
                
            # Get the target concept
            target_id = middle_rel.target_concept
            
            # Skip self-reference
            if target_id == source_id:
                continue
                
            # Calculate confidence for the inferred relation
            inferred_confidence = middle_rel.confidence * confidence_factor
            
            # Add the inferred relation
            self.add_relation(
                source_id=source_id,
                target_identifier=target_id,
                relation_type=inferred_rel_type,
                confidence=inferred_confidence,
                metadata={"inferred_from_rule": rule["name"]},
                inferred=True
            )
    
    def _apply_inheritance_rule(self, source_id: str, parent_id: str, rule: Dict[str, Any]) -> None:
        """
        Apply an inheritance inference rule.
        
        Args:
            source_id: ID of the source concept
            parent_id: ID of the parent concept
            rule: The inference rule to apply
        """
        target_rel_type = rule["target_rel"]
        inferred_rel_type = rule["inferred_rel"]
        confidence_factor = rule["confidence_factor"]
        
        # Find the parent's properties
        if parent_id not in self.relations:
            return
            
        for parent_rel in self.relations[parent_id]:
            if parent_rel.relation_type != target_rel_type:
                continue
                
            # Get the property concept
            property_id = parent_rel.target_concept
            
            # Skip if the child already has this property
            if source_id in self.relations:
                existing = False
                for rel in self.relations[source_id]:
                    if rel.relation_type == inferred_rel_type and rel.target_concept == property_id:
                        existing = True
                        break
                        
                if existing:
                    continue
            
            # Calculate confidence for the inferred relation
            inferred_confidence = parent_rel.confidence * confidence_factor
            
            # Add the inferred relation
            self.add_relation(
                source_id=source_id,
                target_identifier=property_id,
                relation_type=inferred_rel_type,
                confidence=inferred_confidence,
                metadata={"inferred_from_rule": rule["name"], "inherited_from": parent_id},
                inferred=True
            )
    
    def _apply_contradiction_rule(self, source_id: str, opposite_id: str, rule: Dict[str, Any]) -> None:
        """
        Apply a contradiction inference rule.
        
        Args:
            source_id: ID of the source concept
            opposite_id: ID of the opposite concept
            rule: The inference rule to apply
        """
        target_rel_type = rule["target_rel"]
        inferred_rel_type = rule["inferred_rel"]
        confidence_factor = rule["confidence_factor"]
        
        # Find what the opposite is a kind of
        if opposite_id not in self.relations:
            return
            
        for opposite_rel in self.relations[opposite_id]:
            if opposite_rel.relation_type != target_rel_type:
                continue
                
            # Get the category concept
            category_id = opposite_rel.target_concept
            
            # Find other members of this category
            for concept_id, concept_rels in self.relations.items():
                # Skip the opposite concept itself
                if concept_id == opposite_id:
                    continue
                    
                # Look for concepts in the same category
                for rel in concept_rels:
                    if rel.relation_type == target_rel_type and rel.target_concept == category_id:
                        # Infer that the source is opposite of this concept
                        inferred_confidence = rel.confidence * opposite_rel.confidence * confidence_factor
                        
                        # Add the inferred relation
                        self.add_relation(
                            source_id=source_id,
                            target_identifier=concept_id,
                            relation_type=inferred_rel_type,
                            confidence=inferred_confidence,
                            metadata={"inferred_from_rule": rule["name"]},
                            inferred=True
                        )
    
    def add_word(self, term: str, meaning: str) -> str:
        """
        Add a term definition to the codex.
        
        Args:
            term: The term to define
            meaning: The definition or meaning
            
        Returns:
            ID of the created concept
        """
        return self.record(
            topic=term,
            data={"definition": meaning},
            concept_type=ConceptType.TERM
        )
    
    def add_context(self, concept_id: str, context_type: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add contextual information to a concept.
        
        Args:
            concept_id: ID of the concept
            context_type: Type of context (domains, temporal, perspectives, cultures)
            context_data: Context information
            
        Returns:
            Updated concept information
        """
        if concept_id not in self.entries:
            raise ValueError(f"Concept with ID '{concept_id}' not found")
            
        # Prepare update data
        if context_type not in ["domains", "temporal", "perspectives", "cultures"]:
            raise ValueError(f"Invalid context type: {context_type}")
            
        updates = {
            "context": {
                context_type: context_data
            }
        }
        
        # Apply the update
        updated_concept = self.update(
            concept_id=concept_id,
            updates=updates,
            change_description=f"Added {context_type} context"
        )
        
        return updated_concept
    
    def remove(self, concept_id: str) -> bool:
        """
        Remove a concept from the codex.
        
        Args:
            concept_id: ID of the concept to remove
            
        Returns:
            Whether the removal was successful
        """
        if concept_id not in self.entries:
            return False
            
        # Get the concept
        concept = self.entries[concept_id]
        normalized_topic = concept["normalized_topic"]
        concept_type = concept["type"]
        
        # Remove from primary storage
        del self.entries[concept_id]
        
        # Remove from terms index if it matches
        if normalized_topic in self.terms and self.terms[normalized_topic] == concept_id:
            del self.terms[normalized_topic]
        
        # Remove from type index
        if concept_type in self.type_index:
            self.type_index[concept_type].discard(concept_id)
            
        # Remove from domain index
        for domain in concept.get("domains", []):
            domain_lower = domain.lower()
            if domain_lower in self.domain_index:
                self.domain_index[domain_lower].discard(concept_id)
                
        # Remove from tag index
        for tag in concept.get("tags", []):
            tag_lower = tag.lower()
            if tag_lower in self.tag_index:
                self.tag_index[tag_lower].discard(concept_id)
                
        # Remove from clusters
        for cluster in self.concept_clusters.values():
            if concept_id in cluster.concepts:
                cluster.concepts.remove(concept_id)
                
        # Remove relations
        if concept_id in self.relations:
            del self.relations[concept_id]
            
        # Remove inverse relations
        if concept_id in self.inverse_relations:
            del self.inverse_relations[concept_id]
            
        # Update all relations that reference this concept
        for rel_source, relations in list(self.relations.items()):
            updated_relations = [rel for rel in relations if rel.target_concept != concept_id]
            if len(updated_relations) != len(relations):
                self.relations[rel_source] = updated_relations
                
        for rel_target, relations in list(self.inverse_relations.items()):
            updated_relations = [rel for rel in relations if rel.target_concept != concept_id]
            if len(updated_relations) != len(relations):
                self.inverse_relations[rel_target] = updated_relations
                
        # Remove from vector space
        if concept_id in self.concept_vectors:
            del self.concept_vectors[concept_id]
            
        # Remove statistics
        if concept_id in self.access_stats:
            del self.access_stats[concept_id]
            
        # Remove version history
        if concept_id in self.version_history:
            del self.version_history[concept_id]
            
        # Update system metadata
        self.metadata["last_modified"] = datetime.datetime.now().timestamp()
        
        return True
    
    def get_domain_concepts(self, domain: str, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Get concepts belonging to a specific domain.
        
        Args:
            domain: The domain to filter by
            limit: Maximum number of results (0 for all)
            
        Returns:
            List of concepts in the domain
        """
        domain_lower = domain.lower()
        
        if domain_lower not in self.domain_index:
            return []
            
        concept_ids = self.domain_index[domain_lower]
        results = []
        
        for concept_id in concept_ids:
            if concept_id in self.entries:
                concept = self.entries[concept_id]
                results.append({
                    "id": concept_id,
                    "topic": concept["topic"],
                    "type": concept["type"],
                    "description": self._get_concept_description(concept)
                })
        
        # Sort alphabetically by topic
        results.sort(key=lambda x: x["topic"])
        
        # Apply limit if specified
        if limit > 0:
            results = results[:limit]
            
        return results
    
    def get_concepts_by_type(self, concept_type: Union[ConceptType, str], limit: int = 0) -> List[Dict[str, Any]]:
        """
        Get concepts of a specific type.
        
        Args:
            concept_type: The concept type to filter by
            limit: Maximum number of results (0 for all)
            
        Returns:
            List of concepts of the specified type
        """
        # Convert string enum to proper enum
        if isinstance(concept_type, str):
            try:
                concept_type = ConceptType(concept_type)
            except ValueError:
                # Use as custom type
                pass
                
        type_value = concept_type.value if isinstance(concept_type, ConceptType) else concept_type
        
        if type_value not in self.type_index:
            return []
            
        concept_ids = self.type_index[type_value]
        results = []
        
        for concept_id in concept_ids:
            if concept_id in self.entries:
                concept = self.entries[concept_id]
                results.append({
                    "id": concept_id,
                    "topic": concept["topic"],
                    "type": concept["type"],
                    "description": self._get_concept_description(concept)
                })
        
        # Sort alphabetically by topic
        results.sort(key=lambda x: x["topic"])
        
        # Apply limit if specified
        if limit > 0:
            results = results[:limit]
            
        return results
    
    def get_concepts_by_tag(self, tag: str, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Get concepts with a specific tag.
        
        Args:
            tag: The tag to filter by
            limit: Maximum number of results (0 for all)
            
        Returns:
            List of concepts with the specified tag
        """
        tag_lower = tag.lower()
        
        if tag_lower not in self.tag_index:
            return []
            
        concept_ids = self.tag_index[tag_lower]
        results = []
        
        for concept_id in concept_ids:
            if concept_id in self.entries:
                concept = self.entries[concept_id]
                results.append({
                    "id": concept_id,
                    "topic": concept["topic"],
                    "type": concept["type"],
                    "description": self._get_concept_description(concept)
                })
        
        # Sort alphabetically by topic
        results.sort(key=lambda x: x["topic"])
        
        # Apply limit if specified
        if limit > 0:
            results = results[:limit]
            
        return results
    
    def list_topics(self) -> List[Dict[str, str]]:
        """
        List all topics in the codex.
        
        Returns:
            List of topics with their types
        """
        results = []
        
        for concept_id, concept in self.entries.items():
            results.append({
                "id": concept_id,
                "topic": concept["topic"],
                "type": concept["type"]
            })
        
        # Sort alphabetically by topic
        results.sort(key=lambda x: x["topic"])
        
        return results
    
    def get_related_concepts(self, concept_id: str, max_depth: int = 1, 
                            relation_types: List[str] = None) -> Dict[str, Any]:
        """
        Get concepts related to a given concept up to a specified depth.
        
        Args:
            concept_id: ID of the concept to find related concepts for
            max_depth: Maximum relationship depth to traverse
            relation_types: Optional list of relation types to include
            
        Returns:
            Dictionary of related concepts with relationship paths
        """
        if concept_id not in self.entries:
            return {"error": f"Concept with ID '{concept_id}' not found"}
            
        # Get the concept
        concept = self.entries[concept_id]
        
        # Initialize result
        result = {
            "source": {
                "id": concept_id,
                "topic": concept["topic"],
                "type": concept["type"]
            },
            "related": {},
            "relation_counts": {},
            "depth": max_depth
        }
        
        # Convert relation types to a set for faster lookup
        relation_type_set = set(relation_types) if relation_types else None
        
        # Use BFS to find related concepts
        queue = deque([(concept_id, [])])  # (concept_id, path)
        visited = set([concept_id])
        
        while queue:
            current_id, path = queue.popleft()
            current_depth = len(path)
            
            # Stop if we've reached the maximum depth
            if current_depth >= max_depth:
                continue
                
            # Get outgoing relations
            if current_id in self.relations:
                for relation in self.relations[current_id]:
                    # Skip if not in specified relation types
                    if relation_type_set and relation.relation_type not in relation_type_set:
                        continue
                        
                    target_id = relation.target_concept
                    
                    # Skip if already visited
                    if target_id in visited:
                        continue
                        
                    visited.add(target_id)
                    
                    # Add to path
                    new_path = path + [(current_id, relation.relation_type, target_id)]
                    
                    # Add to result
                    if target_id in self.entries:
                        target = self.entries[target_id]
                        
                        result["related"][target_id] = {
                            "topic": target["topic"],
                            "type": target["type"],
                            "path": new_path,
                            "depth": current_depth + 1,
                            "relation_type": relation.relation_type
                        }
                        
                        # Update relation counts
                        if relation.relation_type not in result["relation_counts"]:
                            result["relation_counts"][relation.relation_type] = 0
                        result["relation_counts"][relation.relation_type] += 1
                    
                    # Add to queue for further exploration
                    queue.append((target_id, new_path))
        
        return result
    
    def get_concept_path(self, source_id: str, target_id: str, 
                       max_depth: int = 3) -> Dict[str, Any]:
        """
        Find a path between two concepts through their relationships.
        
        Args:
            source_id: ID of the source concept
            target_id: ID of the target concept
            max_depth: Maximum path depth to search
            
        Returns:
            Path information or error
        """
        if source_id not in self.entries:
            return {"error": f"Source concept with ID '{source_id}' not found"}
            
        if target_id not in self.entries:
            return {"error": f"Target concept with ID '{target_id}' not found"}
            
        # Use BFS to find the shortest path
        queue = deque([(source_id, [])])  # (concept_id, path)
        visited = set([source_id])
        
        while queue:
            current_id, path = queue.popleft()
            current_depth = len(path)
            
            # Stop if we've reached the maximum depth
            if current_depth >= max_depth:
                continue
                
            # Check if we've reached the target
            if current_id == target_id:
                # Format the path for return
                formatted_path = []
                for source, relation_type, target in path:
                    formatted_path.append({
                        "source": {
                            "id": source,
                            "topic": self.entries[source]["topic"]
                        },
                        "relation": relation_type,
                        "target": {
                            "id": target,
                            "topic": self.entries[target]["topic"]
                        }
                    })
                
                return {
                    "source": {
                        "id": source_id,
                        "topic": self.entries[source_id]["topic"]
                    },
                    "target": {
                        "id": target_id,
                        "topic": self.entries[target_id]["topic"]
                    },
                    "path": formatted_path,
                    "length": len(path),
                    "found": True
                }
                
            # Get outgoing relations
            if current_id in self.relations:
                for relation in self.relations[current_id]:
                    relation_target = relation.target_concept
                    
                    # Skip if already visited
                    if relation_target in visited:
                        continue
                        
                    visited.add(relation_target)
                    
                    # Add to path
                    new_path = path + [(current_id, relation.relation_type, relation_target)]
                    
                    # Add to queue for further exploration
                    queue.append((relation_target, new_path))
        
        # If we get here, no path was found
        return {
            "source": {
                "id": source_id,
                "topic": self.entries[source_id]["topic"]
            },
            "target": {
                "id": target_id,
                "topic": self.entries[target_id]["topic"]
            },
            "found": False,
            "message": f"No path found between concepts within depth {max_depth}"
        }
    
    def export(self, path: Optional[str] = None, include_vectors: bool = False) -> Dict[str, Any]:
        """
        Export the codex to a JSON structure.
        
        Args:
            path: Optional file path to save the export to
            include_vectors: Whether to include concept vectors
            
        Returns:
            Dictionary containing the export data
        """
        # Prepare the export data
        export_data = {
            "metadata": self.metadata,
            "entries": self.entries,
            "relations": {},
            "version_history": {},
            "clusters": {}
        }
        
        # Convert relations to serializable format
        for concept_id, relations in self.relations.items():
            export_data["relations"][concept_id] = [asdict(r) for r in relations]
            
        # Convert version history to serializable format
        for concept_id, history in self.version_history.items():
            export_data["version_history"][concept_id] = asdict(history)
            
        # Convert clusters to serializable format
        for cluster_id, cluster in self.concept_clusters.items():
            export_data["clusters"][cluster_id] = asdict(cluster)
            
        # Include vectors if requested
        if include_vectors:
            export_data["vectors"] = {}
            for concept_id, vector in self.concept_vectors.items():
                export_data["vectors"][concept_id] = vector.tolist()
        
        # Add export timestamp
        export_data["export_timestamp"] = datetime.datetime.now().timestamp()
        
        # Save to file if path provided
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error saving export: {str(e)}")
                
        return export_data
    
    def import_data(self, data: Union[Dict[str, Any], str], merge: bool = False) -> Dict[str, Any]:
        """
        Import codex data from a previously exported format or file path.
        
        Args:
            data: Dictionary with codex data or file path to JSON export
            merge: Whether to merge with existing data or replace
            
        Returns:
            Summary of imported data
        """
        # Load from file if string provided
        if isinstance(data, str):
            try:
                with open(data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                return {"error": f"Failed to load file: {str(e)}"}
        
        # Validate data
        if not isinstance(data, dict) or "entries" not in data:
            return {"error": "Invalid import data format"}
            
        # Clear existing data if not merging
        if not merge:
            self.entries = {}
            self.terms = {}
            self.relations = {}
            self.inverse_relations = {}
            self.type_index = defaultdict(set)
            self.domain_index = defaultdict(set)
            self.tag_index = defaultdict(set)
            self.concept_vectors = {}
            self.concept_clusters = {}
            self.version_history = {}
            self.access_stats = {}
        
        # Import entries
        imported_entries = 0
        
        for concept_id, entry in data["entries"].items():
            # Skip if already exists and merging
            if merge and concept_id in self.entries:
                continue
                
            self.entries[concept_id] = entry
            
            # Update indexes
            normalized_topic = entry["normalized_topic"]
            self.terms[normalized_topic] = concept_id
            
            concept_type = entry["type"]
            self.type_index[concept_type].add(concept_id)
            
            for domain in entry.get("domains", []):
                self.domain_index[domain.lower()].add(concept_id)
                
            for tag in entry.get("tags", []):
                self.tag_index[tag.lower()].add(concept_id)
                
            # Initialize access stats
            self.access_stats[concept_id] = AccessStatistics(
                creation_time=entry.get("created", datetime.datetime.now().timestamp()),
                last_accessed=datetime.datetime.now().timestamp()
            )
            
            imported_entries += 1
        
        # Import relations
        imported_relations = 0
        
        if "relations" in data:
            for concept_id, relations_data in data["relations"].items():
                if concept_id not in self.entries:
                    continue
                    
                self.relations[concept_id] = []
                
                for rel_data in relations_data:
                    relation = ConceptRelation(**rel_data)
                    self.relations[concept_id].append(relation)
                    
                    # Update inverse relations
                    target_id = relation.target_concept
                    if target_id in self.entries:
                        if target_id not in self.inverse_relations:
                            self.inverse_relations[target_id] = []
                            
                        inverse_relation = ConceptRelation(
                            relation_type=relation.relation_type,
                            target_concept=concept_id,
                            confidence=relation.confidence,
                            metadata={**relation.metadata, "inverse": True},
                            bidirectional=relation.bidirectional,
                            created=relation.created
                        )
                        
                        self.inverse_relations[target_id].append(inverse_relation)
                        
                    imported_relations += 1
        
        # Import version history
        imported_versions = 0
        
        if "version_history" in data:
            for concept_id, history_data in data["version_history"].items():
                if concept_id not in self.entries:
                    continue
                    
                # Convert dictionary to VersionHistory object
                if isinstance(history_data, dict) and "versions" in history_data:
                    self.version_history[concept_id] = VersionHistory(**history_data)
                    imported_versions += len(history_data["versions"])
        
        # Import clusters
        imported_clusters = 0
        
        if "clusters" in data:
            for cluster_id, cluster_data in data["clusters"].items():
                # Filter out concepts that don't exist
                valid_concepts = [cid for cid in cluster_data.get("concepts", []) if cid in self.entries]
                cluster_data["concepts"] = valid_concepts
                
                self.concept_clusters[cluster_id] = ConceptCluster(**cluster_data)
                imported_clusters += 1
        
        # Import vectors if available
        imported_vectors = 0
        
        if "vectors" in data:
            for concept_id, vector_data in data["vectors"].items():
                if concept_id not in self.entries:
                    continue
                    
                self.concept_vectors[concept_id] = np.array(vector_data)
                imported_vectors += 1
        else:
            # Generate vectors for imported concepts
            for concept_id in [cid for cid in data["entries"] if cid in self.entries]:
                if concept_id not in self.concept_vectors:
                    self.concept_vectors[concept_id] = self._generate_concept_vector(self.entries[concept_id])
                    imported_vectors += 1
        
        # Update system metadata
        self.metadata["last_modified"] = datetime.datetime.now().timestamp()
        
        return {
            "imported_entries": imported_entries,
            "imported_relations": imported_relations,
            "imported_versions": imported_versions,
            "imported_clusters": imported_clusters,
            "imported_vectors": imported_vectors,
            "total_entries": len(self.entries),
            "total_relations": sum(len(rels) for rels in self.relations.values())
        }
    
    def batch_process(self, text: str, max_concepts: int = 10) -> Dict[str, Any]:
        """
        Process a text to extract and record potential concepts and their relationships.
        
        Args:
            text: Text to analyze for concepts
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            Summary of extraction and recording
        """
        # This is a simplified implementation
        # In a real system, you would use NLP techniques for concept extraction
        
        # Extract potential concept phrases (nouns and noun phrases)
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)  # Capitalized words
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){1,3}\b', text)  # Simple noun phrases
        
        # Count occurrences
        word_counts = Counter(words)
        phrase_counts = Counter(phrases)
        
        # Combine and get most frequent
        combined_counts = word_counts + phrase_counts
        top_concepts = combined_counts.most_common(max_concepts)
        
        # Record concepts
        recorded_concepts = []
        
        for concept, count in top_concepts:
            # Skip very short concepts
            if len(concept) < 4:
                continue
                
            # Extract a context for this concept
            context_match = re.search(r'[^.!?]*\b' + re.escape(concept) + r'\b[^.!?]*[.!?]', text)
            context = context_match.group(0).strip() if context_match else ""
            
            # Create a basic description
            if context:
                description = f"Concept extracted from context: '{context}'"
            else:
                description = f"Concept extracted from text with {count} occurrences"
            
            # Record in codex
            concept_id = self.record(
                topic=concept,
                data={
                    "description": description,
                    "context": context,
                    "occurrence_count": count
                },
                confidence=min(0.3 + (count / 10), 0.8),  # Higher confidence for more frequent concepts
                evidence=[EvidenceType.DERIVED.value],
                tags=["auto_extracted"]
            )
            
            recorded_concepts.append({
                "id": concept_id,
                "topic": concept,
                "description": description,
                "occurrences": count
            })
        
        # Try to infer relationships between extracted concepts
        inferred_relations = []
        
        for i, concept1 in enumerate(recorded_concepts):
            for concept2 in recorded_concepts[i+1:]:
                # Check for co-occurrence in the same sentence
                pattern = fr'[^.!?]*\b{re.escape(concept1["topic"])}\b[^.!?]*\b{re.escape(concept2["topic"])}\b[^.!?]*[.!?]'
                co_occurrences = re.findall(pattern, text)
                
                if co_occurrences:
                    # Create a generic relation
                    relation = self.add_relation(
                        source_id=concept1["id"],
                        target_identifier=concept2["id"],
                        relation_type=RelationType.RELATED_TO,
                        confidence=0.6,
                        metadata={"co_occurrences": len(co_occurrences), "auto_extracted": True}
                    )
                    
                    inferred_relations.append({
                        "source": concept1["topic"],
                        "target": concept2["topic"],
                        "type": RelationType.RELATED_TO.value,
                        "co_occurrences": len(co_occurrences)
                    })
        
        # Generate a simple cluster if enough concepts were extracted
        cluster_id = None
        if len(recorded_concepts) >= 3:
            # Extract a name from the text
            cluster_name = text.split('.')[0][:50]  # First sentence, truncated
            if not cluster_name:
                cluster_name = f"Concepts from text ({len(recorded_concepts)})"
                
            cluster_id = self.create_cluster(
                name=cluster_name,
                concept_ids=[c["id"] for c in recorded_concepts],
                description=f"Automatically extracted from text: {text[:100]}..."
            )
        
        return {
            "extracted_concepts": recorded_concepts,
            "inferred_relations": inferred_relations,
            "cluster_id": cluster_id,
            "text_length": len(text),
            "analysis_timestamp": datetime.datetime.now().timestamp()
        }
    
    def merge_concepts(self, primary_id: str, secondary_id: str,
                     merge_strategy: str = "combine") -> Dict[str, Any]:
        """
        Merge two concepts, combining their data and relationships.
        
        Args:
            primary_id: ID of the primary concept (to keep)
            secondary_id: ID of the secondary concept (to merge into primary)
            merge_strategy: Strategy for merging ('combine', 'primary_wins', 'secondary_wins')
            
        Returns:
            Result of the merge operation
        """
        if primary_id not in self.entries:
            return {"error": f"Primary concept with ID '{primary_id}' not found"}
            
        if secondary_id not in self.entries:
            return {"error": f"Secondary concept with ID '{secondary_id}' not found"}
            
        # Get the concepts
        primary = self.entries[primary_id]
        secondary = self.entries[secondary_id]
        
        # Prepare merge summary
        merge_summary = {
            "primary": {
                "id": primary_id,
                "topic": primary["topic"]
            },
            "secondary": {
                "id": secondary_id,
                "topic": secondary["topic"]
            },
            "changes": []
        }
        
        # Merge data based on strategy
        merged_data = {}
        
        if merge_strategy == "combine":
            # Combine data from both concepts
            merged_data = {**secondary["data"], **primary["data"]}
            merge_summary["changes"].append("Combined data from both concepts")
            
        elif merge_strategy == "primary_wins":
            # Keep primary data, only add secondary fields that don't exist
            merged_data = dict(primary["data"])
            for key, value in secondary["data"].items():
                if key not in merged_data:
                    merged_data[key] = value
                    merge_summary["changes"].append(f"Added field '{key}' from secondary concept")
                    
        elif merge_strategy == "secondary_wins":
            # Keep secondary data, only add primary fields that don't exist
            merged_data = dict(secondary["data"])
            for key, value in primary["data"].items():
                if key not in merged_data:
                    merged_data[key] = value
                    merge_summary["changes"].append(f"Added field '{key}' from primary concept")
                    
        else:
            return {"error": f"Invalid merge strategy: {merge_strategy}"}
        
        # Merge domains
        primary_domains = set(primary.get("domains", []))
        secondary_domains = set(secondary.get("domains", []))
        merged_domains = list(primary_domains.union(secondary_domains))
        
        if len(merged_domains) > len(primary_domains):
            merge_summary["changes"].append(f"Added {len(merged_domains) - len(primary_domains)} domains from secondary concept")
        
        # Merge tags
        primary_tags = set(primary.get("tags", []))
        secondary_tags = set(secondary.get("tags", []))
        merged_tags = list(primary_tags.union(secondary_tags))
        
        if len(merged_tags) > len(primary_tags):
            merge_summary["changes"].append(f"Added {len(merged_tags) - len(primary_tags)} tags from secondary concept")
        
        # Update primary concept
        update_data = {
            "data": merged_data,
            "domains": merged_domains,
            "tags": merged_tags
        }
        
        # Calculate merged confidence
        primary_confidence = primary.get("confidence", {}).get("overall", 0.7)
        secondary_confidence = secondary.get("confidence", {}).get("overall", 0.7)
        merged_confidence = max(primary_confidence, secondary_confidence)
        
        update_data["confidence"] = {
            "overall": merged_confidence
        }
        
        # Update the primary concept
        self.update(
            concept_id=primary_id,
            updates=update_data,
            change_description=f"Merged with concept '{secondary['topic']}' using strategy '{merge_strategy}'"
        )
        
        # Transfer relations from secondary to primary
        relations_transferred = 0
        
        if secondary_id in self.relations:
            for relation in self.relations[secondary_id]:
                # Skip self-references and references to primary
                if relation.target_concept == secondary_id or relation.target_concept == primary_id:
                    continue
                    
                # Add relation to primary
                try:
                    self.add_relation(
                        source_id=primary_id,
                        target_identifier=relation.target_concept,
                        relation_type=relation.relation_type,
                        confidence=relation.confidence,
                        metadata={**relation.metadata, "merged_from": secondary_id},
                        bidirectional=relation.bidirectional
                    )
                    relations_transferred += 1
                except Exception as e:
                    logger.warning(f"Error transferring relation: {str(e)}")
        
        if relations_transferred > 0:
            merge_summary["changes"].append(f"Transferred {relations_transferred} relations from secondary concept")
        
        # Update clusters to replace secondary with primary
        clusters_updated = 0
        
        for cluster in self.concept_clusters.values():
            if secondary_id in cluster.concepts:
                if primary_id not in cluster.concepts:
                    cluster.concepts.append(primary_id)
                cluster.concepts.remove(secondary_id)
                clusters_updated += 1
        
        if clusters_updated > 0:
            merge_summary["changes"].append(f"Updated {clusters_updated} clusters to replace secondary concept with primary")
        
        # Remove the secondary concept
        self.remove(secondary_id)
        merge_summary["changes"].append("Removed secondary concept")
        
        # Add final result summary
        merge_summary["result"] = "success"
        
        return merge_summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the codex.
        
        Returns:
            Dictionary with various statistics
        """
        # Basic counts
        total_concepts = len(self.entries)
        total_relations = sum(len(rels) for rels in self.relations.values())
        
        # Count by type
        type_counts = {}
        for type_, concepts in self.type_index.items():
            type_counts[type_] = len(concepts)
            
        # Count by domain
        domain_counts = {}
        for domain, concepts in self.domain_index.items():
            domain_counts[domain] = len(concepts)
            
        # Count by tag
        tag_counts = {}
        for tag, concepts in self.tag_index.items():
            tag_counts[tag] = len(concepts)
            
        # Count relations by type
        relation_type_counts = {}
        for relations in self.relations.values():
            for relation in relations:
                rel_type = relation.relation_type
                if rel_type not in relation_type_counts:
                    relation_type_counts[rel_type] = 0
                relation_type_counts[rel_type] += 1
                
        # Calculate average relations per concept
        avg_relations = total_relations / total_concepts if total_concepts > 0 else 0
        
        # Find most connected concepts
        concept_connections = {}
        for concept_id in self.entries:
            outgoing = len(self.relations.get(concept_id, []))
            incoming = len(self.inverse_relations.get(concept_id, []))
            concept_connections[concept_id] = outgoing + incoming
            
        most_connected = sorted(concept_connections.items(), key=lambda x: x[1], reverse=True)[:10]
        most_connected_info = []
        
        for concept_id, connections in most_connected:
            concept = self.entries[concept_id]
            most_connected_info.append({
                "id": concept_id,
                "topic": concept["topic"],
                "type": concept["type"],
                "connections": connections
            })
            
        # Find oldest and newest concepts
        sorted_by_created = sorted(
            [(cid, c["created"]) for cid, c in self.entries.items()],
            key=lambda x: x[1]
        )
        
        oldest_concepts = []
        for concept_id, timestamp in sorted_by_created[:5]:
            concept = self.entries[concept_id]
            oldest_concepts.append({
                "id": concept_id,
                "topic": concept["topic"],
                "created": timestamp,
                "date": datetime.datetime.fromtimestamp(timestamp).isoformat()
            })
            
        newest_concepts = []
        for concept_id, timestamp in sorted_by_created[-5:]:
            concept = self.entries[concept_id]
            newest_concepts.append({
                "id": concept_id,
                "topic": concept["topic"],
                "created": timestamp,
                "date": datetime.datetime.fromtimestamp(timestamp).isoformat()
            })
            
        # Calculate confidence distribution
        confidence_levels = [
            self.entries[cid].get("confidence", {}).get("overall", 0.5)
            for cid in self.entries
        ]
        
        confidence_distribution = {
            "min": min(confidence_levels) if confidence_levels else 0,
            "max": max(confidence_levels) if confidence_levels else 0,
            "avg": sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0,
            "median": sorted(confidence_levels)[len(confidence_levels) // 2] if confidence_levels else 0
        }
            
        return {
            "total_concepts": total_concepts,
            "total_relations": total_relations,
            "avg_relations_per_concept": avg_relations,
            "type_counts": type_counts,
            "domain_counts": domain_counts,
            "tag_counts": tag_counts,
            "relation_type_counts": relation_type_counts,
            "most_connected": most_connected_info,
            "oldest_concepts": oldest_concepts,
            "newest_concepts": newest_concepts,
            "confidence_distribution": confidence_distribution,
            "clusters": len(self.concept_clusters),
            "contradictions": len(self.contradictions),
            "timestamp": datetime.datetime.now().timestamp()
        }
        
    def update_importance_scores(self) -> Dict[str, Any]:
        """
        Update importance scores for all concepts based on usage patterns.
        
        Returns:
            Summary of the update operation
        """
        updated_count = 0
        
        for concept_id, stats in self.access_stats.items():
            # Skip if concept no longer exists
            if concept_id not in self.entries:
                continue
                
            # Calculate importance score components
            access_factor = min(1.0, stats.access_count / 10.0)  # Max at 10 accesses
            recency_factor = 0.0
            
            # Consider recency (higher if accessed recently)
            if stats.last_accessed:
                time_diff = datetime.datetime.now().timestamp() - stats.last_accessed
                days_diff = time_diff / (60 * 60 * 24)  # Convert to days
                recency_factor = max(0.0, 1.0 - (days_diff / 30))  # Decay over 30 days
                
            # Consider relations
            relation_factor = min(1.0, stats.relation_reference_count / 5.0)  # Max at 5 references
            
            # Search hit factor
            search_factor = min(1.0, stats.search_hit_count / 10.0)  # Max at 10 search hits
            
            # Calculate overall importance (weighted)
            importance = (
                access_factor * 0.3 +
                recency_factor * 0.2 +
                relation_factor * 0.3 +
                search_factor * 0.2
            )
            
            # Update importance score
            stats.importance_score = importance
            updated_count += 1
        
        return {
            "updated_count": updated_count,
            "timestamp": datetime.datetime.now().timestamp()
        }

    def __len__(self) -> int:
        """
        Returns the total number of unique concepts in the codex.
        
        Returns:
            Count of concepts
        """
        return len(self.entries)
        
    def __contains__(self, identifier: str) -> bool:
        """
        Check if a concept exists by ID or topic.
        
        Args:
            identifier: Concept ID or topic
            
        Returns:
            Whether the concept exists
        """
        # Check if it's a direct ID
        if identifier in self.entries:
            return True
            
        # Check as a normalized topic
        normalized = identifier.lower()
        return normalized in self.terms

# Create a global instance for easy access
codex = EnhancedCodex()