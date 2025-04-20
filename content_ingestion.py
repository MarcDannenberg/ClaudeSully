import spacy
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from collections import defaultdict

class LanguageStructureProcessor:
    """
    Advanced linguistic structure processor for Sully's ingestion kernel.
    
    Analyzes and extracts fundamental semantic and syntactic relationships,
    including who, what, where, when, why, how, and attributive relationships.
    """
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the language structure processor with NLP models.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp = spacy.load(model_name)
        
        # Define relationship types
        self.relationship_types = {
            "AGENT": ["who", "subject", "doer"],      # Who performed the action
            "ACTION": ["what", "verb", "action"],     # What was done
            "PATIENT": ["whom", "object", "receiver"], # Who received the action
            "TEMPORAL": ["when", "time", "date"],      # When it happened
            "SPATIAL": ["where", "location", "place"],  # Where it happened
            "CAUSAL": ["why", "reason", "cause"],      # Why it happened
            "MANNER": ["how", "method", "way"],        # How it happened
            "ATTRIBUTIVE": ["is", "of", "attribute"],  # What properties something has
            "QUANTITATIVE": ["how many", "amount", "quantity"], # How much/many
            "CONDITIONAL": ["if", "condition", "requirement"] # Under what conditions
        }
        
        # Special words that signal relationship types
        self.signal_words = {
            "who": "AGENT",
            "whom": "PATIENT",
            "whose": "ATTRIBUTIVE",
            "what": "ACTION",
            "where": "SPATIAL",
            "when": "TEMPORAL",
            "why": "CAUSAL",
            "how": "MANNER",
            "because": "CAUSAL",
            "since": "CAUSAL",
            "due to": "CAUSAL",
            "in order to": "CAUSAL",
            "therefore": "CAUSAL",
            "thus": "CAUSAL",
            "so that": "CAUSAL",
            "at": "TEMPORAL",
            "on": "TEMPORAL",
            "in": "SPATIAL",
            "during": "TEMPORAL",
            "before": "TEMPORAL",
            "after": "TEMPORAL",
            "while": "TEMPORAL",
            "if": "CONDITIONAL",
            "unless": "CONDITIONAL",
            "whether": "CONDITIONAL",
            "although": "CONDITIONAL"
        }
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text to extract fundamental semantic structures.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with extracted semantic relationships
        """
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = list(doc.sents)
        
        # Process each sentence
        processed_sentences = []
        for sent in sentences:
            processed_sent = self._process_sentence(sent)
            processed_sentences.append(processed_sent)
            
        # Find cross-sentence relationships
        cross_sentence_relations = self._extract_cross_sentence_relations(processed_sentences)
        
        return {
            "sentences": processed_sentences,
            "cross_sentence_relations": cross_sentence_relations,
            "entities": self._extract_entities(doc),
            "summary": self._generate_structural_summary(processed_sentences),
            "knowledge_graph": self._build_knowledge_graph(processed_sentences)
        }
    
    def _process_sentence(self, sent) -> Dict[str, Any]:
        """
        Process a single sentence to extract its semantic structure.
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            Dictionary with sentence semantic structure
        """
        # Core elements
        subject = None
        verb = None
        obj = None
        
        # Additional elements
        temporal = []
        spatial = []
        causal = []
        manner = []
        attributive = []
        conditional = []
        
        # Extract subject-verb-object
        for token in sent:
            # Find main verb
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "ccomp", "xcomp"):
                verb = token
                
                # Find subject of this verb
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                        
                    # Find object of this verb
                    elif child.dep_ in ("dobj", "pobj", "attr", "iobj"):
                        obj = child
        
        # Extract other relationship types
        for token in sent:
            # Spatial relationships (where)
            if token.dep_ == "prep" and token.text.lower() in ("in", "at", "on", "near", "by"):
                for child in token.children:
                    if child.dep_ == "pobj":
                        spatial_phrase = self._get_span_text(child)
                        spatial.append({
                            "type": "SPATIAL",
                            "text": spatial_phrase,
                            "preposition": token.text
                        })
            
            # Temporal relationships (when)
            elif (token.dep_ == "prep" and token.text.lower() in ("during", "after", "before", "at", "on")) or (token.ent_type_ in ("DATE", "TIME")):
                if token.dep_ == "prep":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            temporal_phrase = self._get_span_text(child)
                            temporal.append({
                                "type": "TEMPORAL",
                                "text": temporal_phrase,
                                "preposition": token.text
                            })
                else:  # Entity is directly a time/date
                    temporal_phrase = self._get_span_text(token)
                    temporal.append({
                        "type": "TEMPORAL",
                        "text": temporal_phrase,
                        "entity_type": token.ent_type_
                    })
            
            # Causal relationships (why)
            elif token.text.lower() in ("because", "since", "so", "therefore") or (token.dep_ == "prep" and token.text.lower() in ("due", "owing")):
                if token.text.lower() in ("because", "since"):
                    for right_token in token.rights:
                        causal_phrase = self._get_span_text(right_token)
                        causal.append({
                            "type": "CAUSAL",
                            "text": causal_phrase,
                            "marker": token.text
                        })
            
            # Manner relationships (how)
            elif token.dep_ == "advmod" and token.head.pos_ == "VERB":
                manner_phrase = token.text
                manner.append({
                    "type": "MANNER",
                    "text": manner_phrase
                })
                
            # Attributive relationships (is/of)
            elif token.dep_ == "prep" and token.text.lower() == "of":
                for child in token.children:
                    if child.dep_ == "pobj":
                        attributive_phrase = f"{token.head.text} of {self._get_span_text(child)}"
                        attributive.append({
                            "type": "ATTRIBUTIVE",
                            "text": attributive_phrase,
                            "attribute": token.head.text,
                            "entity": self._get_span_text(child)
                        })
            
            # Conditional relationships (if)
            elif token.text.lower() in ("if", "unless", "whether"):
                conditional_clause = ""
                for i, t in enumerate(sent):
                    if t.i >= token.i:  # Tokens at or after the conditional marker
                        if i+1 < len(sent) and sent[i+1].text.lower() in ("then", ","):
                            conditional_clause += t.text + " "
                            break
                        conditional_clause += t.text + " "
                
                conditional.append({
                    "type": "CONDITIONAL",
                    "text": conditional_clause.strip(),
                    "marker": token.text
                })
        
        # Build structured representation of the sentence
        structure = {
            "text": sent.text,
            "core": {
                "subject": self._get_span_text(subject) if subject else None,
                "verb": self._get_span_text(verb) if verb else None,
                "object": self._get_span_text(obj) if obj else None,
            },
            "relationships": {
                "TEMPORAL": temporal,
                "SPATIAL": spatial, 
                "CAUSAL": causal,
                "MANNER": manner,
                "ATTRIBUTIVE": attributive,
                "CONDITIONAL": conditional
            },
            "type": self._determine_sentence_type(sent)
        }
        
        return structure
    
    def _get_span_text(self, token) -> str:
        """
        Get the full span text for a token, including its subtree.
        
        Args:
            token: spaCy Token object
            
        Returns:
            String with the full span text
        """
        if not token:
            return ""
            
        # If token has children, get the full subtree
        if list(token.children):
            # Get the leftmost and rightmost token indices
            left_idx = token.i
            right_idx = token.i
            
            for child in token.subtree:
                left_idx = min(left_idx, child.i)
                right_idx = max(right_idx, child.i)
                
            return token.doc[left_idx:right_idx+1].text
        else:
            return token.text
    
    def _determine_sentence_type(self, sent) -> str:
        """
        Determine the sentence type.
        
        Args:
            sent: spaCy Span object
            
        Returns:
            String indicating sentence type
        """
        text = sent.text.strip()
        
        # Check for questions
        if text.endswith("?"):
            if sent[0].text.lower() in ("who", "what", "where", "when", "why", "how"):
                return "WH_QUESTION"
            else:
                return "YES_NO_QUESTION"
                
        # Check for commands/imperatives
        if sent[0].pos_ == "VERB":
            return "COMMAND"
            
        # Check for exclamations
        if text.endswith("!"):
            return "EXCLAMATION"
            
        # Default to statement
        return "STATEMENT"
    
    def _extract_cross_sentence_relations(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships that span multiple sentences.
        
        Args:
            sentences: List of processed sentence structures
            
        Returns:
            List of cross-sentence relationships
        """
        relations = []
        
        # Look for coreferences by simple word matching
        # (A more sophisticated approach would use a coreference resolution system)
        entities = {}
        
        # First pass: collect entities
        for i, sent in enumerate(sentences):
            # Add subject and object as entities
            subject = sent["core"]["subject"]
            obj = sent["core"]["object"]
            
            if subject:
                if subject.lower() not in entities:
                    entities[subject.lower()] = []
                entities[subject.lower()].append((i, "subject"))
                
            if obj:
                if obj.lower() not in entities:
                    entities[obj.lower()] = []
                entities[obj.lower()].append((i, "object"))
        
        # Second pass: find cross-sentence relationships
        for entity, occurrences in entities.items():
            if len(occurrences) > 1:
                sent_indexes = [idx for idx, role in occurrences]
                
                # Record relationship if occurrences are in different sentences
                if len(set(sent_indexes)) > 1:
                    relations.append({
                        "type": "COREFERENCE",
                        "entity": entity,
                        "occurrences": occurrences
                    })
        
        # Look for causal chains
        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sent = sentences[i+1]
            
            # If current has causal marker at the end or next has causal marker at the beginning
            if (current["relationships"]["CAUSAL"] and "therefore" in current["text"]) or \
               (next_sent["relationships"]["CAUSAL"] and "because" in next_sent["text"][:20]):
                relations.append({
                    "type": "CAUSAL_CHAIN",
                    "sentences": [i, i+1],
                    "text": current["text"] + " " + next_sent["text"]
                })
                
        return relations
    
    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        """
        Extract named entities from the document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
            
        return entities
    
    def _generate_structural_summary(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the text's structural elements.
        
        Args:
            sentences: List of processed sentence structures
            
        Returns:
            Dictionary with structural summary
        """
        subjects = []
        verbs = []
        objects = []
        locations = []
        times = []
        reasons = []
        
        for sent in sentences:
            # Add core elements
            if sent["core"]["subject"]:
                subjects.append(sent["core"]["subject"])
            if sent["core"]["verb"]:
                verbs.append(sent["core"]["verb"])
            if sent["core"]["object"]:
                objects.append(sent["core"]["object"])
                
            # Add relationship elements
            for spatial in sent["relationships"]["SPATIAL"]:
                locations.append(spatial["text"])
                
            for temporal in sent["relationships"]["TEMPORAL"]:
                times.append(temporal["text"])
                
            for causal in sent["relationships"]["CAUSAL"]:
                reasons.append(causal["text"])
        
        # Count sentence types
        sentence_types = {}
        for sent in sentences:
            sent_type = sent["type"]
            sentence_types[sent_type] = sentence_types.get(sent_type, 0) + 1
            
        return {
            "who": list(set(subjects)),  # Remove duplicates
            "what": list(set(verbs)),
            "whom": list(set(objects)),
            "where": list(set(locations)),
            "when": list(set(times)),
            "why": list(set(reasons)),
            "sentence_types": sentence_types,
            "sentence_count": len(sentences)
        }
    
    def _build_knowledge_graph(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a knowledge graph from the extracted structures.
        
        Args:
            sentences: List of processed sentence structures
            
        Returns:
            Dictionary with nodes and edges for a knowledge graph
        """
        G = nx.DiGraph()
        
        # Add nodes and edges from sentences
        for i, sent in enumerate(sentences):
            # Add nodes for core elements
            subj = sent["core"]["subject"]
            verb = sent["core"]["verb"]
            obj = sent["core"]["object"]
            
            # Only add nodes that exist
            if subj:
                if not G.has_node(subj):
                    G.add_node(subj, type="entity")
                    
            if verb:
                if not G.has_node(verb):
                    G.add_node(verb, type="action")
                    
            if obj:
                if not G.has_node(obj):
                    G.add_node(obj, type="entity")
                    
            # Add edges for core structure
            if subj and verb:
                G.add_edge(subj, verb, type="AGENT")
                
            if verb and obj:
                G.add_edge(verb, obj, type="PATIENT")
                
            # Add nodes and edges for relationships
            for rel_type, rels in sent["relationships"].items():
                for rel in rels:
                    text = rel.get("text")
                    if text:
                        if not G.has_node(text):
                            G.add_node(text, type=rel_type.lower())
                            
                        # Connect relationship to relevant core element
                        if rel_type == "TEMPORAL" and verb:
                            G.add_edge(verb, text, type="WHEN")
                        elif rel_type == "SPATIAL" and verb:
                            G.add_edge(verb, text, type="WHERE")
                        elif rel_type == "CAUSAL" and verb:
                            G.add_edge(verb, text, type="WHY")
                        elif rel_type == "MANNER" and verb:
                            G.add_edge(verb, text, type="HOW")
                        elif rel_type == "ATTRIBUTIVE" and "entity" in rel:
                            attribute = rel.get("attribute")
                            entity = rel.get("entity")
                            if entity and attribute:
                                if not G.has_node(entity):
                                    G.add_node(entity, type="entity")
                                if not G.has_node(attribute):
                                    G.add_node(attribute, type="attribute")
                                G.add_edge(entity, attribute, type="HAS_ATTRIBUTE")
        
        # Convert to serializable format
        nodes = [{"id": node, "type": data["type"]} for node, data in G.nodes(data=True)]
        edges = [{"source": u, "target": v, "type": data["type"]} for u, v, data in G.edges(data=True)]
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def extract_structured_knowledge(self, text: str) -> Dict[str, Any]:
        """
        Extract structured knowledge in a format suitable for Sully's ingestion.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with structured knowledge
        """
        # Process the text to get basic structures
        processed = self.process_text(text)
        
        # Extract key facts
        facts = []
        for sent in processed["sentences"]:
            # Extract core fact from subject-verb-object
            if sent["core"]["subject"] and sent["core"]["verb"]:
                fact = {
                    "subject": sent["core"]["subject"],
                    "predicate": sent["core"]["verb"],
                    "object": sent["core"]["object"] if sent["core"]["object"] else None,
                    "confidence": 1.0,  # Could be adjusted based on linguistic certainty markers
                    "source": "text_extraction",
                    "sentence": sent["text"]
                }
                
                # Add contextual information
                context = {}
                
                # Add temporal context (when)
                if sent["relationships"]["TEMPORAL"]:
                    context["when"] = [t["text"] for t in sent["relationships"]["TEMPORAL"]]
                    
                # Add spatial context (where)
                if sent["relationships"]["SPATIAL"]:
                    context["where"] = [s["text"] for s in sent["relationships"]["SPATIAL"]]
                    
                # Add causal context (why)
                if sent["relationships"]["CAUSAL"]:
                    context["why"] = [c["text"] for c in sent["relationships"]["CAUSAL"]]
                    
                # Add manner context (how)
                if sent["relationships"]["MANNER"]:
                    context["how"] = [m["text"] for m in sent["relationships"]["MANNER"]]
                    
                if context:
                    fact["context"] = context
                    
                facts.append(fact)
                
            # Extract attributive facts (X of Y or X is Y)
            for attr in sent["relationships"]["ATTRIBUTIVE"]:
                if "attribute" in attr and "entity" in attr:
                    fact = {
                        "subject": attr["entity"],
                        "predicate": "has_attribute",
                        "object": attr["attribute"],
                        "confidence": 1.0,
                        "source": "text_extraction",
                        "sentence": sent["text"]
                    }
                    facts.append(fact)
        
        # Extract entity information
        entities = {}
        for entity in processed["entities"]:
            entity_name = entity["text"]
            if entity_name not in entities:
                entities[entity_name] = {
                    "type": entity["type"],
                    "mentions": 1,
                    "attributes": []
                }
            else:
                entities[entity_name]["mentions"] += 1
                
        # Add attributes from attributive relationships
        for sent in processed["sentences"]:
            for attr in sent["relationships"]["ATTRIBUTIVE"]:
                if "entity" in attr and "attribute" in attr:
                    entity_name = attr["entity"]
                    if entity_name in entities:
                        entities[entity_name]["attributes"].append(attr["attribute"])
        
        return {
            "facts": facts,
            "entities": entities,
            "summary": processed["summary"],
            "knowledge_graph": processed["knowledge_graph"]
        }


class ContentIngestionKernel:
    """
    Advanced ingestion kernel for Sully that parses text with
    linguistic structure understanding.
    """
    
    def __init__(self, language_processor=None, memory_system=None, codex=None):
        """
        Initialize the ingestion kernel.
        
        Args:
            language_processor: LanguageStructureProcessor instance or None
            memory_system: Optional reference to Sully's memory system 
            codex: Optional reference to Sully's knowledge base
        """
        self.language_processor = language_processor or LanguageStructureProcessor()
        self.memory = memory_system
        self.codex = codex
        
        # Configure processing pipelines
        self.pipelines = {
            "default": [
                self._extract_structure,
                self._extract_concepts,
                self._identify_key_facts,
                self._store_in_memory
            ],
            "document": [
                self._extract_structure,
                self._extract_concepts,
                self._extract_document_sections,
                self._identify_key_facts,
                self._generate_summary,
                self._store_in_memory
            ],
            "conversation": [
                self._extract_structure,
                self._extract_concepts,
                self._identify_intents,
                self._identify_key_facts,
                self._store_in_memory
            ]
        }
    
    def ingest_content(self, content: str, content_type: str = "default") -> Dict[str, Any]:
        """
        Ingest content with full linguistic structure understanding.
        
        Args:
            content: Text content to ingest
            content_type: Type of content ("default", "document", "conversation")
            
        Returns:
            Dictionary with ingestion results
        """
        # Select appropriate pipeline
        pipeline = self.pipelines.get(content_type, self.pipelines["default"])
        
        # Initialize results
        results = {
            "content": content,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "processed_data": {}
        }
        
        # Run the pipeline
        for process in pipeline:
            try:
                process(content, results)
            except Exception as e:
                results["success"] = False
                results["error"] = f"Error in {process.__name__}: {str(e)}"
                break
                
        return results
    
    def _extract_structure(self, content: str, results: Dict[str, Any]) -> None:
        """
        Extract linguistic structure from content.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        # Process with language processor
        knowledge = self.language_processor.extract_structured_knowledge(content)
        
        # Add to results
        results["processed_data"]["structure"] = {
            "facts": knowledge["facts"],
            "entities": knowledge["entities"],
            "summary": knowledge["summary"]
        }
        
        # Add knowledge graph
        results["processed_data"]["knowledge_graph"] = knowledge["knowledge_graph"]
    
    def _extract_concepts(self, content: str, results: Dict[str, Any]) -> None:
        """
        Extract key concepts from content.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        # Use existing structure if available
        if "structure" in results["processed_data"]:
            structure = results["processed_data"]["structure"]
            
            # Extract concepts from subjects, objects, and entities
            concepts = set()
            
            # Add subjects and objects as concepts
            for fact in structure["facts"]:
                if fact["subject"]:
                    concepts.add(fact["subject"].lower())
                if fact["object"]:
                    concepts.add(fact["object"].lower())
                    
            # Add entities
            for entity_name in structure["entities"]:
                concepts.add(entity_name.lower())
                
            # Filter out common words and short concepts
            stop_words = {"the", "a", "an", "in", "on", "at", "by", "for", "with", "about"}
            filtered_concepts = [concept for concept in concepts if concept not in stop_words and len(concept) > 2]
            
            # Add to results
            results["processed_data"]["concepts"] = filtered_concepts
    
    def _identify_key_facts(self, content: str, results: Dict[str, Any]) -> None:
        """
        Identify key facts from the content.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        # Use existing structure if available
        if "structure" in results["processed_data"]:
            structure = results["processed_data"]["structure"]
            
            # Sort facts by confidence (if available) or by completeness
            def fact_importance(fact):
                # Higher score for facts with all core elements
                completeness = 0
                if fact["subject"]:
                    completeness += 1
                if fact["predicate"]:
                    completeness += 1
                if fact["object"]:
                    completeness += 1
                    
                # Higher score for facts with context
                context_score = 0
                if "context" in fact:
                    context_score = len(fact["context"])
                    
                return (fact.get("confidence", 1.0), completeness, context_score)
                
            # Sort facts by importance
            key_facts = sorted(structure["facts"], key=fact_importance, reverse=True)
            
            # Keep top 10 facts
            results["processed_data"]["key_facts"] = key_facts[:10] if len(key_facts) > 10 else key_facts
    
    def _extract_document_sections(self, content: str, results: Dict[str, Any]) -> None:
        """
        Extract document sections for document-type content.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        # Simple section extraction based on line patterns
        lines = content.split("\n")
        sections = []
        current_section = None
        
        # Patterns for potential headers
        header_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^[A-Z][A-Za-z\s]+:$',  # Capitalized words followed by colon
            r'^[IVX]+\.\s+(.+)$',  # Roman numerals
            r'^\d+\.\s+(.+)$',  # Numbered sections
        ]
        
        for line in lines:
            # Check if line matches a header pattern
            is_header = False
            header_text = ""
            
            for pattern in header_patterns:
                match = re.match(pattern, line)
                if match:
                    is_header = True
                    header_text = match.group(1) if match.groups() else line.strip()
                    break
                    
            if is_header:
                # If we have a current section, save it
                if current_section:
                    sections.append(current_section)
                    
                # Start new section
                current_section = {
                    "title": header_text,
                    "content": "",
                    "level": 1  # Default level
                }
                
                # Try to determine section level
                if line.startswith('#'):
                    current_section["level"] = len(re.match(r'^(#+)', line).group(1))
                    
            elif current_section:
                # Add line to current section
                if current_section["content"]:
                    current_section["content"] += "\n" + line
                else:
                    current_section["content"] = line
            else:
                # Start implicit first section
                current_section = {
                    "title": "Introduction",
                    "content": line,
                    "level": 1
                }
                
        # Add the last section
        if current_section:
            sections.append(current_section)
            
        # Add to results
        results["processed_data"]["sections"] = sections
    
    def _identify_intents(self, content: str, results: Dict[str, Any]) -> None:
        """
        Identify conversational intents for conversation-type content.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        # Use existing structure if available
        if "structure" in results["processed_data"]:
            structure = results["processed_data"]["structure"]
            
            # Identify intents based on sentence types and content
            intents = []
            
            for fact in structure["facts"]:
                sentence = fact.get("sentence", "")
                
                # Identify question intent
                if sentence.endswith("?"):
                    intent_type = "question"
                    
                    # Determine question type
                    if any(sentence.lower().startswith(wh) for wh in ["what", "who", "where", "when", "why", "how"]):
                        intent_subtype = "information_seeking"
                    elif sentence.lower().startswith(("do ", "does ", "is ", "are ", "can ", "could ")):
                        intent_subtype = "confirmation_seeking"
                    else:
                        intent_subtype = "general_question"
                        
                    intents.append({
                        "type": intent_type,
                        "subtype": intent_subtype,
                        "content": sentence
                    })
                    
                # Identify command/request intent
                elif fact["predicate"] and fact["predicate"].lower() in ["tell", "show", "give", "explain", "describe"]:
                    intents.append({
                        "type": "request",
                        "subtype": "information_request",
                        "content": sentence
                    })
                    
                # Identify statement intent
                else:
                    # Check for opinion statements
                    opinion_indicators = ["think", "believe", "feel", "opinion", "view"]
                    if fact["predicate"] and any(indicator in fact["predicate"].lower() for indicator in opinion_indicators):
                        intents.append({
                            "type": "statement",
                            "subtype": "opinion_sharing",
                            "content": sentence
                        })
                    # Check for informative statements
                    else:
                        intents.append({
                            "type": "statement",
                            "subtype": "information_sharing",
                            "content": sentence
                        })
            
            # Add to results
            results["processed_data"]["intents"] = intents
    
    def _generate_summary(self, content: str, results: Dict[str, Any]) -> None:
        """
        Generate a summary for document-type content.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        # Use key facts and structure to generate a summary
        if "key_facts" in results["processed_data"] and "structure" in results["processed_data"]:
            key_facts = results["processed_data"]["key_facts"]
            structure = results["processed_data"]["structure"]
            
            # Get key entities and actions
            entities = list(structure["entities"].keys())
            who = structure["summary"]["who"]
            what = structure["summary"]["what"]
            
            # Build summary sentences
            summary_points = []
            
            # Add main entities
            if entities:
                top_entities = entities[:3] if len(entities) > 3 else entities
                summary_points.append(f"This content mentions {', '.join(top_entities)}.")
                
            # Add main subjects and actions
            if who and what:
                top_who = who[:2] if len(who) > 2 else who
                top_what = what[:2] if len(what) > 2 else what
                summary_points.append(f"{', '.join(top_who)} {'/'.join(top_what)}.")
                
            # Add key facts in readable form
            for fact in key_facts[:3]:  # Top 3 facts
                if fact["subject"] and fact["predicate"]:
                    fact_text = f"{fact['subject']} {fact['predicate']}"
                    if fact["object"]:
                        fact_text += f" {fact['object']}"
                    summary_points.append(fact_text + ".")
                    
            # Combine into summary
            summary = " ".join(summary_points)
            
            # Add to results
            results["processed_data"]["summary"] = summary
    
    def _store_in_memory(self, content: str, results: Dict[str, Any]) -> None:
        """
        Store processed content in memory if memory system is available.
        
        Args:
            content: Content to process
            results: Results dictionary to update
        """
        if not self.memory:
            return
            
        try:
            # Extract the most important information
            key_facts = results["processed_data"].get("key_facts", [])
            concepts = results["processed_data"].get("concepts", [])
            
            # Store in memory
            memory_entry = {
                "content_type": results["content_type"],
                "timestamp": results["timestamp"],
                "key_facts": key_facts,
                "concepts": concepts
            }
            
            # Add a summary if available
            if "summary" in results["processed_data"]:
                memory_entry["summary"] = results["processed_data"]["summary"]
                
            # Store in memory with appropriate tags
            self.memory.store_experience(
                content=memory_entry,
                source="content_ingestion",
                importance=0.8,
                concepts=concepts[:5] if len(concepts) > 5 else concepts
            )
            
            # Update results
            results["processed_data"]["memory_stored"] = True
            
        except Exception as e:
            results["processed_data"]["memory_stored"] = False
            results["processed_data"]["memory_error"] = str(e)
    
    def integrate_with_sully(self, sully_instance) -> None:
        """
        Integrate this ingestion kernel with Sully.
        
        Args:
            sully_instance: The main Sully instance
        """
        if not sully_instance:
            return
            
        # Store references to Sully components
        if hasattr(sully_instance, 'memory'):
            self.memory = sully_instance.memory
            
        if hasattr(sully_instance, 'codex'):
            self.codex = sully_instance.codex
            
        # Add a reference to this ingestion kernel in Sully
        sully_instance.ingestion_kernel = self

# Example usage
def enhance_sully_content_ingestion(sully_instance):
    """
    Enhance Sully's content ingestion capabilities with linguistic structure understanding.
    
    Args:
        sully_instance: The main Sully instance
        
    Returns:
        Initialized ContentIngestionKernel
    """
    # Initialize the language structure processor
    language_processor = LanguageStructureProcessor()
    
    # Initialize the ingestion kernel
    ingestion_kernel = ContentIngestionKernel(
        language_processor=language_processor,
        memory_system=sully_instance.memory if hasattr(sully_instance, 'memory') else None,
        codex=sully_instance.codex if hasattr(sully_instance, 'codex') else None
    )
    
    # Integrate with Sully
    ingestion_kernel.integrate_with_sully(sully_instance)
    
    return ingestion_kernel