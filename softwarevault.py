"""
Advanced Software Vault for storing, retrieving, and managing code snippets.

This enterprise-grade module provides a comprehensive vault for storing, versioning,
analyzing, transforming, and collaborating on code snippets with advanced search,
security, and integration capabilities.
"""

import time
import json
import hashlib
import difflib
import re
import uuid
import base64
import datetime
from typing import List, Dict, Optional, Union, Tuple, Set, Any, Callable
from enum import Enum
import concurrent.futures
from dataclasses import dataclass, field, asdict

# Define enums for structured data
class SnippetType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SCRIPT = "script"
    ALGORITHM = "algorithm"
    PATTERN = "pattern"
    OTHER = "other"

class SecurityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    RESTRICTED = "restricted"

class SnippetStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class CodeMetrics:
    """Code quality and complexity metrics."""
    lines_of_code: int = 0
    comment_ratio: float = 0.0
    cyclomatic_complexity: Optional[int] = None
    halstead_complexity: Optional[Dict[str, float]] = None
    cognitive_complexity: Optional[int] = None
    maintainability_index: Optional[float] = None

@dataclass
class Version:
    """Version information for a code snippet."""
    version_id: str
    code: str
    author: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    comment: Optional[str] = None
    change_summary: Optional[Dict[str, Any]] = None

@dataclass
class Tag:
    """Tag with optional hierarchical structure."""
    name: str
    category: Optional[str] = None
    parent_tag: Optional[str] = None

@dataclass
class SnippetMetadata:
    """Extended metadata for a code snippet."""
    type: SnippetType = SnippetType.OTHER
    tags: List[Tag] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    status: SnippetStatus = SnippetStatus.DRAFT
    creation_date: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    authors: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    related_snippets: List[str] = field(default_factory=list)

class AdvancedSoftwareVault:
    """
    Advanced Software Vault with comprehensive features for code snippet management.
    
    Provides enterprise-grade capabilities for storing, versioning, searching,
    analyzing, transforming, and collaborating on code snippets with advanced
    security and integration features.
    """
    
    def __init__(self, encryption_key: Optional[str] = None, enable_metrics: bool = True):
        """
        Initialize an advanced software vault.
        
        Args:
            encryption_key: Optional key for encrypting sensitive snippets
            enable_metrics: Whether to calculate code metrics automatically
        """
        self.library = {}  # Using dict with ID as key for O(1) retrieval
        self.tags = {}  # Tag name -> [snippet_ids]
        self.languages = {}  # Language -> [snippet_ids]
        self.authors = {}  # Author -> [snippet_ids]
        self.types = {}  # SnippetType -> [snippet_ids]
        self.status_index = {}  # SnippetStatus -> [snippet_ids]
        self.security_index = {}  # SecurityLevel -> [snippet_ids]
        self.keyword_index = {}  # Keyword -> [snippet_ids]
        self.version_history = {}  # snippet_id -> [versions]
        self.dependency_graph = {}  # snippet_id -> [dependent_snippet_ids]
        self.reverse_dependencies = {}  # snippet_id -> [depends_on_snippet_ids]
        self.encryption_key = encryption_key
        self.enable_metrics = enable_metrics
        self.executors = {}  # Language -> executor function
        self.transformers = {}  # (source_lang, target_lang) -> transformer function
        self.analytics = {
            "total_snippets_added": 0,
            "total_searches": 0,
            "language_distribution": {},
            "type_distribution": {},
            "access_frequency": {},
        }
        self.integrations = {}  # Integration name -> integration handler
        
        # Initialize default executor for Python
        self.register_executor("python", self._execute_python)
        
    def store_snippet(self, 
                     name: str, 
                     language: str, 
                     description: str, 
                     code: str, 
                     author: Optional[str] = None,
                     snippet_type: Union[SnippetType, str] = SnippetType.OTHER,
                     tags: List[Union[str, Tag]] = None,
                     security_level: Union[SecurityLevel, str] = SecurityLevel.PUBLIC,
                     status: Union[SnippetStatus, str] = SnippetStatus.DRAFT,
                     dependencies: List[str] = None,
                     source: Optional[str] = None) -> Dict:
        """
        Store a code snippet with comprehensive metadata.
        
        Args:
            name: A descriptive name for the snippet
            language: The programming language of the snippet
            description: A detailed description of what the snippet does
            code: The actual code
            author: Optional author of the snippet
            snippet_type: Type of code snippet
            tags: List of tags or tag objects
            security_level: Security/visibility level of the snippet
            status: Current status in the development lifecycle
            dependencies: List of dependent snippet IDs
            source: Optional source reference (URL, book, etc.)
            
        Returns:
            The stored snippet entry
        """
        # Generate a unique ID
        snippet_id = str(uuid.uuid4())
        
        # Process tags
        processed_tags = []
        if tags:
            for tag in tags:
                if isinstance(tag, str):
                    processed_tags.append(Tag(name=tag))
                else:
                    processed_tags.append(tag)
        
        # Convert string enums to proper enum objects if necessary
        if isinstance(snippet_type, str):
            snippet_type = SnippetType(snippet_type)
        if isinstance(security_level, str):
            security_level = SecurityLevel(security_level)
        if isinstance(status, str):
            status = SnippetStatus(status)
        
        # Calculate metrics if enabled
        metrics = CodeMetrics()
        if self.enable_metrics:
            metrics = self._calculate_metrics(code, language)
        
        # Create metadata
        metadata = SnippetMetadata(
            type=snippet_type,
            tags=processed_tags,
            security_level=security_level,
            status=status,
            creation_date=time.time(),
            last_modified=time.time(),
            metrics=metrics,
            authors=[author] if author else [],
            dependencies=dependencies or [],
            related_snippets=[]
        )
        
        # Create the snippet
        snippet = {
            "id": snippet_id,
            "name": name,
            "language": language.lower(),
            "description": description,
            "code": self._encrypt_if_needed(code, security_level),
            "source": source,
            "metadata": asdict(metadata),
            "is_encrypted": security_level in [SecurityLevel.PRIVATE, SecurityLevel.RESTRICTED]
        }
        
        # Store in library
        self.library[snippet_id] = snippet
        
        # Create initial version
        self._create_version(snippet_id, code, author, "Initial version")
        
        # Update indexes
        self._update_indexes(snippet_id, snippet)
        
        # Update dependency graph
        self._update_dependency_graph(snippet_id, dependencies or [])
        
        # Update analytics
        self.analytics["total_snippets_added"] += 1
        self.analytics["language_distribution"][language.lower()] = self.analytics["language_distribution"].get(language.lower(), 0) + 1
        self.analytics["type_distribution"][snippet_type.value] = self.analytics["type_distribution"].get(snippet_type.value, 0) + 1
        
        return snippet
    
    def update_snippet(self, snippet_id: str, 
                      updates: Dict[str, Any],
                      author: Optional[str] = None,
                      comment: Optional[str] = None) -> Dict:
        """
        Update a snippet with new content or metadata.
        
        Args:
            snippet_id: ID of the snippet to update
            updates: Dictionary of fields to update
            author: Optional author of the update
            comment: Optional comment explaining the changes
            
        Returns:
            The updated snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
        
        # Get current snippet
        snippet = self.library[snippet_id]
        
        # Store old values for indexing
        old_language = snippet["language"]
        old_tags = [tag["name"] for tag in snippet["metadata"]["tags"]]
        old_type = snippet["metadata"]["type"]
        old_status = snippet["metadata"]["status"]
        old_security = snippet["metadata"]["security_level"]
        old_authors = snippet["metadata"]["authors"]
        old_dependencies = snippet["metadata"]["dependencies"]
        
        # Save version if code is updated
        if "code" in updates:
            code = updates["code"]
            old_code = self._decrypt_if_needed(snippet["code"], snippet["is_encrypted"])
            self._create_version(snippet_id, old_code, author, comment)
            snippet["code"] = self._encrypt_if_needed(code, snippet["metadata"]["security_level"])
            
            # Recalculate metrics if enabled
            if self.enable_metrics:
                metrics = self._calculate_metrics(code, snippet["language"])
                snippet["metadata"]["metrics"] = asdict(metrics)
        
        # Update other fields
        for key, value in updates.items():
            if key == "code":
                continue  # Already handled above
            elif key == "metadata":
                # Handle nested metadata updates
                for meta_key, meta_value in value.items():
                    snippet["metadata"][meta_key] = meta_value
            else:
                snippet[key] = value
        
        # Always update the last_modified timestamp
        snippet["metadata"]["last_modified"] = time.time()
        
        # Add author if provided and not already in the list
        if author and author not in snippet["metadata"]["authors"]:
            snippet["metadata"]["authors"].append(author)
        
        # Update indexes if relevant fields changed
        if (old_language != snippet["language"] or 
            old_tags != [tag["name"] for tag in snippet["metadata"]["tags"]] or
            old_type != snippet["metadata"]["type"] or
            old_status != snippet["metadata"]["status"] or
            old_security != snippet["metadata"]["security_level"] or
            old_authors != snippet["metadata"]["authors"] or
            old_dependencies != snippet["metadata"]["dependencies"]):
            
            # Remove from old indexes
            self._remove_from_indexes(snippet_id, old_language, old_tags, old_type, old_status, old_security, old_authors)
            
            # Add to new indexes
            self._update_indexes(snippet_id, snippet)
            
            # Update dependency graph if dependencies changed
            if old_dependencies != snippet["metadata"]["dependencies"]:
                self._update_dependency_graph(snippet_id, snippet["metadata"]["dependencies"])
        
        return snippet
    
    def _create_version(self, snippet_id: str, code: str, author: Optional[str], comment: Optional[str]) -> str:
        """
        Create a new version of a snippet.
        
        Args:
            snippet_id: ID of the snippet
            code: Code content for this version
            author: Optional author of this version
            comment: Optional comment about this version
            
        Returns:
            The version ID
        """
        # Initialize version history for this snippet if not exists
        if snippet_id not in self.version_history:
            self.version_history[snippet_id] = []
        
        # Calculate changes if there are previous versions
        change_summary = None
        if self.version_history[snippet_id]:
            prev_version = self.version_history[snippet_id][-1]
            change_summary = self._calculate_diff(prev_version.code, code)
        
        # Create new version
        version_id = f"{snippet_id}_v{len(self.version_history[snippet_id]) + 1}"
        version = Version(
            version_id=version_id,
            code=code,
            author=author,
            timestamp=time.time(),
            comment=comment,
            change_summary=change_summary
        )
        
        # Store version
        self.version_history[snippet_id].append(version)
        
        return version_id
    
    def get_version_history(self, snippet_id: str) -> List[Dict]:
        """
        Get the version history of a snippet.
        
        Args:
            snippet_id: ID of the snippet
            
        Returns:
            List of version information dictionaries
        """
        if snippet_id not in self.version_history:
            return []
        
        # Convert Version objects to dictionaries
        history = []
        for version in self.version_history[snippet_id]:
            history.append({
                "version_id": version.version_id,
                "author": version.author,
                "timestamp": version.timestamp,
                "datetime": datetime.datetime.fromtimestamp(version.timestamp).isoformat(),
                "comment": version.comment,
                "change_summary": version.change_summary
            })
        
        return history
    
    def get_version(self, version_id: str) -> Dict:
        """
        Get a specific version of a snippet.
        
        Args:
            version_id: The version ID
            
        Returns:
            Dictionary with version information and code
        """
        # Parse snippet_id from version_id (format: snippet_id_vX)
        parts = version_id.split("_v")
        if len(parts) != 2:
            raise ValueError(f"Invalid version ID: {version_id}")
        
        snippet_id = parts[0]
        version_num = int(parts[1])
        
        if snippet_id not in self.version_history:
            raise ValueError(f"No version history for snippet: {snippet_id}")
        
        if version_num < 1 or version_num > len(self.version_history[snippet_id]):
            raise ValueError(f"Version number out of range: {version_num}")
        
        version = self.version_history[snippet_id][version_num - 1]
        return {
            "version_id": version.version_id,
            "author": version.author,
            "timestamp": version.timestamp,
            "datetime": datetime.datetime.fromtimestamp(version.timestamp).isoformat(),
            "comment": version.comment,
            "code": version.code,
            "change_summary": version.change_summary
        }
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict:
        """
        Compare two versions of a snippet.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Comparison results
        """
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)
        
        diff = self._calculate_diff(v1["code"], v2["code"])
        
        return {
            "version1": version_id1,
            "version2": version_id2,
            "diff": diff,
            "v1_timestamp": v1["datetime"],
            "v2_timestamp": v2["datetime"],
            "v1_author": v1["author"],
            "v2_author": v2["author"]
        }
    
    def _calculate_diff(self, code1: str, code2: str) -> Dict[str, Any]:
        """
        Calculate the differences between two versions of code.
        
        Args:
            code1: Original code version
            code2: New code version
            
        Returns:
            Dictionary summarizing the changes
        """
        lines1 = code1.splitlines()
        lines2 = code2.splitlines()
        
        # Calculate diff
        diff = list(difflib.unified_diff(lines1, lines2, lineterm=''))
        
        # Count changes
        additions = len([line for line in diff if line.startswith('+')])
        deletions = len([line for line in diff if line.startswith('-')])
        
        # Calculate change percentage
        total_lines = max(len(lines1), len(lines2))
        change_percentage = ((additions + deletions) / total_lines * 100) if total_lines > 0 else 0
        
        return {
            "diff": diff,
            "additions": additions,
            "deletions": deletions,
            "change_percentage": change_percentage,
            "total_lines_before": len(lines1),
            "total_lines_after": len(lines2)
        }
    
    def search_by_name(self, query: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Search snippets by name with improved matching.
        
        Args:
            query: The search term to look for in snippet names
            case_sensitive: Whether to perform case-sensitive matching
            
        Returns:
            A list of matching snippets with relevance scores
        """
        self.analytics["total_searches"] += 1
        
        results = []
        for snippet_id, snippet in self.library.items():
            name = snippet["name"]
            relevance = self._calculate_relevance(query, name, case_sensitive)
            if relevance > 0:
                # Make a copy of the snippet to add relevance
                result = dict(snippet)
                result["relevance"] = relevance
                results.append(result)
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x["relevance"], reverse=True)
    
    def search_by_language(self, language: str) -> List[Dict]:
        """
        Search snippets by programming language.
        
        Args:
            language: The programming language to filter by
            
        Returns:
            A list of matching snippets
        """
        self.analytics["total_searches"] += 1
        
        language = language.lower()
        if language not in self.languages:
            return []
        
        results = []
        for snippet_id in self.languages[language]:
            if snippet_id in self.library:
                results.append(self.library[snippet_id])
        
        return results
    
    def search_by_text(self, query: str, case_sensitive: bool = False, fuzzy: bool = True) -> List[Dict]:
        """
        Search snippets by any text field with fuzzy matching.
        
        Args:
            query: The search term to look for in any text field
            case_sensitive: Whether to perform case-sensitive matching
            fuzzy: Whether to use fuzzy matching for better results
            
        Returns:
            A list of matching snippets with relevance scores
        """
        self.analytics["total_searches"] += 1
        
        results = []
        for snippet_id, snippet in self.library.items():
            relevance = 0
            
            # Check name
            name_relevance = self._calculate_relevance(query, snippet["name"], case_sensitive, fuzzy)
            relevance = max(relevance, name_relevance * 1.5)  # Weight name matches higher
            
            # Check description
            desc_relevance = self._calculate_relevance(query, snippet["description"], case_sensitive, fuzzy)
            relevance = max(relevance, desc_relevance)
            
            # Check code if not encrypted
            if not snippet.get("is_encrypted", False):
                code = snippet["code"]
                code_relevance = self._calculate_relevance(query, code, case_sensitive, fuzzy)
                relevance = max(relevance, code_relevance * 0.7)  # Weight code matches lower
            
            if relevance > 0:
                # Make a copy of the snippet to add relevance
                result = dict(snippet)
                result["relevance"] = relevance
                results.append(result)
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x["relevance"], reverse=True)
    
    def search_by_tags(self, tags: List[str], require_all: bool = False) -> List[Dict]:
        """
        Search snippets by tags.
        
        Args:
            tags: List of tags to search for
            require_all: Whether all tags must be present (AND) or any tag (OR)
            
        Returns:
            A list of matching snippets
        """
        self.analytics["total_searches"] += 1
        
        if not tags:
            return []
        
        matching_ids = set()
        first = True
        
        for tag in tags:
            tag = tag.lower()
            if tag in self.tags:
                if first or not require_all:
                    matching_ids.update(self.tags[tag])
                    first = False
                else:
                    matching_ids.intersection_update(self.tags[tag])
        
        results = []
        for snippet_id in matching_ids:
            if snippet_id in self.library:
                results.append(self.library[snippet_id])
        
        return results
    
    def search_by_author(self, author: str) -> List[Dict]:
        """
        Search snippets by author.
        
        Args:
            author: Author name to search for
            
        Returns:
            A list of matching snippets
        """
        self.analytics["total_searches"] += 1
        
        if author not in self.authors:
            return []
        
        results = []
        for snippet_id in self.authors[author]:
            if snippet_id in self.library:
                results.append(self.library[snippet_id])
        
        return results
    
    def search_by_metrics(self, 
                        min_complexity: Optional[int] = None,
                        max_complexity: Optional[int] = None,
                        min_maintainability: Optional[float] = None) -> List[Dict]:
        """
        Search snippets by code quality metrics.
        
        Args:
            min_complexity: Minimum cyclomatic complexity
            max_complexity: Maximum cyclomatic complexity
            min_maintainability: Minimum maintainability index
            
        Returns:
            A list of matching snippets
        """
        self.analytics["total_searches"] += 1
        
        results = []
        for snippet_id, snippet in self.library.items():
            metrics = snippet["metadata"]["metrics"]
            
            # Check complexity bounds if specified
            if min_complexity is not None and (metrics.get("cyclomatic_complexity") is None or
                                              metrics.get("cyclomatic_complexity") < min_complexity):
                continue
                
            if max_complexity is not None and (metrics.get("cyclomatic_complexity") is None or
                                              metrics.get("cyclomatic_complexity") > max_complexity):
                continue
                
            if min_maintainability is not None and (metrics.get("maintainability_index") is None or
                                                   metrics.get("maintainability_index") < min_maintainability):
                continue
            
            results.append(snippet)
        
        return results
    
    def advanced_search(self, 
                       keywords: Optional[List[str]] = None,
                       languages: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None,
                       authors: Optional[List[str]] = None,
                       types: Optional[List[Union[str, SnippetType]]] = None,
                       statuses: Optional[List[Union[str, SnippetStatus]]] = None,
                       security_levels: Optional[List[Union[str, SecurityLevel]]] = None,
                       min_date: Optional[float] = None,
                       max_date: Optional[float] = None,
                       fuzzy_search: bool = True) -> List[Dict]:
        """
        Perform an advanced search with multiple criteria.
        
        Args:
            keywords: Optional list of keywords to search for in text
            languages: Optional list of languages to filter by
            tags: Optional list of tags to filter by
            authors: Optional list of authors to filter by
            types: Optional list of snippet types to filter by
            statuses: Optional list of statuses to filter by
            security_levels: Optional list of security levels to filter by
            min_date: Optional minimum last modified timestamp
            max_date: Optional maximum last modified timestamp
            fuzzy_search: Whether to use fuzzy matching for keywords
            
        Returns:
            A list of matching snippets with relevance scores
        """
        self.analytics["total_searches"] += 1
        
        results = {}  # snippet_id -> (snippet, relevance score)
        
        # Convert enum strings to proper enum values
        if types:
            processed_types = []
            for t in types:
                if isinstance(t, str):
                    processed_types.append(SnippetType(t))
                else:
                    processed_types.append(t)
            types = processed_types
            
        if statuses:
            processed_statuses = []
            for s in statuses:
                if isinstance(s, str):
                    processed_statuses.append(SnippetStatus(s))
                else:
                    processed_statuses.append(s)
            statuses = processed_statuses
            
        if security_levels:
            processed_security = []
            for s in security_levels:
                if isinstance(s, str):
                    processed_security.append(SecurityLevel(s))
                else:
                    processed_security.append(s)
            security_levels = processed_security
        
        # Process each snippet
        for snippet_id, snippet in self.library.items():
            # Skip if doesn't match any specified language
            if languages and snippet["language"].lower() not in [lang.lower() for lang in languages]:
                continue
            
            # Skip if doesn't match any specified type
            if types and SnippetType(snippet["metadata"]["type"]) not in types:
                continue
            
            # Skip if doesn't match any specified status
            if statuses and SnippetStatus(snippet["metadata"]["status"]) not in statuses:
                continue
            
            # Skip if doesn't match any specified security level
            if security_levels and SecurityLevel(snippet["metadata"]["security_level"]) not in security_levels:
                continue
            
            # Skip if doesn't match any specified author
            if authors:
                snippet_authors = snippet["metadata"]["authors"]
                if not any(author in snippet_authors for author in authors):
                    continue
            
            # Skip if doesn't match date range
            last_modified = snippet["metadata"]["last_modified"]
            if min_date is not None and last_modified < min_date:
                continue
            if max_date is not None and last_modified > max_date:
                continue
            
            # Skip if doesn't match tags
            if tags:
                snippet_tags = [tag["name"].lower() for tag in snippet["metadata"]["tags"]]
                found_tag = False
                for tag in tags:
                    if tag.lower() in snippet_tags:
                        found_tag = True
                        break
                if not found_tag:
                    continue
            
            # Calculate keyword relevance
            relevance = 0
            if keywords:
                # Check name
                for keyword in keywords:
                    name_relevance = self._calculate_relevance(keyword, snippet["name"], False, fuzzy_search)
                    relevance += name_relevance * 1.5  # Weight name matches higher
                
                # Check description
                for keyword in keywords:
                    desc_relevance = self._calculate_relevance(keyword, snippet["description"], False, fuzzy_search)
                    relevance += desc_relevance
                
                # Check code if not encrypted
                if not snippet.get("is_encrypted", False):
                    code = snippet["code"]
                    for keyword in keywords:
                        code_relevance = self._calculate_relevance(keyword, code, False, fuzzy_search)
                        relevance += code_relevance * 0.7  # Weight code matches lower
            else:
                # If no keywords specified, give a default relevance
                relevance = 1.0
            
            # Only include if there's some relevance
            if relevance > 0:
                results[snippet_id] = (snippet, relevance)
        
        # Sort by relevance and convert to list
        sorted_results = []
        for snippet_id, (snippet, relevance) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
            result = dict(snippet)
            result["relevance"] = relevance
            sorted_results.append(result)
            
        return sorted_results
    
    def get_best_for_goal(self, goal: str, limit: int = 5) -> List[Dict]:
        """
        Find snippets that best match a specified goal using advanced matching.
        
        Args:
            goal: A description of what the user wants to accomplish
            limit: Maximum number of results to return
            
        Returns:
            Up to 'limit' snippets that best match the goal
        """
        self.analytics["total_searches"] += 1
        
        # Split goal into keywords for better matching
        keywords = re.findall(r'\w+', goal.lower())
        important_words = [w for w in keywords if len(w) > 3 and w not in [
            "the", "and", "that", "for", "with", "this", "from", "have", "has", "what", "when", "where", "which"
        ]]
        
        # Calculate scores for each snippet
        scored_snippets = []
        for snippet_id, snippet in self.library.items():
            score = 0
            description = snippet["description"].lower()
            
            # Score based on keyword presence in description
            for word in important_words:
                if word in description:
                    score += 1
                    
                    # Boost score if the word appears in key positions
                    if description.startswith(word) or re.search(r'^\s*' + re.escape(word), description):
                        score += 0.5
            
            # Score based on tag matches
            for tag in snippet["metadata"]["tags"]:
                tag_name = tag["name"].lower()
                if any(word in tag_name for word in important_words):
                    score += 1.5
            
            # Score based on exact phrase match
            if goal.lower() in description:
                score += 5
                
            # Score based on fuzzy match of the whole goal
            similarity = self._calculate_relevance(goal, description, False, True)
            score += similarity * 3
            
            # Add name match bonus
            for word in important_words:
                if word in snippet["name"].lower():
                    score += 2
            
            if score > 0:
                scored_snippets.append((score, snippet))
        
        # Sort by score (descending) and return top 'limit'
        sorted_snippets = [item for _, item in sorted(scored_snippets, key=lambda x: x[0], reverse=True)]
        return sorted_snippets[:limit]
    
    def get_similar_snippets(self, snippet_id: str, limit: int = 5) -> List[Dict]:
        """
        Find snippets similar to a given snippet.
        
        Args:
            snippet_id: ID of the reference snippet
            limit: Maximum number of similar snippets to return
            
        Returns:
            Up to 'limit' snippets most similar to the reference
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
        
        snippet = self.library[snippet_id]
        
        # Get snippets with same language
        same_language = self.search_by_language(snippet["language"])
        
        # Get snippets with shared tags
        tags = [tag["name"] for tag in snippet["metadata"]["tags"]]
        tagged_snippets = self.search_by_tags(tags)
        
        # Get snippets by description similarity
        text_similar = self.search_by_text(snippet["description"])
        
        # Calculate a composite similarity score
        similarities = {}
        
        for s in same_language + tagged_snippets + text_similar:
            # Skip the reference snippet itself
            if s["id"] == snippet_id:
                continue
                
            # Calculate similarity score components
            if s["id"] not in similarities:
                similarities[s["id"]] = 0
                
            # Language match
            if s["language"] == snippet["language"]:
                similarities[s["id"]] += 1
                
            # Tag overlap
            s_tags = [tag["name"] for tag in s["metadata"]["tags"]]
            tag_overlap = len(set(tags).intersection(s_tags))
            similarities[s["id"]] += tag_overlap * 0.5
                
            # Text similarity - if available from prior searches
            if "relevance" in s:
                similarities[s["id"]] += s["relevance"]
                
            # Type match
            if s["metadata"]["type"] == snippet["metadata"]["type"]:
                similarities[s["id"]] += 0.5
        
        # Sort by similarity score
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top snippets
        similar_snippets = []
        for s_id, score in sorted_similarities[:limit]:
            s = self.library[s_id]
            s_copy = dict(s)
            s_copy["similarity_score"] = score
            similar_snippets.append(s_copy)
            
        return similar_snippets
    
    def get_all(self) -> List[Dict]:
        """
        Get all snippets in the vault.
        
        Returns:
            A list of all snippets
        """
        return list(self.library.values())
    
    def get_by_id(self, snippet_id: str) -> Optional[Dict]:
        """
        Get a snippet by its ID.
        
        Args:
            snippet_id: The ID of the snippet to retrieve
            
        Returns:
            The snippet or None if not found
        """
        return self.library.get(snippet_id)
    
    def get_by_status(self, status: Union[str, SnippetStatus]) -> List[Dict]:
        """
        Get snippets by their status.
        
        Args:
            status: The status to filter by
            
        Returns:
            List of snippets with the specified status
        """
        if isinstance(status, str):
            status = SnippetStatus(status)
            
        if status.value not in self.status_index:
            return []
            
        return [self.library[snippet_id] for snippet_id in self.status_index[status.value] 
                if snippet_id in self.library]
    
    def get_by_type(self, snippet_type: Union[str, SnippetType]) -> List[Dict]:
        """
        Get snippets by their type.
        
        Args:
            snippet_type: The type to filter by
            
        Returns:
            List of snippets with the specified type
        """
        if isinstance(snippet_type, str):
            snippet_type = SnippetType(snippet_type)
            
        if snippet_type.value not in self.types:
            return []
            
        return [self.library[snippet_id] for snippet_id in self.types[snippet_type.value] 
                if snippet_id in self.library]
    
    def get_by_security_level(self, level: Union[str, SecurityLevel]) -> List[Dict]:
        """
        Get snippets by their security level.
        
        Args:
            level: The security level to filter by
            
        Returns:
            List of snippets with the specified security level
        """
        if isinstance(level, str):
            level = SecurityLevel(level)
            
        if level.value not in self.security_index:
            return []
            
        return [self.library[snippet_id] for snippet_id in self.security_index[level.value] 
                if snippet_id in self.library]
    
    def get_snippets_by_language_stats(self) -> Dict[str, int]:
        """
        Get statistics on snippets by language.
        
        Returns:
            Dictionary mapping languages to snippet counts
        """
        return {lang: len(snippets) for lang, snippets in self.languages.items()}
    
    def get_snippets_by_type_stats(self) -> Dict[str, int]:
        """
        Get statistics on snippets by type.
        
        Returns:
            Dictionary mapping types to snippet counts
        """
        return {type_: len(snippets) for type_, snippets in self.types.items()}
    
    def get_snippets_by_status_stats(self) -> Dict[str, int]:
        """
        Get statistics on snippets by status.
        
        Returns:
            Dictionary mapping statuses to snippet counts
        """
        return {status: len(snippets) for status, snippets in self.status_index.items()}
    
    def delete_snippet(self, snippet_id: str) -> bool:
        """
        Delete a snippet by its ID.
        
        Args:
            snippet_id: The ID of the snippet to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if snippet_id not in self.library:
            return False
        
        # Get snippet for index cleanup
        snippet = self.library[snippet_id]
        
        # Remove from indexes
        language = snippet["language"]
        tags = [tag["name"] for tag in snippet["metadata"]["tags"]]
        type_ = snippet["metadata"]["type"]
        status = snippet["metadata"]["status"]
        security = snippet["metadata"]["security_level"]
        authors = snippet["metadata"]["authors"]
        
        self._remove_from_indexes(snippet_id, language, tags, type_, status, security, authors)
        
        # Remove from dependency graphs
        if snippet_id in self.dependency_graph:
            del self.dependency_graph[snippet_id]
        
        for s_id in self.dependency_graph:
            if snippet_id in self.dependency_graph[s_id]:
                self.dependency_graph[s_id].remove(snippet_id)
                
        if snippet_id in self.reverse_dependencies:
            del self.reverse_dependencies[snippet_id]
            
        for s_id in self.reverse_dependencies:
            if snippet_id in self.reverse_dependencies[s_id]:
                self.reverse_dependencies[s_id].remove(snippet_id)
        
        # Remove from library
        del self.library[snippet_id]
        
        # Keep version history for potential recovery
        
        return True
    
    def archive_snippet(self, snippet_id: str, archive_comment: Optional[str] = None) -> Dict:
        """
        Archive a snippet instead of deleting it.
        
        Args:
            snippet_id: The ID of the snippet to archive
            archive_comment: Optional comment explaining the archiving
            
        Returns:
            The archived snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
        
        # Update snippet status to ARCHIVED
        updates = {
            "metadata": {
                "status": SnippetStatus.ARCHIVED.value,
                "archive_date": time.time(),
                "archive_comment": archive_comment
            }
        }
        
        updated_snippet = self.update_snippet(snippet_id, updates)
        return updated_snippet
    
    def execute_snippet(self, snippet_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a code snippet with optional parameters.
        
        Args:
            snippet_id: The ID of the snippet to execute
            params: Optional parameters to pass to the snippet
            
        Returns:
            Result of the execution
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
        
        snippet = self.library[snippet_id]
        language = snippet["language"].lower()
        
        # Check if we have an executor for this language
        if language not in self.executors:
            raise ValueError(f"No executor available for language: {language}")
        
        # Decrypt code if needed
        code = self._decrypt_if_needed(snippet["code"], snippet["is_encrypted"])
        
        # Execute the code
        executor = self.executors[language]
        start_time = time.time()
        try:
            result = executor(code, params or {})
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            end_time = time.time()
            
        # Record execution in analytics
        self.analytics.setdefault("executions", {})
        self.analytics["executions"].setdefault(language, 0)
        self.analytics["executions"][language] += 1
        
        return {
            "snippet_id": snippet_id,
            "language": language,
            "success": success,
            "result": result,
            "error": error,
            "execution_time": end_time - start_time
        }
    
    def register_executor(self, language: str, executor_func: Callable) -> None:
        """
        Register an executor function for a specific language.
        
        Args:
            language: The programming language
            executor_func: Function that executes code in that language
        """
        self.executors[language.lower()] = executor_func
    
    def transform_snippet(self, snippet_id: str, target_language: str) -> Dict[str, Any]:
        """
        Transform a snippet from its original language to a target language.
        
        Args:
            snippet_id: The ID of the snippet to transform
            target_language: The target programming language
            
        Returns:
            The transformed snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
        
        snippet = self.library[snippet_id]
        source_language = snippet["language"].lower()
        target_language = target_language.lower()
        
        # Check if we have a transformer for this language pair
        transformer_key = (source_language, target_language)
        if transformer_key not in self.transformers:
            raise ValueError(f"No transformer available from {source_language} to {target_language}")
        
        # Decrypt code if needed
        code = self._decrypt_if_needed(snippet["code"], snippet["is_encrypted"])
        
        # Transform the code
        transformer = self.transformers[transformer_key]
        try:
            transformed_code = transformer(code)
            success = True
            error = None
        except Exception as e:
            transformed_code = None
            success = False
            error = str(e)
        
        return {
            "original_snippet_id": snippet_id,
            "original_language": source_language,
            "target_language": target_language,
            "success": success,
            "transformed_code": transformed_code,
            "error": error,
            "original_name": snippet["name"],
            "original_description": snippet["description"]
        }
    
    def register_transformer(self, source_language: str, target_language: str, transformer_func: Callable) -> None:
        """
        Register a transformer function for a specific language pair.
        
        Args:
            source_language: The source programming language
            target_language: The target programming language
            transformer_func: Function that transforms code from source to target
        """
        self.transformers[(source_language.lower(), target_language.lower())] = transformer_func
    
    def add_tag(self, snippet_id: str, tag: Union[str, Tag]) -> Dict:
        """
        Add a tag to a snippet.
        
        Args:
            snippet_id: The ID of the snippet
            tag: The tag to add (string or Tag object)
            
        Returns:
            The updated snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
        
        # Process tag
        if isinstance(tag, str):
            tag_obj = Tag(name=tag)
        else:
            tag_obj = tag
            
        # Get current tags
        snippet = self.library[snippet_id]
        current_tags = snippet["metadata"]["tags"]
        
        # Skip if tag already exists
        if any(t["name"].lower() == tag_obj.name.lower() for t in current_tags):
            return snippet
            
        # Add tag
        updates = {
            "metadata": {
                "tags": current_tags + [asdict(tag_obj)]
            }
        }
        
        return self.update_snippet(snippet_id, updates)
    
    def remove_tag(self, snippet_id: str, tag_name: str) -> Dict:
        """
        Remove a tag from a snippet.
        
        Args:
            snippet_id: The ID of the snippet
            tag_name: The name of the tag to remove
            
        Returns:
            The updated snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
            
        # Get current tags
        snippet = self.library[snippet_id]
        current_tags = snippet["metadata"]["tags"]
        
        # Remove tag
        updated_tags = [t for t in current_tags if t["name"].lower() != tag_name.lower()]
        
        # Skip if no change
        if len(updated_tags) == len(current_tags):
            return snippet
            
        updates = {
            "metadata": {
                "tags": updated_tags
            }
        }
        
        return self.update_snippet(snippet_id, updates)
    
    def add_dependency(self, snippet_id: str, depends_on_id: str) -> Dict:
        """
        Add a dependency relationship between snippets.
        
        Args:
            snippet_id: The ID of the dependent snippet
            depends_on_id: The ID of the snippet it depends on
            
        Returns:
            The updated snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
            
        if depends_on_id not in self.library:
            raise ValueError(f"Dependency snippet with ID {depends_on_id} not found")
            
        # Get current dependencies
        snippet = self.library[snippet_id]
        current_deps = snippet["metadata"]["dependencies"]
        
        # Skip if dependency already exists
        if depends_on_id in current_deps:
            return snippet
            
        # Add dependency
        updates = {
            "metadata": {
                "dependencies": current_deps + [depends_on_id]
            }
        }
        
        updated_snippet = self.update_snippet(snippet_id, updates)
        
        # Also update the dependency graph
        self._update_dependency_graph(snippet_id, updated_snippet["metadata"]["dependencies"])
        
        return updated_snippet
    
    def remove_dependency(self, snippet_id: str, depends_on_id: str) -> Dict:
        """
        Remove a dependency relationship between snippets.
        
        Args:
            snippet_id: The ID of the dependent snippet
            depends_on_id: The ID of the snippet to remove as a dependency
            
        Returns:
            The updated snippet
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
            
        # Get current dependencies
        snippet = self.library[snippet_id]
        current_deps = snippet["metadata"]["dependencies"]
        
        # Skip if dependency doesn't exist
        if depends_on_id not in current_deps:
            return snippet
            
        # Remove dependency
        updated_deps = [d for d in current_deps if d != depends_on_id]
        
        updates = {
            "metadata": {
                "dependencies": updated_deps
            }
        }
        
        updated_snippet = self.update_snippet(snippet_id, updates)
        
        # Also update the dependency graph
        self._update_dependency_graph(snippet_id, updated_snippet["metadata"]["dependencies"])
        
        return updated_snippet
    
    def get_dependencies(self, snippet_id: str, recursive: bool = False) -> List[Dict]:
        """
        Get the dependencies of a snippet.
        
        Args:
            snippet_id: The ID of the snippet
            recursive: Whether to include indirect dependencies
            
        Returns:
            List of dependency snippets
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
            
        snippet = self.library[snippet_id]
        direct_deps = snippet["metadata"]["dependencies"]
        
        if not recursive:
            return [self.library[dep_id] for dep_id in direct_deps if dep_id in self.library]
            
        # Get recursive dependencies
        all_deps = set()
        
        def collect_deps(s_id):
            deps = self.library[s_id]["metadata"]["dependencies"]
            for dep_id in deps:
                if dep_id in self.library and dep_id not in all_deps:
                    all_deps.add(dep_id)
                    collect_deps(dep_id)
        
        collect_deps(snippet_id)
        
        return [self.library[dep_id] for dep_id in all_deps]
    
    def get_dependents(self, snippet_id: str, recursive: bool = False) -> List[Dict]:
        """
        Get the snippets that depend on a given snippet.
        
        Args:
            snippet_id: The ID of the snippet
            recursive: Whether to include indirect dependents
            
        Returns:
            List of dependent snippets
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
            
        if snippet_id not in self.reverse_dependencies:
            return []
            
        direct_dependents = self.reverse_dependencies[snippet_id]
        
        if not recursive:
            return [self.library[dep_id] for dep_id in direct_dependents if dep_id in self.library]
            
        # Get recursive dependents
        all_dependents = set()
        
        def collect_dependents(s_id):
            if s_id in self.reverse_dependencies:
                deps = self.reverse_dependencies[s_id]
                for dep_id in deps:
                    if dep_id in self.library and dep_id not in all_dependents:
                        all_dependents.add(dep_id)
                        collect_dependents(dep_id)
        
        collect_dependents(snippet_id)
        
        return [self.library[dep_id] for dep_id in all_dependents]
    
    def export_to_json(self, file_path: Optional[str] = None, include_versions: bool = True) -> Dict:
        """
        Export the vault to JSON.
        
        Args:
            file_path: Optional file path to save the export to
            include_versions: Whether to include version history
            
        Returns:
            Dictionary containing the export data
        """
        # Prepare export data
        data = {
            "snippets": self.library,
            "metadata": {
                "exported_at": time.time(),
                "total_snippets": len(self.library),
                "analytics": self.analytics
            }
        }
        
        if include_versions:
            # Convert version objects to dictionaries
            versions_dict = {}
            for snippet_id, versions in self.version_history.items():
                versions_dict[snippet_id] = [asdict(v) for v in versions]
            
            data["version_history"] = versions_dict
        
        # Save to file if path provided
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
        return data
    
    def import_from_json(self, data: Union[str, Dict], merge: bool = False) -> Dict:
        """
        Import snippets from JSON.
        
        Args:
            data: JSON string or dictionary with export data
            merge: Whether to merge with existing snippets or replace
            
        Returns:
            Summary of the import operation
        """
        # Parse JSON if string provided
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # Try as file path
                try:
                    with open(data, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except:
                    raise ValueError("Invalid JSON or file path")
        
        # Validate data
        if "snippets" not in data:
            raise ValueError("Invalid import data: missing 'snippets' key")
            
        # Clear existing data if not merging
        if not merge:
            self.library = {}
            self.tags = {}
            self.languages = {}
            self.authors = {}
            self.types = {}
            self.status_index = {}
            self.security_index = {}
            self.version_history = {}
            self.dependency_graph = {}
            self.reverse_dependencies = {}
            
        # Import snippets
        imported_count = 0
        for snippet_id, snippet in data["snippets"].items():
            self.library[snippet_id] = snippet
            self._update_indexes(snippet_id, snippet)
            
            # Update dependency graph
            if "metadata" in snippet and "dependencies" in snippet["metadata"]:
                self._update_dependency_graph(snippet_id, snippet["metadata"]["dependencies"])
                
            imported_count += 1
            
        # Import version history if available
        imported_versions = 0
        if "version_history" in data:
            for snippet_id, versions_data in data["version_history"].items():
                versions = []
                for v_data in versions_data:
                    versions.append(Version(**v_data))
                    imported_versions += 1
                
                self.version_history[snippet_id] = versions
        
        return {
            "status": "success",
            "imported_snippets": imported_count,
            "imported_versions": imported_versions
        }
    
    def bulk_update(self, updates: Dict[str, Dict[str, Any]], author: Optional[str] = None) -> Dict:
        """
        Update multiple snippets at once.
        
        Args:
            updates: Dictionary mapping snippet IDs to update dictionaries
            author: Optional author to attribute the updates to
            
        Returns:
            Summary of the batch update operation
        """
        success_count = 0
        failed_updates = {}
        
        for snippet_id, update in updates.items():
            try:
                if snippet_id in self.library:
                    self.update_snippet(snippet_id, update, author)
                    success_count += 1
                else:
                    failed_updates[snippet_id] = "Snippet not found"
            except Exception as e:
                failed_updates[snippet_id] = str(e)
                
        return {
            "total_updates": len(updates),
            "successful_updates": success_count,
            "failed_updates": failed_updates
        }
    
    def analyze_code(self, snippet_id: str) -> Dict:
        """
        Perform in-depth code analysis on a snippet.
        
        Args:
            snippet_id: The ID of the snippet to analyze
            
        Returns:
            Analysis results
        """
        if snippet_id not in self.library:
            raise ValueError(f"Snippet with ID {snippet_id} not found")
            
        snippet = self.library[snippet_id]
        
        # Decrypt code if needed
        code = self._decrypt_if_needed(snippet["code"], snippet["is_encrypted"])
        language = snippet["language"].lower()
        
        # Basic structural analysis
        lines = code.splitlines()
        line_count = len(lines)
        
        # Character analysis
        char_count = len(code)
        
        # Whitespace analysis
        whitespace_count = len(re.findall(r'\s', code))
        whitespace_ratio = whitespace_count / char_count if char_count > 0 else 0
        
        # Comment analysis (simplified)
        comment_patterns = {
            "python": r'#.*$|""".*?"""|\'\'\'.*?\'\'\'',
            "javascript": r'//.*$|/\*.*?\*/',
            "java": r'//.*$|/\*.*?\*/',
            "c": r'//.*$|/\*.*?\*/',
            "cpp": r'//.*$|/\*.*?\*/',
            "csharp": r'//.*$|/\*.*?\*/',
            "php": r'//.*$|/\*.*?\*/|#.*$',
            "ruby": r'#.*$|=begin.*?=end',
            "go": r'//.*$|/\*.*?\*/',
            "swift": r'//.*$|/\*.*?\*/',
            "rust": r'//.*$|/\*.*?\*/',
        }
        
        comment_count = 0
        if language in comment_patterns:
            comments = re.findall(comment_patterns[language], code, re.MULTILINE | re.DOTALL)
            comment_count = len(comments)
        
        comment_ratio = comment_count / line_count if line_count > 0 else 0
        
        # Function/method analysis (simplified)
        function_patterns = {
            "python": r'def\s+\w+\s*\(',
            "javascript": r'function\s+\w+\s*\(|const\s+\w+\s*=\s*\(.*?\)\s*=>',
            "java": r'(public|private|protected)?\s*\w+\s+\w+\s*\(',
            "c": r'\w+\s+\w+\s*\(',
            "cpp": r'\w+\s+\w+\s*\(',
            "csharp": r'(public|private|protected)?\s*\w+\s+\w+\s*\(',
            "php": r'function\s+\w+\s*\(',
            "ruby": r'def\s+\w+\s*',
            "go": r'func\s+\w+\s*\(',
            "swift": r'func\s+\w+\s*\(',
            "rust": r'fn\s+\w+\s*\(',
        }
        
        function_count = 0
        if language in function_patterns:
            functions = re.findall(function_patterns[language], code, re.MULTILINE)
            function_count = len(functions)
            
        # Calculate complexity (simplified)
        complexity = self._calculate_metrics(code, language)
        
        return {
            "snippet_id": snippet_id,
            "language": language,
            "line_count": line_count,
            "character_count": char_count,
            "whitespace_ratio": whitespace_ratio,
            "comment_count": comment_count,
            "comment_ratio": comment_ratio,
            "function_count": function_count,
            "complexity": asdict(complexity),
            "analysis_timestamp": time.time()
        }
    
    def get_analytics(self) -> Dict:
        """
        Get usage analytics of the vault.
        
        Returns:
            Dictionary with analytics data
        """
        # Prepare summary analytics
        summary = {
            "total_snippets": len(self.library),
            "languages": self.get_snippets_by_language_stats(),
            "types": self.get_snippets_by_type_stats(),
            "statuses": self.get_snippets_by_status_stats(),
            "searches": self.analytics.get("total_searches", 0),
            "executions": self.analytics.get("executions", {}),
            "top_tags": self._get_top_tags(10),
            "security_levels": {level: len(snippets) for level, snippets in self.security_index.items()},
            "average_metrics": self._calculate_average_metrics()
        }
        
        return summary
    
    def _get_top_tags(self, limit: int = 10) -> Dict[str, int]:
        """
        Get the most frequently used tags.
        
        Args:
            limit: Maximum number of tags to return
            
        Returns:
            Dictionary mapping top tags to their counts
        """
        tag_counts = {tag: len(snippets) for tag, snippets in self.tags.items()}
        
        # Sort by count (descending)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 'limit' tags
        return dict(sorted_tags[:limit])
    
    def _calculate_average_metrics(self) -> Dict[str, Any]:
        """
        Calculate average metrics across all snippets.
        
        Returns:
            Dictionary with average metric values
        """
        metrics = {
            "lines_of_code": [],
            "comment_ratio": [],
            "cyclomatic_complexity": [],
            "maintainability_index": []
        }
        
        for snippet in self.library.values():
            snippet_metrics = snippet["metadata"]["metrics"]
            
            if "lines_of_code" in snippet_metrics:
                metrics["lines_of_code"].append(snippet_metrics["lines_of_code"])
                
            if "comment_ratio" in snippet_metrics:
                metrics["comment_ratio"].append(snippet_metrics["comment_ratio"])
                
            if "cyclomatic_complexity" in snippet_metrics and snippet_metrics["cyclomatic_complexity"] is not None:
                metrics["cyclomatic_complexity"].append(snippet_metrics["cyclomatic_complexity"])
                
            if "maintainability_index" in snippet_metrics and snippet_metrics["maintainability_index"] is not None:
                metrics["maintainability_index"].append(snippet_metrics["maintainability_index"])
        
        # Calculate averages
        averages = {}
        for metric, values in metrics.items():
            if values:
                averages[metric] = sum(values) / len(values)
            else:
                averages[metric] = None
                
        return averages
    
    def _calculate_metrics(self, code: str, language: str) -> CodeMetrics:
        """
        Calculate code quality metrics for a snippet.
        
        Args:
            code: The code to analyze
            language: The programming language
            
        Returns:
            CodeMetrics object with calculated metrics
        """
        metrics = CodeMetrics()
        
        # Count lines of code
        lines = code.splitlines()
        metrics.lines_of_code = len(lines)
        
        # Calculate comment ratio (simplified)
        comment_count = 0
        comment_patterns = {
            "python": r'#.*$|""".*?"""|\'\'\'.*?\'\'\'',
            "javascript": r'//.*$|/\*.*?\*/',
            "java": r'//.*$|/\*.*?\*/',
            "c": r'//.*$|/\*.*?\*/',
            "cpp": r'//.*$|/\*.*?\*/',
            "csharp": r'//.*$|/\*.*?\*/',
            "php": r'//.*$|/\*.*?\*/|#.*$',
            "ruby": r'#.*$|=begin.*?=end',
            "go": r'//.*$|/\*.*?\*/',
            "swift": r'//.*$|/\*.*?\*/',
            "rust": r'//.*$|/\*.*?\*/',
        }
        
        if language.lower() in comment_patterns:
            comments = re.findall(comment_patterns[language.lower()], code, re.MULTILINE | re.DOTALL)
            comment_count = len(comments)
        
        metrics.comment_ratio = comment_count / len(lines) if len(lines) > 0 else 0
        
        # Simplified cyclomatic complexity calculation (based on branches)
        branch_patterns = {
            "python": r'\bif\b|\belse\b|\belif\b|\bfor\b|\bwhile\b|\btry\b|\bexcept\b|\bwith\b',
            "javascript": r'\bif\b|\belse\b|\bfor\b|\bwhile\b|\bdo\b|\bswitch\b|\bcase\b|\btry\b|\bcatch\b',
            "java": r'\bif\b|\belse\b|\bfor\b|\bwhile\b|\bdo\b|\bswitch\b|\bcase\b|\btry\b|\bcatch\b',
            "c": r'\bif\b|\belse\b|\bfor\b|\bwhile\b|\bdo\b|\bswitch\b|\bcase\b',
            "cpp": r'\bif\b|\belse\b|\bfor\b|\bwhile\b|\bdo\b|\bswitch\b|\bcase\b|\btry\b|\bcatch\b',
            "csharp": r'\bif\b|\belse\b|\bfor\b|\bwhile\b|\bdo\b|\bswitch\b|\bcase\b|\btry\b|\bcatch\b',
            "php": r'\bif\b|\belse\b|\belseif\b|\bfor\b|\bwhile\b|\bdo\b|\bswitch\b|\bcase\b|\btry\b|\bcatch\b',
            "ruby": r'\bif\b|\belse\b|\belsif\b|\bunless\b|\bwhile\b|\buntil\b|\bfor\b|\bcase\b|\bwhen\b|\bbegin\b|\brescue\b',
            "go": r'\bif\b|\belse\b|\bfor\b|\bswitch\b|\bcase\b|\bdefer\b|\bselect\b',
            "swift": r'\bif\b|\belse\b|\bguard\b|\bfor\b|\bwhile\b|\bswitch\b|\bcase\b|\bdo\b|\bcatch\b',
            "rust": r'\bif\b|\belse\b|\bloop\b|\bwhile\b|\bfor\b|\bmatch\b|\bcatch\b',
        }
        
        complexity = 1  # Base complexity
        if language.lower() in branch_patterns:
            branches = re.findall(branch_patterns[language.lower()], code)
            complexity += len(branches)
        
        metrics.cyclomatic_complexity = complexity
        
        # Simplified maintainability index calculation
        if metrics.lines_of_code > 0:
            # Simple formula: 100 - 20 * log(complexity) - 10 * log(lines)
            import math
            try:
                metrics.maintainability_index = 100 - 20 * math.log(complexity) - 10 * math.log(metrics.lines_of_code)
                metrics.maintainability_index = max(0, min(100, metrics.maintainability_index))
            except:
                metrics.maintainability_index = 50  # Default mid-range value if calculation fails
        
        return metrics
    
    def _update_indexes(self, snippet_id: str, snippet: Dict) -> None:
        """
        Update all indexes for a snippet.
        
        Args:
            snippet_id: The ID of the snippet
            snippet: The snippet data
        """
        # Language index
        language = snippet["language"].lower()
        self.languages.setdefault(language, set()).add(snippet_id)
        
        # Tags index
        for tag in snippet["metadata"]["tags"]:
            tag_name = tag["name"].lower()
            self.tags.setdefault(tag_name, set()).add(snippet_id)
        
        # Type index
        snippet_type = snippet["metadata"]["type"]
        self.types.setdefault(snippet_type, set()).add(snippet_id)
        
        # Status index
        status = snippet["metadata"]["status"]
        self.status_index.setdefault(status, set()).add(snippet_id)
        
        # Security level index
        security_level = snippet["metadata"]["security_level"]
        self.security_index.setdefault(security_level, set()).add(snippet_id)
        
        # Authors index
        for author in snippet["metadata"]["authors"]:
            self.authors.setdefault(author, set()).add(snippet_id)
        
        # Keyword index
        keywords = self._extract_keywords(snippet)
        for keyword in keywords:
            self.keyword_index.setdefault(keyword, set()).add(snippet_id)
    
    def _remove_from_indexes(self, snippet_id: str, language: str, tags: List[str], 
                           type_: str, status: str, security: str, authors: List[str]) -> None:
        """
        Remove a snippet from all indexes.
        
        Args:
            snippet_id: The ID of the snippet
            language: The snippet's language
            tags: The snippet's tags
            type_: The snippet's type
            status: The snippet's status
            security: The snippet's security level
            authors: The snippet's authors
        """
        # Language index
        if language in self.languages:
            self.languages[language].discard(snippet_id)
            if not self.languages[language]:
                del self.languages[language]
        
        # Tags index
        for tag in tags:
            tag = tag.lower()
            if tag in self.tags:
                self.tags[tag].discard(snippet_id)
                if not self.tags[tag]:
                    del self.tags[tag]
        
        # Type index
        if type_ in self.types:
            self.types[type_].discard(snippet_id)
            if not self.types[type_]:
                del self.types[type_]
        
        # Status index
        if status in self.status_index:
            self.status_index[status].discard(snippet_id)
            if not self.status_index[status]:
                del self.status_index[status]
        
        # Security level index
        if security in self.security_index:
            self.security_index[security].discard(snippet_id)
            if not self.security_index[security]:
                del self.security_index[security]
        
        # Authors index
        for author in authors:
            if author in self.authors:
                self.authors[author].discard(snippet_id)
                if not self.authors[author]:
                    del self.authors[author]
        
        # Keyword index
        for keyword, snippets in list(self.keyword_index.items()):
            if snippet_id in snippets:
                snippets.discard(snippet_id)
                if not snippets:
                    del self.keyword_index[keyword]
    
    def _update_dependency_graph(self, snippet_id: str, dependencies: List[str]) -> None:
        """
        Update the dependency graph for a snippet.
        
        Args:
            snippet_id: The ID of the snippet
            dependencies: List of dependency snippet IDs
        """
        # Clear existing dependencies
        self.dependency_graph[snippet_id] = set(dependencies)
        
        # Update reverse dependencies
        for dep_id in dependencies:
            self.reverse_dependencies.setdefault(dep_id, set()).add(snippet_id)
            
        # Remove old reverse dependencies
        for dep_id, dependents in list(self.reverse_dependencies.items()):
            if dep_id != snippet_id and snippet_id in dependents and dep_id not in dependencies:
                dependents.discard(snippet_id)
                if not dependents:
                    del self.reverse_dependencies[dep_id]
    
    def _extract_keywords(self, snippet: Dict) -> Set[str]:
        """
        Extract important keywords from a snippet for indexing.
        
        Args:
            snippet: The snippet data
            
        Returns:
            Set of extracted keywords
        """
        keywords = set()
        
        # Add words from name
        name_words = re.findall(r'\w+', snippet["name"].lower())
        keywords.update(w for w in name_words if len(w) > 3)
        
        # Add words from description
        desc_words = re.findall(r'\w+', snippet["description"].lower())
        keywords.update(w for w in desc_words if len(w) > 3)
        
        # Add tags
        for tag in snippet["metadata"]["tags"]:
            keywords.add(tag["name"].lower())
            if tag.get("category"):
                keywords.add(tag["category"].lower())
        
        # Add type and other metadata
        keywords.add(snippet["metadata"]["type"].lower())
        keywords.add(snippet["metadata"]["status"].lower())
        keywords.add(snippet["language"].lower())
        
        return keywords
    
    def _calculate_relevance(self, query: str, text: str, case_sensitive: bool = False, fuzzy: bool = False) -> float:
        """
        Calculate relevance score for a query against text.
        
        Args:
            query: The search query
            text: The text to search in
            case_sensitive: Whether to perform case-sensitive matching
            fuzzy: Whether to use fuzzy matching
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not query or not text:
            return 0.0
            
        if not case_sensitive:
            query = query.lower()
            text = text.lower()
            
        # Exact match gets highest score
        if query == text:
            return 1.0
            
        # Contains check
        if query in text:
            # If query is found at the start, give higher score
            if text.startswith(query):
                return 0.9
            else:
                # Score based on position (earlier is better)
                position = text.find(query)
                position_factor = 1.0 - (position / len(text))
                
                # Score based on length ratio (longer match relative to text is better)
                length_ratio = len(query) / len(text)
                
                return 0.7 * position_factor + 0.2 * length_ratio
                
        # If fuzzy matching enabled, calculate similarity
        if fuzzy:
            # Simple word matching
            query_words = set(re.findall(r'\w+', query))
            text_words = set(re.findall(r'\w+', text))
            
            if not query_words:
                return 0.0
                
            # Calculate Jaccard similarity
            intersection = query_words.intersection(text_words)
            union = query_words.union(text_words)
            
            if not union:
                return 0.0
                
            jaccard = len(intersection) / len(union)
            
            # Calculate ratio of matched words
            if query_words:
                word_match_ratio = len(intersection) / len(query_words)
            else:
                word_match_ratio = 0.0
                
            # More weight on word match ratio
            return 0.3 * jaccard + 0.4 * word_match_ratio
                
        return 0.0
    
    def _encrypt_if_needed(self, code: str, security_level: str) -> str:
        """
        Encrypt code if required by security level.
        
        Args:
            code: The code to potentially encrypt
            security_level: The security level
            
        Returns:
            Original or encrypted code
        """
        if security_level in [SecurityLevel.PRIVATE.value, SecurityLevel.RESTRICTED.value]:
            if self.encryption_key:
                # Simple encryption for demonstration (use proper encryption in production)
                return base64.b64encode(code.encode("utf-8")).decode("utf-8")
        return code
    
    def _decrypt_if_needed(self, code: str, is_encrypted: bool) -> str:
        """
        Decrypt code if it's encrypted.
        
        Args:
            code: The potentially encrypted code
            is_encrypted: Whether the code is encrypted
            
        Returns:
            Decrypted or original code
        """
        if is_encrypted:
            if self.encryption_key:
                # Simple decryption for demonstration (use proper decryption in production)
                try:
                    return base64.b64decode(code.encode("utf-8")).decode("utf-8")
                except:
                    return code
        return code
    
    def _execute_python(self, code: str, params: Dict[str, Any]) -> Any:
        """
        Execute Python code with parameters.
        
        Args:
            code: The Python code to execute
            params: Parameters to pass to the code
            
        Returns:
            Execution result
        """
        # This is simplified and unsafe - real implementation would use better isolation
        # Use subprocess, Docker containers, or other secure execution environments in production
        local_env = params.copy()
        
        try:
            # Add params to locals for execution
            result = {}
            exec(code, {"params": params}, result)
            
            # Look for return_value in the execution result
            if "return_value" in result:
                return result["return_value"]
            else:
                return None
        except Exception as e:
            raise RuntimeError(f"Python execution error: {str(e)}")

    def __len__(self) -> int:
        """
        Get the number of snippets in the vault.
        
        Returns:
            Number of snippets
        """
        return len(self.library)
    
    def __contains__(self, snippet_id: str) -> bool:
        """
        Check if a snippet ID exists in the vault.
        
        Args:
            snippet_id: The ID to check
            
        Returns:
            Whether the ID exists
        """
        return snippet_id in self.library

# Create a global instance for easy access
vault = AdvancedSoftwareVault()