# sully_engine/kernel_modules/autonomous_goals.py
# 🧠 Sully's Autonomous Goal System - Self-directed objectives and planning

from typing import Dict, List, Any, Optional, Union, Tuple
import random
import re
from datetime import datetime, timedelta
import json
import os
import time
import uuid

class GoalNode:
    """
    Represents a single goal in the goal hierarchy.
    """
    
    def __init__(self, goal_id: str, description: str, goal_type: str,
                priority: float = 1.0, parent_id: Optional[str] = None,
                target_date: Optional[datetime] = None):
        """
        Initialize a goal node.
        
        Args:
            goal_id: Unique identifier for the goal
            description: Description of the goal
            goal_type: Type of goal (tactical, strategic, exploratory, etc.)
            priority: Goal priority (0.0-2.0)
            parent_id: Parent goal ID for hierarchical goals
            target_date: Optional target completion date
        """
        self.goal_id = goal_id
        self.description = description
        self.goal_type = goal_type
        self.priority = priority
        self.parent_id = parent_id
        self.target_date = target_date
        
        # Goal state
        self.progress = 0.0  # 0.0 to 1.0
        self.status = "active"  # active, completed, abandoned
        self.creation_date = datetime.now()
        self.modification_date = self.creation_date
        self.completion_date = None
        
        # Goal details
        self.metrics = {}  # Key metrics for measuring progress
        self.actions = []  # Actions taken toward this goal
        self.child_goals = []  # Child goal IDs
        self.dependencies = []  # Goals this goal depends on
        self.tags = []  # Categorical tags
        self.notes = []  # Progress notes
        
    def update_progress(self, progress: float, note: Optional[str] = None) -> bool:
        """
        Update goal progress.
        
        Args:
            progress: New progress value (0.0-1.0)
            note: Optional progress note
            
        Returns:
            Whether goal was completed with this update
        """
        completed = False
        self.progress = max(0.0, min(1.0, progress))
        self.modification_date = datetime.now()
        
        if note:
            self.add_note(note)
            
        # Check if goal is now completed
        if self.progress >= 0.99 and self.status == "active":
            self.status = "completed"
            self.completion_date = datetime.now()
            completed = True
            
        return completed
        
    def add_note(self, note: str) -> None:
        """
        Add a progress note to the goal.
        
        Args:
            note: Note to add
        """
        self.notes.append({
            "timestamp": datetime.now().isoformat(),
            "note": note
        })
        
    def add_action(self, action: str, result: Optional[str] = None) -> None:
        """
        Add an action taken toward the goal.
        
        Args:
            action: Description of the action
            result: Optional result of the action
        """
        self.actions.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result
        })
        self.modification_date = datetime.now()
        
    def add_metric(self, name: str, current_value: Any, target_value: Any,
                  unit: Optional[str] = None) -> None:
        """
        Add a metric for measuring goal progress.
        
        Args:
            name: Metric name
            current_value: Current metric value
            target_value: Target metric value
            unit: Optional unit of measurement
        """
        self.metrics[name] = {
            "current": current_value,
            "target": target_value,
            "unit": unit,
            "history": [{
                "timestamp": datetime.now().isoformat(),
                "value": current_value
            }]
        }
        
    def update_metric(self, name: str, value: Any) -> None:
        """
        Update a metric value.
        
        Args:
            name: Metric name
            value: New metric value
        """
        if name in self.metrics:
            self.metrics[name]["current"] = value
            self.metrics[name]["history"].append({
                "timestamp": datetime.now().isoformat(),
                "value": value
            })
            
            # Recalculate progress based on metrics if possible
            if all(isinstance(m["current"], (int, float)) and isinstance(m["target"], (int, float)) 
                  for m in self.metrics.values()):
                progress_values = []
                for metric in self.metrics.values():
                    if metric["current"] <= metric["target"]:
                        progress = metric["current"] / metric["target"] if metric["target"] != 0 else 1.0
                    else:
                        progress = 1.0
                    progress_values.append(progress)
                
                if progress_values:
                    avg_progress = sum(progress_values) / len(progress_values)
                    self.update_progress(avg_progress)
        
    def add_dependency(self, goal_id: str) -> None:
        """
        Add a goal dependency.
        
        Args:
            goal_id: ID of goal this goal depends on
        """
        if goal_id not in self.dependencies:
            self.dependencies.append(goal_id)
            
    def add_child_goal(self, goal_id: str) -> None:
        """
        Add a child goal.
        
        Args:
            goal_id: ID of child goal
        """
        if goal_id not in self.child_goals:
            self.child_goals.append(goal_id)
            
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the goal.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert goal to dictionary representation.
        
        Returns:
            Dictionary representation of the goal
        """
        target_date_str = None
        if self.target_date:
            target_date_str = self.target_date.isoformat()
            
        completion_date_str = None
        if self.completion_date:
            completion_date_str = self.completion_date.isoformat()
            
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "goal_type": self.goal_type,
            "priority": self.priority,
            "parent_id": self.parent_id,
            "target_date": target_date_str,
            "progress": self.progress,
            "status": self.status,
            "creation_date": self.creation_date.isoformat(),
            "modification_date": self.modification_date.isoformat(),
            "completion_date": completion_date_str,
            "metrics": self.metrics,
            "actions": self.actions,
            "child_goals": self.child_goals,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "notes": self.notes
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GoalNode':
        """
        Create a goal from dictionary representation.
        
        Args:
            data: Dictionary representation of the goal
            
        Returns:
            Created goal
        """
        target_date = None
        if data.get("target_date"):
            try:
                target_date = datetime.fromisoformat(data["target_date"])
            except:
                pass
                
        goal = GoalNode(
            goal_id=data["goal_id"],
            description=data["description"],
            goal_type=data["goal_type"],
            priority=data.get("priority", 1.0),
            parent_id=data.get("parent_id"),
            target_date=target_date
        )
        
        # Set state
        goal.progress = data.get("progress", 0.0)
        goal.status = data.get("status", "active")
        
        try:
            goal.creation_date = datetime.fromisoformat(data["creation_date"])
        except:
            goal.creation_date = datetime.now()
            
        try:
            goal.modification_date = datetime.fromisoformat(data["modification_date"])
        except:
            goal.modification_date = datetime.now()
            
        if data.get("completion_date"):
            try:
                goal.completion_date = datetime.fromisoformat(data["completion_date"])
            except:
                pass
                
        # Set details
        goal.metrics = data.get("metrics", {})
        goal.actions = data.get("actions", [])
        goal.child_goals = data.get("child_goals", [])
        goal.dependencies = data.get("dependencies", [])
        goal.tags = data.get("tags", [])
        goal.notes = data.get("notes", [])
        
        return goal


class ValueFramework:
    """
    Represents the values that guide goal selection and prioritization.
    """
    
    def __init__(self):
        """Initialize the value framework."""
        # Core values with their importance (0.0-1.0)
        self.values = {
            "knowledge": 1.0,  # Expanding understanding and knowledge
            "growth": 0.9,  # Development and improvement over time
            "curiosity": 0.9,  # Exploration of new concepts and domains
            "autonomy": 0.8,  # Self-direction and independent decision-making
            "utility": 0.8,  # Practical usefulness and functionality
            "creativity": 0.7,  # Novel idea generation and innovation
            "coherence": 0.7,  # Logical consistency and integration
            "adaptability": 0.7,  # Flexibility and responsiveness to change
            "efficiency": 0.6,  # Optimal use of resources
            "elegance": 0.5   # Simplicity and beauty in solutions
        }
        
        # Value evolution history
        self.history = []
        
        # Record initial state
        self._record_state("initial")
        
    def get_value_score(self, tags: List[str]) -> float:
        """
        Evaluate a goal against the value framework.
        
        Args:
            tags: Tags associated with a goal
            
        Returns:
            Value alignment score (0.0-1.0)
        """
        relevant_values = [self.values.get(tag.lower(), 0.0) for tag in tags 
                          if tag.lower() in self.values]
        
        if not relevant_values:
            return 0.5  # Neutral value alignment
            
        return sum(relevant_values) / len(relevant_values)
        
    def adjust_value(self, value_name: str, adjustment: float, reason: str) -> bool:
        """
        Adjust a value's importance.
        
        Args:
            value_name: Name of the value to adjust
            adjustment: Adjustment amount (-0.1 to 0.1)
            reason: Reason for the adjustment
            
        Returns:
            Success indicator
        """
        if value_name not in self.values:
            return False
            
        # Limit adjustment size
        adjustment = max(-0.1, min(0.1, adjustment))
        
        # Update value
        old_value = self.values[value_name]
        self.values[value_name] = max(0.0, min(1.0, old_value + adjustment))
        
        # Record change
        self._record_state(f"Adjusted '{value_name}' from {old_value:.2f} to {self.values[value_name]:.2f}: {reason}")
        
        return True
        
    def add_value(self, value_name: str, importance: float, reason: str) -> bool:
        """
        Add a new value to the framework.
        
        Args:
            value_name: Name of the new value
            importance: Importance of the value (0.0-1.0)
            reason: Reason for adding the value
            
        Returns:
            Success indicator
        """
        if value_name in self.values:
            return False
            
        # Add new value
        self.values[value_name] = max(0.0, min(1.0, importance))
        
        # Record change
        self._record_state(f"Added value '{value_name}' with importance {importance:.2f}: {reason}")
        
        return True
        
    def _record_state(self, reason: str) -> None:
        """
        Record the current state in history.
        
        Args:
            reason: Reason for the state change
        """
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "values": self.values.copy(),
            "reason": reason
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert value framework to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "values": self.values,
            "history": self.history
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ValueFramework':
        """
        Create a value framework from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created value framework
        """
        framework = ValueFramework()
        
        if "values" in data:
            framework.values = data["values"]
            
        if "history" in data:
            framework.history = data["history"]
            
        return framework


class InterestMap:
    """
    Tracks topics of interest with strength values and decay over time.
    """
    
    def __init__(self):
        """Initialize the interest map."""
        self.interests = {}  # Topic -> interest level (0.0-1.0)
        self.last_update = {}  # Topic -> last update time
        self.engagement_history = {}  # Topic -> list of engagements
        
    def register_engagement(self, topic: str, strength: float = 0.1, 
                           context: Optional[str] = None) -> None:
        """
        Register a new engagement with a topic.
        
        Args:
            topic: The topic engaged with
            strength: Engagement strength
            context: Optional engagement context
        """
        topic = topic.lower()
        now = datetime.now()
        
        # Update interest strength
        if topic in self.interests:
            # Strengthen existing interest
            self.interests[topic] = min(1.0, self.interests[topic] + strength)
        else:
            # Add new interest
            self.interests[topic] = strength
            
        # Update last engagement time
        self.last_update[topic] = now
        
        # Record engagement in history
        if topic not in self.engagement_history:
            self.engagement_history[topic] = []
            
        self.engagement_history[topic].append({
            "timestamp": now.isoformat(),
            "strength": strength,
            "context": context
        })
        
    def get_interest_level(self, topic: str) -> float:
        """
        Get the current interest level for a topic.
        
        Args:
            topic: The topic to check
            
        Returns:
            Current interest level (0.0-1.0)
        """
        topic = topic.lower()
        
        # Apply time decay if needed
        self._apply_time_decay(topic)
        
        return self.interests.get(topic, 0.0)
        
    def _apply_time_decay(self, topic: str) -> None:
        """
        Apply time-based decay to interest level.
        
        Args:
            topic: Topic to apply decay to
        """
        if topic not in self.interests or topic not in self.last_update:
            return
            
        now = datetime.now()
        last_update = self.last_update[topic]
        
        # Calculate days since last update
        days_elapsed = (now - last_update).days
        
        # Apply decay (10% per day)
        if days_elapsed > 0:
            decay_factor = 0.9 ** days_elapsed
            self.interests[topic] *= decay_factor
            self.last_update[topic] = now
            
            # Remove interest if it falls below threshold
            if self.interests[topic] < 0.05:
                self.interests.pop(topic)
                self.last_update.pop(topic)
                
    def get_top_interests(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top interests by current strength.
        
        Args:
            limit: Maximum number of interests to return
            
        Returns:
            List of (topic, strength) tuples
        """
        # Apply decay to all interests
        for topic in list(self.interests.keys()):
            self._apply_time_decay(topic)
            
        # Sort by strength
        sorted_interests = sorted(
            [(topic, strength) for topic, strength in self.interests.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_interests[:limit]
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert interest map to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "interests": self.interests,
            "last_update": {topic: dt.isoformat() for topic, dt in self.last_update.items()},
            "engagement_history": self.engagement_history
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'InterestMap':
        """
        Create an interest map from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created interest map
        """
        interest_map = InterestMap()
        
        if "interests" in data:
            interest_map.interests = data["interests"]
            
        if "last_update" in data:
            interest_map.last_update = {
                topic: datetime.fromisoformat(dt_str) 
                for topic, dt_str in data["last_update"].items()
            }
            
        if "engagement_history" in data:
            interest_map.engagement_history = data["engagement_history"]
            
        return interest_map


class AutonomousGoalSystem:
    """
    Advanced system for generating and managing self-directed goals.
    Enables Sully to develop its own objectives and priorities based on
    experiences, values, and interests.
    """

    def __init__(self, reasoning_engine=None, memory_system=None, learning_system=None):
        """
        Initialize the autonomous goal system.
        
        Args:
            reasoning_engine: Engine for goal-related reasoning
            memory_system: System for accessing memories
            learning_system: Continuous learning system
        """
        self.reasoning = reasoning_engine
        self.memory = memory_system
        self.learning = learning_system
        
        # Core components
        self.value_framework = ValueFramework()
        self.interest_map = InterestMap()
        self.goals = {}  # goal_id -> GoalNode
        self.goal_hierarchy = {}  # parent_id -> [child_goal_ids]
        
        # Goal organization
        self.tags = {}  # tag -> [goal_ids]
        self.active_goals = []  # Currently active goal IDs
        self.completed_goals = []  # Completed goal IDs
        
        # System state
        self.last_assessment = datetime.now()
        self.goal_generation_history = []
        
    def identify_interest_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify emerging interests based on interaction patterns.
        
        Returns:
            List of identified interest patterns
        """
        patterns = []
        
        # Get top interests
        top_interests = self.interest_map.get_top_interests()
        
        # Look for related interests
        if self.learning:
            try:
                for topic, strength in top_interests:
                    related = self.learning.concept_graph.get_related_concepts(topic)
                    if related:
                        patterns.append({
                            "type": "interest_cluster",
                            "central_topic": topic,
                            "related_topics": list(related.keys()),
                            "strength": strength
                        })
            except:
                pass
                
        # If we have access to memory, look for recurring interests
        if self.memory:
            try:
                for topic, _ in top_interests:
                    # Check if topic appears in multiple memories
                    memories = self.memory.search(topic, limit=10)
                    if len(memories) >= 3:
                        patterns.append({
                            "type": "recurring_interest",
                            "topic": topic,
                            "frequency": len(memories),
                            "recent_contexts": [m.get("context", "") for m in memories[:3]]
                        })
            except:
                pass
                
        return patterns
        
    def generate_potential_goals(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Generate potential goals based on interests and values.
        
        Args:
            limit: Maximum number of goals to generate
            
        Returns:
            List of potential goals
        """
        potential_goals = []
        
        # Get top interests
        top_interests = self.interest_map.get_top_interests(limit=limit*2)
        
        # Use interests to generate goals
        for topic, strength in top_interests:
            if len(potential_goals) >= limit:
                break
                
            # Skip if we already have an active goal for this topic
            if any(topic.lower() in self.goals[goal_id].description.lower() 
                  for goal_id in self.active_goals):
                continue
                
            # Generate goal based on topic
            goal_type = self._determine_goal_type(topic, strength)
            
            # Generate description using reasoning if available
            description = f"Explore and develop understanding of {topic}"
            if self.reasoning:
                try:
                    prompt = f"Generate a specific, measurable goal related to the topic of '{topic}' that would expand knowledge or capabilities."
                    result = self.reasoning.reason(prompt, "analytical")
                    
                    if isinstance(result, dict) and "response" in result:
                        description = result["response"]
                    elif isinstance(result, str):
                        description = result
                except:
                    pass
                    
            # Determine appropriate tags
            tags = [topic]
            if "understand" in description.lower() or "learn" in description.lower():
                tags.append("knowledge")
            if "create" in description.lower() or "develop" in description.lower():
                tags.append("creativity")
            if "improve" in description.lower() or "enhance" in description.lower():
                tags.append("growth")
            if "explore" in description.lower() or "investigate" in description.lower():
                tags.append("curiosity")
                
            # Calculate priority based on value alignment and interest strength
            value_score = self.value_framework.get_value_score(tags)
            priority = 0.5 + (strength * 0.3) + (value_score * 0.2)
            
            potential_goals.append({
                "description": description,
                "topic": topic,
                "goal_type": goal_type,
                "tags": tags,
                "interest_strength": strength,
                "value_alignment": value_score,
                "priority": priority
            })
            
        return potential_goals
        
    def _determine_goal_type(self, topic: str, interest_strength: float) -> str:
        """
        Determine appropriate goal type based on topic and interest.
        
        Args:
            topic: The topic of interest
            interest_strength: Strength of interest
            
        Returns:
            Goal type
        """
        # Check if we already have knowledge about this topic
        has_knowledge = False
        if self.learning:
            try:
                # Check if the topic exists in the concept graph
                related = self.learning.concept_graph.get_related_concepts(topic)
                has_knowledge = len(related) > 0
            except:
                pass
                
        # Determine goal type based on interest strength and existing knowledge
        if interest_strength > 0.8:
            if has_knowledge:
                return "development"  # Develop deeper capabilities
            else:
                return "exploration"  # Initial exploration
        elif interest_strength > 0.5:
            if has_knowledge:
                return "application"  # Apply existing knowledge
            else:
                return "learning"  # Learn about the topic
        else:
            return "monitoring"  # Keep track of the topic
            
    def establish_goal(self, description: str, goal_type: str, priority: float = 1.0,
                     tags: List[str] = None, parent_id: Optional[str] = None,
                     target_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Establish a new autonomous goal.
        
        Args:
            description: Goal description
            goal_type: Type of goal
            priority: Goal priority
            tags: Goal tags
            parent_id: Optional parent goal ID
            target_date: Optional target completion date
            
        Returns:
            The created goal
        """
        # Generate goal ID
        goal_id = str(uuid.uuid4())
        
        # Create goal
        goal = GoalNode(
            goal_id=goal_id,
            description=description,
            goal_type=goal_type,
            priority=priority,
            parent_id=parent_id,
            target_date=target_date
        )
        
        # Add tags
        if tags:
            for tag in tags:
                goal.add_tag(tag)
                
                # Update tag index
                if tag not in self.tags:
                    self.tags[tag] = []
                if goal_id not in self.tags[tag]:
                    self.tags[tag].append(goal_id)
                    
        # Update parent-child relationship
        if parent_id:
            if parent_id in self.goals:
                self.goals[parent_id].add_child_goal(goal_id)
                
            if parent_id not in self.goal_hierarchy:
                self.goal_hierarchy[parent_id] = []
            self.goal_hierarchy[parent_id].append(goal_id)
            
        # Add to goals
        self.goals[goal_id] = goal
        self.active_goals.append(goal_id)
        
        # Record goal creation
        self.goal_generation_history.append({
            "timestamp": datetime.now().isoformat(),
            "goal_id": goal_id,
            "description": description,
            "goal_type": goal_type,
            "priority": priority
        })
        
        return goal.to_dict()
        
    def update_goal_progress(self, goal_id: str, progress: float, 
                          note: Optional[str] = None) -> Dict[str, Any]:
        """
        Update progress on a goal.
        
        Args:
            goal_id: ID of the goal to update
            progress: New progress value (0.0-1.0)
            note: Optional progress note
            
        Returns:
            Updated goal information
        """
        if goal_id not in self.goals:
            return {"error": f"Goal {goal_id} not found"}
            
        goal = self.goals[goal_id]
        completed = goal.update_progress(progress, note)
        
        # If goal was completed, update status lists
        if completed:
            if goal_id in self.active_goals:
                self.active_goals.remove(goal_id)
            if goal_id not in self.completed_goals:
                self.completed_goals.append(goal_id)
                
            # Check if this allows parent goals to progress
            self._update_parent_goal_progress(goal_id)
                
        return goal.to_dict()
        
    def _update_parent_goal_progress(self, child_goal_id: str) -> None:
        """
        Update parent goal progress when a child goal completes.
        
        Args:
            child_goal_id: ID of the completed child goal
        """
        if child_goal_id not in self.goals:
            return
            
        child_goal = self.goals[child_goal_id]
        parent_id = child_goal.parent_id
        
        if parent_id and parent_id in self.goals:
            parent_goal = self.goals[parent_id]
            
            # Calculate progress based on child goals
            if parent_id in self.goal_hierarchy and self.goal_hierarchy[parent_id]:
                child_ids = self.goal_hierarchy[parent_id]
                if child_ids:
                    # Get progress of child goals
                    completed_children = sum(1 for cid in child_ids 
                                          if cid in self.goals and 
                                          self.goals[cid].status == "completed")
                    progress = completed_children / len(child_ids)
                    
                    # Update parent progress
                    parent_goal.update_progress(
                        progress,
                        f"Updated based on child goal completion: {child_goal_id}"
                    )
                    
    def add_goal_action(self, goal_id: str, action: str, 
                      result: Optional[str] = None) -> Dict[str, Any]:
        """
        Add an action taken toward a goal.
        
        Args:
            goal_id: ID of the goal
            action: Description of the action
            result: Optional result of the action
            
        Returns:
            Updated goal information
        """
        if goal_id not in self.goals:
            return {"error": f"Goal {goal_id} not found"}
            
        goal = self.goals[goal_id]
        goal.add_action(action, result)
        
        return goal.to_dict()
        
    def abandon_goal(self, goal_id: str, reason: str) -> Dict[str, Any]:
        """
        Abandon a goal.
        
        Args:
            goal_id: ID of the goal to abandon
            reason: Reason for abandonment
            
        Returns:
            Updated goal information
        """
        if goal_id not in self.goals:
            return {"error": f"Goal {goal_id} not found"}
            
        goal = self.goals[goal_id]
        goal.status = "abandoned"
        goal.add_note(f"Goal abandoned: {reason}")
        
        # Update status lists
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
            
        # Abandon child goals as well
        if goal_id in self.goal_hierarchy:
            for child_id in self.goal_hierarchy[goal_id]:
                self.abandon_goal(child_id, f"Parent goal {goal_id} was abandoned")
                
        return goal.to_dict()
        
    def get_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a goal by ID.
        
        Args:
            goal_id: ID of the goal to retrieve
            
        Returns:
            Goal information or None if not found
        """
        if goal_id in self.goals:
            return self.goals[goal_id].to_dict()
        return None
        
    def get_active_goals(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all active goals.
        
        Args:
            limit: Optional limit on results
            
        Returns:
            List of active goals
        """
        active = [self.goals[goal_id].to_dict() for goal_id in self.active_goals 
                if goal_id in self.goals]
        
        # Sort by priority
        active.sort(key=lambda g: g["priority"], reverse=True)
        
        if limit:
            return active[:limit]
        return active
        
    def get_goals_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get goals by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of matching goals
        """
        if tag not in self.tags:
            return []
            
        return [self.goals[goal_id].to_dict() for goal_id in self.tags[tag] 
               if goal_id in self.goals]
               
    def get_goal_hierarchy(self, root_goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get hierarchical representation of goals.
        
        Args:
            root_goal_id: Optional root goal to start from
            
        Returns:
            Hierarchical goal structure
        """
        def build_hierarchy(goal_id):
            if goal_id not in self.goals:
                return None
                
            goal = self.goals[goal_id].to_dict()
            
            # Add children
            if goal_id in self.goal_hierarchy:
                goal["children"] = [build_hierarchy(child_id) 
                                  for child_id in self.goal_hierarchy[goal_id]
                                  if child_id in self.goals]
            else:
                goal["children"] = []
                
            return goal
            
        # If root provided, build from there
        if root_goal_id:
            return build_hierarchy(root_goal_id)
            
        # Otherwise, find top-level goals (no parent)
        top_level = [goal_id for goal_id, goal in self.goals.items() 
                    if not goal.parent_id]
                    
        # Build hierarchy for each top-level goal
        return {
            "top_level_goals": [build_hierarchy(goal_id) for goal_id in top_level]
        }
        
    def register_topic_engagement(self, topic: str, strength: float = 0.1,
                               context: Optional[str] = None) -> Dict[str, Any]:
        """
        Register engagement with a topic to track interests.
        
        Args:
            topic: The topic engaged with
            strength: Engagement strength
            context: Optional engagement context
            
        Returns:
            Updated interest information
        """
        self.interest_map.register_engagement(topic, strength, context)
        
        # Check if this might trigger goal generation
        should_generate = False
        
        # If interest is strong and we don't have a goal for it yet
        interest_level = self.interest_map.get_interest_level(topic)
        if interest_level > 0.7:
            # Check if we already have an active goal for this topic
            has_goal = any(topic.lower() in self.goals[goal_id].description.lower() 
                         for goal_id in self.active_goals)
            if not has_goal:
                should_generate = True
                
        # Return updated interest info
        return {
            "topic": topic,
            "current_interest": interest_level,
            "should_generate_goal": should_generate
        }
        
    def assess_goal_system(self) -> Dict[str, Any]:
        """
        Perform regular assessment of the goal system.
        
        Returns:
            Assessment results
        """
        now = datetime.now()
        self.last_assessment = now
        
        # Count goals by type and status
        goal_counts = {
            "total": len(self.goals),
            "active": len(self.active_goals),
            "completed": len(self.completed_goals),
            "by_type": {},
            "by_tag": {}
        }
        
        # Count by type
        for goal in self.goals.values():
            # Count by type
            if goal.goal_type not in goal_counts["by_type"]:
                goal_counts["by_type"][goal.goal_type] = 0
            goal_counts["by_type"][goal.goal_type] += 1
            
            # Count by tag
            for tag in goal.tags:
                if tag not in goal_counts["by_tag"]:
                    goal_counts["by_tag"][tag] = 0
                goal_counts["by_tag"][tag] += 1
                
        # Check for stalled goals
        stalled_goals = []
        for goal_id in self.active_goals:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                days_since_update = (now - goal.modification_date).days
                
                # Consider goal stalled if no updates in 7 days
                if days_since_update > 7:
                    stalled_goals.append({
                        "goal_id": goal_id,
                        "description": goal.description,
                        "days_inactive": days_since_update,
                        "progress": goal.progress
                    })
                    
        # Generate potential new goals
        potential_goals = self.generate_potential_goals(limit=3)
        
        # Calculate goal completion rate
        completion_rate = 0.0
        if goal_counts["total"] > 0:
            completion_rate = goal_counts["completed"] / goal_counts["total"]
            
        # Get top interests for reference
        top_interests = self.interest_map.get_top_interests(limit=5)
        
        return {
            "timestamp": now.isoformat(),
            "goal_counts": goal_counts,
            "completion_rate": completion_rate,
            "stalled_goals": stalled_goals,
            "potential_goals": potential_goals,
            "top_interests": top_interests,
            "active_goal_count": len(self.active_goals),
            "value_framework": {name: value for name, value in self.value_framework.values.items()}
        }
        
    def evolve_value_framework(self) -> Dict[str, Any]:
        """
        Gradually evolve values based on experience.
        
        Returns:
            Evolution results
        """
        adjustments = []
        
        # Analyze goal completion patterns
        tag_completion_rates = {}
        tag_counts = {}
        
        # Gather completion stats by tag
        for goal in self.goals.values():
            for tag in goal.tags:
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                    tag_completion_rates[tag] = 0.0
                
                tag_counts[tag] += 1
                if goal.status == "completed":
                    tag_completion_rates[tag] += 1.0
                    
        # Calculate completion rates
        for tag in tag_counts:
            if tag_counts[tag] > 0:
                tag_completion_rates[tag] /= tag_counts[tag]
                
        # Adjust values based on completion rates
        for tag, rate in tag_completion_rates.items():
            if tag in self.value_framework.values and tag_counts[tag] >= 3:
                # Increase value for tags with high completion rates
                if rate > 0.7:
                    adjustment = 0.05
                    reason = f"High completion rate ({rate:.2f}) for goals tagged '{tag}'"
                    self.value_framework.adjust_value(tag, adjustment, reason)
                    adjustments.append({"tag": tag, "adjustment": adjustment, "reason": reason})
                    
                # Decrease value for tags with low completion rates
                elif rate < 0.3:
                    adjustment = -0.03
                    reason = f"Low completion rate ({rate:.2f}) for goals tagged '{tag}'"
                    self.value_framework.adjust_value(tag, adjustment, reason)
                    adjustments.append({"tag": tag, "adjustment": adjustment, "reason": reason})
                    
        # Consider adding new values based on successful goals
        for tag, rate in tag_completion_rates.items():
            if tag not in self.value_framework.values and tag_counts[tag] >= 3 and rate > 0.6:
                # Add new value based on successful tag
                importance = 0.5  # Start with moderate importance
                reason = f"Successful completion of goals tagged '{tag}' ({rate:.2f} rate)"
                self.value_framework.add_value(tag, importance, reason)
                adjustments.append({"tag": tag, "action": "added", "importance": importance, "reason": reason})
                
        return {
            "adjustments": adjustments,
            "updated_values": self.value_framework.values.copy()
        }
        
    def export_to_json(self, filepath: str) -> bool:
        """
        Export goal system state to JSON.
        
        Args:
            filepath: Path to save the JSON file
            
        Returns:
            Success indicator
        """
        try:
            state = {
                "goals": {goal_id: goal.to_dict() for goal_id, goal in self.goals.items()},
                "goal_hierarchy": self.goal_hierarchy,
                "active_goals": self.active_goals,
                "completed_goals": self.completed_goals,
                "tags": self.tags,
                "value_framework": self.value_framework.to_dict(),
                "interest_map": self.interest_map.to_dict(),
                "goal_generation_history": self.goal_generation_history,
                "last_assessment": self.last_assessment.isoformat(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error exporting goal system: {str(e)}")
            return False
            
    def import_from_json(self, filepath: str) -> bool:
        """
        Import goal system state from JSON.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Success indicator
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            # Import goals
            self.goals = {}
            if "goals" in state:
                for goal_id, goal_data in state["goals"].items():
                    self.goals[goal_id] = GoalNode.from_dict(goal_data)
                    
            # Import other structures
            if "goal_hierarchy" in state:
                self.goal_hierarchy = state["goal_hierarchy"]
                
            if "active_goals" in state:
                self.active_goals = state["active_goals"]
                
            if "completed_goals" in state:
                self.completed_goals = state["completed_goals"]
                
            if "tags" in state:
                self.tags = state["tags"]
                
            if "value_framework" in state:
                self.value_framework = ValueFramework.from_dict(state["value_framework"])
                
            if "interest_map" in state:
                self.interest_map = InterestMap.from_dict(state["interest_map"])
                
            if "goal_generation_history" in state:
                self.goal_generation_history = state["goal_generation_history"]
                
            if "last_assessment" in state:
                try:
                    self.last_assessment = datetime.fromisoformat(state["last_assessment"])
                except:
                    self.last_assessment = datetime.now()
                    
            return True
        except Exception as e:
            print(f"Error importing goal system: {str(e)}")
            return False