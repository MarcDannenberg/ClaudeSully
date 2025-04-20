# sully_engine/kernel_modules/visual_cognition.py
# 🧠 Sully's Visual Cognition System - Understanding and reasoning about visual input

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import uuid
import base64
from datetime import datetime
import io
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
import re

# Note: In a production system, you might use libraries like:
# - TensorFlow/PyTorch for neural network processing
# - OpenCV for computer vision tasks
# - CLIP or similar models for image-text understanding
# For this implementation, we'll define interfaces that could connect to such systems

class VisualObject:
    """
    Represents a detected object in a visual scene.
    """
    
    def __init__(self, label: str, confidence: float, 
               bbox: Optional[List[float]] = None,
               attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize a visual object.
        
        Args:
            label: Object class label
            confidence: Detection confidence (0.0-1.0)
            bbox: Bounding box coordinates [x1, y1, x2, y2] normalized to 0-1
            attributes: Additional object attributes
        """
        self.label = label
        self.confidence = confidence
        self.bbox = bbox or [0.0, 0.0, 0.0, 0.0]
        self.attributes = attributes or {}
        self.relationships = []  # Relationships to other objects
        self.object_id = str(uuid.uuid4())
        
    def add_relationship(self, relation_type: str, target_object: 'VisualObject',
                       confidence: float = 1.0) -> None:
        """
        Add a relationship to another object.
        
        Args:
            relation_type: Type of relationship (e.g., "above", "contains", "next_to")
            target_object: The related object
            confidence: Relationship confidence (0.0-1.0)
        """
        self.relationships.append({
            "type": relation_type,
            "target_id": target_object.object_id,
            "target_label": target_object.label,
            "confidence": confidence
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "object_id": self.object_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "attributes": self.attributes,
            "relationships": self.relationships
        }


class VisualScene:
    """
    Represents a complete visual scene with objects and their relationships.
    """
    
    def __init__(self, scene_id: Optional[str] = None):
        """
        Initialize a visual scene.
        
        Args:
            scene_id: Optional scene identifier
        """
        self.scene_id = scene_id or str(uuid.uuid4())
        self.objects = {}  # object_id -> VisualObject
        self.global_attributes = {}
        self.creation_time = datetime.now()
        self.source_image = None  # Could store path or reference to source
        self.width = 0
        self.height = 0
        
    def add_object(self, visual_object: VisualObject) -> None:
        """
        Add an object to the scene.
        
        Args:
            visual_object: Object to add
        """
        self.objects[visual_object.object_id] = visual_object
        
    def get_object_by_id(self, object_id: str) -> Optional[VisualObject]:
        """
        Get an object by ID.
        
        Args:
            object_id: Object identifier
            
        Returns:
            The object or None if not found
        """
        return self.objects.get(object_id)
        
    def get_objects_by_label(self, label: str) -> List[VisualObject]:
        """
        Get objects by label.
        
        Args:
            label: Object label to find
            
        Returns:
            List of matching objects
        """
        return [obj for obj in self.objects.values() if obj.label.lower() == label.lower()]
        
    def set_dimensions(self, width: int, height: int) -> None:
        """
        Set scene dimensions.
        
        Args:
            width: Scene width
            height: Scene height
        """
        self.width = width
        self.height = height
        
    def set_source(self, source: str) -> None:
        """
        Set source image reference.
        
        Args:
            source: Source image reference
        """
        self.source_image = source
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "scene_id": self.scene_id,
            "creation_time": self.creation_time.isoformat(),
            "width": self.width,
            "height": self.height,
            "source_image": self.source_image,
            "global_attributes": self.global_attributes,
            "objects": [obj.to_dict() for obj in self.objects.values()]
        }
        
    def describe(self) -> str:
        """
        Generate a textual description of the scene.
        
        Returns:
            Scene description
        """
        # Count objects by type
        object_counts = {}
        for obj in self.objects.values():
            label = obj.label
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
            
        # Generate overall description
        description = f"Visual scene with {len(self.objects)} objects: "
        
        # List objects by type
        object_descriptions = []
        for label, count in object_counts.items():
            if count == 1:
                object_descriptions.append(f"1 {label}")
            else:
                object_descriptions.append(f"{count} {label}s")
                
        description += ", ".join(object_descriptions)
        
        # Add spatial relationships if available
        if any(obj.relationships for obj in self.objects.values()):
            description += ". Key relationships: "
            
            # Get top relationships
            relationships = []
            for obj in self.objects.values():
                for rel in obj.relationships:
                    if rel["confidence"] > 0.7:  # Only include high-confidence relationships
                        relationships.append(
                            f"{obj.label} {rel['type']} {rel['target_label']}"
                        )
                        
            # Add top relationships to description
            if relationships:
                description += ", ".join(relationships[:3])
                if len(relationships) > 3:
                    description += f" and {len(relationships) - 3} more"
                    
        return description
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VisualScene':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created visual scene
        """
        scene = VisualScene(scene_id=data.get("scene_id"))
        
        # Set basic properties
        scene.global_attributes = data.get("global_attributes", {})
        scene.width = data.get("width", 0)
        scene.height = data.get("height", 0)
        scene.source_image = data.get("source_image")
        
        # Try to parse creation time
        if "creation_time" in data:
            try:
                scene.creation_time = datetime.fromisoformat(data["creation_time"])
            except Exception:
                scene.creation_time = datetime.now()
                
        # Create objects
        for obj_data in data.get("objects", []):
            obj = VisualObject(
                label=obj_data.get("label", "unknown"),
                confidence=obj_data.get("confidence", 1.0),
                bbox=obj_data.get("bbox"),
                attributes=obj_data.get("attributes")
            )
            
            # Set object ID if available
            if "object_id" in obj_data:
                obj.object_id = obj_data["object_id"]
                
            # Set relationships
            obj.relationships = obj_data.get("relationships", [])
            
            # Add to scene
            scene.add_object(obj)
            
        return scene


class ObjectRecognitionModule:
    """
    Module for recognizing objects in images.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the object recognition module.
        
        Args:
            model_path: Optional path to a recognition model
        """
        self.model_path = model_path
        self.initialized = False
        self.supported_labels = self._load_supported_labels()
        
        # Dictionary to map model prediction indices to human-readable labels
        self.label_map = {i: label for i, label in enumerate(self.supported_labels)}
        
    def _load_supported_labels(self) -> List[str]:
        """
        Load supported object labels.
        
        Returns:
            List of supported labels
        """
        # In a real implementation, this would load from the model
        # For this implementation, we'll use a predefined list
        base_labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        return base_labels
        
    def load_model(self) -> bool:
        """
        Load the recognition model.
        
        Returns:
            Success indicator
        """
        # In a real implementation, this would load a machine learning model
        # For this implementation, we'll simulate successful loading
        self.initialized = True
        return True
        
    def detect_objects(self, image: Union[str, Image.Image, np.ndarray]) -> List[VisualObject]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of detected objects
        """
        # Ensure model is loaded
        if not self.initialized:
            self.load_model()
            
        # Prepare image
        try:
            if isinstance(image, str):
                # Load from path
                if os.path.exists(image):
                    img = Image.open(image)
                else:
                    raise ValueError(f"Image path not found: {image}")
            elif isinstance(image, Image.Image):
                # Use PIL image directly
                img = image
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                img = Image.fromarray(image.astype('uint8'))
            else:
                raise ValueError("Unsupported image type")
        except Exception as e:
            return []  # Return empty list on error
            
        # For this implementation, we'll simulate object detection
        # In a real implementation, this would use the loaded model
        
        # Get image dimensions
        width, height = img.size
        
        # Simulate random object detection
        # In a real system, this would call a proper object detection model
        detected_objects = []
        
        # Simulate 2-5 random objects from our supported labels
        num_objects = np.random.randint(2, 6)
        for _ in range(num_objects):
            # Select random label
            label_idx = np.random.randint(0, len(self.supported_labels))
            label = self.supported_labels[label_idx]
            
            # Generate random confidence (biased toward high confidence)
            confidence = 0.7 + 0.3 * np.random.random()
            
            # Generate random bounding box
            x1 = np.random.random() * 0.8
            y1 = np.random.random() * 0.8
            w = np.random.random() * 0.3 + 0.1  # width between 0.1 and 0.4
            h = np.random.random() * 0.3 + 0.1  # height between 0.1 and 0.4
            x2 = min(x1 + w, 1.0)
            y2 = min(y1 + h, 1.0)
            
            # Create object with some random attributes
            attributes = {"size": np.random.choice(["small", "medium", "large"])}
            
            # Add color attribute for some object types
            if label in ["car", "shirt", "book", "bicycle"]:
                attributes["color"] = np.random.choice(["red", "blue", "green", "yellow", "black", "white"])
                
            # Create the visual object
            obj = VisualObject(
                label=label,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                attributes=attributes
            )
            
            detected_objects.append(obj)
            
        return detected_objects
        
    def visualize_detections(self, image: Union[str, Image.Image], 
                           objects: List[VisualObject]) -> Image.Image:
        """
        Visualize detected objects on an image.
        
        Args:
            image: Input image (path or PIL Image)
            objects: Detected objects
            
        Returns:
            Annotated image
        """
        # Prepare image
        try:
            if isinstance(image, str):
                # Load from path
                if os.path.exists(image):
                    img = Image.open(image).convert("RGB")
                else:
                    raise ValueError(f"Image path not found: {image}")
            elif isinstance(image, Image.Image):
                # Use PIL image directly
                img = image.convert("RGB")
            else:
                raise ValueError("Unsupported image type")
        except Exception as e:
            # Return blank image on error
            return Image.new("RGB", (400, 300), color=(240, 240, 240))
            
        # Create drawing context
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            # Use default font if arial not available
            font = ImageFont.load_default()
            
        # Draw each detection
        for obj in objects:
            # Get normalized coordinates
            x1, y1, x2, y2 = obj.bbox
            
            # Convert to image coordinates
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            # Choose color based on confidence
            if obj.confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif obj.confidence > 0.5:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
                
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label_text = f"{obj.label} ({obj.confidence:.2f})"
            
            # In newer versions of PIL, textsize requires the font parameter
            try:
                text_width, text_height = draw.textsize(label_text, font=font)
            except TypeError:
                # For older PIL versions that don't require the font parameter
                try:
                    text_width, text_height = draw.textsize(label_text)
                except Exception:
                    # Fallback to estimated size
                    text_width, text_height = len(label_text) * 7, 15
            
            # Draw background for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(0, 0, 0), font=font)
            
        return img


class SceneUnderstandingModule:
    """
    Module for understanding relationships and context in visual scenes.
    """
    
    def __init__(self):
        """Initialize the scene understanding module."""
        self.spatial_relationships = [
            "above", "below", "left_of", "right_of", "inside", "contains",
            "touching", "near", "far_from", "in_front_of", "behind",
            "centered", "aligned_with"
        ]
        
    def analyze_scene(self, objects: List[VisualObject], 
                     image_width: int, image_height: int) -> VisualScene:
        """
        Analyze a scene with detected objects to understand relationships.
        
        Args:
            objects: Detected objects
            image_width: Image width
            image_height: Image height
            
        Returns:
            Analyzed visual scene
        """
        # Create a new scene
        scene = VisualScene()
        scene.set_dimensions(image_width, image_height)
        
        # Add objects to scene
        for obj in objects:
            scene.add_object(obj)
            
        # Analyze spatial relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Identify relationships
                    relationships = self._identify_spatial_relationships(obj1, obj2)
                    
                    # Add relationships to object
                    for rel_type, confidence in relationships:
                        obj1.add_relationship(rel_type, obj2, confidence)
                        
        # Add global scene attributes
        scene.global_attributes["object_count"] = len(objects)
        scene.global_attributes["primary_objects"] = self._identify_primary_objects(objects)
        scene.global_attributes["scene_type"] = self._classify_scene_type(objects)
        
        return scene
        
    def _identify_spatial_relationships(self, obj1: VisualObject, 
                                      obj2: VisualObject) -> List[Tuple[str, float]]:
        """
        Identify spatial relationships between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            List of (relationship_type, confidence) tuples
        """
        relationships = []
        
        # Get bounding box coordinates
        x1_1, y1_1, x2_1, y2_1 = obj1.bbox
        x1_2, y1_2, x2_2, y2_2 = obj2.bbox
        
        # Calculate centers
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Check "above" relationship
        if y2_1 < y1_2:
            confidence = min(1.0, (y1_2 - y2_1) * 5)  # Higher confidence for larger vertical separation
            relationships.append(("above", confidence))
            
        # Check "below" relationship
        if y1_1 > y2_2:
            confidence = min(1.0, (y1_1 - y2_2) * 5)
            relationships.append(("below", confidence))
            
        # Check "left_of" relationship
        if x2_1 < x1_2:
            confidence = min(1.0, (x1_2 - x2_1) * 5)
            relationships.append(("left_of", confidence))
            
        # Check "right_of" relationship
        if x1_1 > x2_2:
            confidence = min(1.0, (x1_1 - x2_2) * 5)
            relationships.append(("right_of", confidence))
            
        # Check "contains" relationship
        if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
            # Calculate containment ratio (area of obj2 / area of obj1)
            if area1 > 0:
                containment_ratio = area2 / area1
                # Higher confidence for smaller contained objects
                confidence = min(1.0, max(0.5, 1.0 - containment_ratio))
                relationships.append(("contains", confidence))
                
        # Check "inside" relationship
        if x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1:
            # Calculate containment ratio (area of obj1 / area of obj2)
            if area2 > 0:
                containment_ratio = area1 / area2
                # Higher confidence for smaller contained objects
                confidence = min(1.0, max(0.5, 1.0 - containment_ratio))
                relationships.append(("inside", confidence))
                
        # Check "near" relationship
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
        if distance < 0.2:  # Threshold for "near"
            confidence = min(1.0, max(0.5, 1.0 - distance * 5))
            relationships.append(("near", confidence))
            
        # Check "touching" relationship
        # Simple approximation: consider objects touching if their bounding boxes are very close
        horizontal_overlap = (x1_1 <= x2_2 and x2_1 >= x1_2)
        vertical_overlap = (y1_1 <= y2_2 and y2_1 >= y1_2)
        
        if horizontal_overlap and vertical_overlap:
            # Calculate overlap area
            overlap_width = min(x2_1, x2_2) - max(x1_1, x1_2)
            overlap_height = min(y2_1, y2_2) - max(y1_1, y1_2)
            overlap_area = overlap_width * overlap_height
            
            # If overlap area is small, they might be touching
            if 0 < overlap_area < 0.05 * min(area1, area2):
                relationships.append(("touching", 0.7))
                
        return relationships
        
    def _identify_primary_objects(self, objects: List[VisualObject]) -> List[str]:
        """
        Identify primary objects in the scene.
        
        Args:
            objects: Detected objects
            
        Returns:
            List of primary object labels
        """
        if not objects:
            return []
            
        # Sort objects by size and confidence
        sorted_objects = sorted(
            objects,
            key=lambda obj: (
                (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1]),  # Area
                obj.confidence
            ),
            reverse=True
        )
        
        # Return top 3 objects
        return [obj.label for obj in sorted_objects[:3]]
        
    def _classify_scene_type(self, objects: List[VisualObject]) -> str:
        """
        Classify the type of scene based on objects.
        
        Args:
            objects: Detected objects
            
        Returns:
            Scene type classification
        """
        # Simple rule-based classification
        if not objects:
            return "unknown"
            
        # Count object types
        object_types = {}
        for obj in objects:
            if obj.label not in object_types:
                object_types[obj.label] = 0
            object_types[obj.label] += 1
            
        # Check for common scene types
        if "person" in object_types and object_types["person"] >= 2:
            return "social"
            
        if any(label in object_types for label in ["car", "truck", "bus", "motorcycle"]):
            return "transportation"
            
        if any(label in object_types for label in ["chair", "couch", "bed", "table"]):
            return "indoor"
            
        if any(label in object_types for label in ["tree", "plant", "flower", "grass"]):
            return "nature"
            
        if any(label in object_types for label in ["food", "fruit", "vegetable", "dish"]):
            return "food"
            
        return "general"
        
    def describe_scene(self, scene: VisualScene) -> str:
        """
        Generate a comprehensive description of a scene.
        
        Args:
            scene: The visual scene to describe
            
        Returns:
            Scene description
        """
        # Start with basic scene information
        scene_type = scene.global_attributes.get("scene_type", "general")
        object_count = scene.global_attributes.get("object_count", len(scene.objects))
        
        description = f"This appears to be a {scene_type} scene containing {object_count} objects. "
        
        # Mention primary objects
        primary_objects = scene.global_attributes.get("primary_objects", [])
        if primary_objects:
            description += f"The main elements are: {', '.join(primary_objects)}. "
            
        # Describe spatial composition
        spatial_description = self._generate_spatial_description(scene)
        if spatial_description:
            description += spatial_description
            
        return description
        
    def _generate_spatial_description(self, scene: VisualScene) -> str:
        """
        Generate description of spatial relationships in the scene.
        
        Args:
            scene: The visual scene
            
        Returns:
            Spatial description
        """
        if not scene.objects:
            return ""
            
        # Find significant relationships
        significant_relations = []
        
        for obj_id, obj in scene.objects.items():
            for rel in obj.relationships:
                if rel["confidence"] > 0.7:  # Only high-confidence relationships
                    target_obj = scene.get_object_by_id(rel["target_id"])
                    if target_obj:
                        significant_relations.append(
                            (obj.label, rel["type"], target_obj.label, rel["confidence"])
                        )
                        
        # Sort by confidence
        significant_relations.sort(key=lambda x: x[3], reverse=True)
        
        # Generate description from top relations
        if significant_relations:
            relations_text = []
            
            for obj1, rel_type, obj2, _ in significant_relations[:5]:  # Top 5 relations
                if rel_type == "above":
                    relations_text.append(f"the {obj1} is above the {obj2}")
                elif rel_type == "below":
                    relations_text.append(f"the {obj1} is below the {obj2}")
                elif rel_type == "left_of":
                    relations_text.append(f"the {obj1} is to the left of the {obj2}")
                elif rel_type == "right_of":
                    relations_text.append(f"the {obj1} is to the right of the {obj2}")
                elif rel_type == "contains":
                    relations_text.append(f"the {obj1} contains the {obj2}")
                elif rel_type == "inside":
                    relations_text.append(f"the {obj1} is inside the {obj2}")
                elif rel_type == "near":
                    relations_text.append(f"the {obj1} is near the {obj2}")
                elif rel_type == "touching":
                    relations_text.append(f"the {obj1} is touching the {obj2}")
                    
            if relations_text:
                return "In terms of spatial arrangement, " + "; ".join(relations_text) + "."
                
        return ""


class VisualCognitionSystem:
    """
    Advanced system for processing, understanding, and reasoning about visual inputs.
    Integrates object recognition, scene understanding, and conceptual mapping.
    """

    def __init__(self, reasoning_engine=None, codex=None):
        """
        Initialize the visual cognition system.
        
        Args:
            reasoning_engine: Optional reasoning engine for high-level analysis
            codex: Optional knowledge base for conceptual mapping
        """
        # Core components
        self.object_recognition = ObjectRecognitionModule()
        self.scene_understanding = SceneUnderstandingModule()
        
        # External connections
        self.reasoning = reasoning_engine
        self.codex = codex
        
        # State
        self.processed_scenes = {}  # scene_id -> VisualScene
        self.scene_history = []  # Recent scenes
        self.visual_memory = {}  # Persistent visual concepts
        
    def process_image(self, image: Union[str, Image.Image, np.ndarray], 
                     scene_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image to extract objects, relationships, and meaning.
        
        Args:
            image: The image to process
            scene_id: Optional scene identifier
            
        Returns:
            Processing results
        """
        try:
            # Load image if path provided
            if isinstance(image, str) and os.path.exists(image):
                img = Image.open(image)
                image_path = image
            elif isinstance(image, Image.Image):
                img = image
                image_path = None
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image.astype('uint8'))
                image_path = None
            else:
                return {"error": "Invalid image input"}
                
            # Get image dimensions
            width, height = img.size
            
            # Detect objects
            detected_objects = self.object_recognition.detect_objects(img)
            
            # Create scene
            scene_id = scene_id or str(uuid.uuid4())
            
            # Analyze scene relationships
            scene = self.scene_understanding.analyze_scene(
                detected_objects,
                width,
                height
            )
            
            # Set scene properties
            scene.scene_id = scene_id
            scene.set_source(image_path)
            
            # Generate visualized image
            try:
                visualized_img = self.object_recognition.visualize_detections(img, detected_objects)
                # Could save visualized image if needed:
                # visualized_img.save(f"visualized_{scene_id}.jpg")
            except Exception:
                visualized_img = None
                
            # Generate scene description
            description = self.scene_understanding.describe_scene(scene)
            
            # Store in processed scenes
            self.processed_scenes[scene_id] = scene
            
            # Add to history
            self.scene_history.append({
                "scene_id": scene_id,
                "timestamp": datetime.now().isoformat(),
                "object_count": len(detected_objects)
            })
            
            # Limit history size
            if len(self.scene_history) > 100:
                self.scene_history = self.scene_history[-100:]
                
            # Prepare response
            result = {
                "success": True,
                "scene_id": scene_id,
                "width": width,
                "height": height,
                "object_count": len(detected_objects),
                "objects": [obj.to_dict() for obj in detected_objects],
                "scene_type": scene.global_attributes.get("scene_type", "unknown"),
                "description": description
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Image processing failed: {str(e)}",
                "success": False
            }
            
    def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a processed scene by ID.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene data or None if not found
        """
        if scene_id in self.processed_scenes:
            scene = self.processed_scenes[scene_id]
            return scene.to_dict()
        return None
        
    def search_visual_memory(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for visual concepts in memory.
        
        Args:
            query: Search query
            
        Returns:
            Matching results
        """
        results = []
        
        # Search in processed scenes
        for scene_id, scene in self.processed_scenes.items():
            match_score = 0
            
            # Check for object label matches
            for obj in scene.objects.values():
                if query.lower() in obj.label.lower():
                    match_score += 1
                    
            # Check scene type match
            if "scene_type" in scene.global_attributes:
                if query.lower() in scene.global_attributes["scene_type"].lower():
                    match_score += 2
                    
            if match_score > 0:
                results.append({
                    "scene_id": scene_id,
                    "match_score": match_score,
                    "scene_type": scene.global_attributes.get("scene_type", "unknown"),
                    "object_count": len(scene.objects),
                    "creation_time": scene.creation_time.isoformat()
                })
                
        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return results
        
    def integrate_with_conceptual_knowledge(self, scene_id: str) -> Dict[str, Any]:
        """
        Connect visual scene with conceptual knowledge.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Integration results
        """
        if scene_id not in self.processed_scenes:
            return {"error": f"Scene {scene_id} not found"}
            
        scene = self.processed_scenes[scene_id]
        
        # Need codex for conceptual integration
        if not self.codex:
            return {
                "error": "Codex required for conceptual integration",
                "scene_id": scene_id
            }
            
        try:
            # Connect objects to concepts
            object_concepts = {}
            
            for obj_id, obj in scene.objects.items():
                # Search codex for object concept
                concept_results = self.codex.search(obj.label)
                
                if concept_results:
                    # Get top concept match
                    top_concept = next(iter(concept_results))
                    
                    object_concepts[obj_id] = {
                        "object_label": obj.label,
                        "concept": top_concept,
                        "concept_data": concept_results[top_concept]
                    }
                    
            # If reasoning engine available, generate insights
            insights = []
            if self.reasoning and object_concepts:
                # Get object labels
                object_labels = [obj.label for obj in scene.objects.values()]
                
                # Generate insights prompt
                prompt = f"""
                Generate insights about the relationship between these visual elements:
                {', '.join(object_labels)}
                
                Consider their symbolic meaning, potential conceptual relationships, and
                possible interpretations based on their arrangement in a {scene.global_attributes.get('scene_type', 'general')} scene.
                """
                
                # Get insights
                insight_result = self.reasoning.reason(prompt, "creative")
                
                if isinstance(insight_result, dict) and "response" in insight_result:
                    insights.append(insight_result["response"])
                elif isinstance(insight_result, str):
                    insights.append(insight_result)
                    
            return {
                "success": True,
                "scene_id": scene_id,
                "object_concepts": object_concepts,
                "insights": insights
            }
            
        except Exception as e:
            return {
                "error": f"Conceptual integration failed: {str(e)}",
                "scene_id": scene_id
            }
            
    def generate_visual_prediction(self, scene_id: str, action: str) -> Dict[str, Any]:
        """
        Predict visual outcome of an action on a scene.
        
        Args:
            scene_id: Scene identifier
            action: Action to perform
            
        Returns:
            Prediction results
        """
        if scene_id not in self.processed_scenes:
            return {"error": f"Scene {scene_id} not found"}
            
        scene = self.processed_scenes[scene_id]
        
        # Need reasoning engine for predictions
        if not self.reasoning:
            return {
                "error": "Reasoning engine required for visual predictions",
                "scene_id": scene_id
            }
            
        try:
            # Generate prediction using reasoning engine
            scene_desc = self.scene_understanding.describe_scene(scene)
            
            # Create prediction prompt
            prompt = f"""
            Visual Scene: {scene_desc}
            
            Predict what would happen visually if the following action occurred:
            {action}
            
            Describe:
            1. How the objects would change in appearance or position
            2. Any new objects that might appear
            3. Any objects that might disappear or be altered
            4. The overall visual outcome
            """
            
            # Get prediction
            prediction_result = self.reasoning.reason(prompt, "analytical")
            
            if isinstance(prediction_result, dict) and "response" in prediction_result:
                prediction = prediction_result["response"]
            elif isinstance(prediction_result, str):
                prediction = prediction_result
            else:
                prediction = "Unable to generate prediction."
                
            return {
                "success": True,
                "scene_id": scene_id,
                "action": action,
                "prediction": prediction
            }
            
        except Exception as e:
            return {
                "error": f"Visual prediction failed: {str(e)}",
                "scene_id": scene_id,
                "action": action
            }
            
    def create_mental_imagery(self, concept: str) -> Dict[str, Any]:
        """
        Generate a mental image description based on a concept.
        
        Args:
            concept: Concept to visualize
            
        Returns:
            Mental imagery description
        """
        # Need reasoning engine for mental imagery
        if not self.reasoning:
            return {"error": "Reasoning engine required for mental imagery"}
            
        try:
            # Create mental imagery prompt
            prompt = f"""
            Generate a detailed visual description of the concept: {concept}
            
            Describe:
            1. What objects would appear in this mental image
            2. Their spatial arrangement and visual properties
            3. The overall scene composition
            4. Colors, lighting, and atmosphere
            5. Any movement or action in the scene
            
            Make the description detailed enough that someone could visualize it clearly.
            """
            
            # Get imagery description
            imagery_result = self.reasoning.reason(prompt, "visual")
            
            if isinstance(imagery_result, dict) and "response" in imagery_result:
                imagery = imagery_result["response"]
            elif isinstance(imagery_result, str):
                imagery = imagery_result
            else:
                imagery = f"A simple visual representation of {concept}."
                
            return {
                "success": True,
                "concept": concept,
                "mental_imagery": imagery
            }
            
        except Exception as e:
            return {
                "error": f"Mental imagery generation failed: {str(e)}",
                "concept": concept
            }
            
    def detect_visual_anomalies(self, scene_id: str) -> Dict[str, Any]:
        """
        Detect unusual or unexpected visual elements in a scene.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Detected anomalies
        """
        if scene_id not in self.processed_scenes:
            return {"error": f"Scene {scene_id} not found"}
            
        scene = self.processed_scenes[scene_id]
        
        try:
            anomalies = []
            
            # Check for unusual object combinations
            scene_type = scene.global_attributes.get("scene_type", "general")
            
            # Define expected objects for scene types
            expected_objects = {
                "indoor": ["chair", "table", "couch", "lamp", "book", "tv", "clock"],
                "nature": ["tree", "plant", "flower", "rock", "grass", "water"],
                "transportation": ["car", "road", "truck", "bus", "motorcycle", "bicycle"],
                "social": ["person", "chair", "table", "food", "drink"],
                "food": ["plate", "food", "table", "bowl", "utensil", "cup"]
            }
            
            # Check for unexpected objects in the scene
            if scene_type in expected_objects:
                expected = set(expected_objects[scene_type])
                
                for obj_id, obj in scene.objects.items():
                    if obj.label not in expected:
                        anomalies.append({
                            "type": "unexpected_object",
                            "object_id": obj_id,
                            "object_label": obj.label,
                            "confidence": obj.confidence,
                            "description": f"Unexpected {obj.label} in a {scene_type} scene"
                        })
                        
            # Check for unusual spatial relationships
            for obj_id, obj in scene.objects.items():
                for rel in obj.relationships:
                    rel_type = rel["type"]
                    target_label = rel["target_label"]
                    
                    # Define some unlikely relationships
                    unlikely_pairs = [
                        ("person", "inside", "cup"),
                        ("car", "above", "bird"),
                        ("book", "contains", "table"),
                        ("food", "above", "ceiling")
                    ]
                    
                    if any((obj.label, rel_type, target_label) == pair for pair in unlikely_pairs):
                        anomalies.append({
                            "type": "unusual_relationship",
                            "object_label": obj.label,
                            "relationship": rel_type,
                            "target_label": target_label,
                            "confidence": rel["confidence"],
                            "description": f"Unusual relationship: {obj.label} {rel_type} {target_label}"
                        })
                        
            return {
                "success": True,
                "scene_id": scene_id,
                "anomalies": anomalies,
                "anomaly_count": len(anomalies)
            }
            
        except Exception as e:
            return {
                "error": f"Anomaly detection failed: {str(e)}",
                "scene_id": scene_id
            }
            
    def save_visual_memory(self, filepath: str) -> Dict[str, Any]:
        """
        Save visual memory to file.
        
        Args:
            filepath: Path to save the memory
            
        Returns:
            Save results
        """
        try:
            # Prepare data for saving
            memory_data = {
                "scenes": {
                    scene_id: scene.to_dict() 
                    for scene_id, scene in self.processed_scenes.items()
                },
                "history": self.scene_history,
                "visual_memory": self.visual_memory
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2)
                
            return {
                "success": True,
                "filepath": filepath,
                "scenes_saved": len(self.processed_scenes)
            }
            
        except Exception as e:
            return {
                "error": f"Failed to save visual memory: {str(e)}",
                "filepath": filepath
            }
            
    def load_visual_memory(self, filepath: str) -> Dict[str, Any]:
        """
        Load visual memory from file.
        
        Args:
            filepath: Path to load the memory from
            
        Returns:
            Load results
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return {
                    "error": f"File not found: {filepath}",
                    "success": False
                }
                
            # Load from file
            with open(filepath, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
                
            # Load scenes
            loaded_scenes = 0
            if "scenes" in memory_data:
                for scene_id, scene_data in memory_data["scenes"].items():
                    scene = VisualScene.from_dict(scene_data)
                    self.processed_scenes[scene_id] = scene
                    loaded_scenes += 1
                    
            # Load history
            if "history" in memory_data:
                self.scene_history = memory_data["history"]
                
            # Load visual memory
            if "visual_memory" in memory_data:
                self.visual_memory = memory_data["visual_memory"]
                
            return {
                "success": True,
                "filepath": filepath,
                "scenes_loaded": loaded_scenes
            }
            
        except Exception as e:
            return {
                "error": f"Failed to load visual memory: {str(e)}",
                "filepath": filepath
            }