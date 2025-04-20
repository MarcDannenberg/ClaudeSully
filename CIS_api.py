"""
Sully Integration API Routes
===========================

This file defines the FastAPI routes for Sully's revolutionary Cognitive Integration System,
connecting the system's capabilities to the external world through a REST API.
"""

from fastapi import APIRouter, Body, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
import asyncio
import logging
import json
from datetime import datetime
import traceback

# Import the Cognitive Integration System
from CognitiveIntegrationSystem import (
    CognitiveIntegrationSystem, 
    CognitiveMode, 
    CognitiveLayer,
    CognitiveEvent,
    CognitiveThread
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Initialize the cognitive integration system
cognitive_system = CognitiveIntegrationSystem()
system_initialized = False

# Background task to initialize the system
async def initialize_cognitive_system():
    global system_initialized
    try:
        logger.info("Initializing Cognitive Integration System...")
        await cognitive_system.initialize_system()
        system_initialized = True
        logger.info("Cognitive Integration System initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Cognitive Integration System: {str(e)}")
        logger.error(traceback.format_exc())

# Start initialization when the module is imported
asyncio.create_task(initialize_cognitive_system())

# Helper function to check if system is initialized
def check_system_initialized():
    if not system_initialized:
        raise HTTPException(
            status_code=503,
            detail="Cognitive Integration System is still initializing"
        )

# Routes

@router.get("/status")
async def get_status():
    """Get the current status of the Cognitive Integration System."""
    try:
        if not system_initialized:
            return {
                "status": "initializing",
                "message": "Cognitive Integration System is still initializing"
            }
            
        status = cognitive_system.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system status: {str(e)}"
        )

@router.post("/process")
async def process_input(
    input_type: str = Body(..., description="Type of input (text, document, image, etc.)"),
    content: Any = Body(..., description="The input content"),
    context: Optional[Dict[str, Any]] = Body(None, description="Optional contextual information")
):
    """Process an input through the Cognitive Integration System."""
    check_system_initialized()
    
    try:
        result = await cognitive_system.process_input(input_type, content, context)
        return result
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing input: {str(e)}"
        )

@router.post("/thread/create")
async def create_thread(
    name: str = Body(..., description="Descriptive name of the thread"),
    purpose: str = Body(..., description="The goal or purpose of this cognitive thread"),
    cognitive_mode: str = Body(..., description="The primary cognitive mode of operation"),
    priority: float = Body(0.5, description="Thread priority (0.0-1.0)"),
    ttl: Optional[int] = Body(None, description="Time to live in seconds (None for persistent threads)")
):
    """Create a new cognitive thread."""
    check_system_initialized()
    
    try:
        # Convert string mode to enum
        try:
            mode = CognitiveMode(cognitive_mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cognitive mode: {cognitive_mode}"
            )
            
        thread = cognitive_system.create_thread(
            name=name,
            purpose=purpose,
            cognitive_mode=mode,
            priority=priority,
            ttl=ttl
        )
        
        return {
            "success": True,
            "thread_id": thread.id,
            "name": thread.name,
            "cognitive_mode": thread.cognitive_mode.value
        }
    except Exception as e:
        logger.error(f"Error creating thread: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error creating thread: {str(e)}"
        )

@router.get("/thread/{thread_id}")
async def get_thread(thread_id: str):
    """Get information about a specific thread."""
    check_system_initialized()
    
    try:
        # Check active threads # Check active threads
        if thread_id in cognitive_system.active_threads:
            thread = cognitive_system.active_threads[thread_id]
            return {
                "thread_id": thread.id,
                "name": thread.name,
                "purpose": thread.purpose,
                "cognitive_mode": thread.cognitive_mode.value,
                "priority": thread.priority,
                "creation_time": thread.creation_time.isoformat(),
                "ttl": thread.ttl,
                "active": thread.active,
                "events_count": len(thread.events),
                "involved_modules": list(thread.involved_modules),
                "status": "active"
            }
            
        # Check completed threads
        if thread_id in cognitive_system.completed_threads:
            thread = cognitive_system.completed_threads[thread_id]
            return {
                "thread_id": thread.id,
                "name": thread.name,
                "purpose": thread.purpose,
                "cognitive_mode": thread.cognitive_mode.value,
                "priority": thread.priority,
                "creation_time": thread.creation_time.isoformat(),
                "ttl": thread.ttl,
                "active": thread.active,
                "events_count": len(thread.events),
                "involved_modules": list(thread.involved_modules),
                "results": thread.results,
                "status": "completed"
            }
            
        # Thread not found
        raise HTTPException(
            status_code=404,
            detail=f"Thread {thread_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting thread: {str(e)}"
        )

@router.get("/thread/{thread_id}/events")
async def get_thread_events(thread_id: str):
    """Get events for a specific thread."""
    check_system_initialized()
    
    try:
        # Check active threads
        if thread_id in cognitive_system.active_threads:
            thread = cognitive_system.active_threads[thread_id]
        # Check completed threads
        elif thread_id in cognitive_system.completed_threads:
            thread = cognitive_system.completed_threads[thread_id]
        else:
            # Thread not found
            raise HTTPException(
                status_code=404,
                detail=f"Thread {thread_id} not found"
            )
            
        # Return events
        return {
            "thread_id": thread_id,
            "events": [
                event.to_dict() if hasattr(event, 'to_dict') else event 
                for event in thread.events
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread events: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting thread events: {str(e)}"
        )

@router.post("/event")
async def add_event(
    event_type: str = Body(..., description="Type of cognitive event"),
    description: str = Body(..., description="Detailed description of the event"),
    source_modules: List[str] = Body(..., description="List of modules involved in generating the event"),
    importance: float = Body(0.5, description="Subjective importance of the event (0.0-1.0)"),
    related_concepts: Optional[List[str]] = Body(None, description="List of concepts related to this event"),
    thread_ids: Optional[List[str]] = Body(None, description="Optional list of specific thread IDs to add this event to")
):
    """Add a cognitive event to the system and relevant threads."""
    check_system_initialized()
    
    try:
        event = cognitive_system.add_event(
            event_type=event_type,
            description=description,
            source_modules=source_modules,
            importance=importance,
            related_concepts=related_concepts,
            thread_ids=thread_ids
        )
        
        return {
            "success": True,
            "event_id": event.id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding event: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error adding event: {str(e)}"
        )

@router.post("/module/register")
async def register_module(
    name: str = Body(..., description="Unique identifier for the module"),
    capabilities: List[str] = Body(..., description="List of capabilities provided by this module"),
    interfaces: Dict[str, str] = Body(..., description="Dictionary mapping interface names to methods"),
    module_info: Dict[str, Any] = Body({}, description="Additional module information")
):
    """Register a module with the cognitive system."""
    check_system_initialized()
    
    try:
        # Create mock module object for registration
        class MockModule:
            def __init__(self, info):
                self.info = info
                
        module = MockModule(module_info)
        
        # Create callable interfaces
        callable_interfaces = {}
        for interface_name, method_name in interfaces.items():
            # Create a placeholder callable that returns information about the call
            def create_callable(interface, method):
                async def mock_callable(*args, **kwargs):
                    return {
                        "module": name,
                        "interface": interface,
                        "method": method,
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "timestamp": datetime.now().isoformat(),
                        "result": f"Mock implementation of {method}"
                    }
                return mock_callable
                
            callable_interfaces[interface_name] = create_callable(interface_name, method_name)
        
        success = cognitive_system.register_module(
            name=name,
            module=module,
            capabilities=capabilities,
            interfaces=callable_interfaces
        )
        
        return {
            "success": success,
            "name": name,
            "capabilities": capabilities,
            "interfaces": list(interfaces.keys())
        }
    except Exception as e:
        logger.error(f"Error registering module: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error registering module: {str(e)}"
        )

@router.get("/modules")
async def get_modules():
    """Get all registered modules."""
    check_system_initialized()
    
    try:
        return cognitive_system.registry.to_dict()
    except Exception as e:
        logger.error(f"Error getting modules: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting modules: {str(e)}"
        )

@router.get("/threads")
async def get_threads(active_only: bool = False):
    """Get all cognitive threads."""
    check_system_initialized()
    
    try:
        if active_only:
            threads = {
                thread_id: {
                    "id": thread.id,
                    "name": thread.name,
                    "purpose": thread.purpose,
                    "cognitive_mode": thread.cognitive_mode.value,
                    "creation_time": thread.creation_time.isoformat()
                }
                for thread_id, thread in cognitive_system.active_threads.items()
            }
            return {"active_threads": threads}
        else:
            active_threads = {
                thread_id: {
                    "id": thread.id,
                    "name": thread.name,
                    "purpose": thread.purpose,
                    "cognitive_mode": thread.cognitive_mode.value,
                    "creation_time": thread.creation_time.isoformat()
                }
                for thread_id, thread in cognitive_system.active_threads.items()
            }
            
            completed_threads = {
                thread_id: {
                    "id": thread.id,
                    "name": thread.name,
                    "purpose": thread.purpose,
                    "cognitive_mode": thread.cognitive_mode.value,
                    "creation_time": thread.creation_time.isoformat()
                }
                for thread_id, thread in cognitive_system.completed_threads.items()
            }
            
            return {
                "active_threads": active_threads,
                "completed_threads": completed_threads
            }
    except Exception as e:
        logger.error(f"Error getting threads: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting threads: {str(e)}"
        )

@router.post("/reflection/trigger")
async def trigger_reflection(background_tasks: BackgroundTasks):
    """Trigger a system-wide reflection."""
    check_system_initialized()
    
    try:
        # Create a reflection thread
        reflection_thread = cognitive_system.create_thread(
            name="Manual System-wide Reflection",
            purpose="Analyze system state and optimize cognitive processes",
            cognitive_mode=CognitiveMode.REFLECTION,
            priority=0.9,
            ttl=600  # 10 minutes
        )
        
        # Store system state in thread's working memory
        reflection_thread.working_memory["active_threads"] = len(cognitive_system.active_threads)
        reflection_thread.working_memory["completed_threads"] = len(cognitive_system.completed_threads)
        reflection_thread.working_memory["events"] = len(cognitive_system.event_history)
        reflection_thread.working_memory["mode_weights"] = cognitive_system.mode_weights.copy()
        reflection_thread.working_memory["layer_activation"] = cognitive_system.layer_activation.copy()
        reflection_thread.working_memory["modules"] = cognitive_system.registry.to_dict()
        reflection_thread.working_memory["manual_trigger"] = True
        
        # Start reflection in background
        background_tasks.add_task(
            cognitive_system._perform_system_reflection,
            reflection_thread
        )
        
        return {
            "success": True,
            "thread_id": reflection_thread.id,
            "message": "System-wide reflection triggered"
        }
    except Exception as e:
        logger.error(f"Error triggering reflection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering reflection: {str(e)}"
        )

@router.get("/emergence/properties")
async def get_emergent_properties():
    """Get detected emergent properties."""
    check_system_initialized()
    
    try:
        return {
            "emergent_properties": cognitive_system.emergence_candidates
        }
    except Exception as e:
        logger.error(f"Error getting emergent properties: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting emergent properties: {str(e)}"
        )

@router.post("/emergence/detect", response_model=Dict[str, Any])
async def trigger_emergence_detection(background_tasks: BackgroundTasks):
    """Trigger emergence detection."""
    check_system_initialized()
    
    try:
        # Start emergence detection in background
        background_tasks.add_task(cognitive_system._basic_emergence_detection)
        
        return {
            "success": True,
            "message": "Emergence detection triggered"
        }
    except Exception as e:
        logger.error(f"Error triggering emergence detection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering emergence detection: {str(e)}"
        )

@router.post("/state/export")
async def export_system_state():
    """Export the current state of the cognitive integration system."""
    check_system_initialized()
    
    try:
        state = cognitive_system.export_state()
        
        # Convert to JSON-compatible format
        state_json = json.dumps(state, default=str)
        
        return {
            "success": True,
            "state": json.loads(state_json),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting system state: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting system state: {str(e)}"
        )

@router.post("/state/import")
async def import_system_state(
    state: Dict[str, Any] = Body(..., description="The system state to import")
):
    """Import a previously exported system state."""
    check_system_initialized()
    
    try:
        success = await cognitive_system.import_state(state)
        
        return {
            "success": success,
            "message": "System state imported successfully" if success else "Error importing system state"
        }
    except Exception as e:
        logger.error(f"Error importing system state: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error importing system state: {str(e)}"
        )

@router.post("/system/restart")
async def restart_system(background_tasks: BackgroundTasks):
    """Restart the cognitive integration system."""
    global system_initialized
    
    try:
        # Mark system as uninitializing
        system_initialized = False
        
        # Start initialization in background
        background_tasks.add_task(initialize_cognitive_system)
        
        return {
            "success": True,
            "message": "System restart triggered"
        }
    except Exception as e:
        logger.error(f"Error restarting system: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error restarting system: {str(e)}"
        )

# Additional route for integrating with core Sully systems

@router.post("/integrate/conversation")
async def integrate_conversation(
    message: str = Body(..., description="User message"),
    context: Optional[Dict[str, Any]] = Body(None, description="Conversation context")
):
    """Process a conversation message through the cognitive integration system."""
    check_system_initialized()
    
    try:
        result = await cognitive_system.process_input("text", message, context)
        
        # Extract the response for conversation interface
        response = result.get("response", "I'm processing your message through my cognitive integration system.")
        
        return {
            "response": response,
            "thread_id": result.get("thread_id"),
            "confidence": result.get("confidence", 0.5),
            "processing_info": {
                "cognitive_events": len(result.get("cognitive_events", [])),
                "processing_stages": len(result.get("processing_stages", [])),
            }
        }
    except Exception as e:
        logger.error(f"Error integrating conversation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error integrating conversation: {str(e)}"
        )

@router.post("/integrate/document")
async def integrate_document(
    file_path: str = Body(..., description="Path to the document file"),
    document_type: str = Body(..., description="Type of document"),
    metadata: Optional[Dict[str, Any]] = Body(None, description="Document metadata")
):
    """Process a document through the cognitive integration system."""
    check_system_initialized()
    
    try:
        # Create context with document type and metadata
        context = {
            "document_type": document_type,
            "metadata": metadata or {}
        }
        
        # Process document as file path
        result = await cognitive_system.process_input("document", file_path, context)
        
        return {
            "success": True,
            "thread_id": result.get("thread_id"),
            "processing_status": "completed",
            "document_path": file_path,
            "cognitive_events": len(result.get("cognitive_events", [])),
            "integration_level": result.get("confidence", 0.5)
        }
    except Exception as e:
        logger.error(f"Error integrating document: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error integrating document: {str(e)}"
        )

@router.post("/integrate/goal")
async def integrate_goal(
    description: str = Body(..., description="Goal description"),
    goal_type: str = Body(..., description="Type of goal"),
    priority: float = Body(1.0, description="Goal priority"),
    tags: List[str] = Body([], description="Goal tags")
):
    """Integrate an autonomous goal with the cognitive integration system."""
    check_system_initialized()
    
    try:
        # Find autonomous goals module
        goal_modules = cognitive_system.registry.find_modules_by_capability("goal_management")
        
        if not goal_modules:
            # Create a thread for goal integration
            thread = cognitive_system.create_thread(
                name=f"Goal Integration: {description[:30]}...",
                purpose=f"Integrate new goal: {description}",
                cognitive_mode=CognitiveMode.MOTIVATION,
                priority=priority,
                ttl=300  # 5 minutes
            )
            
            # Store goal in thread's working memory
            thread.working_memory["goal_description"] = description
            thread.working_memory["goal_type"] = goal_type
            thread.working_memory["goal_priority"] = priority
            thread.working_memory["goal_tags"] = tags
            
            # Add event for goal creation
            event = cognitive_system.add_event(
                event_type="goal_creation",
                description=f"New goal created: {description}",
                source_modules=["CognitiveIntegrationSystem"],
                importance=0.8,
                related_concepts=tags,
                thread_ids=[thread.id]
            )
            
            return {
                "success": True,
                "thread_id": thread.id,
                "message": "Goal integrated through cognitive thread",
                "event_id": event.id
            }
        else:
            # Use the first available goal module
            module_name = goal_modules[0]
            establish_goal = cognitive_system.registry.get_interface(module_name, "establish_goal")
            
            if not establish_goal:
                raise HTTPException(
                    status_code=500,
                    detail=f"Module {module_name} does not provide establish_goal interface"
                )
                
            # Establish the goal
            result = await establish_goal({
                "description": description,
                "goal_type": goal_type,
                "priority": priority,
                "tags": tags
            })
            
            return {
                "success": True,
                "goal_id": result.get("goal_id"),
                "module": module_name,
                "message": "Goal established through goal module"
            }
    except Exception as e:
        logger.error(f"Error integrating goal: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error integrating goal: {str(e)}"
        )

@router.post("/integrate/memory")
async def integrate_memory(
    content: str = Body(..., description="Memory content"),
    source: str = Body(..., description="Memory source"),
    tags: List[str] = Body([], description="Memory tags"),
    importance: float = Body(0.5, description="Memory importance")
):
    """Integrate a memory with the cognitive integration system."""
    check_system_initialized()
    
    try:
        # Find memory modules
        memory_modules = cognitive_system.registry.find_modules_by_capability("memory_storage")
        
        if not memory_modules:
            # Create a thread for memory integration
            thread = cognitive_system.create_thread(
                name=f"Memory Integration: {content[:30]}...",
                purpose=f"Integrate new memory from {source}",
                cognitive_mode=CognitiveMode.MEMORY,
                priority=importance,
                ttl=300  # 5 minutes
            )
            
            # Store memory in thread's working memory
            thread.working_memory["memory_content"] = content
            thread.working_memory["memory_source"] = source
            thread.working_memory["memory_tags"] = tags
            thread.working_memory["memory_importance"] = importance
            
            # Add event for memory creation
            event = cognitive_system.add_event(
                event_type="memory_creation",
                description=f"New memory created from {source}",
                source_modules=["CognitiveIntegrationSystem"],
                importance=importance,
                related_concepts=tags,
                thread_ids=[thread.id]
            )
            
            return {
                "success": True,
                "thread_id": thread.id,
                "message": "Memory integrated through cognitive thread",
                "event_id": event.id
            }
        else:
            # Use the first available memory module
            module_name = memory_modules[0]
            store_memory = cognitive_system.registry.get_interface(module_name, "store")
            
            if not store_memory:
                raise HTTPException(
                    status_code=500,
                    detail=f"Module {module_name} does not provide store interface"
                )
                
            # Store the memory
            result = await store_memory({
                "content": content,
                "source": source,
                "tags": tags,
                "importance": importance
            })
            
            return {
                "success": True,
                "memory_id": result.get("memory_id"),
                "module": module_name,
                "message": "Memory stored through memory module"
            }
    except Exception as e:
        logger.error(f"Error integrating memory: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error integrating memory: {str(e)}"
        )

# Include the integration router in your main application
# In sully_api2.py, add:
# from sully_integration_api import router as integration_router
# api_app.include_router(integration_router, prefix="/integration")