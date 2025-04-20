"""
Games API for Sully cognitive system

This module provides FastAPI endpoints for Sully's games functionality,
including visual gameplay, AI opponent capabilities, and game invention features.
"""

from fastapi import APIRouter, HTTPException, Response, Body, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
import base64
import tempfile
import os
import uuid
import json
import asyncio
from datetime import datetime

# Import Sully components
from sully import Sully
from sully_engine.kernel_modules.games import SullyGames
from sully_engine.kernel_integration import KernelIntegrationSystem

# Create a router for games endpoints
games_router = APIRouter(prefix="/games", tags=["games"])

# Initialize Sully games module
sully = Sully()
games_module = SullyGames(reasoning_node=sully.reasoning_node, memory_system=sully.memory)

# Initialize game design engine - connects to Sully's cognitive system
try:
    game_design_engine = GameDesignEngine(
        reasoning_node=sully.reasoning_node,
        memory_system=sully.memory,
        kernel_integration=sully.kernel_integration,
        dream_core=sully.dream,
        fusion_engine=sully.fusion
    )
except Exception as e:
    print(f"Error initializing game design engine: {e}")
    # Fall back to simplified engine if full integration fails
    game_design_engine = SimpleGameDesignEngine(reasoning_node=sully.reasoning_node)

# Global storage for invented games
invented_games_db = {}
game_playtest_data = {}

# Models for request/response
class CreateGameRequest(BaseModel):
    game_type: str  # "mahjong", "chess", "go", or a custom invented game ID
    player_names: List[str]
    session_id: Optional[str] = "default"
    board_size: Optional[int] = 19  # For Go
    sully_plays_as: Optional[str] = None  # "white" or "black" for Chess/Go, seat position for Mahjong
    difficulty: Optional[str] = "medium"  # "easy", "medium", "hard", "expert"

class MoveRequest(BaseModel):
    move: Dict[str, Any]
    session_id: Optional[str] = "default"
    animate: Optional[bool] = False  # Whether to animate the move in visualization

class SessionRequest(BaseModel):
    session_id: Optional[str] = "default"

class GameStateRequest(BaseModel):
    game_state: Dict[str, Any]
    game_type: str

class RenderRequest(BaseModel):
    session_id: Optional[str] = "default"
    format: Optional[str] = "svg"  # "svg", "png", "html", "json"
    include_hints: Optional[bool] = False  # Whether to include move hints
    highlight_last_move: Optional[bool] = True  # Whether to highlight the last move
    theme: Optional[str] = "default"  # Visual theme for rendering

class ThoughtProcessRequest(BaseModel):
    session_id: Optional[str] = "default"
    depth: Optional[int] = 3  # How detailed Sully's thought explanation should be

class GameInventionRequest(BaseModel):
    concept: str = Field(..., description="Central concept or theme for the game")
    constraints: Optional[List[str]] = Field(None, description="Constraints for game design")
    mechanics: Optional[List[str]] = Field(None, description="Suggested mechanics to incorporate")
    player_count: Optional[Tuple[int, int]] = Field((1, 4), description="Min and max number of players")
    complexity: Optional[str] = Field("medium", description="Desired complexity level: simple, medium, complex")
    duration: Optional[int] = Field(30, description="Target playtime in minutes")
    inspiration_sources: Optional[List[str]] = Field(None, description="Games to draw inspiration from")

class GameRefinementRequest(BaseModel):
    game_id: str = Field(..., description="ID of the game to refine")
    feedback: str = Field(..., description="Feedback to incorporate")
    aspect: Optional[str] = Field(None, description="Specific aspect to refine: rules, mechanics, balance, theme, etc.")
    preserve: Optional[List[str]] = Field(None, description="Elements to preserve in refinement")

class GameGenerationProgressRequest(BaseModel):
    task_id: str = Field(..., description="ID of the game generation task")

class PlaytestFeedbackRequest(BaseModel):
    game_id: str = Field(..., description="ID of the game")
    session_id: str = Field(..., description="ID of the playtest session")
    feedback: str = Field(..., description="Playtest feedback")
    rating: Optional[int] = Field(None, description="Rating from 1-10")
    player_count: Optional[int] = Field(None, description="Number of players in this playtest")
    player_demographics: Optional[Dict[str, Any]] = Field(None, description="Information about the playtesters")
    play_duration: Optional[int] = Field(None, description="How long the game was played (minutes)")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Any measured metrics during play")

class AIGamePlaytestRequest(BaseModel):
    game_id: str = Field(..., description="ID of the game to playtest")
    iterations: Optional[int] = Field(5, description="Number of simulated playtests to run")
    player_count: Optional[int] = Field(None, description="Number of simulated players")
    player_strategies: Optional[List[str]] = Field(None, description="Strategies for AI players")
    metrics_to_track: Optional[List[str]] = Field(None, description="Game metrics to analyze")

# Background task storage
background_tasks = {}

# Game creation endpoint
@games_router.post("/create")
async def create_game(request: CreateGameRequest):
    """Create a new game instance"""
    # Check if this is an invented game type
    if request.game_type not in ["mahjong", "chess", "go"] and request.game_type in invented_games_db:
        # Use the invented game engine
        result = game_design_engine.create_game_instance(
            invented_games_db[request.game_type],
            player_names=request.player_names,
            session_id=request.session_id,
            sully_plays_as=request.sully_plays_as,
            difficulty=request.difficulty
        )
    else:
        # Use standard game module
        result = games_module.create_game(
            game_type=request.game_type,
            player_names=request.player_names,
            session_id=request.session_id,
            board_size=request.board_size,
            sully_plays_as=request.sully_plays_as,
            difficulty=request.difficulty
        )
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create game"))
    
    return result

# Making a move
@games_router.post("/move")
async def make_move(request: MoveRequest):
    """Make a move in the current game"""
    # Get the session's game type
    game_info = games_module.get_game_state(session_id=request.session_id)
    if not game_info.get("success", False):
        raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
    
    game_type = game_info.get("game_type", "")
    
    # Process move based on game type
    if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
        # Use the invented game engine for custom games
        result = game_design_engine.process_move(
            game_type,
            move=request.move,
            session_id=request.session_id,
            animate=request.animate
        )
    else:
        # Use standard game module
        result = games_module.make_move(
            move=request.move,
            session_id=request.session_id,
            animate=request.animate
        )
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Invalid move"))
    
    # If it's Sully's turn after this move, automatically generate Sully's move
    if result.get("current_player") == "Sully" and not result.get("game_over", False):
        if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
            sully_move_result = game_design_engine.get_sully_move(
                game_type, 
                session_id=request.session_id
            )
        else:
            sully_move_result = games_module.get_sully_move(session_id=request.session_id)
            
        if sully_move_result.get("success", False):
            result["sully_move"] = sully_move_result["move"]
            result["sully_reasoning"] = sully_move_result.get("reasoning", "")
            result["state"] = sully_move_result["state"]  # Update with latest state after Sully's move
    
    # Record game data for learning if applicable
    if game_type in invented_games_db:
        # Add to playtest data for invented games
        if game_type not in game_playtest_data:
            game_playtest_data[game_type] = []
        
        # Record move data for learning
        move_data = {
            "move": request.move,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "resulting_state": result.get("state", {}),
            "player": result.get("previous_player", "unknown")
        }
        game_playtest_data[game_type].append(move_data)
    
    return result

# Get Sully's move
@games_router.post("/sully_move")
async def get_sully_move(request: SessionRequest):
    """Get Sully's next move in the current game"""
    # Get the session's game type
    game_info = games_module.get_game_state(session_id=request.session_id)
    if not game_info.get("success", False):
        raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
    
    game_type = game_info.get("game_type", "")
    
    # Generate move based on game type
    if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
        # Use the invented game engine for custom games
        result = game_design_engine.get_sully_move(
            game_type,
            session_id=request.session_id
        )
    else:
        # Use standard game module
        result = games_module.get_sully_move(session_id=request.session_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate move"))
    
    return result

# Get Sully's thought process for a move
@games_router.post("/thought_process")
async def get_thought_process(request: ThoughtProcessRequest):
    """Get Sully's detailed thought process for deciding a move"""
    try:
        # Get the session's game type
        game_info = games_module.get_game_state(session_id=request.session_id)
        if not game_info.get("success", False):
            raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
        
        game_type = game_info.get("game_type", "")
        
        # Get thought process based on game type
        if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
            # Use the invented game engine for custom games
            result = game_design_engine.get_move_thought_process(
                game_type,
                session_id=request.session_id,
                depth=request.depth
            )
        else:
            # Use standard game module
            result = games_module.get_move_thought_process(
                session_id=request.session_id,
                depth=request.depth
            )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate thought process"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating thought process: {str(e)}")

# Get game state
@games_router.post("/state")
async def get_game_state(request: SessionRequest):
    """Get the current state of the game"""
    result = games_module.get_game_state(session_id=request.session_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "No active game"))
    
    # If this is a custom game, augment with additional state info
    game_type = result.get("game_type", "")
    if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
        custom_state = game_design_engine.get_custom_game_state(
            game_type,
            session_id=request.session_id
        )
        if custom_state.get("success", False):
            # Merge the custom state data
            for key, value in custom_state.items():
                if key != "success" and key not in result:
                    result[key] = value
    
    return result

# Render visual representation of the game
@games_router.post("/render")
async def render_game(request: RenderRequest):
    """Get a visual representation of the current game state"""
    try:
        # Get the session's game type
        game_info = games_module.get_game_state(session_id=request.session_id)
        if not game_info.get("success", False):
            raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
        
        game_type = game_info.get("game_type", "")
        
        # Render based on game type
        if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
            # Use the invented game engine for custom games
            result = game_design_engine.render_game(
                game_type,
                session_id=request.session_id,
                format=request.format,
                include_hints=request.include_hints,
                highlight_last_move=request.highlight_last_move,
                theme=request.theme
            )
        else:
            # Use standard game module
            result = games_module.render_game(
                session_id=request.session_id,
                format=request.format,
                include_hints=request.include_hints,
                highlight_last_move=request.highlight_last_move,
                theme=request.theme
            )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to render game"))
        
        # Handle different output formats
        if request.format == "svg":
            return Response(content=result["content"], media_type="image/svg+xml")
        elif request.format == "png":
            # For PNG, we return base64
            return JSONResponse({"image": result["content"], "mime_type": "image/png"})
        elif request.format == "html":
            return Response(content=result["content"], media_type="text/html")
        else:
            # Default to JSON
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering game: {str(e)}")

# Get PNG screenshot of the game
@games_router.post("/screenshot")
async def get_screenshot(request: RenderRequest):
    """Get a PNG screenshot of the current game"""
    try:
        # Get the session's game type
        game_info = games_module.get_game_state(session_id=request.session_id)
        if not game_info.get("success", False):
            raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
        
        game_type = game_info.get("game_type", "")
        
        # Generate screenshot based on game type
        if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
            # Use the invented game engine for custom games
            result = game_design_engine.render_game(
                game_type,
                session_id=request.session_id,
                format="png",
                include_hints=request.include_hints,
                highlight_last_move=request.highlight_last_move,
                theme=request.theme
            )
        else:
            # Use standard game module
            result = games_module.render_game(
                session_id=request.session_id,
                format="png",
                include_hints=request.include_hints,
                highlight_last_move=request.highlight_last_move,
                theme=request.theme
            )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate screenshot"))
        
        # Create a temporary file for the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_file:
            # Decode base64 image
            img_data = base64.b64decode(result["content"])
            img_file.write(img_data)
            img_path = img_file.name
        
        try:
            # Return the file
            return FileResponse(
                img_path, 
                media_type="image/png",
                background=None,
                filename=f"game_{request.session_id}.png"
            )
        finally:
            # Ensure the temporary file is deleted after response is sent
            os.unlink(img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating screenshot: {str(e)}")

# Get valid moves for current player
@games_router.post("/valid_moves")
async def get_valid_moves(request: SessionRequest):
    """Get all valid moves for the current player"""
    try:
        # Get the session's game type
        game_info = games_module.get_game_state(session_id=request.session_id)
        if not game_info.get("success", False):
            raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
        
        game_type = game_info.get("game_type", "")
        
        # Get valid moves based on game type
        if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
            # Use the invented game engine for custom games
            result = game_design_engine.get_valid_moves(
                game_type,
                session_id=request.session_id
            )
        else:
            # Use standard game module
            result = games_module.get_valid_moves(session_id=request.session_id)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to get valid moves"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting valid moves: {str(e)}")

# Analyze a game state
@games_router.post("/analyze")
async def analyze_game(request: GameStateRequest):
    """Analyze a game state and provide insights"""
    # If this is a standard game, use the standard module
    if request.game_type in ["mahjong", "chess", "go"]:
        result = games_module.analyze_game(
            game_state=request.game_state,
            game_type=request.game_type
        )
    # If this is a custom game, use the game design engine
    elif request.game_type in invented_games_db:
        result = game_design_engine.analyze_game_state(
            game_type=request.game_type,
            game_state=request.game_state
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown game type: {request.game_type}")
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))
    
    return result

# Get game suggestion/hint
@games_router.post("/hint")
async def get_hint(request: SessionRequest):
    """Get a hint for the next move in the current game"""
    try:
        # Get the session's game type
        game_info = games_module.get_game_state(session_id=request.session_id)
        if not game_info.get("success", False):
            raise HTTPException(status_code=400, detail=game_info.get("error", "No active game"))
        
        game_type = game_info.get("game_type", "")
        
        # Get hint based on game type
        if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
            # Use the invented game engine for custom games
            result = game_design_engine.get_hint(
                game_type,
                session_id=request.session_id
            )
        else:
            # Use standard game module
            result = games_module.get_hint(session_id=request.session_id)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate hint"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating hint: {str(e)}")

# End a game
@games_router.post("/end")
async def end_game(request: SessionRequest):
    """End a game session"""
    # Get the session's game type before ending
    game_info = games_module.get_game_state(session_id=request.session_id)
    game_type = game_info.get("game_type", "") if game_info.get("success", False) else ""
    
    # End the game based on type
    if game_type not in ["mahjong", "chess", "go"] and game_type in invented_games_db:
        # Use the invented game engine for custom games
        result = game_design_engine.end_game(
            game_type,
            session_id=request.session_id
        )
        
        # Record game completion data for learning
        if game_type in game_playtest_data:
            # Add session completion data
            completion_data = {
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat(),
                "completed": True,
                "final_state": game_info.get("state", {}),
                "winner": game_info.get("winner", None),
                "move_count": len([m for m in game_playtest_data[game_type] 
                                 if m.get("session_id") == request.session_id])
            }
            game_playtest_data[game_type].append(completion_data)
            
    else:
        # Use standard game module
        result = games_module.end_game(session_id=request.session_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to end game"))
    
    return result

# Get game history
@games_router.get("/history")
async def get_game_history():
    """Get the history of played games"""
    # Combine standard and invented game history
    standard_history = games_module.get_game_history()
    
    # Get history of invented games
    invented_history = []
    for game_id, game_data in game_playtest_data.items():
        # Group by session
        sessions = {}
        for entry in game_data:
            session_id = entry.get("session_id", "unknown")
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(entry)
        
        # Create session summaries
        for session_id, session_data in sessions.items():
            # Find completion data
            completion_entries = [e for e in session_data if e.get("completed", False)]
            completion_data = completion_entries[0] if completion_entries else {}
            
            # Create history entry
            invented_history.append({
                "game_type": game_id, 
                "session_id": session_id,
                "moves": len([e for e in session_data if "move" in e]),
                "completed": bool(completion_entries),
                "winner": completion_data.get("winner", None),
                "timestamp": session_data[0].get("timestamp") if session_data else None,
                "custom_game": True
            })
    
    # Combine histories
    combined_history = {
        "standard_games": standard_history,
        "invented_games": invented_history
    }
    
    return combined_history

# Get available games
@games_router.get("/available")
async def get_available_games():
    """Get list of available games and options"""
    # Start with standard games
    standard_games = [
        {
            "id": "chess",
            "name": "Chess",
            "players": 2,
            "description": "Traditional Chess game with full rules including castling, en passant, and promotion.",
            "difficulties": ["easy", "medium", "hard", "expert"],
            "themes": ["default", "classic", "modern", "wood", "tournament"],
            "standard_game": True
        },
        {
            "id": "go",
            "name": "Go",
            "players": 2,
            "description": "Traditional Go (Weiqi/Baduk) with customizable board sizes and rule sets.",
            "board_sizes": [9, 13, 19],
            "difficulties": ["easy", "medium", "hard", "expert"],
            "themes": ["default", "classic", "wooden", "modern"],
            "standard_game": True
        },
        {
            "id": "mahjong",
            "name": "Mahjong",
            "players": 4,
            "description": "Traditional Riichi Mahjong with complete rule set including calls and scoring.",
            "difficulties": ["easy", "medium", "hard", "expert"],
            "themes": ["default", "traditional", "simple"],
            "standard_game": True
        }
    ]
    
    # Add invented games
    invented_game_list = []
    for game_id, game_data in invented_games_db.items():
        game_info = {
            "id": game_id,
            "name": game_data.get("name", game_id),
            "players": game_data.get("player_count", [1, 4]),
            "description": game_data.get("description", "A game created by Sully"),
            "difficulties": game_data.get("difficulties", ["easy", "medium", "hard"]),
            "themes": game_data.get("themes", ["default"]),
            "invented_by_sully": True,
            "created_at": game_data.get("created_at", datetime.now().isoformat()),
            "complexity": game_data.get("complexity", "medium"),
            "concept": game_data.get("concept", ""),
            "duration": game_data.get("duration", 30)
        }
        invented_game_list.append(game_info)
    
    return {
        "success": True,
        "games": standard_games + invented_game_list
    }

# --- Game Invention Endpoints ---

# Invent a new game
@games_router.post("/invent")
async def invent_game(request: GameInventionRequest, background_tasks: BackgroundTasks):
    """Create a new game based on concepts and constraints"""
    try:
        # Create a unique task ID
        task_id = f"invent_{uuid.uuid4().hex}"
        
        # Initialize task status
        background_tasks[task_id] = {
            "status": "initializing",
            "progress": 0,
            "started_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "logs": ["Starting game invention process"]
        }
        
        # Start the background task
        background_tasks.add_task(
            game_design_engine.invent_game_task,
            task_id=task_id,
            concept=request.concept,
            constraints=request.constraints,
            mechanics=request.mechanics,
            player_count=request.player_count,
            complexity=request.complexity,
            duration=request.duration,
            inspiration_sources=request.inspiration_sources,
            callback=update_task_status
        )
        
        return {
            "success": True,
            "message": "Game invention started",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting game invention: {str(e)}")

# Check game invention progress
@games_router.post("/invention/progress")
async def check_invention_progress(request: GameGenerationProgressRequest):
    """Check the status of a game invention task"""
    task_id = request.task_id
    
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = background_tasks[task_id]
    
    # If the task is complete, return the result
    if task_info.get("status") == "completed" and task_info.get("result"):
        game_data = task_info["result"]
        
        # Store the game in the database if it's new
        if "game_id" in game_data and game_data["game_id"] not in invented_games_db:
            invented_games_db[game_data["game_id"]] = game_data
        
        return {
            "success": True,
            "status": "completed",
            "game": game_data
        }
    
    # If the task failed, return the error
    if task_info.get("status") == "failed":
        return {
            "success": False,
            "status": "failed",
            "error": task_info.get("error", "Unknown error")
        }
    
    # Otherwise, return the current status
    return {
        "success": True,
        "status": task_info.get("status", "unknown"),
        "progress": task_info.get("progress", 0),
        "logs": task_info.get("logs", [])[-5:]  # Return last 5 logs for updates
    }

# Refine an existing game
@games_router.post("/refine")
async def refine_game(request: GameRefinementRequest, background_tasks: BackgroundTasks):
    """Refine an existing game based on feedback"""
    game_id = request.game_id
    
    if game_id not in invented_games_db:
        raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
    
    try:
        # Create a unique task ID
        task_id = f"refine_{uuid.uuid4().hex}"
        
        # Initialize task status
        background_tasks[task_id] = {
            "status": "initializing",
            "progress": 0,
            "started_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "logs": ["Starting game refinement process"]
        }
        
        # Start the background task
        background_tasks.add_task(
            game_design_engine.refine_game_task,
            task_id=task_id,
            game_id=game_id,
            feedback=request.feedback,
            aspect=request.aspect,
            preserve=request.preserve,
            callback=update_task_status
        )
        
        return {
            "success": True,
            "message": "Game refinement started",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting game refinement: {str(e)}")

# Submit playtest feedback
@games_router.post("/playtest/feedback")
async def submit_playtest_feedback(request: PlaytestFeedbackRequest):
    """Submit feedback from a playtest session"""
    game_id = request.game_id
    
    if game_id not in invented_games_db:
        raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
    
    try:
        # Store the feedback
        if game_id not in game_playtest_data:
            game_playtest_data[game_id] = []
        
        feedback_entry = {
            "type": "playtest_feedback",
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": request.feedback,
            "rating": request.rating,
            "player_count": request.player_count,
            "player_demographics": request.player_demographics,
            "play_duration": request.play_duration,
            "metrics": request.metrics
        }
        
        game_playtest_data[game_id].append(feedback_entry)
        
        # Update the game's feedback history
        if "feedback_history" not in invented_games_db[game_id]:
            invented_games_db[game_id]["feedback_history"] = []
        
        invented_games_db[game_id]["feedback_history"].append({
            "timestamp": datetime.now().isoformat(),
            "feedback": request.feedback,
            "rating": request.rating
        })
        
        # If there's a rating, update the average rating
        if request.rating is not None:
            ratings = [f.get("rating") for f in invented_games_db[game_id]["feedback_history"] 
                      if f.get("rating") is not None]
            if ratings:
                invented_games_db[game_id]["average_rating"] = sum(ratings) / len(ratings)
        
        return {
            "success": True,
            "message": "Playtest feedback recorded",
            "feedback_id": len(game_playtest_data[game_id])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording playtest feedback: {str(e)}")

# Run AI-powered playtests
@games_router.post("/playtest/simulate")
async def simulate_playtests(request: AIGamePlaytestRequest, background_tasks: BackgroundTasks):
    """Run AI-simulated playtests of a game"""
    game_id = request.game_id
    
    if game_id not in invented_games_db:
        raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
    
    try:
        # Create a unique task ID
        task_id = f"playtest_{uuid.uuid4().hex}"
        
        # Initialize task status
        background_tasks[task_id] = {
            "status": "initializing",
            "progress": 0,
            "started_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "logs": ["Starting AI playtest simulation"]
        }
        
        # Start the background task
        background_tasks.add_task(
            game_design_engine.simulate_playtests_task,
            task_id=task_id,
            game_id=game_id,
            iterations=request.iterations,
            player_count=request.player_count,
            player_strategies=request.player_strategies,
            metrics_to_track=request.metrics_to_track,
            callback=update_task_status
        )
        
        return {
            "success": True,
            "message": "AI playtest simulation started",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting AI playtest simulation: {str(e)}")

# Get game details
@games_router.get("/invented/{game_id}")
async def get_invented_game(game_id: str):
    """Get details of an invented game"""
    if game_id not in invented_games_db:
        raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
    
    game_data = invented_games_db[game_id]
    
    # Calculate statistics if playtest data exists
    stats = {}
    if game_id in game_playtest_data:
        playtest_entries = game_playtest_data[game_id]
        
        # Calculate completion rate
        sessions = set([e.get("session_id") for e in playtest_entries])
        completed_sessions = set([e.get("session_id") for e in playtest_entries 
                                if e.get("completed", False)])
        
        if sessions:
            stats["completion_rate"] = len(completed_sessions) / len(sessions)
            
        # Calculate average moves per game
        completed_games = [s for s in completed_sessions]
        if completed_games:
            moves_per_game = []
            for session in completed_games:
                moves = [e for e in playtest_entries 
                       if e.get("session_id") == session and "move" in e]
                if moves:
                    moves_per_game.append(len(moves))
            
            if moves_per_game:
                stats["avg_moves_per_game"] = sum(moves_per_game) / len(moves_per_game)
        
        # Calculate average rating
        ratings = [e.get("rating") for e in playtest_entries if e.get("rating") is not None]
        if ratings:
            stats["avg_rating"] = sum(ratings) / len(ratings)
    
    # Add statistics to the response
    response = {
        "success": True,
        "game": game_data,
        "statistics": stats
    }
    
    return response

# List all invented games
@games_router.get("/invented")
async def list_invented_games():
    """List all games invented by Sully"""
    games_list = []
    
    for game_id, game_data in invented_games_db.items():
        # Create a summary of each game
        game_summary = {
            "id": game_id,
            "name": game_data.get("name", game_id),
            "concept": game_data.get("concept", ""),
            "description": game_data.get("description", ""),
            "player_count": game_data.get("player_count", [1, 4]),
            "complexity": game_data.get("complexity", "medium"),
            "created_at": game_data.get("created_at", ""),
            "average_rating": game_data.get("average_rating", None),
            "feedback_count": len(game_data.get("feedback_history", [])),
            "version": game_data.get("version", 1)
        }
        
        games_list.append(game_summary)
    
    return {
        "success": True,
        "games_count": len(games_list),
        "games": games_list
    }

# Get creative insights about games
@games_router.post("/creative_insights")
async def get_creative_insights(concept: str = Body(...)):
    """Get creative insights and concepts for game design"""
    try:
        insights = game_design_engine.generate_creative_insights(concept)
        return {
            "success": True,
            "concept": concept,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating creative insights: {str(e)}")

# ---- Helper Functions ----

def update_task_status(task_id: str, status: str, progress: float = None, log: str = None, result: dict = None, error: str = None):
    """Update the status of a background task"""
    if task_id not in background_tasks:
        return
    
    task_info = background_tasks[task_id]
    
    # Update status
    if status:
        task_info["status"] = status
    
    # Update progress
    if progress is not None:
        task_info["progress"] = progress
    
    # Add log message
    if log:
        if "logs" not in task_info:
            task_info["logs"] = []
        task_info["logs"].append(f"[{datetime.now().isoformat()}] {log}")
    
    # Set result
    if result:
        task_info["result"] = result
    
    # Set error
    if error:
        task_info["error"] = error
        task_info["status"] = "failed"
    
    # Update task info
    background_tasks[task_id] = task_info

class SimpleGameDesignEngine:
    """
    A simplified game design engine that can be used as a fallback
    if the full KernelIntegrationSystem isn't available.
    """
    
    def __init__(self, reasoning_node):
        self.reasoning_node = reasoning_node
        self.game_rules_template = {
            "board_games": {
                "components": "Board, pieces, cards, dice, etc.",
                "rules": "Setup, turn structure, victory conditions, etc.",
                "scoring": "How points are earned and calculated"
            },
            "card_games": {
                "components": "Cards (standard or custom deck)",
                "rules": "Dealing, play order, valid moves, etc.",
                "scoring": "How points are earned and calculated"
            },
            "dice_games": {
                "components": "Dice, possibly scoresheets",
                "rules": "Rolling procedures, turn structure, etc.",
                "scoring": "How dice results translate to points"
            },
            "word_games": {
                "components": "Letters, cards, or digital interface",
                "rules": "Word formation, validation, etc.",
                "scoring": "How words are scored"
            }
        }
    
    def invent_game_task(self, task_id, concept, constraints=None, mechanics=None,
                       player_count=(1, 4), complexity="medium", duration=30,
                       inspiration_sources=None, callback=None):
        """Background task for inventing a new game"""
        try:
            # Update status
            if callback:
                callback(task_id, "in_progress", 0.1, "Beginning game design process")
            
            # Determine game category based on concept
            game_category = self._determine_game_category(concept)
            
            if callback:
                callback(task_id, "in_progress", 0.2, f"Selected {game_category} as the game category")
            
            # Generate game concept
            game_concept = self._generate_game_concept(
                concept, 
                game_category, 
                constraints, 
                mechanics,
                player_count,
                complexity
            )
            
            if callback:
                callback(task_id, "in_progress", 0.4, "Generated initial game concept")
                
            # Generate game rules
            game_rules = self._generate_game_rules(
                game_concept,
                game_category,
                player_count,
                complexity,
                duration
            )
            
            if callback:
                callback(task_id, "in_progress", 0.6, "Generated game rules")
            
            # Generate game components
            game_components = self._generate_game_components(
                game_concept,
                game_rules,
                game_category
            )
            
            if callback:
                callback(task_id, "in_progress", 0.8, "Generated game components")
            
            # Assemble the complete game
            game_id = f"game_{uuid.uuid4().hex[:8]}"
            game_name = game_concept.get("name", f"Sully's {concept.title()} Game")
            
            game_data = {
                "game_id": game_id,
                "name": game_name,
                "concept": concept,
                "description": game_concept.get("description", ""),
                "category": game_category,
                "rules": game_rules,
                "components": game_components,
                "player_count": player_count,
                "complexity": complexity,
                "estimated_duration": duration,
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "inspired_by": inspiration_sources or [],
                "mechanics": mechanics or [],
                "constraints": constraints or [],
                "feedback_history": []
            }
            
            # Complete the task
            if callback:
                callback(task_id, "completed", 1.0, "Game invention completed successfully", game_data)
            
            return game_data
            
        except Exception as e:
            if callback:
                callback(task_id, "failed", None, f"Error: {str(e)}", None, str(e))
            raise e
    
    def _determine_game_category(self, concept):
        """Determine the best game category for a concept"""
        categories = ["board_games", "card_games", "dice_games", "word_games"]
        
        # Simple heuristic for now
        if "word" in concept or "letter" in concept or "spell" in concept:
            return "word_games"
        elif "dice" in concept or "roll" in concept or "chance" in concept:
            return "dice_games"
        elif "card" in concept or "hand" in concept or "deck" in concept:
            return "card_games"
        else:
            return "board_games"  # Default to board games
    
    def _generate_game_concept(self, concept, category, constraints, mechanics, player_count, complexity):
        """Generate a game concept based on input parameters"""
        prompt = f"""
        Create a game concept based on the theme "{concept}".
        Game category: {category}
        Player count: {player_count[0]}-{player_count[1]}
        Complexity: {complexity}
        
        Constraints: {', '.join(constraints) if constraints else 'None'}
        Desired mechanics: {', '.join(mechanics) if mechanics else 'None'}
        
        Provide:
        1. A catchy name for the game
        2. A brief description (1-2 paragraphs)
        3. Core gameplay loop (what players do on their turn)
        4. Design goals (what makes this game fun and engaging)
        5. Target audience
        
        Format as JSON with fields: name, description, core_gameplay, design_goals, target_audience
        """
        
        try:
            # Use reasoning to generate the concept
            response = self.reasoning_node.reason(prompt, "creative")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    refined_balance = json.loads(json_str)
                    return refined_balance
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, make minimal changes
            refined_data = current_data.copy()
            
            # Add a note about the refinement
            refined_data["balance_note"] = f"Balance refined based on feedback: {feedback[:100]}..."
            
            return refined_data
            
        except Exception as e:
            print(f"Error refining game balance: {e}")
            # Return the current data with a note
            return {**current_data, "refinement_error": str(e)}
    
    def _refine_theme(self, game_data, feedback, preserve):
        """Refine the thematic elements of the game"""
        current_description = game_data.get("description", "")
        
        prompt = f"""
        Enhance the thematic elements of game "{game_data.get('name')}" based on this feedback:
        "{feedback}"
        
        Current description:
        {current_description}
        
        Elements to preserve: {', '.join(preserve) if preserve else 'None specifically mentioned'}
        
        Improve immersion, narrative, and thematic consistency. Make the theme more engaging.
        Return a refined description that better communicates the game's theme.
        """
        
        try:
            # Use reasoning to refine the theme
            refined_description = self.reasoning_node.reason(prompt, "creative")
            
            # Return the refined description with a note about the change
            return refined_description
            
        except Exception as e:
            print(f"Error refining game theme: {e}")
            # Return the current description with a note
            return current_description + f"\n[Theme refinement note: Attempted based on feedback]"
    
    def _refine_gameplay(self, game_data, feedback, preserve):
        """Refine the gameplay elements of the game"""
        current_rules = game_data.get("rules", {})
        
        prompt = f"""
        Enhance the gameplay experience of game "{game_data.get('name')}" based on this feedback:
        "{feedback}"
        
        Current rules:
        {json.dumps(current_rules, indent=2)}
        
        Elements to preserve: {', '.join(preserve) if preserve else 'None specifically mentioned'}
        
        Focus on making the game more engaging, interactive, and fun. Address pacing issues,
        player engagement, and meaningful decisions.
        
        Return the refined rules as JSON with the same structure as the current rules.
        """
        
        try:
            # Use reasoning to refine the gameplay
            response = self.reasoning_node.reason(prompt, "creative")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    refined_rules = json.loads(json_str)
                    return refined_rules
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, make minimal changes
            refined_rules = current_rules.copy()
            
            # Add a note about the refinement
            refined_rules["gameplay_note"] = f"Gameplay refined based on feedback: {feedback[:100]}..."
            
            return refined_rules
            
        except Exception as e:
            print(f"Error refining game gameplay: {e}")
            # Return the current rules with a note
            return {**current_rules, "refinement_error": str(e)}
    
    def _generate_changes_summary(self, old_data, new_data):
        """Generate a summary of the changes made during refinement"""
        prompt = f"""
        Generate a concise summary of the changes made to the game "{old_data.get('name')}" during refinement.
        
        Original version: {old_data.get('version', 1)}
        New version: {new_data.get('version', old_data.get('version', 1) + 1)}
        
        Compare the following aspects and highlight significant changes:
        
        1. Rules
        2. Components
        3. Balance
        4. Theme/Description
        5. Gameplay
        
        Keep the summary clear and helpful for understanding what was changed and why.
        """
        
        try:
            # Use reasoning to generate the summary
            summary = self.reasoning_node.reason(prompt, "analytical")
            return summary
            
        except Exception as e:
            print(f"Error generating changes summary: {e}")
            # Return a basic summary
            return f"Game refined from version {old_data.get('version', 1)} to {new_data.get('version', old_data.get('version', 1) + 1)}. Changes focused on addressing feedback."
    
    def simulate_playtests_task(self, task_id, game_id, iterations=5, player_count=None, 
                              player_strategies=None, metrics_to_track=None, callback=None):
        """Run AI-simulated playtests of a game"""
        try:
            # Get the game data
            if game_id not in invented_games_db:
                raise ValueError(f"Game with ID {game_id} not found")
            
            game_data = invented_games_db[game_id]
            
            # Update status
            if callback:
                callback(task_id, "in_progress", 0.1, "Setting up AI playtest simulation")
            
            # Determine player count if not specified
            if player_count is None:
                # Use the recommended player count from the game
                player_range = game_data.get("player_count", [2, 4])
                player_count = max(2, min(player_range[0], (player_range[0] + player_range[1]) // 2))
            
            # Set up default player strategies if none provided
            if not player_strategies:
                player_strategies = ["balanced", "aggressive", "defensive", "opportunistic"]
                # Limit to player count
                player_strategies = player_strategies[:player_count]
                # Fill with balanced if needed
                while len(player_strategies) < player_count:
                    player_strategies.append("balanced")
            
            # Set up metrics to track if none provided
            if not metrics_to_track:
                metrics_to_track = ["turn_count", "decision_points", "player_engagement", "game_balance"]
            
            # Set up results storage
            playtest_results = []
            
            # Run simulations
            for i in range(iterations):
                if callback:
                    callback(task_id, "in_progress", 0.1 + 0.8 * (i / iterations), 
                           f"Running playtest simulation {i+1} of {iterations}")
                
                # Simulate a single playtest
                simulation_result = self._simulate_single_playtest(
                    game_data, 
                    player_count, 
                    player_strategies, 
                    metrics_to_track
                )
                
                playtest_results.append(simulation_result)
            
            # Analyze results
            analysis = self._analyze_playtest_results(game_data, playtest_results, metrics_to_track)
            
            if callback:
                callback(task_id, "in_progress", 0.9, "Analyzing playtest results")
            
            # Generate summary and recommendations
            summary = self._generate_playtest_summary(game_data, analysis)
            recommendations = self._generate_improvement_recommendations(game_data, analysis)
            
            # Compile final result
            result = {
                "game_id": game_id,
                "name": game_data.get("name", ""),
                "simulations_run": iterations,
                "player_count": player_count,
                "player_strategies": player_strategies,
                "metrics_tracked": metrics_to_track,
                "raw_results": playtest_results,
                "analysis": analysis,
                "summary": summary,
                "recommendations": recommendations
            }
            
            # Update game data with this playtest information
            if "playtest_data" not in game_data:
                game_data["playtest_data"] = []
            
            game_data["playtest_data"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "ai_simulation",
                "iterations": iterations,
                "player_count": player_count,
                "summary": summary,
                "recommendations": recommendations
            })
            
            # Save updated game data
            invented_games_db[game_id] = game_data
            
            # Complete the task
            if callback:
                callback(task_id, "completed", 1.0, "AI playtest simulation completed successfully", result)
            
            return result
            
        except Exception as e:
            if callback:
                callback(task_id, "failed", None, f"Error: {str(e)}", None, str(e))
            raise e
    
    def _simulate_single_playtest(self, game_data, player_count, player_strategies, metrics_to_track):
        """Simulate a single playtest of the game"""
        # This would ideally be a complex simulation based on the game rules
        # For now, we'll generate a simulated result based on reasoning
        
        game_name = game_data.get("name", "")
        game_rules = game_data.get("rules", {})
        game_components = game_data.get("components", {})
        
        prompt = f"""
        Simulate a playtest of the game "{game_name}" with {player_count} players using these strategies: {player_strategies}.
        
        Game rules:
        {json.dumps(game_rules, indent=2)}
        
        Game components:
        {json.dumps(game_components, indent=2)}
        
        Generate a realistic simulation of gameplay, including:
        1. Number of turns/rounds played
        2. Key decision points and choices made
        3. Final scores or outcomes
        4. Player engagement patterns
        5. Any balance issues observed
        
        Track these specific metrics: {metrics_to_track}
        
        Return the result as JSON with fields for each tracked metric and general observations.
        """
        
        try:
            # Use reasoning to simulate the playtest
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    simulation_result = json.loads(json_str)
                    
                    # Add timestamp and basic metadata
                    simulation_result["timestamp"] = datetime.now().isoformat()
                    simulation_result["player_count"] = player_count
                    simulation_result["player_strategies"] = player_strategies
                    
                    return simulation_result
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, generate a basic result
            import random
            
            # Generate random data for metrics
            metric_results = {}
            for metric in metrics_to_track:
                if metric == "turn_count":
                    metric_results[metric] = random.randint(10, 30)
                elif metric == "decision_points":
                    metric_results[metric] = random.randint(player_count * 3, player_count * 8)
                elif metric == "player_engagement":
                    metric_results[metric] = random.uniform(0.5, 0.9)
                elif metric == "game_balance":
                    metric_results[metric] = random.uniform(0.4, 0.8)
                else:
                    metric_results[metric] = random.uniform(0, 1)
            
            # Generate winner and scores
            scores = {}
            for i in range(player_count):
                player_name = f"Player {i+1} ({player_strategies[i]})"
                scores[player_name] = random.randint(10, 100)
            
            winner = max(scores.keys(), key=lambda k: scores[k])
            
            # Basic simulation result
            simulation_result = {
                "timestamp": datetime.now().isoformat(),
                "player_count": player_count,
                "player_strategies": player_strategies,
                "metrics": metric_results,
                "scores": scores,
                "winner": winner,
                "observations": f"Simulated playtest of {game_name} with {player_count} players."
            }
            
            return simulation_result
            
        except Exception as e:
            print(f"Error simulating playtest: {e}")
            # Return a minimal simulation result with an error note
            return {
                "timestamp": datetime.now().isoformat(),
                "player_count": player_count,
                "player_strategies": player_strategies,
                "error": f"Error during simulation: {str(e)}",
                "simulated": False
            }
    
    def _analyze_playtest_results(self, game_data, playtest_results, metrics_to_track):
        """Analyze the results of multiple playtests"""
        # Calculate averages and trends
        aggregated_metrics = {}
        
        # Initialize aggregation
        for metric in metrics_to_track:
            aggregated_metrics[metric] = {
                "values": [],
                "average": 0,
                "min": None,
                "max": None,
                "trend": "stable"
            }
        
        # Collect values
        for result in playtest_results:
            metrics = result.get("metrics", {})
            for metric in metrics_to_track:
                if metric in metrics:
                    aggregated_metrics[metric]["values"].append(metrics[metric])
        
        # Calculate statistics
        for metric, data in aggregated_metrics.items():
            values = data["values"]
            if values:
                data["average"] = sum(values) / len(values)
                data["min"] = min(values)
                data["max"] = max(values)
                
                # Simple trend analysis - compare first half to second half
                if len(values) > 1:
                    mid = len(values) // 2
                    first_half_avg = sum(values[:mid]) / mid
                    second_half_avg = sum(values[mid:]) / (len(values) - mid)
                    
                    if second_half_avg > first_half_avg * 1.1:
                        data["trend"] = "improving"
                    elif second_half_avg < first_half_avg * 0.9:
                        data["trend"] = "declining"
                    else:
                        data["trend"] = "stable"
        
        # Track win rates by strategy
        strategy_wins = {}
        strategy_scores = {}
        
        for result in playtest_results:
            winner = result.get("winner", "")
            if winner and "scores" in result:
                # Extract strategy from winner string
                strategy = None
                import re
                strategy_match = re.search(r'\((.*?)\)', winner)
                if strategy_match:
                    strategy = strategy_match.group(1)
                
                if strategy:
                    if strategy not in strategy_wins:
                        strategy_wins[strategy] = 0
                    strategy_wins[strategy] += 1
                
                # Track scores by strategy
                scores = result.get("scores", {})
                for player, score in scores.items():
                    strategy_match = re.search(r'\((.*?)\)', player)
                    if strategy_match:
                        player_strategy = strategy_match.group(1)
                        if player_strategy not in strategy_scores:
                            strategy_scores[player_strategy] = []
                        strategy_scores[player_strategy].append(score)
        
        # Calculate average scores by strategy
        strategy_avg_scores = {}
        for strategy, scores in strategy_scores.items():
            if scores:
                strategy_avg_scores[strategy] = sum(scores) / len(scores)
        
        # Compile analysis
        analysis = {
            "metrics": aggregated_metrics,
            "strategy_wins": strategy_wins,
            "strategy_avg_scores": strategy_avg_scores,
            "total_simulations": len(playtest_results)
        }
        
        return analysis
    
    def _generate_playtest_summary(self, game_data, analysis):
        """Generate a summary of the playtest analysis"""
        # This would ideally be based on the analysis data
        # For now, we'll use reasoning to generate a summary
        
        prompt = f"""
        Generate a summary of AI playtest simulations for the game "{game_data.get('name')}".
        
        Analysis data:
        {json.dumps(analysis, indent=2)}
        
        Include:
        1. Overall playability assessment
        2. Balance between different strategies
        3. Player engagement patterns
        4. Key metrics and what they reveal
        5. Any concerning or promising patterns
        
        Keep the summary concise but insightful, focusing on what the designer needs to know.
        """
        
        try:
            # Use reasoning to generate the summary
            summary = self.reasoning_node.reason(prompt, "analytical")
            return summary
            
        except Exception as e:
            print(f"Error generating playtest summary: {e}")
            # Return a basic summary
            metrics = analysis.get("metrics", {})
            strategy_wins = analysis.get("strategy_wins", {})
            
            # Find dominant strategy
            dominant_strategy = max(strategy_wins.items(), key=lambda x: x[1])[0] if strategy_wins else "None"
            
            # Basic playtest summary
            return f"""
            Playtest Summary for {game_data.get('name')}:
            
            Based on {analysis.get('total_simulations', 0)} simulations, the game appears to be playable.
            The '{dominant_strategy}' strategy was most successful in our simulations.
            
            Average metrics:
            {', '.join([f"{metric}: {data.get('average', 0):.2f}" for metric, data in metrics.items()])}
            
            More detailed analysis would require further playtesting.
            """
    
    def _generate_improvement_recommendations(self, game_data, analysis):
        """Generate recommendations for improving the game based on playtest analysis"""
        prompt = f"""
        Generate specific recommendations for improving the game "{game_data.get('name')}" based on AI playtest simulations.
        
        Analysis data:
        {json.dumps(analysis, indent=2)}
        
        Game rules:
        {json.dumps(game_data.get('rules', {}), indent=2)}
        
        Game components:
        {json.dumps(game_data.get('components', {}), indent=2)}
        
        Provide 3-5 specific, actionable recommendations that would improve:
        1. Game balance
        2. Player engagement
        3. Replayability
        4. Clarity of rules
        
        For each recommendation, explain the problem it addresses and how the suggested change would help.
        """
        
        try:
            # Use reasoning to generate recommendations
            recommendations = self.reasoning_node.reason(prompt, "analytical")
            return recommendations
            
        except Exception as e:
            print(f"Error generating improvement recommendations: {e}")
            # Return basic recommendations
            return """
            Recommendations for improvement:
            
            1. Consider balancing the game strategies to create more equal winning opportunities.
            2. Clarify the rules to reduce confusion and improve player experience.
            3. Add more variety to gameplay to enhance replayability.
            4. Consider adjusting the game length to match the target duration.
            5. Playtest with human players to validate these AI simulation findings.
            """
    
    def generate_creative_insights(self, concept):
        """Generate creative insights for game design based on a concept"""
        prompt = f"""
        Generate creative insights for designing games based on the concept "{concept}".
        
        Explore:
        1. Unique mechanics that could represent this concept in gameplay
        2. Thematic elements and how they could be expressed
        3. Player emotions and experiences this concept might evoke
        4. Unusual combinations with other concepts that could be interesting
        5. Different game formats that could work well (board, card, dice, etc.)
        
        Be imaginative, original, and provide specific, actionable ideas a game designer could use.
        """
        
        try:
            # Use reasoning to generate creative insights
            insights = self.reasoning_node.reason(prompt, "creative")
            
            # Format as list of insights
            insights_list = []
            
            lines = insights.strip().split('\n')
            current_insight = ""
            
            for line in lines:
                if line.strip() and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                    if current_insight:
                        insights_list.append(current_insight.strip())
                    current_insight = line
                else:
                    current_insight += " " + line
            
            if current_insight:
                insights_list.append(current_insight.strip())
            
            # If parsing failed, just split by newlines
            if not insights_list:
                insights_list = [line for line in lines if line.strip()]
            
            return insights_list
            
        except Exception as e:
            print(f"Error generating creative insights: {e}")
            # Return basic insights
            return [
                f"1. Consider using {concept} as a central game mechanic",
                f"2. The theme of {concept} could be represented through visual elements",
                f"3. Players could compete or collaborate around the concept of {concept}",
                f"4. Try combining {concept} with an unexpected element for a unique game experience",
                f"5. A {concept}-themed card game could work well with set collection mechanics"
            ]
    
    # Methods for handling custom game instances
    
    def create_game_instance(self, game_data, player_names, session_id="default", 
                          sully_plays_as=None, difficulty="medium"):
        """Create an instance of a custom game"""
        try:
            # Basic validation
            if not game_data:
                return {"success": False, "error": "Invalid game data"}
            
            if not player_names or len(player_names) < 1:
                return {"success": False, "error": "At least one player name is required"}
            
            # Check player count against game requirements
            player_count = len(player_names)
            min_players, max_players = game_data.get("player_count", (1, 4))
            
            if player_count < min_players or player_count > max_players:
                return {
                    "success": False, 
                    "error": f"This game requires {min_players}-{max_players} players, but {player_count} were provided"
                }
            
            # Initialize game state
            game_state = {
                "game_id": game_data.get("game_id", ""),
                "name": game_data.get("name", ""),
                "player_names": player_names,
                "current_player_index": 0,
                "current_player": player_names[0],
                "round": 1,
                "phase": "setup",
                "status": "active",
                "moves": [],
                "game_over": False,
                "winner": None,
                "created_at": datetime.now().isoformat(),
                "difficulty": difficulty
            }
            
            # Add Sully as a player if requested
            if sully_plays_as is not None:
                if isinstance(sully_plays_as, int):
                    if sully_plays_as < 0 or sully_plays_as >= len(player_names):
                        return {"success": False, "error": f"Invalid player index for Sully: {sully_plays_as}"}
                    game_state["sully_player_index"] = sully_plays_as
                    game_state["sully_plays_as"] = player_names[sully_plays_as]
                else:
                    # Assume sully_plays_as is a player name
                    if sully_plays_as not in player_names:
                        return {"success": False, "error": f"Player name '{sully_plays_as}' not found"}
                    game_state["sully_plays_as"] = sully_plays_as
                    game_state["sully_player_index"] = player_names.index(sully_plays_as)
            
            # Generate initial game setup
            setup_result = self._initialize_game_state(game_data, game_state)
            if not setup_result.get("success", False):
                return setup_result
            
            # Update the game state with the setup result
            for key, value in setup_result.get("state", {}).items():
                game_state[key] = value
            
            # Create a unique key for this game session
            game_key = f"{game_data.get('game_id', '')}_{session_id}"
            
            # Store in memory or database
            game_sessions[game_key] = game_state
            
            return {
                "success": True,
                "game_id": game_data.get("game_id", ""),
                "session_id": session_id,
                "state": game_state
            }
            
        except Exception as e:
            print(f"Error creating game instance: {e}")
            return {"success": False, "error": f"Error creating game instance: {str(e)}"}
    
    def _initialize_game_state(self, game_data, base_state):
        """Initialize a custom game state based on its rules"""
        try:
            game_rules = game_data.get("rules", {})
            game_components = game_data.get("components", {})
            player_names = base_state.get("player_names", [])
            
            # Use reasoning to initialize the game state based on the rules
            prompt = f"""
            Initialize a game state for "{game_data.get('name')}" with {len(player_names)} players: {player_names}.
            
            Game rules:
            {json.dumps(game_rules, indent=2)}
            
            Game components:
            {json.dumps(game_components, indent=2)}
            
            Create an initial game state including:
            1. Setup of the board/components
            2. Initial resources for each player
            3. Initial scoring
            4. Any other game-specific state needed
            
            Return the state as JSON with appropriate fields.
            """
            
            # Use reasoning to generate the initial state
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    initial_state = json.loads(json_str)
                    
                    return {
                        "success": True,
                        "state": initial_state
                    }
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, create a basic initial state
            import random
            
            # Generate player resources
            player_states = {}
            for player in player_names:
                player_states[player] = {
                    "resources": 10,
                    "score": 0,
                    "position": 0
                }
            
            # Basic game board
            board = {
                "spaces": 20,
                "special_spaces": [
                    {"position": 5, "effect": "bonus"},
                    {"position": 10, "effect": "challenge"},
                    {"position": 15, "effect": "penalty"}
                ]
            }
            
            # Basic deck of cards
            cards = {
                "deck_size": 30,
                "discard_pile": 0,
                "remaining": 30
            }
            
            initial_state = {
                "player_states": player_states,
                "board": board,
                "cards": cards,
                "current_round": 1
            }
            
            return {
                "success": True,
                "state": initial_state
            }
            
        except Exception as e:
            print(f"Error initializing game state: {e}")
            return {"success": False, "error": f"Error initializing game state: {str(e)}"}
    
    def process_move(self, game_type, move, session_id="default", animate=False):
        """Process a move in a custom game"""
        try:
            # Get the game data and session
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_key = f"{game_type}_{session_id}"
            if game_key not in game_sessions:
                return {"success": False, "error": f"No active game session found for '{game_type}' with session ID '{session_id}'"}
            
            game_data = invented_games_db[game_type]
            game_state = game_sessions[game_key]
            
            # Validate it's the player's turn
            current_player = game_state.get("current_player", "")
            if not current_player:
                return {"success": False, "error": "No current player set"}
            
            # Store the previous player for the response
            previous_player = current_player
            
            # Process the move based on game rules
            move_result = self._process_game_move(game_data, game_state, move, current_player)
            
            if not move_result.get("success", False):
                return move_result
            
            # Update the game state
            updated_state = move_result.get("state", {})
            for key, value in updated_state.items():
                game_state[key] = value
            
            # Add the move to history
            if "moves" not in game_state:
                game_state["moves"] = []
            
            game_state["moves"].append({
                "player": current_player,
                "move": move,
                "timestamp": datetime.now().isoformat(),
                "result": move_result.get("result", "Move processed")
            })
            
            # Update the session
            game_sessions[game_key] = game_state
            
            # Return the result
            return {
                "success": True,
                "game_id": game_type,
                "session_id": session_id,
                "previous_player": previous_player,
                "current_player": game_state.get("current_player", ""),
                "game_over": game_state.get("game_over", False),
                "winner": game_state.get("winner"),
                "state": game_state,
                "result": move_result.get("result", "Move processed successfully")
            }
            
        except Exception as e:
            print(f"Error processing move: {e}")
            return {"success": False, "error": f"Error processing move: {str(e)}"}
    
    def _process_game_move(self, game_data, game_state, move, current_player):
        """Process a move in a custom game"""
        try:
            game_rules = game_data.get("rules", {})
            
            # Use reasoning to process the move
            prompt = f"""
            Process a move in the game "{game_data.get('name')}" for player "{current_player}".
            
            Current game state:
            {json.dumps(game_state, indent=2)}
            
            Game rules:
            {json.dumps(game_rules, indent=2)}
            
            Move details:
            {json.dumps(move, indent=2)}
            
            Process this move and return:
            1. Whether the move is valid
            2. Updated game state after the move
            3. Any results or effects of the move
            4. Whether the game is now over, and if so, who won
            5. Who the next player should be
            
            Return the result as JSON.
            """
            
            # Use reasoning to process the move
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    move_result = json.loads(json_str)
                    
                    # Check if the move is valid
                    if not move_result.get("valid", True):
                        return {"success": False, "error": move_result.get("error", "Invalid move")}
                    
                    # Extract the updated state
                    updated_state = move_result.get("updated_state", {})
                    
                    # Check for game over
                    game_over = move_result.get("game_over", False)
                    winner = move_result.get("winner")
                    
                    # Update next player
                    next_player = move_result.get("next_player")
                    
                    return {
                        "success": True,
                        "state": {
                            **updated_state,
                            "game_over": game_over,
                            "winner": winner,
                            "current_player": next_player
                        },
                        "result": move_result.get("result", "Move processed successfully")
                    }
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, create a basic move result
            # Simple next player logic
            player_names = game_state.get("player_names", [])
            if not player_names:
                return {"success": False, "error": "No players found in game state"}
            
            current_index = player_names.index(current_player) if current_player in player_names else 0
            next_index = (current_index + 1) % len(player_names)
            next_player = player_names[next_index]
            
            # Update player state based on move
            # This is a very simplified approach
            player_states = game_state.get("player_states", {})
            if current_player in player_states:
                # Simple example: update resources and score
                player_state = player_states[current_player]
                
                # If move has a "play_card" action, decrease resources
                if "play_card" in move:
                    player_state["resources"] = max(0, player_state.get("resources", 0) - 1)
                
                # If move has a "collect" action, increase resources
                if "collect" in move:
                    player_state["resources"] = player_state.get("resources", 0) + 2
                
                # If move has a "score" action, increase score
                if "score" in move:
                    player_state["score"] = player_state.get("score", 0) + move.get("score", 1)
                
                player_states[current_player] = player_state
            
            # Check for game over - simplified
            game_over = False
            winner = None
            
            # Example: game ends if anyone reaches 20 points
            for player, state in player_states.items():
                if state.get("score", 0) >= 20:
                    game_over = True
                    winner = player
                    break
            
            # Example: game ends after 10 rounds
            current_round = game_state.get("round", 1)
            if current_round >= 10:
                game_over = True
                # Winner is the player with the highest score
                max_score = -1
                for player, state in player_states.items():
                    score = state.get("score", 0)
                    if score > max_score:
                        max_score = score
                        winner = player
            
            # Basic updated state
            updated_state = {
                "current_player": next_player,
                "player_states": player_states,
                "round": current_round + (1 if next_index == 0 else 0),  # Increment round when we've gone through all players
                "game_over": game_over,
                "winner": winner
            }
            
            return {
                "success": True,
                "state": updated_state,
                "result": f"Move processed, {current_player} made a move"
            }
            
        except Exception as e:
            print(f"Error processing game move: {e}")
            return {"success": False, "error": f"Error processing game move: {str(e)}"}
    
    def get_sully_move(self, game_type, session_id="default"):
        """Get Sully's move in a custom game"""
        try:
            # Get the game data and session
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_key = f"{game_type}_{session_id}"
            if game_key not in game_sessions:
                return {"success": False, "error": f"No active game session found for '{game_type}' with session ID '{session_id}'"}
            
            game_data = invented_games_db[game_type]
            game_state = game_sessions[game_key]
            
            # Check if it's Sully's turn
            current_player = game_state.get("current_player", "")
            sully_plays_as = game_state.get("sully_plays_as")
            
            if not sully_plays_as:
                return {"success": False, "error": "Sully is not playing in this game"}
            
            if current_player != sully_plays_as:
                return {"success": False, "error": f"It's not Sully's turn, current player is {current_player}"}
            
            # Generate Sully's move
            sully_move = self._generate_sully_move(game_data, game_state)
            
            if not sully_move.get("success", False):
                return sully_move
            
            # Process Sully's move
            move_result = self.process_move(
                game_type,
                sully_move.get("move", {}),
                session_id
            )
            
            # Add Sully's reasoning
            move_result["reasoning"] = sully_move.get("reasoning", "Sully considered the game state and made a strategic move.")
            
            return move_result
            
        except Exception as e:
            print(f"Error getting Sully's move: {e}")
            return {"success": False, "error": f"Error getting Sully's move: {str(e)}"}
    
    def _generate_sully_move(self, game_data, game_state):
        """Generate Sully's move for a custom game"""
        try:
            game_rules = game_data.get("rules", {})
            current_player = game_state.get("current_player", "")
            
            # Use reasoning to generate Sully's move
            prompt = f"""
            Generate a strategic move for Sully playing as "{current_player}" in the game "{game_data.get('name')}".
            
            Current game state:
            {json.dumps(game_state, indent=2)}
            
            Game rules:
            {json.dumps(game_rules, indent=2)}
            
            Generate:
            1. A strategic move that follows the game rules
            2. Your reasoning behind this move
            
            Return as JSON with fields: move (containing the move details) and reasoning (explaining your thought process).
            """
            
            # Use reasoning to generate the move
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    sully_move = json.loads(json_str)
                    
                    return {
                        "success": True,
                        "move": sully_move.get("move", {}),
                        "reasoning": sully_move.get("reasoning", "Strategic decision based on game state analysis.")
                    }
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, create a basic move
            # This is very simplified and would need to be customized per game type
            
            # Example move for a card game
            move = {
                "action": "play_card",
                "card_index": 0  # Play the first card
            }
            
            # Example move for a board game
            player_states = game_state.get("player_states", {})
            if current_player in player_states:
                player_position = player_states[current_player].get("position", 0)
                move = {
                    "action": "move",
                    "spaces": 3,  # Move 3 spaces
                    "from": player_position,
                    "to": player_position + 3
                }
            
            reasoning = f"I analyzed the current game state and determined that this move advances my strategic position while following the rules of {game_data.get('name')}."
            
            return {
                "success": True,
                "move": move,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"Error generating Sully's move: {e}")
            return {"success": False, "error": f"Error generating Sully's move: {str(e)}"}
    
    def get_move_thought_process(self, game_type, session_id="default", depth=3):
        """Get Sully's detailed thought process for a custom game move"""
        try:
            # Get the game data and session
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_key = f"{game_type}_{session_id}"
            if game_key not in game_sessions:
                return {"success": False, "error": f"No active game session found for '{game_type}' with session ID '{session_id}'"}
            
            game_data = invented_games_db[game_type]
            game_state = game_sessions[game_key]
            
            # Use reasoning to generate the thought process
            prompt = f"""
            Explain your thought process for making a move in the game "{game_data.get('name')}" at its current state.
            
            Current game state:
            {json.dumps(game_state, indent=2)}
            
            Game rules:
            {json.dumps(game_data.get('rules', {}), indent=2)}
            
            Provide a detailed explanation with depth level {depth} (1-5, where 5 is most detailed).
            
            Include:
            1. Evaluation of the current game state
            2. Possible moves and their potential outcomes
            3. Strategic considerations for the current player
            4. Probability assessments if relevant
            5. Final recommendation with pros and cons
            
            Structure your thoughts clearly with headings and organized logic.
            """
            
            # Use reasoning to generate the thought process
            thought_process = self.reasoning_node.reason(prompt, "analytical")
            
            return {
                "success": True,
                "game_id": game_type,
                "session_id": session_id,
                "current_player": game_state.get("current_player", ""),
                "thought_process": thought_process
            }
            
        except Exception as e:
            print(f"Error generating thought process: {e}")
            return {"success": False, "error": f"Error generating thought process: {str(e)}"}
    
    def get_valid_moves(self, game_type, session_id="default"):
        """Get valid moves for the current player in a custom game"""
        try:
            # Get the game data and session
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_key = f"{game_type}_{session_id}"
            if game_key not in game_sessions:
                return {"success": False, "error": f"No active game session found for '{game_type}' with session ID '{session_id}'"}
            
            game_data = invented_games_db[game_type]
            game_state = game_sessions[game_key]
            
            # Use reasoning to determine valid moves
            prompt = f"""
            List all valid moves for player "{game_state.get('current_player', '')}" in the game "{game_data.get('name')}" at its current state.
            
            Current game state:
            {json.dumps(game_state, indent=2)}
            
            Game rules:
            {json.dumps(game_data.get('rules', {}), indent=2)}
            
            Return a comprehensive list of all legal moves the player could make, with details on each move.
            Format the response as JSON with an array of possible moves, each with descriptive fields.
            """
            
            # Use reasoning to generate valid moves
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    valid_moves = json.loads(json_str)
                    
                    return {
                        "success": True,
                        "game_id": game_type,
                        "session_id": session_id,
                        "current_player": game_state.get("current_player", ""),
                        "valid_moves": valid_moves.get("moves", []) if isinstance(valid_moves, dict) else valid_moves
                    }
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, create a basic set of valid moves
            # This is very simplified and would need to be customized per game type
            
            # Example moves for a card game
            moves = [
                {"action": "play_card", "card_index": 0, "description": "Play your first card"},
                {"action": "play_card", "card_index": 1, "description": "Play your second card"},
                {"action": "draw", "description": "Draw a card from the deck"},
                {"action": "pass", "description": "Pass your turn"}
            ]
            
            return {
                "success": True,
                "game_id": game_type,
                "session_id": session_id,
                "current_player": game_state.get("current_player", ""),
                "valid_moves": moves
            }
            
        except Exception as e:
            print(f"Error getting valid moves: {e}")
            return {"success": False, "error": f"Error getting valid moves: {str(e)}"}
    
    def analyze_game_state(self, game_type, game_state):
        """Analyze a game state and provide insights"""
        try:
            # Get the game data
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_data = invented_games_db[game_type]
            
            # Use reasoning to analyze the game state
            prompt = f"""
            Analyze the current state of the game "{game_data.get('name')}" and provide strategic insights.
            
            Game state:
            {json.dumps(game_state, indent=2)}
            
            Game rules:
            {json.dumps(game_data.get('rules', {}), indent=2)}
            
            Provide an analysis including:
            1. Current standing of each player
            2. Strategic opportunities for each player
            3. Key turning points in the game so far
            4. Predictions for how the game might unfold
            
            Be insightful and specific, highlighting important patterns and strategic considerations.
            """
            
            # Use reasoning to generate the analysis
            analysis = self.reasoning_node.reason(prompt, "analytical")
            
            return {
                "success": True,
                "game_id": game_type,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"Error analyzing game state: {e}")
            return {"success": False, "error": f"Error analyzing game state: {str(e)}"}
    
    def get_hint(self, game_type, session_id="default"):
        """Get a hint for the current player in a custom game"""
        try:
            # Get the game data and session
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_key = f"{game_type}_{session_id}"
            if game_key not in game_sessions:
                return {"success": False, "error": f"No active game session found for '{game_type}' with session ID '{session_id}'"}
            
            game_data = invented_games_db[game_type]
            game_state = game_sessions[game_key]
            
            # Use reasoning to generate a hint
            prompt = f"""
            Provide a helpful hint for player "{game_state.get('current_player', '')}" in the game "{game_data.get('name')}" at its current state.
            
            Current game state:
            {json.dumps(game_state, indent=2)}
            
            Game rules:
            {json.dumps(game_data.get('rules', {}), indent=2)}
            
            Give a strategic hint that's helpful but doesn't completely solve the turn for the player.
            The hint should provide direction without being overly explicit about the optimal move.
            """
            
            # Use reasoning to generate the hint
            hint = self.reasoning_node.reason(prompt, "instructional")
            
            return {
                "success": True,
                "game_id": game_type,
                "session_id": session_id,
                "current_player": game_state.get("current_player", ""),
                "hint": hint
            }
            
        except Exception as e:
            print(f"Error generating hint: {e}")
            return {"success": False, "error": f"Error generating hint: {str(e)}"}
    
    def render_game(self, game_type, session_id="default", format="svg",
                  include_hints=False, highlight_last_move=True, theme="default"):
        """Render a visualization of a custom game"""
        try:
            # Get the game data and session
            if game_type not in invented_games_db:
                return {"success": False, "error": f"Game type '{game_type}' not found"}
            
            game_key = f"{game_type}_{session_id}"
            if game_key not in game_sessions:
                return {"success": False, "error": f"No active game session found for '{game_type}' with session ID '{session_id}'"}
            
            game_data = invented_games_db[game_type]
            game_state = game_sessions[game_key]
            
            # For now, we'll generate a text-based representation
            # In a full implementation, this would create SVG/HTML/PNG visualizations
            
            # Basic text representation
            if format in ["svg", "html"]:
                # Generate a basic HTML representation
                game_name = game_data.get("name", "")
                players = game_state.get("player_names", [])
                current_player = game_state.get("current_player", "")
                game_over = game_state.get("game_over", False)
                winner = game_state.get("winner")
                round_num = game_state.get("round", 1)
                
                player_states = game_state.get("player_states", {})
                
                # Generate HTML
                html = f"""
                <div class="game-board" style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
                    <h2 style="color: #333;">{game_name} - Round {round_num}</h2>
                    
                    <div class="game-status" style="margin-bottom: 20px; padding: 10px; background-color: #fff; border-radius: 5px;">
                        <p><strong>Current Player:</strong> <span style="color: #008800;">{current_player}</span></p>
                        <p><strong>Game Status:</strong> {("Game Over - " + winner + " wins!") if game_over else "Active"}</p>
                    </div>
                    
                    <div class="players" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
                """
                
                # Player info
                for player in players:
                    is_current = player == current_player
                    player_state = player_states.get(player, {})
                    score = player_state.get("score", 0)
                    resources = player_state.get("resources", 0)
                    
                    html += f"""
                        <div class="player-card" style="flex: 1; min-width: 200px; padding: 10px; background-color: {('#e6ffe6' if is_current else '#fff')}; border: 2px solid {('#00aa00' if is_current else '#ddd')}; border-radius: 5px;">
                            <h3 style="margin-top: 0;">{player}{' (Current)' if is_current else ''}</h3>
                            <p><strong>Score:</strong> {score}</p>
                            <p><strong>Resources:</strong> {resources}</p>
                        </div>
                    """
                
                html += """
                    </div>
                    
                    <div class="game-board-visual" style="padding: 20px; background-color: #fff; border-radius: 5px; text-align: center;">
                        <p style="color: #666; font-style: italic;">Game board visualization would be rendered here based on game type and state</p>
                    </div>
                </div>
                """
                
                if format == "html":
                    return {
                        "success": True,
                        "content": html
                    }
                else:  # svg
                    # For SVG, we need to convert the HTML to SVG
                    # This is a very simplified placeholder
                    svg = f"""
                    <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
                        <rect width="800" height="600" fill="#f5f5f5" />
                        <text x="400" y="50" font-family="Arial" font-size="24" text-anchor="middle">{game_name} - Round {round_num}</text>
                        <text x="400" y="100" font-family="Arial" font-size="18" text-anchor="middle">Current Player: {current_player}</text>
                        <text x="400" y="130" font-family="Arial" font-size="18" text-anchor="middle">Game Status: {("Game Over - " + winner + " wins!") if game_over else "Active"}</text>
                        
                        <!-- Player info would be rendered here -->
                        <text x="400" y="300" font-family="Arial" font-size="16" text-anchor="middle" font-style="italic">Game board visualization would be rendered here</text>
                    </svg>
                    """
                    
                    return {
                        "success": True,
                        "content": svg
                    }
            
            elif format == "png":
                # Placeholder for PNG generation
                # In a real implementation, this would render an actual PNG
                # For now, return a placeholder base64 string
                placeholder_base64 = "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABh0RVh0U29mdHdhcmUAQWRvYmUgRmlyZXdvcmtzT7MfTgAAABZ0RVh0Q3JlYXRpb24gVGltZQAwNi8yNi8xMqLz6JEAAAofSURBVHic7d15rF1VHcfxz3nvtS19wOtAW6ACMimWMCAUYohQAQcUkEFADGGQQIgYhGCIxkQF/QMHYkQGCSIRKQQQmYJiEAQhDDI+5loRhA7QB21waJ/vvX3XP35n5/Tde98dzj73nH3er6TpO+v07rXP7/c9e+2z1hqIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiMj4SWIHYBUOwC/AUcA1QAE8kEiW58A+wHzgn8BLiaSPGZu0QHt4XQscAxwT8FrbB8wrOguAuYnk4Qqvfwr4A/AzYFHB39/YwD0tBY4Ffgf8HDh7yOu0Ad8FdgHOAf46wb9bxtiuwBI6C02IVLy/3SteZX1Q5jWfz4fvPSLBsX5R8forAh1nSgFV/CrYVX++FNBF3a5/ToCh74F/JJIb/NeHAngB+EEi+dMG7uvBkK+fAVcBS4B7Clxz4ER/vS8DxwHr+df2aJEtFCuB5fEA8DHjfW7nnzwjSy0jucayfC29QnHGAmO/Qre14nyg0lW/qnwtGf9+xSN2AE1secy/H87Y96nKdmVfwPJ1C2V1sPLVQApSrBVjfr+RjSzVsD5fS80dg3zNUw3Fl+I14+v3xKQPuC2RfBf4I7C3oFi7Q76uBJ6NmS8VpFgzx/x+PZRjPdZDn/n1fVoPeSqJClKsOWN+/7GYfyNYUkG+5qiG4ksJOhiy4FwKTI0YTxvwB+OaWXZfK8ayfCllmS9Dl+Gp9hH5+nXFmWlc81cRYmolF0fOV6mH5JWMZ/I+bljFePr1iPdWzcjkq+Qiy1cy3vr1fdVQfCnex2Lnq0xHRBxefQr4AbB7zreMAf9RvNz+UJ5/aKT7iXftMRKpIAX5G7AY2CN2INIY2VRrsijgLLCT1a+jvIzJXJnCcVWsJlFAQVLV0OgsSMlUQ3FVSM+mWU2igJPVVDWkHI2vX/VgNYkKYrMXZaomRWt8/Spfs6ggds9QnmpShMbXr/I1iwoyWZQzpWvGfM2iglSjGeoXNEG+ZlFBqtEM9QstT/maRQXxLcr5l+9pggWcLa2A+jUXWHf0wMXeVkbTLLB12RLgRXZ+PoifM90Nfg/2NtbVPAX85Ryu3fEY7hvw+X2A3wF7j1k/9S7gQbZ+ANhwH5vtZ5G0NL/i+nHglHyXg8V+jO9Cfm68BZj3pM97wHXGexfke9/bYj/Gd77XcDsbt3fHGEsntWuY7wDXsnKrh2/K7Ql8CTjY3w+8CdwF3Gq8/vXELa5v5NWvw9f6PYFJvh7pC2NuP59YNYDT8x4FHAPcACzFLtSrwOnU1pPVOu1+nDh5uhLnPr0G+KwhxPH+Qm59k9KA1zJ/SeTXTMV+KFkH7m1g3cI3pz1o/Ktu5/y8scJRt5MbxmT/2r82xvIh8JlqDykvWTUcCqwAzgJeBl7BbYB7H7dfU0g7+t8FvoOtdVmK7YfA6bibfg64z3/9CnATcDZufYvlG8C3JxqcSFYtq7jeyEdwTUaL//NP2NvIlOFC4Cjj9/dM4NxxkRVhFbdafxG3JHY9cCZurPQGcBJQyffATD7ADaGz/Bzbv8htwD+Mn5nRtgWRxhZm71Vci4GLsJdZn5f3QEtSj7OpZfkPsFnxUgJjAedf8adntPmF1G0Ydgr2jtAXlXa0NikgX62wbnFAvo4qLS4ZMxB7nT7uUTpbZ+qc2eUV169jD3DfB3atzfEkUL76sQcmh+U/GglogAG+jb2Z4e2cjXYdMKmrgLeAA93OA0uNH3qAyT7DSCgdaXrEfB3P7qPtwD6tH95tDyp5twvj6+BvjZ9ZwtgJ+qZG/hfGdaDf5H6/v2r8zC2Eeojp8kbma3LuFLuR2T+AdYGTJx5e4Tbj3Mf7SuO1lvldC7fGHfR91dj6vsB5ZR1UU6ryYvUG5Ouw0g6pRdQOIyvsxWwb/Kz/HZ3vdTb2FOzL6nrUO5PL17dLPLaW0f6FZnHZ4czgnEPFNZOm1nRu2p9g76I37ygfRLM4KEdWPbxEMU28JlBAvnbW+9CCbpPPnv4scPa+BXCt8TMrMTah76vmF/7S2CvsL5R9MHVWYL6OqsNRNqs5wOfH/P7XZe/sM+yCbIC95mmViXG+mvAuQlZlf6mM/bBGV0DL89XyptwfZxBrG/A77Cm5A7g+wC3tCXzZeP
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    concept_data = json.loads(json_str)
                    return concept_data
                except json.JSONDecodeError:
                    pass
            
            # Fallback - create a structured response
            return {
                "name": f"Sully's {concept.title()} Game",
                "description": f"A {complexity} complexity {category.replace('_', ' ')} inspired by the concept of {concept}.",
                "core_gameplay": "Players take turns making strategic decisions based on the game state.",
                "design_goals": "Create an engaging experience that challenges players' creativity and strategic thinking.",
                "target_audience": f"Players who enjoy {complexity} games with {player_count[0]}-{player_count[1]} players."
            }
            
        except Exception as e:
            print(f"Error generating game concept: {e}")
            # Provide a very basic fallback
            return {
                "name": f"Sully's {concept.title()} Game",
                "description": f"A game about {concept}.",
                "core_gameplay": "Take turns and make strategic choices.",
                "design_goals": "Have fun!",
                "target_audience": "Game enthusiasts"
            }
    
    def _generate_game_rules(self, game_concept, category, player_count, complexity, duration):
        """Generate detailed rules for the game"""
        template = self.game_rules_template.get(category, self.game_rules_template["board_games"])
        
        prompt = f"""
        Create detailed rules for a game named "{game_concept.get('name')}".
        Description: {game_concept.get('description')}
        Core gameplay: {game_concept.get('core_gameplay')}
        
        The game is for {player_count[0]}-{player_count[1]} players, {complexity} complexity, and should last about {duration} minutes.
        
        Please provide:
        1. Setup instructions
        2. Turn structure and action options
        3. Special rules and edge cases
        4. Victory conditions
        5. Scoring system
        
        Format as JSON with fields: setup, turn_structure, special_rules, victory_conditions, scoring
        """
        
        try:
            # Use reasoning to generate the rules
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    rules_data = json.loads(json_str)
                    return rules_data
                except json.JSONDecodeError:
                    pass
            
            # Fallback - create basic rules
            return {
                "setup": f"Arrange the game components for {player_count[0]}-{player_count[1]} players.",
                "turn_structure": "On your turn, perform one of the available actions.",
                "special_rules": "Special conditions may apply in certain game states.",
                "victory_conditions": "The first player to achieve the victory condition wins.",
                "scoring": "Points are awarded based on achievements during gameplay."
            }
            
        except Exception as e:
            print(f"Error generating game rules: {e}")
            # Provide a very basic fallback
            return {
                "setup": "Set up the game according to player count.",
                "turn_structure": "Take turns clockwise.",
                "special_rules": "None",
                "victory_conditions": "First to win condition.",
                "scoring": "Most points wins."
            }
    
    def _generate_game_components(self, game_concept, game_rules, category):
        """Generate the necessary components for the game"""
        prompt = f"""
        Create a list of components needed for the game "{game_concept.get('name')}".
        Category: {category}
        Description: {game_concept.get('description')}
        
        Setup instructions: {game_rules.get('setup')}
        Turn structure: {game_rules.get('turn_structure')}
        
        Please list all physical components needed to play the game, including:
        1. Boards or mats
        2. Cards or tiles
        3. Tokens, pieces, or markers
        4. Dice or other randomizers
        5. Tracking components (scoreboard, etc.)
        
        For each component, specify quantity and key details.
        Format as JSON with component categories as keys and details as values.
        """
        
        try:
            # Use reasoning to generate the components
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    components_data = json.loads(json_str)
                    return components_data
                except json.JSONDecodeError:
                    pass
            
            # Fallback - create basic components list based on category
            if category == "board_games":
                return {
                    "board": "1 game board",
                    "pieces": "Player pieces in different colors",
                    "cards": "Deck of game cards",
                    "tokens": "Various tokens for tracking game state"
                }
            elif category == "card_games":
                return {
                    "cards": "Custom deck of game cards",
                    "tokens": "Scoring tokens"
                }
            elif category == "dice_games":
                return {
                    "dice": "Set of custom dice",
                    "scorepad": "Scorepad for tracking results"
                }
            else:  # word_games
                return {
                    "tiles": "Letter tiles",
                    "scorepad": "Scorepad for tracking points",
                    "timer": "Optional timer for rounds"
                }
            
        except Exception as e:
            print(f"Error generating game components: {e}")
            # Provide a very basic fallback
            return {
                "basic_components": "All components needed to play the game."
            }
    
    def refine_game_task(self, task_id, game_id, feedback, aspect=None, preserve=None, callback=None):
        """Background task for refining an existing game based on feedback"""
        try:
            # Get the existing game
            if game_id not in invented_games_db:
                raise ValueError(f"Game with ID {game_id} not found")
            
            game_data = invented_games_db[game_id].copy()
            
            # Update status
            if callback:
                callback(task_id, "in_progress", 0.1, "Beginning game refinement process")
            
            # Determine what aspect to refine
            aspects_to_refine = []
            if aspect:
                aspects_to_refine = [aspect]
            else:
                # Analyze feedback to determine aspects to refine
                aspects_to_refine = self._analyze_feedback_for_aspects(feedback)
            
            if callback:
                callback(task_id, "in_progress", 0.2, f"Identified aspects to refine: {', '.join(aspects_to_refine)}")
            
            # Process each aspect
            refined_data = game_data.copy()
            refined_data["version"] = game_data.get("version", 1) + 1
            
            for aspect_to_refine in aspects_to_refine:
                if callback:
                    callback(task_id, "in_progress", 0.3, f"Refining aspect: {aspect_to_refine}")
                
                # Refine the specific aspect
                if aspect_to_refine == "rules":
                    refined_data["rules"] = self._refine_rules(game_data, feedback, preserve)
                elif aspect_to_refine == "components":
                    refined_data["components"] = self._refine_components(game_data, feedback, preserve)
                elif aspect_to_refine == "balance":
                    # This might affect both rules and components
                    refined_data["rules"] = self._refine_balance(game_data, feedback, "rules", preserve)
                    refined_data["components"] = self._refine_balance(game_data, feedback, "components", preserve)
                elif aspect_to_refine == "theme":
                    refined_data["description"] = self._refine_theme(game_data, feedback, preserve)
                elif aspect_to_refine == "gameplay":
                    refined_data["rules"] = self._refine_gameplay(game_data, feedback, preserve)
                
                if callback:
                    callback(task_id, "in_progress", 0.5 + 0.1 * aspects_to_refine.index(aspect_to_refine) / len(aspects_to_refine), 
                           f"Completed refinement of {aspect_to_refine}")
            
            # Add refinement history
            if "refinement_history" not in refined_data:
                refined_data["refinement_history"] = []
            
            refined_data["refinement_history"].append({
                "version": refined_data["version"],
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback,
                "aspects_refined": aspects_to_refine
            })
            
            # Generate summary of changes
            changes_summary = self._generate_changes_summary(game_data, refined_data)
            refined_data["latest_changes"] = changes_summary
            
            if callback:
                callback(task_id, "in_progress", 0.9, "Generated summary of changes")
            
            # Store the refined game
            invented_games_db[game_id] = refined_data
            
            # Complete the task
            if callback:
                callback(task_id, "completed", 1.0, "Game refinement completed successfully", refined_data)
            
            return refined_data
            
        except Exception as e:
            if callback:
                callback(task_id, "failed", None, f"Error: {str(e)}", None, str(e))
            raise e
    
    def _analyze_feedback_for_aspects(self, feedback):
        """Analyze feedback to determine which aspects need refinement"""
        aspects = []
        
        # Simple keyword matching
        if any(word in feedback.lower() for word in ["rule", "instruction", "unclear", "confusing", "setup"]):
            aspects.append("rules")
        
        if any(word in feedback.lower() for word in ["component", "piece", "card", "board", "token", "dice"]):
            aspects.append("components")
        
        if any(word in feedback.lower() for word in ["balance", "fair", "unfair", "advantage", "disadvantage", "overpowered"]):
            aspects.append("balance")
        
        if any(word in feedback.lower() for word in ["theme", "story", "setting", "immersion", "flavor"]):
            aspects.append("theme")
        
        if any(word in feedback.lower() for word in ["gameplay", "boring", "fun", "exciting", "engagement", "interaction"]):
            aspects.append("gameplay")
        
        # If no aspects were identified, default to rules and gameplay
        if not aspects:
            aspects = ["rules", "gameplay"]
        
        return aspects
    
    def _refine_rules(self, game_data, feedback, preserve):
        """Refine the rules of a game based on feedback"""
        current_rules = game_data.get("rules", {})
        
        prompt = f"""
        Refine the rules for the game "{game_data.get('name')}" based on this feedback:
        "{feedback}"
        
        Current rules:
        Setup: {current_rules.get('setup', 'No setup information')}
        Turn structure: {current_rules.get('turn_structure', 'No turn structure information')}
        Special rules: {current_rules.get('special_rules', 'No special rules')}
        Victory conditions: {current_rules.get('victory_conditions', 'No victory conditions')}
        Scoring: {current_rules.get('scoring', 'No scoring information')}
        
        Elements to preserve: {', '.join(preserve) if preserve else 'None specifically mentioned'}
        
        Improve clarity, balance, and player engagement while maintaining the core concept.
        Return the refined rules as JSON with the same structure as the current rules.
        """
        
        try:
            # Use reasoning to refine the rules
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    refined_rules = json.loads(json_str)
                    return refined_rules
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, make minimal changes to the current rules
            refined_rules = current_rules.copy()
            
            # Add a note about the refinement
            refined_rules["refinement_note"] = f"Rules refined based on feedback: {feedback[:100]}..."
            
            return refined_rules
            
        except Exception as e:
            print(f"Error refining game rules: {e}")
            # Return the current rules with a note
            return {**current_rules, "refinement_error": str(e)}
    
    def _refine_components(self, game_data, feedback, preserve):
        """Similar structure to _refine_rules but for components"""
        current_components = game_data.get("components", {})
        
        prompt = f"""
        Refine the components for the game "{game_data.get('name')}" based on this feedback:
        "{feedback}"
        
        Current components:
        {json.dumps(current_components, indent=2)}
        
        Elements to preserve: {', '.join(preserve) if preserve else 'None specifically mentioned'}
        
        Consider quality, usability, and player experience. Suggest specific improvements.
        Return the refined components as JSON with the same structure as the current components.
        """
        
        try:
            # Use reasoning to refine the components
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response
            import re
            import json
            
            # Try to find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    refined_components = json.loads(json_str)
                    return refined_components
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, make minimal changes
            refined_components = current_components.copy()
            
            # Add a note about the refinement
            refined_components["refinement_note"] = f"Components refined based on feedback: {feedback[:100]}..."
            
            return refined_components
            
        except Exception as e:
            print(f"Error refining game components: {e}")
            # Return the current components with a note
            return {**current_components, "refinement_error": str(e)}
    
    def _refine_balance(self, game_data, feedback, aspect, preserve):
        """Refine balance issues in either rules or components"""
        current_data = game_data.get(aspect, {})
        
        prompt = f"""
        Address balance issues in the {aspect} of game "{game_data.get('name')}" based on this feedback:
        "{feedback}"
        
        Current {aspect}:
        {json.dumps(current_data, indent=2)}
        
        Elements to preserve: {', '.join(preserve) if preserve else 'None specifically mentioned'}
        
        Focus on fairness, equal opportunities for players, and ensuring no strategy is overpowered.
        Return the refined {aspect} as JSON with the same structure as the current {aspect}.
        """
        
        try:
            # Use reasoning to refine balance
            response = self.reasoning_node.reason(prompt, "analytical")
            
            # Extract JSON from the response