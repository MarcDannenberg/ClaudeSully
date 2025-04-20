"""
Simple Pygame UI for the Games Kernel

This module provides a graphical user interface for various board games
using Pygame. It supports Chess, Go, and Mahjong games with customizable
settings and responsive design.
"""

import pygame
import sys
import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("games.visualizer")

# Try to import game modules, with graceful error handling
try:
    # Import the games kernel
    from games_kernel import GameEngine, Player
    from mahjong_game import MahjongGame, MahjongPlayer, Wind, TileType, Tile
    from chess_game import ChessGame, ChessPlayer, PieceColor, PieceType, ChessPiece
    from go_game import GoGame, GoPlayer, Stone
except ImportError as e:
    logger.error(f"Failed to import game modules: {e}")
    print(f"Error: Could not import required game modules. {e}")
    print("Please ensure all required game modules are installed.")
    sys.exit(1)

# Initialize pygame
pygame.init()

# Get screen info for responsive design
display_info = pygame.display.Info()
DEFAULT_SCREEN_WIDTH = min(1024, display_info.current_w - 100)
DEFAULT_SCREEN_HEIGHT = min(768, display_info.current_h - 100)

# Load configuration if available
@dataclass
class GameConfig:
    """Configuration settings for the game visualizer."""
    screen_width: int = DEFAULT_SCREEN_WIDTH
    screen_height: int = DEFAULT_SCREEN_HEIGHT
    board_size: int = 600
    margin: int = 50
    fps: int = 30
    fullscreen: bool = False
    music_enabled: bool = True
    sound_enabled: bool = True
    theme: str = "default"
    debug: bool = False

# Try to load config from file
CONFIG_PATH = Path("config.json")
config = GameConfig()

try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
except Exception as e:
    logger.warning(f"Failed to load configuration: {e}")
    logger.info("Using default configuration")

# Constants
SCREEN_WIDTH = config.screen_width
SCREEN_HEIGHT = config.screen_height
BOARD_SIZE = config.board_size
MARGIN = config.margin
FPS = config.fps

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
LIGHT_BROWN = (222, 184, 135)
DARK_BROWN = (160, 120, 80)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Asset paths
ASSET_DIR = Path("assets")
FONT_DIR = ASSET_DIR / "fonts"
IMAGE_DIR = ASSET_DIR / "images"
SOUND_DIR = ASSET_DIR / "sounds"

class Button:
    """A clickable button for the GUI."""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: Tuple[int, int, int] = GRAY, 
                 text_color: Tuple[int, int, int] = BLACK,
                 hover_color: Tuple[int, int, int] = LIGHT_GRAY,
                 font: Optional[pygame.font.Font] = None):
        """
        Initialize a button.
        
        Args:
            x: X position
            y: Y position
            width: Button width
            height: Button height
            text: Button text
            color: Button color
            text_color: Text color
            hover_color: Color when mouse hovers over button
            font: Font to use for text
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = hover_color
        self.font = font if font else pygame.font.SysFont("Arial", 16)
        self.is_hovered = False
        
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the button on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        # Draw the button
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw the text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def update(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Update button state based on mouse position.
        
        Args:
            mouse_pos: Current mouse position (x, y)
        """
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
    def is_clicked(self, mouse_pos: Tuple[int, int], mouse_clicked: bool) -> bool:
        """
        Check if the button is clicked.
        
        Args:
            mouse_pos: Current mouse position (x, y)
            mouse_clicked: Whether the mouse is clicked
            
        Returns:
            True if the button is clicked, False otherwise
        """
        return self.rect.collidepoint(mouse_pos) and mouse_clicked

class GameGUI:
    """Base class for game GUIs"""
    
    def __init__(self, game: Any):
        """
        Initialize the game GUI.
        
        Args:
            game: Game instance
        """
        self.game = game
        
        # Set up the display
        flags = pygame.RESIZABLE
        if config.fullscreen:
            flags |= pygame.FULLSCREEN
            
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        pygame.display.set_caption(f"Games Kernel - {game.name}")
        
        # Set up the clock
        self.clock = pygame.time.Clock()
        
        # Load fonts
        try:
            self.font = pygame.font.SysFont("Arial", 16)
            self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        except Exception as e:
            logger.error(f"Failed to load fonts: {e}")
            sys.exit(1)
            
        # Initialize state
        self.selected = None
        self.is_running = True
        self.fps_display = config.debug
        self.show_help = False
        
        # Create buttons
        self.create_buttons()
        
    def create_buttons(self) -> None:
        """Create common UI buttons."""
        # Exit button
        self.exit_button = Button(
            SCREEN_WIDTH - 90, 10, 80, 30, "Exit",
            color=GRAY, hover_color=RED
        )
        
        # Help button
        self.help_button = Button(
            SCREEN_WIDTH - 180, 10, 80, 30, "Help",
            color=GRAY, hover_color=BLUE
        )
        
        # FPS toggle button (debug mode only)
        if config.debug:
            self.fps_button = Button(
                SCREEN_WIDTH - 270, 10, 80, 30, "FPS: ON" if self.fps_display else "FPS: OFF",
                color=GRAY, hover_color=GREEN
            )
            
    def handle_window_resize(self, new_size: Tuple[int, int]) -> None:
        """
        Handle window resize event.
        
        Args:
            new_size: New window size (width, height)
        """
        global SCREEN_WIDTH, SCREEN_HEIGHT
        SCREEN_WIDTH, SCREEN_HEIGHT = new_size
        
        # Update button positions
        self.exit_button.rect.x = SCREEN_WIDTH - 90
        self.help_button.rect.x = SCREEN_WIDTH - 180
        
        if config.debug and hasattr(self, 'fps_button'):
            self.fps_button.rect.x = SCREEN_WIDTH - 270
            
        # Game-specific resize handling
        self.handle_game_resize(new_size)
        
    def handle_game_resize(self, new_size: Tuple[int, int]) -> None:
        """
        Handle game-specific resize operations.
        To be overridden by subclasses.
        
        Args:
            new_size: New window size (width, height)
        """
        pass
    
    def draw_text(self, text: str, font: pygame.font.Font, 
                 color: Tuple[int, int, int], x: int, y: int, 
                 align: str = "left") -> pygame.Rect:
        """
        Draw text on the screen.
        
        Args:
            text: Text to draw
            font: Font to use
            color: Text color
            x: X position
            y: Y position
            align: Text alignment (left, center, right)
            
        Returns:
            Text rectangle
        """
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        if align == "center":
            text_rect.center = (x, y)
        elif align == "right":
            text_rect.right, text_rect.top = x, y
        else:
            text_rect.left, text_rect.top = x, y
        
        self.screen.blit(text_surface, text_rect)
        return text_rect
    
    def draw_game_info(self) -> None:
        """Draw general game information."""
        # Draw game title
        self.draw_text(self.game.name, self.title_font, BLACK, SCREEN_WIDTH // 2, 20, align="center")
        
        # Draw current player
        player_text = f"Current Player: {self.game.current_player.name}"
        self.draw_text(player_text, self.font, BLACK, MARGIN, MARGIN // 2)
        
        # Draw game status
        if self.game.is_game_over:
            if self.game.winner:
                status_text = f"Game Over - Winner: {self.game.winner.name}"
            else:
                status_text = "Game Over - Draw"
            self.draw_text(status_text, self.font, RED, SCREEN_WIDTH - MARGIN, MARGIN // 2, align="right")
        
        # Draw FPS counter if enabled
        if self.fps_display:
            fps = int(self.clock.get_fps())
            fps_text = f"FPS: {fps}"
            self.draw_text(fps_text, self.font, GREEN if fps > 30 else YELLOW if fps > 15 else RED,
                          SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20, align="right")
    
    def handle_events(self) -> None:
        """Handle pygame events."""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            
            elif event.type == pygame.VIDEORESIZE:
                self.handle_window_resize(event.size)
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_game()
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_f and config.debug:
                    self.fps_display = not self.fps_display
                    if hasattr(self, 'fps_button'):
                        self.fps_button.text = "FPS: ON" if self.fps_display else "FPS: OFF"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
                
            # Let subclasses handle game-specific events
            self.handle_game_events(event)
        
        # Update buttons
        self.exit_button.update(mouse_pos)
        self.help_button.update(mouse_pos)
        
        if config.debug and hasattr(self, 'fps_button'):
            self.fps_button.update(mouse_pos)
        
        # Handle button clicks
        if mouse_clicked:
            if self.exit_button.is_clicked(mouse_pos, mouse_clicked):
                self.quit_game()
            
            if self.help_button.is_clicked(mouse_pos, mouse_clicked):
                self.show_help = not self.show_help
                
            if config.debug and hasattr(self, 'fps_button') and self.fps_button.is_clicked(mouse_pos, mouse_clicked):
                self.fps_display = not self.fps_display
                self.fps_button.text = "FPS: ON" if self.fps_display else "FPS: OFF"
    
    def handle_game_events(self, event: pygame.event.Event) -> None:
        """
        Handle game-specific events.
        To be overridden by subclasses.
        
        Args:
            event: Pygame event
        """
        pass
    
    def draw(self) -> None:
        """Draw the game."""
        self.screen.fill(WHITE)
        self.draw_game_info()
        self.draw_game_content()
        
        # Draw buttons
        self.exit_button.draw(self.screen)
        self.help_button.draw(self.screen)
        
        if config.debug and hasattr(self, 'fps_button'):
            self.fps_button.draw(self.screen)
        
        # Draw help overlay if enabled
        if self.show_help:
            self.draw_help()
        
        pygame.display.flip()
    
    def draw_game_content(self) -> None:
        """
        Draw game-specific content.
        To be overridden by subclasses.
        """
        pass
    
    def draw_help(self) -> None:
        """Draw help overlay."""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Black with 70% opacity
        self.screen.blit(overlay, (0, 0))
        
        # Draw help title
        self.draw_text("Help", self.title_font, WHITE, SCREEN_WIDTH // 2, 100, align="center")
        
        # Draw common controls
        help_text = [
            "ESC - Quit game",
            "H - Toggle help",
            "Click 'Exit' button to quit",
            "Click 'Help' button to toggle help"
        ]
        
        # Add game-specific help text
        game_help = self.get_game_help()
        if game_help:
            help_text.append("")
            help_text.append("Game-specific controls:")
            help_text.extend(game_help)
        
        # Draw help text
        for i, line in enumerate(help_text):
            self.draw_text(line, self.font, WHITE, SCREEN_WIDTH // 2, 150 + i * 30, align="center")
    
    def get_game_help(self) -> List[str]:
        """
        Get game-specific help text.
        To be overridden by subclasses.
        
        Returns:
            List of help text strings
        """
        return []
    
    def quit_game(self) -> None:
        """Clean up resources and quit the game."""
        self.is_running = False
    
    def run(self) -> None:
        """Main game loop."""
        try:
            while self.is_running:
                self.handle_events()
                self.draw()
                self.clock.tick(FPS)
        except Exception as e:
            logger.error(f"Error in game loop: {e}", exc_info=True)
            print(f"Error: {e}")
        finally:
            pygame.quit()
            sys.exit()

class MahjongGUI(GameGUI):
    """GUI for Mahjong game"""
    
    def __init__(self, game: MahjongGame):
        """
        Initialize the Mahjong GUI.
        
        Args:
            game: MahjongGame instance
        """
        super().__init__(game)
        
        # Tile dimensions
        self.tile_width = 40
        self.tile_height = 60
        
        # Selected tile index
        self.selected_tile_idx = None
        
        # Button rectangles
        self.button_rects = []
        
        # Load images for tiles
        self.tile_images = self._load_tile_images()
    
    def _load_tile_images(self) -> Dict[Tuple[TileType, Any], pygame.Surface]:
        """
        Load tile images for Mahjong.
        
        Returns:
            Dictionary mapping tile types to images
        """
        try:
            # Try to load actual images if available
            tile_image_path = IMAGE_DIR / "mahjong"
            if tile_image_path.exists():
                logger.info(f"Loading tile images from {tile_image_path}")
                # Code to load actual images would go here
                # For now, we'll use placeholder images
            
            # Create placeholder tiles for each type
            images = {}
            
            # Suited tiles
            for suit in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
                for value in range(1, 10):
                    key = (suit, value)
                    # Create a surface with the suit and value text
                    img = pygame.Surface((self.tile_width, self.tile_height))
                    img.fill(WHITE)
                    pygame.draw.rect(img, BLACK, (0, 0, self.tile_width, self.tile_height), 2)
                    
                    # Add text for suit and value
                    font = pygame.font.SysFont("Arial", 14)
                    value_text = font.render(str(value), True, BLACK)
                    suit_text = font.render(suit.value[0], True, BLACK)
                    
                    img.blit(value_text, (self.tile_width // 2 - value_text.get_width() // 2, 5))
                    img.blit(suit_text, (self.tile_width // 2 - suit_text.get_width() // 2, 25))
                    
                    images[key] = img
            
            # Honor tiles (winds and dragons)
            for wind in Wind:
                key = (TileType.WIND, wind)
                img = pygame.Surface((self.tile_width, self.tile_height))
                img.fill(WHITE)
                pygame.draw.rect(img, BLACK, (0, 0, self.tile_width, self.tile_height), 2)
                
                font = pygame.font.SysFont("Arial", 14)
                text = font.render(wind.value[0], True, BLUE)
                img.blit(text, (self.tile_width // 2 - text.get_width() // 2, self.tile_height // 2 - text.get_height() // 2))
                
                images[key] = img
            
            return images
        except Exception as e:
            logger.error(f"Failed to load tile images: {e}", exc_info=True)
            # Return an empty dictionary as fallback
            return {}
    
    def handle_game_resize(self, new_size: Tuple[int, int]) -> None:
        """
        Handle window resize for Mahjong.
        
        Args:
            new_size: New window size (width, height)
        """
        # Adjust tile size based on new window dimensions
        width, height = new_size
        self.tile_width = max(30, min(50, width // 25))
        self.tile_height = int(self.tile_width * 1.5)
    
    def draw_game_content(self) -> None:
        """Draw the Mahjong game content."""
        # Draw the wall (simplified)
        wall_text = f"Wall: {len(self.game.wall)} tiles"
        self.draw_text(wall_text, self.font, BLACK, MARGIN, MARGIN * 2)
        
        # Draw dora indicators
        dora_text = "Dora Indicators: " + ", ".join(str(tile) for tile in self.game.dora_indicators)
        self.draw_text(dora_text, self.font, BLACK, MARGIN, MARGIN * 3)
        
        # Draw current discard
        discard_text = f"Current Discard: {self.game.current_discard}"
        self.draw_text(discard_text, self.font, BLACK, MARGIN, MARGIN * 4)
        
        # Draw player hands
        self.draw_player_hands()
        
        # Draw action buttons
        self.draw_action_buttons()
    
    def draw_player_hands(self) -> None:
        """Draw the hands of all players."""
        for i, player in enumerate(self.game.players):
            # Position for each player's hand
            positions = [
                (SCREEN_WIDTH // 2, SCREEN_HEIGHT - MARGIN * 2),  # Bottom (current player)
                (MARGIN * 2, SCREEN_HEIGHT // 2),  # Left
                (SCREEN_WIDTH // 2, MARGIN * 5),  # Top
                (SCREEN_WIDTH - MARGIN * 2, SCREEN_HEIGHT // 2)   # Right
            ]
            
            x, y = positions[i]
            
            # Highlight current player
            color = RED if i == self.game.current_player_idx else BLACK
            self.draw_text(f"{player.name} ({player.wind.value})", self.font, color, x, y - 20, align="center")
            
            # If this is the current player, show the full hand
            if i == self.game.current_player_idx:
                self.draw_player_tiles(player, x, y)
            else:
                # For other players, just show the number of tiles
                tile_count = len(player.hand.tiles)
                self.draw_text(f"{tile_count} tiles", self.font, BLACK, x, y, align="center")
    
    def draw_player_tiles(self, player: MahjongPlayer, x: int, y: int) -> None:
        """
        Draw a player's tiles.
        
        Args:
            player: Player whose tiles to draw
            x: Center x position
            y: Y position
        """
        tiles = player.hand.tiles
        total_width = len(tiles) * self.tile_width
        start_x = x - total_width // 2
        
        for i, tile in enumerate(tiles):
            tile_x = start_x + i * self.tile_width
            
            # Draw a background rectangle to show selection
            if i == self.selected_tile_idx:
                pygame.draw.rect(self.screen, GREEN, (tile_x - 2, y - 2, self.tile_width + 4, self.tile_height + 4))
            
            # Draw the tile
            if (tile.type, tile.value) in self.tile_images:
                self.screen.blit(self.tile_images[(tile.type, tile.value)], (tile_x, y))
            else:
                # Fallback for missing images
                img = pygame.Surface((self.tile_width, self.tile_height))
                img.fill(WHITE)
                pygame.draw.rect(img, BLACK, (0, 0, self.tile_width, self.tile_height), 2)
                self.screen.blit(img, (tile_x, y))
                self.draw_text(str(tile), self.font, BLACK, tile_x + 5, y + 20)
    
    def draw_action_buttons(self) -> None:
        """Draw action buttons for the current player."""
        button_width = 100
        button_height = 30
        button_margin = 10
        start_x = MARGIN
        start_y = SCREEN_HEIGHT - MARGIN * 4
        
        buttons = ["Discard", "Chi", "Pon", "Kan", "Win"]
        self.button_rects = []
        
        for i, text in enumerate(buttons):
            button_x = start_x + i * (button_width + button_margin)
            button_rect = pygame.Rect(button_x, start_y, button_width, button_height)
            
            # Draw the button
            pygame.draw.rect(self.screen, GRAY, button_rect)
            pygame.draw.rect(self.screen, BLACK, button_rect, 2)
            self.draw_text(text, self.font, BLACK, button_rect.centerx, button_rect.centery, align="center")
            
            self.button_rects.append(button_rect)
    
    def handle_game_events(self, event: pygame.event.Event) -> None:
        """
        Handle Mahjong-specific events.
        
        Args:
            event: Pygame event
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if a tile was clicked
            player = self.game.current_player
            tiles = player.hand.tiles
            total_width = len(tiles) * self.tile_width
            x, y = SCREEN_WIDTH // 2, SCREEN_HEIGHT - MARGIN * 2
            start_x = x - total_width // 2
            
            for i in range(len(tiles)):
                tile_x = start_x + i * self.tile_width
                tile_rect = pygame.Rect(tile_x, y, self.tile_width, self.tile_height)
                
                if tile_rect.collidepoint(event.pos):
                    self.selected_tile_idx = i
                    break
            
            # Check if an action button was clicked
            for i, button_rect in enumerate(self.button_rects):
                if button_rect.collidepoint(event.pos):
                    self.handle_action_button(i)
    
    def handle_action_button(self, button_idx: int) -> None:
        """
        Handle action button clicks.
        
        Args:
            button_idx: Index of the clicked button
        """
        try:
            if button_idx == 0:  # Discard
                if self.selected_tile_idx is not None:
                    move = {"type": "discard", "tile_idx": self.selected_tile_idx}
                    if self.game.is_valid_move(move):
                        self.game.make_move(move)
                        self.selected_tile_idx = None
            
            elif button_idx == 4:  # Win
                move = {"type": "win", "from_discard": False}
                if self.game.is_valid_move(move):
                    self.game.make_move(move)
        except Exception as e:
            logger.error(f"Error handling action button: {e}", exc_info=True)
    
    def get_game_help(self) -> List[str]:
        """
        Get Mahjong-specific help text.
        
        Returns:
            List of help text strings
        """
        return [
            "Click on a tile to select it",
            "Click 'Discard' to discard the selected tile",
            "Click 'Win' to declare a win if your hand is complete"
        ]

class ChessGUI(GameGUI):
    """GUI for Chess game"""
    
    def __init__(self, game: ChessGame):
        """
        Initialize the Chess GUI.
        
        Args:
            game: ChessGame instance
        """
        super().__init__(game)
        
        # Chess square size
        self.square_size = BOARD_SIZE // 8
        
        # Board offset
        self.board_offset_x = (SCREEN_WIDTH - BOARD_SIZE) // 2
        self.board_offset_y = (SCREEN_HEIGHT - BOARD_SIZE) // 2
        
        # Selected square
        self.selected_square = None
        
        # Load chess piece images
        self.piece_images = self._load_piece_images()
    
    def _load_piece_images(self) -> Dict[Tuple[PieceType, PieceColor], pygame.Surface]:
        """
        Load chess piece images.
        
        Returns:
            Dictionary mapping piece types to images
        """
        try:
            # Try to load actual images if available
            piece_image_path = IMAGE_DIR / "chess"
            if piece_image_path.exists():
                logger.info(f"Loading piece images from {piece_image_path}")
                # Code to load actual images would go here
                # For now, we'll use placeholder images
            
            images = {}
            
            for color in [PieceColor.WHITE, PieceColor.BLACK]:
                for piece_type in PieceType:
                    if piece_type == PieceType.EMPTY:
                        continue
                    
                    # Create a placeholder image
                    img = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                    
                    # Draw the piece
                    if piece_type == PieceType.PAWN:
                        self._draw_pawn(img, color)
                    elif piece_type == PieceType.KNIGHT:
                        self._draw_knight(img, color)
                    elif piece_type == PieceType.BISHOP:
                        self._draw_bishop(img, color)
                    elif piece_type == PieceType.ROOK:
                        self._draw_rook(img, color)
                    elif piece_type == PieceType.QUEEN:
                        self._draw_queen(img, color)
                    elif piece_type == PieceType.KING:
                        self._draw_king(img, color)
                    
                    images[(piece_type, color)] = img
            
            return images
        except Exception as e:
            logger.error(f"Failed to load piece images: {e}", exc_info=True)
            # Return an empty dictionary as fallback
            return {}
    
    def _draw_pawn(self, surface: pygame.Surface, color: PieceColor) -> None:
        """
        Draw a pawn piece.
        
        Args:
            surface: Surface to draw on
            color: Piece color
        """
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 4)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 4, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("P", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_knight(self, surface: pygame.Surface, color: PieceColor) -> None:
        """
        Draw a knight piece.
        
        Args:
            surface: Surface to draw on
            color: Piece color
        """
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("N", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_bishop(self, surface: pygame.Surface, color: PieceColor) -> None:
        """
        Draw a bishop piece.
        
        Args:
            surface: Surface to draw on
            color: Piece color
        """
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("B", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_rook(self, surface: pygame.Surface, color: PieceColor) -> None:
        """
        Draw a rook piece.
        
        Args:
            surface: Surface to draw on
            color: Piece color
        """
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("R", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_queen(self, surface: pygame.Surface, color: PieceColor) -> None:
        """
        Draw a queen piece.
        
        Args:
            surface: Surface to draw on
            color: Piece color
        """
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("Q", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_king(self, surface: pygame.Surface, color: PieceColor) -> None:
        """
        Draw a king piece.
        
        Args:
            surface: Surface to draw on
            color: Piece color
        """
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("K", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def handle_game_resize(self, new_size: Tuple[int, int]) -> None:
        """
        Handle window resize for Chess.
        
        Args:
            new_size: New window size (width, height)
        """
        width, height = new_size
        self.square_size = min(width, height) // 10  # Allow for margins
        self.board_offset_x = (width - self.square_size * 8) // 2
        self.board_offset_y = (height - self.square_size * 8) // 2
        
        # Reload piece images with new size
        self.piece_images = self._load_piece_images()
    
    def draw_game_content(self) -> None:
        """Draw the Chess game content."""
        # Draw the board
        self.draw_board()
        
        # Draw game status
        if self.game.check_status:
            self.draw_text("CHECK!", self.font, RED, SCREEN_WIDTH - MARGIN, MARGIN * 2, align="right")
        
        # Draw captured pieces
        white_captured = [str(p) for p in self.game.players[0].captured_pieces]
        black_captured = [str(p) for p in self.game.players[1].captured_pieces]
        
        self.draw_text(f"White Captured: {' '.join(white_captured)}", self.font, BLACK, MARGIN, MARGIN * 2)
        self.draw_text(f"Black Captured: {' '.join(black_captured)}", self.font, BLACK, MARGIN, MARGIN * 3)
    
    def draw_board(self) -> None:
        """Draw the chess board."""
        # Draw the squares
        for row in range(8):
            for col in range(8):
                x = self.board_offset_x + col * self.square_size
                y = self.board_offset_y + (7 - row) * self.square_size  # Flip y-axis to match chess coordinates
                
                # Determine square color
                color = WHITE if (row + col) % 2 == 0 else GRAY
                
                # Draw the square
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                
                # Highlight selected square
                if self.selected_square == (col, row):
                    pygame.draw.rect(self.screen, GREEN, (x, y, self.square_size, self.square_size), 3)
                
                # Draw the piece
                piece = self.game.board[row][col]
                if piece.type != PieceType.EMPTY:
                    if (piece.type, piece.color) in self.piece_images:
                        self.screen.blit(self.piece_images[(piece.type, piece.color)], (x, y))
        
        # Draw board border
        pygame.draw.rect(self.screen, BLACK, (self.board_offset_x, self.board_offset_y, 
                                             self.square_size * 8, self.square_size * 8), 2)
        
        # Draw coordinates
        for i in range(8):
            # File labels (a-h)
            file_text = chr(97 + i)
            self.draw_text(file_text, self.font, BLACK, 
                          self.board_offset_x + i * self.square_size + self.square_size // 2, 
                          self.board_offset_y + 8 * self.square_size + 5, 
                          align="center")
            
            # Rank labels (1-8)
            rank_text = str(8 - i)
            self.draw_text(rank_text, self.font, BLACK, 
                          self.board_offset_x - 15, 
                          self.board_offset_y + i * self.square_size + self.square_size // 2, 
                          align="center")
    
    def handle_game_events(self, event: pygame.event.Event) -> None:
        """
        Handle Chess-specific events.
        
        Args:
            event: Pygame event
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the board was clicked
            x, y = event.pos
            
            if (self.board_offset_x <= x < self.board_offset_x + self.square_size * 8 and 
                self.board_offset_y <= y < self.board_offset_y + self.square_size * 8):
                
                # Calculate board coordinates
                col = (x - self.board_offset_x) // self.square_size
                row = 7 - (y - self.board_offset_y) // self.square_size  # Flip y-axis
                
                if 0 <= col < 8 and 0 <= row < 8:
                    # If no square is selected, select this one
                    if self.selected_square is None:
                        piece = self.game.board[row][col]
                        if piece.type != PieceType.EMPTY and piece.color == self.game.current_player.color:
                            self.selected_square = (col, row)
                    else:
                        # If a square is already selected, try to move
                        from_col, from_row = self.selected_square
                        
                        # Check if clicking the same square (deselect)
                        if (col, row) == self.selected_square:
                            self.selected_square = None
                        else:
                            # Try to move
                            move = {
                                "from_x": from_col,
                                "from_y": from_row,
                                "to_x": col,
                                "to_y": row
                            }
                            
                            # Check for special moves
                            piece = self.game.board[from_row][from_col]
                            
                            # Promotion
                            if (piece.type == PieceType.PAWN and 
                                ((piece.color == PieceColor.WHITE and row == 7) or 
                                 (piece.color == PieceColor.BLACK and row == 0))):
                                move["promotion"] = "Q"  # Default to queen
                            
                            # Castling
                            if (piece.type == PieceType.KING and abs(col - from_col) == 2):
                                move["castling"] = True
                            
                            # En passant
                            if (piece.type == PieceType.PAWN and 
                                abs(col - from_col) == 1 and 
                                self.game.board[row][col].type == PieceType.EMPTY):
                                move["en_passant"] = True
                            
                            try:
                                if self.game.is_valid_move(move):
                                    self.game.make_move(move)
                                    self.selected_square = None
                                else:
                                    # If invalid move, try selecting the new square instead
                                    piece = self.game.board[row][col]
                                    if piece.type != PieceType.EMPTY and piece.color == self.game.current_player.color:
                                        self.selected_square = (col, row)
                                    else:
                                        self.selected_square = None
                            except Exception as e:
                                logger.error(f"Error making move: {e}", exc_info=True)
                                self.selected_square = None
    
    def get_game_help(self) -> List[str]:
        """
        Get Chess-specific help text.
        
        Returns:
            List of help text strings
        """
        return [
            "Click on a piece to select it",
            "Click on a destination square to move",
            "Clicking on the same piece deselects it",
            "Pawns are automatically promoted to Queens",
            "Castling is detected automatically"
        ]

class GoGUI(GameGUI):
    """GUI for Go game"""
    
    def __init__(self, game: GoGame):
        """
        Initialize the Go GUI.
        
        Args:
            game: GoGame instance
        """
        super().__init__(game)
        
        # Go board size
        self.board_size = game.board_size
        self.cell_size = min(BOARD_SIZE // self.board_size, 30)
        
        # Board dimensions
        self.board_width = self.cell_size * self.board_size
        self.board_height = self.cell_size * self.board_size
        
        # Board offset
        self.board_offset_x = (SCREEN_WIDTH - self.board_width) // 2
        self.board_offset_y = (SCREEN_HEIGHT - self.board_height) // 2
        
        # Create pass button
        self.pass_button = Button(
            SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50, 100, 30, "Pass",
            color=GRAY, hover_color=LIGHT_GRAY
        )
    
    def handle_game_resize(self, new_size: Tuple[int, int]) -> None:
        """
        Handle window resize for Go.
        
        Args:
            new_size: New window size (width, height)
        """
        width, height = new_size
        board_dim = min(width, height) // 1.5
        self.cell_size = int(board_dim // self.board_size)
        self.board_width = self.cell_size * self.board_size
        self.board_height = self.cell_size * self.board_size
        self.board_offset_x = (width - self.board_width) // 2
        self.board_offset_y = (height - self.board_height) // 2
        
        # Update pass button position
        self.pass_button.rect.x = width - 150
        self.pass_button.rect.y = height - 50
    
    def draw_game_content(self) -> None:
        """Draw the Go game content."""
        # Draw the board
        self.draw_board()
        
        # Draw scores
        black_score = self.game.players[0].score
        white_score = self.game.players[1].score
        
        if black_score > 0 or white_score > 0:
            self.draw_text(f"Black Score: {black_score}", self.font, BLACK, MARGIN, MARGIN * 2)
            self.draw_text(f"White Score: {white_score}", self.font, BLACK, MARGIN, MARGIN * 3)
        
        # Draw captures
        black_captures = self.game.players[0].captures
        white_captures = self.game.players[1].captures
        
        self.draw_text(f"Black Captures: {black_captures}", self.font, BLACK, MARGIN, MARGIN * 4)
        self.draw_text(f"White Captures: {white_captures}", self.font, BLACK, MARGIN, MARGIN * 5)
        
        # Draw "Pass" button
        self.pass_button.update(pygame.mouse.get_pos())
        self.pass_button.draw(self.screen)
    
    def draw_board(self) -> None:
        """Draw the Go board."""
        # Draw the board background
        pygame.draw.rect(self.screen, LIGHT_BROWN, 
                        (self.board_offset_x - self.cell_size // 2, 
                         self.board_offset_y - self.cell_size // 2,
                         self.board_width + self.cell_size,
                         self.board_height + self.cell_size))
        
        # Draw the grid lines
        for i in range(self.board_size):
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                           (self.board_offset_x, self.board_offset_y + i * self.cell_size),
                           (self.board_offset_x + self.board_width - self.cell_size, 
                            self.board_offset_y + i * self.cell_size))
            
            # Vertical lines
            pygame.draw.line(self.screen, BLACK,
                           (self.board_offset_x + i * self.cell_size, self.board_offset_y),
                           (self.board_offset_x + i * self.cell_size, 
                            self.board_offset_y + self.board_height - self.cell_size))
        
        # Draw star points (hoshi)
        star_points = self._get_star_points()
        for point in star_points:
            x, y = point
            center_x = self.board_offset_x + x * self.cell_size
            center_y = self.board_offset_y + y * self.cell_size
            pygame.draw.circle(self.screen, BLACK, (center_x, center_y), self.cell_size // 5)
        
        # Draw the stones
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.game.board[y][x] != Stone.EMPTY:
                    center_x = self.board_offset_x + x * self.cell_size
                    center_y = self.board_offset_y + y * self.cell_size
                    color = BLACK if self.game.board[y][x] == Stone.BLACK else WHITE
                    pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 2 - 1)
                    pygame.draw.circle(self.screen, BLACK, (center_x, center_y), self.cell_size // 2 - 1, 1)
        
        # Draw board coordinates
        for i in range(self.board_size):
            # Column labels (A, B, C, ...)
            column_text = chr(65 + i) if i < 8 else chr(66 + i)  # Skip 'I'
            self.draw_text(column_text, self.font, BLACK, 
                          self.board_offset_x + i * self.cell_size, 
                          self.board_offset_y - 20, 
                          align="center")
            
            # Row labels (1, 2, 3, ...)
            row_text = str(i + 1)
            self.draw_text(row_text, self.font, BLACK, 
                          self.board_offset_x - 20, 
                          self.board_offset_y + i * self.cell_size, 
                          align="center")
    
    def _get_star_points(self) -> List[Tuple[int, int]]:
        """
        Get the positions of star points based on board size.
        
        Returns:
            List of (x, y) coordinates for star points
        """
        star_points = []
        
        if self.board_size == 19:
            # Traditional 19x19 board has 9 star points
            points = [3, 9, 15]
            for y in points:
                for x in points:
                    star_points.append((x, y))
        elif self.board_size == 13:
            # 13x13 board has 5 star points
            points = [3, 6, 9]
            middle = 6
            star_points.append((middle, middle))
            for point in points:
                if point != middle:
                    star_points.append((point, middle))
                    star_points.append((middle, point))
        elif self.board_size == 9:
            # 9x9 board has 5 star points
            points = [2, 4, 6]
            middle = 4
            star_points.append((middle, middle))
            for point in points:
                if point != middle:
                    star_points.append((point, middle))
                    star_points.append((middle, point))
        
        return star_points
    
    def handle_game_events(self, event: pygame.event.Event) -> None:
        """
        Handle Go-specific events.
        
        Args:
            event: Pygame event
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Check if the pass button was clicked
            if self.pass_button.is_clicked(mouse_pos, True):
                try:
                    move = {"type": "pass"}
                    self.game.make_move(move)
                except Exception as e:
                    logger.error(f"Error making pass move: {e}", exc_info=True)
                return
            
            # Check if the board was clicked
            x, y = mouse_pos
            
            if (self.board_offset_x - self.cell_size // 2 <= x < self.board_offset_x + self.board_width + self.cell_size // 2 and 
                self.board_offset_y - self.cell_size // 2 <= y < self.board_offset_y + self.board_height + self.cell_size // 2):
                
                # Calculate the closest intersection
                board_x = round((x - self.board_offset_x) / self.cell_size)
                board_y = round((y - self.board_offset_y) / self.cell_size)
                
                if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                    # Try to place a stone
                    try:
                        move = {"type": "place", "x": board_x, "y": board_y}
                        if self.game.is_valid_move(move):
                            self.game.make_move(move)
                    except Exception as e:
                        logger.error(f"Error placing stone: {e}", exc_info=True)
    
    def get_game_help(self) -> List[str]:
        """
        Get Go-specific help text.
        
        Returns:
            List of help text strings
        """
        return [
            "Click on an intersection to place a stone",
            "Click 'Pass' button to pass your turn",
            "Stones are automatically captured when surrounded",
            "Game ends after two consecutive passes",
            f"Current komi (points given to White): {self.game.komi}"
        ]

def main() -> None:
    """Main entry point for the pygame UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Games Kernel Pygame UI")
    parser.add_argument("game", choices=["mahjong", "chess", "go"], help="Game to play")
    parser.add_argument("--board-size", "-b", type=int, default=19, help="Board size for Go (default: 19)")
    parser.add_argument("--fullscreen", "-f", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config.fullscreen = args.fullscreen
    config.debug = args.debug
    
    # Create game engine
    try:
        engine = GameEngine()
        engine.register_game(MahjongGame)
        engine.register_game(ChessGame)
        engine.register_game(GoGame)
    except Exception as e:
        logger.error(f"Failed to initialize game engine: {e}", exc_info=True)
        print(f"Error: Failed to initialize game engine. {e}")
        sys.exit(1)
    
    # Create game instance
    try:
        if args.game == "mahjong":
            winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
            players = [MahjongPlayer(f"Player {i+1}", wind) for i, wind in enumerate(winds)]
            game = MahjongGame(players)
            gui = MahjongGUI(game)
        elif args.game == "chess":
            players = [
                ChessPlayer("White Player", PieceColor.WHITE),
                ChessPlayer("Black Player", PieceColor.BLACK)
            ]
            game = ChessGame(players)
            gui = ChessGUI(game)
        elif args.game == "go":
            players = [
                GoPlayer("Black Player", Stone.BLACK),
                GoPlayer("White Player", Stone.WHITE)
            ]
            game = GoGame(players, board_size=args.board_size)
            gui = GoGUI(game)
        
        game.initialize_game()
        gui.run()
    except Exception as e:
        logger.error(f"Failed to create game: {e}", exc_info=True)
        print(f"Error: Failed to create {args.game} game. {e}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()