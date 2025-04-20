def _create_tile_set(self) -> List[Tile]:
        """Create a complete set of Mahjong tiles based on game settings"""
        tiles = []
        
        # Suited tiles (1-9 in three suits, four of each)
        for suit in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            for value in range(1, 10):
                for _ in range(4):
                    # If using red fives, make one of each suit's fives red
                    if self.use_red_fives and value == 5 and _ == 0:
                        # In a real implementation, we'd add a "red" attribute to the Tile class
                        # For simplicity, we're just adding regular fives
                        pass
                    tiles.append(Tile(suit, value))
        
        # Winds (4 of each)
        for wind in Wind:
            for _ in range(4):
                tiles.append(Tile(TileType.WIND, wind))
        
        # Dragons (4 of each)
        for dragon in Dragon:
            for _ in range(4):
                tiles.append(Tile(TileType.DRAGON, dragon))
        
        # Flowers and Seasons (1 of each, numbered 1-4)
        if self.include_flowers:
            for i in range(1, 5):
                tiles.append(Tile(TileType.FLOWER, i))
                
        if self.include_seasons:
            for i in range(1, 5):
                tiles.append(Tile(TileType.SEASON, i))
        
        return tiles
    
    def draw_tile(self) -> Optional[Tile]:
        """Draw a tile from the wall"""
        if not self.wall:
            return None  # Wall is empty (exhaustive draw)
        return self.wall.pop(0)
    
    def is_valid_move(self, move: Dict[str, Any]) -> bool:
        """Check if a move is valid"""
        move_type = move.get("type")
        player_idx = move.get("player_idx", self.current_player_idx)
        player = self.players[player_idx]
        
        if move_type == "discard":
            # Check if it's this player's turn and they have the tile
            tile_idx = move.get("tile_idx")
            return player_idx == self.current_player_idx and 0 <= tile_idx < len(player.hand.tiles)
        
        elif move_type == "chi":
            # Check if the player can call chi on the current discard
            # Must be the next player's turn
            return (self.current_discard and 
                   player_idx == (self.current_player_idx + 1) % 4 and
                   player.hand.can_chi(self.current_discard))
        
        elif move_type == "pon":
            # Any player can pon if they have two matching tiles
            return (self.current_discard and 
                   player.hand.can_pon(self.current_discard))
        
        elif move_type == "kan":
            # Open kan from discard or closed kan from hand
            is_closed = move.get("is_closed", False)
            if is_closed:
                # Closed kan must be on player's turn
                tile = player.hand.tiles[move.get("tile_idx")]
                return player_idx == self.current_player_idx and player.hand.can_kan(tile, True)
            else:
                # Open kan from discard
                return self.current_discard and player.hand.can_kan(self.current_discard)
        
        elif move_type == "win":
            # Check if the player can win with the current discard or self-draw
            tile = self.current_discard if move.get("from_discard") else None
            return player.hand.can_win(tile)
        
        return False
    
    def get_valid_moves(self) -> List[Dict[str, Any]]:
        """Get all valid moves for the current player"""
        valid_moves = []
        player = self.current_player
        
        # Check for win (self-draw)
        if player.hand.can_win():
            valid_moves.append({"type": "win", "from_discard": False})
        
        # Check for closed kan
        for i, tile in enumerate(player.hand.tiles):
            tile_count = sum(1 for t in player.hand.tiles if t == tile)
            if tile_count == 4:
                valid_moves.append({"type": "kan", "is_closed": True, "tile_idx": i})
        
        # Check for discard
        for i in range(len(player.hand.tiles)):
            valid_moves.append({"type": "discard", "tile_idx": i})
        
        return valid_moves
    
    def get_valid_calls(self, player_idx: int) -> List[Dict[str, Any]]:
        """Get valid calls for a player after another player discards"""
        if not self.current_discard:
            return []
            
        valid_calls = []
        player = self.players[player_idx]
        
        # Check for win (from discard)
        if player.hand.can_win(self.current_discard):
            valid_calls.append({"type": "win", "from_discard": True})
        
        # Check for kan
        if player.hand.can_kan(self.current_discard, False):
            valid_calls.append({"type": "kan", "is_closed": False})
        
        # Check for pon
        if player.hand.can_pon(self.current_discard):
            valid_calls.append({"type": "pon"})
        
        # Check for chi (only the next player can chi)
        if player_idx == (self.current_player_idx + 1) % 4 and player.hand.can_chi(self.current_discard):
            valid_calls.append({"type": "chi"})
        
        return valid_calls
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a player's move"""
        if not self.is_valid_move(move):
            return False
        
        move_type = move.get("type")
        player_idx = move.get("player_idx", self.current_player_idx)
        player = self.players[player_idx]
        
        # Record the move
        self.record_move(move, player_idx)
        
        if move_type == "discard":
            # Discard a tile from hand
            tile_idx = move.get("tile_idx")
            discarded_tile = player.hand.discard_tile(tile_idx)
            if discarded_tile:
                self.current_discard = discarded_tile
                self.discards.append(discarded_tile)
                self.last_action = "discard"
                
                # Update statistics
                self.stats["discards_by_player"][player.id] += 1
                
                # Move to next player
                self.next_player()
                
                # For AI opponents, we'll handle their moves elsewhere
                if not self._handle_calls_after_discard():
                    # If no calls, next player draws a tile
                    next_player = self.players[self.current_player_idx]
                    drawn_tile = self.draw_tile()
                    if drawn_tile:
                        next_player.hand.add_tile(drawn_tile)
                        
                        # Auto-win for AI if possible and enabled
                        if self.auto_win and isinstance(next_player, MahjongAIPlayer):
                            if next_player.hand.can_win():
                                win_move = {"type": "win", "from_discard": False}
                                return self.make_move(win_move)
                    else:
                        # Wall is exhausted - draw game
                        self.is_game_over = True
                return True
        
        elif move_type == "chi":
            # Claim a sequence with the current discard
            sequence_tiles = move.get("sequence_tiles", [])  # Indices of tiles in hand
            if len(sequence_tiles) == 2:
                tiles = [player.hand.tiles[i] for i in sequence_tiles]
                # Form the sequence
                sequence = [self.current_discard] + tiles
                # Remove the tiles from hand
                for i in sorted(sequence_tiles, reverse=True):
                    player.hand.tiles.pop(i)
                # Add the revealed set
                player.hand.revealed_sets.append(("chi", sequence))
                self.current_discard = None
                # Set this player as current
                self.current_player_idx = player_idx
                self.last_action = "chi"
                self.stats["calls"]["chi"] += 1
                return True
        
        elif move_type == "pon":
            # Claim a triplet with the current discard
            # Find two matching tiles in hand
            matching_tiles = [i for i, t in enumerate(player.hand.tiles) if t == self.current_discard]
            if len(matching_tiles) >= 2:
                # Remove the tiles from hand
                tiles = [player.hand.tiles[matching_tiles[0]], player.hand.tiles[matching_tiles[1]]]
                player.hand.tiles.pop(matching_tiles[1])
                player.hand.tiles.pop(matching_tiles[0])
                # Add the revealed set
                triplet = [self.current_discard] + tiles
                player.hand.revealed_sets.append(("pon", triplet))
                self.current_discard = None
                # Set this player as current
                self.current_player_idx = player_idx
                self.last_action = "pon"
                self.stats["calls"]["pon"] += 1
                return True
        
        elif move_type == "kan":
            is_closed = move.get("is_closed", False)
            if is_closed:
                # Closed kan from hand
                tile_idx = move.get("tile_idx")
                if tile_idx < len(player.hand.tiles):
                    tile = player.hand.tiles[tile_idx]
                    # Find all four of this tile in hand
                    matching_indices = [i for i, t in enumerate(player.hand.tiles) if t == tile]
                    if len(matching_indices) == 4:
                        # Remove the tiles from hand
                        kan_tiles = []
                        for i in sorted(matching_indices, reverse=True):
                            kan_tiles.append(player.hand.tiles.pop(i))
                        # Add the revealed set
                        player.hand.revealed_sets.append(("closed_kan", kan_tiles))
                        # Draw replacement tile from dead wall
                        if self.dead_wall:
                            player.hand.add_tile(self.dead_wall.pop(0))
                        self.last_action = "closed_kan"
                        # Add new dora indicator
                        if len(self.dora_indicators) < 5 and self.dead_wall:
                            self.dora_indicators.append(self.dead_wall[4 + len(self.dora_indicators)])
                        self.stats["calls"]["kan"] += 1
                        return True
            else:
                # Open kan from discard
                matching_indices = [i for i, t in enumerate(player.hand.tiles) if t == self.current_discard]
                if len(matching_indices) >= 3:
                    # Remove the tiles from hand
                    kan_tiles = []
                    for i in sorted(matching_indices[:3], reverse=True):
                        kan_tiles.append(player.hand.tiles.pop(i))
                    # Add the current discard
                    kan_tiles.append(self.current_discard)
                    # Add the revealed set
                    player.hand.revealed_sets.append(("open_kan", kan_tiles))
                    self.current_discard = None
                    # Set this player as current
                    self.current_player_idx = player_idx
                    # Draw replacement tile from dead wall
                    if self.dead_wall:
                        player.hand.add_tile(self.dead_wall.pop(0))
                    self.last_action = "open_kan"
                    # Add new dora indicator
                    if len(self.dora_indicators) < 5 and self.dead_wall:
                        self.dora_indicators.append(self.dead_wall[4 + len(self.dora_indicators)])
                    self.stats["calls"]["kan"] += 1
                    return True
        
        elif move_type == "win":
            # Player wins the game
            from_discard = move.get("from_discard", False)
            win_tile = self.current_discard if from_discard else None
            if player.hand.can_win(win_tile):
                self.winner = player
                self.is_game_over = True
                self.last_action = "win"
                self.stats["calls"]["win"] += 1
                
                # Record the winning hand
                winning_tiles = player.hand.tiles.copy()
                if win_tile:
                    winning_tiles.append(win_tile)
                    
                self.stats["winning_hands"].append({
                    "player": player.name,
                    "tiles": [str(tile) for tile in winning_tiles],
                    "revealed_sets": player.hand.revealed_sets,
                    "win_tile": str(win_tile) if win_tile else "self-draw"
                })
                
                # Calculate and update scores (simplified scoring)
                self._calculate_scores(player_idx, from_discard)
                
                return True
        
        return False
    
    def _handle_calls_after_discard(self) -> bool:
        """Handle player calls after a discard, returns True if any calls were made"""
        # Priority order: Win > Kan > Pon > Chi
        
        # Check for other players calling Win (Ron)
        for i in range(1, 4):
            check_idx = (self.current_player_idx + i) % 4
            player = self.players[check_idx]
            
            if player.hand.can_win(self.current_discard):
                # AI players will automatically claim win
                if isinstance(player, MahjongAIPlayer) and self.auto_win:
                    win_move = {"type": "win", "from_discard": True, "player_idx": check_idx}
                    self.make_move(win_move)
                    return True
                else:
                    # For human players, we'll wait for their input
                    # In a real implementation, we'd set a state indicating win is available
                    pass
        
        # Check for other players calling Kan
        for i in range(1, 4):
            check_idx = (self.current_player_idx + i) % 4
            player = self.players[check_idx]
            
            if player.hand.can_kan(self.current_discard, False):
                # AI players will decide whether to claim
                if isinstance(player, MahjongAIPlayer):
                    decision = player.decide_call(
                        self.current_discard, 
                        False,  # can_chi
                        False,  # can_pon
                        True,   # can_kan
                        False   # can_win
                    )
                    
                    if decision == "kan":
                        kan_move = {"type": "kan", "is_closed": False, "player_idx": check_idx}
                        self.make_move(kan_move)
                        return True
                else:
                    # For human players, we'll wait for their input
                    pass
        
        # Check for other players calling Pon
        for i in range(1, 4):
            check_idx = (self.current_player_idx + i) % 4
            player = self.players[check_idx]
            
            if player.hand.can_pon(self.current_discard):
                # AI players will decide whether to claim
                if isinstance(player, MahjongAIPlayer):
                    decision = player.decide_call(
                        self.current_discard, 
                        False,  # can_chi
                        True,   # can_pon
                        False,  # can_kan
                        False   # can_win
                    )
                    
                    if decision == "pon":
                        pon_move = {"type": "pon", "player_idx": check_idx}
                        self.make_move(pon_move)
                        return True
                else:
                    # For human players, we'll wait for their input
                    pass
        
        # Check for next player calling Chi
        next_idx = (self.current_player_idx + 1) % 4
        next_player = self.players[next_idx]
        
        if next_player.hand.can_chi(self.current_discard):
            # AI player will decide whether to claim
            if isinstance(next_player, MahjongAIPlayer):
                decision = next_player.decide_call(
                    self.current_discard, 
                    True,   # can_chi
                    False,  # can_pon
                    False,  # can_kan
                    False   # can_win
                )
                
                if decision == "chi":
                    # Find sequence tiles for Chi
                    sequence_options = []
                    
                    # Check for possible sequences
                    val = self.current_discard.value
                    tile_type = self.current_discard.type
                    
                    # Check if we can form val-1, val, val+1
                    if (Tile(tile_type, val-1) in next_player.hand.tiles and 
                        Tile(tile_type, val+1) in next_player.hand.tiles):
                        indices = []
                        for i, t in enumerate(next_player.hand.tiles):
                            if t == Tile(tile_type, val-1) or t == Tile(tile_type, val+1):
                                indices.append(i)
                                if len(indices) == 2:
                                    break
                        sequence_options.append(indices)
                    
                    # Check if we can form val, val+1, val+2
                    if (Tile(tile_type, val+1) in next_player.hand.tiles and 
                        Tile(tile_type, val+2) in next_player.hand.tiles):
                        indices = []
                        for i, t in enumerate(next_player.hand.tiles):
                            if t == Tile(tile_type, val+1) or t == Tile(tile_type, val+2):
                                indices.append(i)
                                if len(indices) == 2:
                                    break
                        sequence_options.append(indices)
                    
                    # Check if we can form val-2, val-1, val
                    if (Tile(tile_type, val-2) in next_player.hand.tiles and 
                        Tile(tile_type, val-1) in next_player.hand.tiles):
                        indices = []
                        for i, t in enumerate(next_player.hand.tiles):
                            if t == Tile(tile_type, val-2) or t == Tile(tile_type, val-1):
                                indices.append(i)
                                if len(indices) == 2:
                                    break
                        sequence_options.append(indices)
                    
                    # Choose one sequence option
                    if sequence_options:
                        chi_move = {
                            "type": "chi", 
                            "player_idx": next_idx,
                            "sequence_tiles": sequence_options[0]
                        }
                        self.make_move(chi_move)
                        return True
            else:
                # For human players, we'll wait for their input
                pass
        
        # No calls were made
        return False
    
    def _calculate_scores(self, winner_idx: int, from_discard: bool):
        """Calculate and update scores (simplified scoring)"""
        # In a real implementation, this would involve complex scoring rules
        # based on the hand composition, dora, etc.
        
        # Simple scoring: 
        # - 30 points for winning
        # - +10 for self-draw
        # - Dealer gets/pays double
        score = 30
        if not from_discard:
            score += 10
            
        winner = self.players[winner_idx]
        dealer_idx = 0  # East player is dealer
        
        # Update scores
        if winner_idx == dealer_idx:
            # Dealer wins
            dealer_bonus = 2
            for i, player in enumerate(self.players):
                if i != winner_idx:
                    self.scores[player.id] -= score * dealer_bonus
                    self.scores[winner.id] += score * dealer_bonus
        elif from_discard:
            # Someone else discarded the winning tile
            discarder_idx = self.current_player_idx
            discarder = self.players[discarder_idx]
            
            payment = score
            if discarder_idx == dealer_idx:
                payment *= 2  # Dealer pays double
                
            self.scores[discarder.id] -= payment
            self.scores[winner.id] += payment
        else:
            # Self-draw win (pay all)
            for i, player in enumerate(self.players):
                if i != winner_idx:
                    payment = score
                    if i == dealer_idx:
                        payment *= 2  # Dealer pays double
                        
                    self.scores[player.id] -= payment
                    self.scores[winner.id] += payment
    
    def check_game_over(self) -> bool:
        """Check if the game has ended"""
        # Game can end by someone winning or exhaustive draw
        return self.is_game_over
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        return {
            "current_player": self.current_player_idx,
            "round_wind": self.round_wind.value,
            "dora_indicators": [str(tile) for tile in self.dora_indicators],
            "wall_size": len(self.wall),
            "dead_wall_size": len(self.dead_wall),
            "discards": [str(tile) for tile in self.discards],
            "current_discard": str(self.current_discard) if self.current_discard else None,
            "players": [{
                "name": player.name,
                "id": player.id,
                "wind": player.wind.value,
                "hand": [str(tile) for tile in player.hand.tiles],
                "revealed_sets": player.hand.revealed_sets,
                "discards": [str(tile) for tile in player.hand.discards],
                "score": self.scores.get(player.id, 0)
            } for player in self.players],
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None,
            "last_action": self.last_action,
            "scores": self.scores,
            "stats": self.stats
        }
    
    def get_player_hand_view(self, player_idx: int) -> Dict[str, Any]:
        """Get a view of the game state from a player's perspective"""
        player = self.players[player_idx]
        
        # Create a limited view of other players' hands
        players_view = []
        for i, p in enumerate(self.players):
            if i == player_idx:
                # Full info for this player
                players_view.append({
                    "name": p.name,
                    "id": p.id,
                    "wind": p.wind.value,
                    "hand": [str(tile) for tile in p.hand.tiles],
                    "hand_size": len(p.hand.tiles),
                    "revealed_sets": p.hand.revealed_sets,
                    "discards": [str(tile) for tile in p.hand.discards],
                    "score": self.scores.get(p.id, 0),
                    "is_current": i == self.current_player_idx
                })
            else:
                # Limited info for other players
                players_view.append({
                    "name": p.name,
                    "id": p.id,
                    "wind": p.wind.value,
                    "hand_size": len(p.hand.tiles),
                    "revealed_sets": p.hand.revealed_sets,
                    "discards": [str(tile) for tile in p.hand.discards],
                    "score": self.scores.get(p.id, 0),
                    "is_current": i == self.current_player_idx
                })
        
        # Determine valid moves
        valid_moves = []
        valid_calls = []
        
        if self.current_player_idx == player_idx:
            valid_moves = self.get_valid_moves()
        
        if self.current_discard and self.current_player_idx != player_idx:
            valid_calls = self.get_valid_calls(player_idx)
        
        return {
            "player": player.name,
            "player_idx": player_idx,
            "current_player": self.current_player_idx,
            "round_wind": self.round_wind.value,
            "dora_indicators": [str(tile) for tile in self.dora_indicators],
            "wall_size": len(self.wall),
            "dead_wall_size": len(self.dead_wall),
            "discards": [str(tile) for tile in self.discards],
            "current_discard": str(self.current_discard) if self.current_discard else None,
            "players": players_view,
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None,
            "last_action": self.last_action,
            "valid_moves": valid_moves,
            "valid_calls": valid_calls
        }
    
    def get_ai_move(self, player_idx: int) -> Dict[str, Any]:
        """Get a move for an AI player"""
        player = self.players[player_idx]
        
        if not isinstance(player, MahjongAIPlayer):
            return {"error": "Not an AI player"}
        
        # Check if it's for a call response
        if self.current_discard and self.current_player_idx != player_idx:
            valid_calls = self.get_valid_calls(player_idx)
            
            if valid_calls:
                # Determine if we can win
                can_win = any(call["type"] == "win" for call in valid_calls)
                can_kan = any(call["type"] == "kan" for call in valid_calls)
                can_pon = any(call["type"] == "pon" for call in valid_calls)
                can_chi = any(call["type"] == "chi" for call in valid_calls)
                
                decision = player.decide_call(
                    self.current_discard,
                    can_chi,
                    can_pon,
                    can_kan,
                    can_win
                )
                
                if decision == "win" and can_win:
                    return {"type": "win", "from_discard": True, "player_idx": player_idx}
                elif decision == "kan" and can_kan:
                    return {"type": "kan", "is_closed": False, "player_idx": player_idx}
                elif decision == "pon" and can_pon:
                    return {"type": "pon", "player_idx": player_idx}
                elif decision == "chi" and can_chi:
                    # Find sequence tiles for Chi (simplified)
                    sequence_options = []
                    
                    # Check for possible sequences
                    val = self.current_discard.value
                    tile_type = self.current_discard.type
                    
                    # Check if we can form val-1, val, val+1
                    if (Tile(tile_type, val-1) in player.hand.tiles and 
                        Tile(tile_type, val+1) in player.hand.tiles):
                        indices = []
                        for i, t in enumerate(player.hand.tiles):
                            if t == Tile(tile_type, val-1) or t == Tile(tile_type, val+1):
                                indices.append(i)
                                if len(indices) == 2:
                                    break
                        sequence_options.append(indices)
                    
                    # Additional sequence checks would go here...
                    
                    if sequence_options:
                        return {
                            "type": "chi", 
                            "player_idx": player_idx,
                            "sequence_tiles": sequence_options[0]
                        }
            
            # No call or "pass" decision
            return {"type": "pass", "player_idx": player_idx}
        
        # Regular turn move
        if self.current_player_idx == player_idx:
            # Check for win (self-draw)
            if player.hand.can_win() and self.auto_win:
                return {"type": "win", "from_discard": False, "player_idx": player_idx}
            
            # Check for closed kan
            for i, tile in enumerate(player.hand.tiles):
                tile_count = sum(1 for t in player.hand.tiles if t == tile)
                if tile_count == 4:
                    # AI would evaluate whether to declare kan
                    # For simplicity, always declare kan
                    return {"type": "kan", "is_closed": True, "tile_idx": i, "player_idx": player_idx}
            
            # Choose a tile to discard
            tile_idx = player.choose_tile_to_discard(player.hand)
            if tile_idx >= 0:
                return {"type": "discard", "tile_idx": tile_idx, "player_idx": player_idx}
        
        return {"error": "No valid AI move found"}
    
    def display(self):
        """Display the current game state"""
        print(f"=== Mahjong Game ===")
        print(f"Round Wind: {self.round_wind.value}")
        print(f"Dora Indicator(s): {', '.join(str(tile) for tile in self.dora_indicators)}")
        print(f"Wall: {len(self.wall)} tiles remaining")
        print(f"Current Discard: {self.current_discard}")
        print(f"Scores: {self.scores}")
        print("\nPlayers:")
        for i, player in enumerate(self.players):
            current = "→ " if i == self.current_player_idx else "  "
            print(f"{current}{player}")
        
        if self.is_game_over:
            if self.winner:
                print(f"\nGame Over! Winner: {self.winner.name}")
                print(f"Final Scores: {self.scores}")
            else:
                print("\nGame Over! Exhaustive Draw")


"""
Go Game Implementation
"""

class Stone(Enum):
    """Enum representing stone colors in Go"""
    BLACK = "●"
    WHITE = "○"
    EMPTY = "·"

class GoPlayer(Player):
    """Player in a Go game"""
    
    def __init__(self, name: str, stone: Stone, id: str = None):
        super().__init__(name, id)
        self.stone = stone
        self.captures = 0
    
    def __str__(self):
        return f"{self.name} ({self.stone.value}, Captures: {self.captures})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = super().to_dict()
        data["stone"] = self.stone.value
        data["captures"] = self.captures
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoPlayer':
        """Create from dictionary"""
        stone_value = data.get("stone", "●")
        stone = Stone.BLACK if stone_value == "●" else Stone.WHITE
        
        player = cls(data["name"], stone, data.get("id"))
        player.captures = data.get("captures", 0)
        
        # Set base player attributes
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player

class GoAIPlayer(GoPlayer, AIPlayer):
    """AI player for Go games"""
    
    def __init__(self, name: str, stone: Stone, difficulty: GameDifficulty = GameDifficulty.MEDIUM,
                 adaptive: bool = True, learning_rate: float = 0.05, id: str = None):
        GoPlayer.__init__(self, name, stone, id)
        AIPlayer.__init__(self, name, difficulty, adaptive, learning_rate)
        self.territory_value = 1.0  # Value of territory vs. captures
        self.influence_value = 0.8  # Value of influence
        self.position_evaluations = {}  # Cache for position evaluations
        
    def evaluate_position(self, board, position_value_cache=None):
        """Evaluate a board position for the current player"""
        # This would be a complex function in a real implementation
        # using machine learning techniques for stronger AI
        
        # For the scope of this example, we'll implement a simplified heuristic
        if position_value_cache is None:
            position_value_cache = {}
            
        # Create a board hash for caching
        board_hash = str(board)
        if board_hash in self.position_evaluations:
            return self.position_evaluations[board_hash]
            
        # Basic evaluation: count stones, liberties, and potential territory
        my_stones = 0
        opponent_stones = 0
        my_liberties = 0
        opponent_liberties = 0
        
        board_size = len(board)
        opponent_stone = Stone.WHITE if self.stone == Stone.BLACK else Stone.BLACK
        
        # Count stones and estimate liberties
        for y in range(board_size):
            for x in range(board_size):
                if board[y][x] == self.stone:
                    my_stones += 1
                    # Count liberties for this stone
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < board_size and 0 <= ny < board_size and board[ny][nx] == Stone.EMPTY:
                            my_liberties += 1
                elif board[y][x] == opponent_stone:
                    opponent_stones += 1
                    # Count liberties for opponent stone
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < board_size and 0 <= ny < board_size and board[ny][nx] == Stone.EMPTY:
                            opponent_liberties += 1
        
        # Estimate territory control (simplified)
        my_territory = 0
        opponent_territory = 0
        
        for y in range(board_size):
            for x in range(board_size):
                if board[y][x] == Stone.EMPTY:
                    # Count surrounding stones to estimate territory
                    my_influence = 0
                    opponent_influence = 0
                    
                    for dist in range(1, 4):  # Look at 3 steps distance
                        influence_value = 1.0 / dist  # Closer stones have more influence
                        
                        for dx, dy in [(0, dist), (dist, 0), (0, -dist), (-dist, 0), 
                                        (dist, dist), (-dist, dist), (dist, -dist), (-dist, -dist)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < board_size and 0 <= ny < board_size:
                                if board[ny][nx] == self.stone:
                                    my_influence += influence_value
                                elif board[ny][nx] == opponent_stone:
                                    opponent_influence += influence_value
                    
                    if my_influence > opponent_influence * 1.5:
                        my_territory += 1
                    elif opponent_influence > my_influence * 1.5:
                        opponent_territory += 1
        
        # Calculate positional advantage based on the difficulty level
        if self.difficulty == GameDifficulty.BEGINNER:
            # Beginner just counts stones and captures
            value = (my_stones - opponent_stones) + (self.captures * 2)
        elif self.difficulty == GameDifficulty.EASY:
            # Easy adds basic liberty counting
            value = (my_stones - opponent_stones) + (self.captures * 1.5) + (my_liberties - opponent_liberties) * 0.3
        elif self.difficulty == GameDifficulty.MEDIUM:
            # Medium adds territory estimation
            value = (my_stones - opponent_stones) + (self.captures * 1.2) + \
                    (my_liberties - opponent_liberties) * 0.4 + \
                    (my_territory - opponent_territory) * self.territory_value
        else:
            # Hard and Expert add positional influence and more sophisticated evaluation
            center_control = 0
            center_x, center_y = board_size // 2, board_size // 2
            
            # Value control of the center and key points
            for y in range(board_size):
                for x in range(board_size):
                    if board[y][x] == self.stone:
                        # Distance from center affects value
                        dist_from_center = abs(x - center_x) + abs(y - center_y)
                        center_control += max(0, (board_size - dist_from_center)) * 0.1
            
            value = (my_stones - opponent_stones) + (self.captures) + \
                    (my_liberties - opponent_liberties) * 0.5 + \
                    (my_territory - opponent_territory) * self.territory_value + \
                    center_control * self.influence_value
        
        # Adjust based on komi if playing white
        if self.stone == Stone.WHITE:
            value += 6.5  # Standard komi
        
        # Cache the evaluation
        self.position_evaluations[board_hash] = value
        return value
    
    def choose_move(self, board, valid_moves):
        """Choose a move based on AI difficulty"""
        board_size = len(board)
        
        # For beginners, mostly random play
        if self.difficulty == GameDifficulty.BEGINNER:
            # 80% random, 20% capture if possible
            if random.random() < 0.8 or not valid_moves:
                # Random move
                if not valid_moves:
                    return {"type": "pass"}
                return random.choice(valid_moves)
            else:
                # Try to find a capturing move
                for move in valid_moves:
                    if move["type"] == "place":
                        x, y = move["x"], move["y"]
                        # Check if this move captures stones
                        test_board = [row[:] for row in board]
                        test_board[y][x] = self.stone
                        captures = self._find_potential_captures(test_board, x, y)
                        if captures:
                            return move
                
                # If no capturing move, just random
                return random.choice(valid_moves)
        
        # For easy and better difficulty, evaluate positions
        # Create copy of board for simulation
        best_move = None
        best_value = float("-inf")
        
        # Add randomness based on difficulty
        randomness = {
            GameDifficulty.EASY: 0.4,
            GameDifficulty.MEDIUM: 0.2,
            GameDifficulty.HARD: 0.1,
            GameDifficulty.EXPERT: 0.05,
            GameDifficulty.MASTER: 0.02
        }.get(self.difficulty, 0)
        
        # Check for pass move
        pass_move = {"type": "pass"}
        has_pass = False
        for move in valid_moves:
            if move["type"] == "pass":
                has_pass = True
                break
        
        # Analyze each possible move
        for move in valid_moves:
            # Skip pass move for normal evaluation
            if move["type"] == "pass":
                continue
                
            # Place stone and evaluate resulting position
            x, y = move["x"], move["y"]
            test_board = [row[:] for row in board]
            test_board[y][x] = self.stone
            
            # Check for captures
            captures = self._find_potential_captures(test_board, x, y)
            for cx, cy in captures:
                test_board[cy][cx] = Stone.EMPTY
            
            # Evaluate the resulting position
            value = self.evaluate_position(test_board)
            
            # Add some randomness based on difficulty
            value += random.uniform(0, randomness * 10)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        # If passing is better than the best move, or no valid moves, pass
        if not best_move or (has_pass and random.random() < 0.05):  # Small chance to pass even with valid moves
            return pass_move
            
        return best_move
    
    def _find_potential_captures(self, board, x, y):
        """Find opponent stones that would be captured by placing at (x,y)"""
        board_size = len(board)
        opponent_stone = Stone.WHITE if self.stone == Stone.BLACK else Stone.BLACK
        captures = []
        
        # Check adjacent intersections for opponent stones
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if the new coordinates are valid
            if not (0 <= nx < board_size and 0 <= ny < board_size):
                continue
            
            # If there's an opponent stone, check if it/its group has liberties
            if board[ny][nx] == opponent_stone:
                group = []
                checked = set()
                self._collect_group(board, nx, ny, opponent_stone, group, checked)
                
                has_liberty = False
                for gx, gy in group:
                    # Check adjacent intersections for liberties
                    for gdx, gdy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        gnx, gny = gx + gdx, gy + gdy
                        
                        # Check if the new coordinates are valid
                        if not (0 <= gnx < board_size and 0 <= gny < board_size):
                            continue
                        
                        # If there's an empty intersection, the group has a liberty
                        if board[gny][gnx] == Stone.EMPTY:
                            has_liberty = True
                            break
                    
                    if has_liberty:
                        break
                
                # If the group has no liberties, it would be captured
                if not has_liberty:
                    captures.extend(group)
        
        return captures
    
    def _collect_group(self, board, x, y, stone, group, checked):
        """Collect all stones in a connected group"""
        if (x, y) in checked:
            return
        
        checked.add((x, y))
        board_size = len(board)
        
        if board[y][x] == stone:
            group.append((x, y))
            
            # Check adjacent intersections
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check if the new coordinates are valid
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    self._collect_group(board, nx, ny, stone, group, checked)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = GoPlayer.to_dict(self)
        
        # Add AI-specific data
        data["ai_settings"] = {
            "difficulty": self.difficulty.name,
            "adaptive": self.adaptive,
            "learning_rate": self.learning_rate,
            "territory_value": self.territory_value,
            "influence_value": self.influence_value
        }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoAIPlayer':
        """Create from dictionary"""
        stone_value = data.get("stone", "●")
        stone = Stone.BLACK if stone_value == "●" else Stone.WHITE
        
        # Extract AI settings
        ai_settings = data.get("ai_settings", {})
        difficulty_name = ai_settings.get("difficulty", "MEDIUM")
        try:
            difficulty = GameDifficulty[difficulty_name]
        except KeyError:
            difficulty = GameDifficulty.MEDIUM
            
        adaptive = ai_settings.get("adaptive", True)
        learning_rate = ai_settings.get("learning_rate", 0.05)
        
        # Create instance
        player = cls(
            data["name"], 
            stone, 
            difficulty=difficulty,
            adaptive=adaptive,
            learning_rate=learning_rate,
            id=data.get("id")
        )
        
        # Set additional AI parameters
        player.territory_value = ai_settings.get("territory_value", 1.0)
        player.influence_value = ai_settings.get("influence_value", 0.8)
        player.captures = data.get("captures", 0)
        
        # Set base player attributes
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player

class GoGame(Game):
    """Enhanced implementation of the Go board game"""
    
    def __init__(self, players: List[Player], board_size: int = 19, 
                komi: float = 6.5, ruleset: str = "chinese", game_id: str = None):
        """
        Initialize a Go game
        
        Args:
            players: List of players (must be exactly 2)
            board_size: Size of the board (typically 9, 13, or 19)
            komi: Compensation points for white (typically 6.5)
            ruleset: Ruleset to use (chinese, japanese, etc.)
            game_id: Optional game ID
        """
        if len(players) != 2:
            raise ValueError("Go requires exactly 2 players")
        
        # Convert regular Players to GoPlayers if needed
        stones = [Stone.BLACK, Stone.WHITE]
        for i, player in enumerate(players):
            if not isinstance(player, GoPlayer):
                players[i] = GoPlayer(player.name, stones[i], player.id if hasattr(player, 'id') else None)
        
        super().__init__("Go", players, game_id)
        self.board_size = board_size
        self.komi = komi
        self.ruleset = ruleset
        self.board = None
        self.previous_board = None  # For ko rule checking
        self.consecutive_passes = 0
        self.territories = {"black": 0, "white": 0}
        
        # Game stats
        self.stats = {
            "captures": {self.players[0].id: 0, self.players[1].id: 0},
            "moves_played": 0,
            "passes": 0,
            "move_distribution": [[0 for _ in range(board_size)] for _ in range(board_size)],
            "territory_history": []
        }
    
    def initialize_game(self):
        """Set up the game state"""
        logger.info(f"Initializing Go game {self.game_id} with {self.board_size}x{self.board_size} board")
        self.is_game_over = False
        self.winner = None
        # Create empty board
        self.board = [[Stone.EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.previous_board = copy.deepcopy(self.board)
        self.consecutive_passes = 0
        self.territories = {"black": 0, "white": 0}
        
        # Reset player captures
        for player in self.players:
            player.captures = 0
            
        # Reset stats
        self.stats = {
            "captures": {self.players[0].id: 0, self.players[1].id: 0},
            "moves_played": 0,
            "passes": 0,
            "move_distribution": [[0 for _ in range(self.board_size)] for _ in range(self.board_size)],
            "territory_history": []
        }
        
        # Black plays first in Go
        self.current_player_idx = 0
        logger.info(f"Go game {self.game_id} initialized")
    
    def is_valid_move(self, move: Dict[str, Any]) -> bool:
        """Check if a move is valid"""
        move_type = move.get("type")
        
        if move_type == "pass":
            # Passing is always valid
            return True
        
        elif move_type == "place":
            x, y = move.get("x"), move.get("y")
            
            # Check if coordinates are valid
            if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                return False
            
            # Check if the intersection is empty
            if self.board[y][x] != Stone.EMPTY:
                return False
            
            # Create a hypothetical board for testing
            test_board = copy.deepcopy(self.board)
            test_board[y][x] = self.current_player.stone
            
            # Check if the move would capture any stones
            captured = self._find_captures(test_board, x, y)
            
            # If no captures, check if the move would result in a suicide
            if not captured:
                # Check if the placed stone has liberties
                if not self._has_liberties(test_board, x, y):
                    return False
            
            # Check for ko rule - simple ko check
            # If the move would exactly recreate the previous board position, it's invalid
            if self._would_violate_ko(test_board, captured, x, y):
                return False
            
            return True
        
        return False
    
    def get_valid_moves(self) -> List[Dict[str, Any]]:
        """Return a list of all valid moves for the current player"""
        valid_moves = [{"type": "pass"}]  # Passing is always valid
        
        # Check each position on the board
        for y in range(self.board_size):
            for x in range(self.board_size):
                move = {"type": "place", "x": x, "y": y}
                if self.is_valid_move(move):
                    valid_moves.append(move)
        
        return valid_moves
    
    def _would_violate_ko(self, test_board: List[List[Stone]], captured: List[Tuple[int, int]], x: int, y: int) -> bool:
        """Check if a move would violate the ko rule"""
        # If there's exactly one capture and that capture would recreate the previous board position
        if len(captured) == 1:
            # Apply captures to test board
            for cx, cy in captured:
                test_board[cy][cx] = Stone.EMPTY
            
            # Check if the resulting board is identical to the previous one
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if test_board[i][j] != self.previous_board[i][j]:
                        return False
            
            return True
        
        return False
    
    def _has_liberties(self, board: List[List[Stone]], x: int, y: int) -> bool:
        """Check if a stone or group has liberties (empty adjacent intersections)"""
        stone = board[y][x]
        if stone == Stone.EMPTY:
            return True
        
        checked = set()
        return self._check_liberties(board, x, y, stone, checked)
    
    def _check_liberties(self, board: List[List[Stone]], x: int, y: int, stone: Stone, checked: Set[Tuple[int, int]]) -> bool:
        """Recursive function to check if a stone or group has liberties"""
        if (x, y) in checked:
            return False
        
        checked.add((x, y))
        
        # Check adjacent intersections
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if the new coordinates are valid
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                continue
            
            # If there's an empty intersection, the group has a liberty
            if board[ny][nx] == Stone.EMPTY:
                return True
            
            # If there's a stone of the same color, check if it has liberties
            if board[ny][nx] == stone and self._check_liberties(board, nx, ny, stone, checked):
                return True
        
        return False
    
    def _find_captures(self, board: List[List[Stone]], x: int, y: int) -> List[Tuple[int, int]]:
        """Find any opponent stones that would be captured by placing a stone at (x, y)"""
        stone = board[y][x]
        opponent_stone = Stone.WHITE if stone == Stone.BLACK else Stone.BLACK
        captures = []
        
        # Check adjacent intersections for opponent stones
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if the new coordinates are valid
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                continue
            
            # If there's an opponent stone, check if it/its group has liberties
            if board[ny][nx] == opponent_stone:
                group = self._find_group(board, nx, ny)
                has_liberty = False
                
                for gx, gy in group:
                    # Check adjacent intersections for liberties
                    for gdx, gdy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        gnx, gny = gx + gdx, gy + gdy
                        
                        # Check if the new coordinates are valid
                        if not (0 <= gnx < self.board_size and 0 <= gny < self.board_size):
                            continue
                        
                        # If there's an empty intersection, the group has a liberty
                        if board[gny][gnx] == Stone.EMPTY:
                            has_liberty = True
                            break
                    
                    if has_liberty:
                        break
                
                # If the group has no liberties, it would be captured
                if not has_liberty:
                    captures.extend(group)
        
        return captures
    
    def _find_group(self, board: List[List[Stone]], x: int, y: int) -> List[Tuple[int, int]]:
        """Find all stones in a connected group"""
        stone = board[y][x]
        if stone == Stone.EMPTY:
            return []
        
        group = []
        checked = set()
        self._collect_group(board, x, y, stone, group, checked)
        return group
    
    def _collect_group(self, board: List[List[Stone]], x: int, y: int, stone: Stone, 
                      group: List[Tuple[int, int]], checked: Set[Tuple[int, int]]):
        """Recursive function to collect all stones in a connected group"""
        if (x, y) in checked:
            return
        
        checked.add((x, y))
        
        if board[y][x] == stone:
            group.append((x, y))
            
            # Check adjacent intersections
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check if the new coordinates are valid
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    self._collect_group(board, nx, ny, stone, group, checked)
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a player's move"""
        if not self.is_valid_move(move):
            return False
        
        # Record the move
        self.record_move(move, self.current_player_idx)
        
        move_type = move.get("type")
        
        if move_type == "pass":
            # Record the pass
            self.consecutive_passes += 1
            self.stats["passes"] += 1
            
            # Check if both players passed consecutively
            if self.consecutive_passes >= 2:
                self.is_game_over = True
                # Score the game
                self._score_game()
            else:
                # Move to the next player
                self.next_player()
            
            return True
        
        elif move_type == "place":
            x, y = move.get("x"), move.get("y")
            
            # Save the current board for ko rule checking
            self.previous_board = copy.deepcopy(self.board)
            
            # Place the stone
            self.board[y][x] = self.current_player.stone
            
            # Update stats
            self.stats["moves_played"] += 1
            self.stats["move_distribution"][y][x] += 1
            
            # Find and remove any captured stones
            captures = self._find_captures(self.board, x, y)
            for cx, cy in captures:
                self.board[cy][cx] = Stone.EMPTY
            
            # Update the player's capture count
            self.current_player.captures += len(captures)
            self.stats["captures"][self.current_player.id] += len(captures)
            
            # Reset consecutive passes
            self.consecutive_passes = 0
            
            # Calculate current territories (for stats)
            if self.stats["moves_played"] % 5 == 0:  # Every 5 moves
                territories = self._calculate_territories()
                self.stats["territory_history"].append({
                    "move": self.stats["moves_played"],
                    "black": territories["black"],
                    "white": territories["white"]
                })
            
            # Move to the next player
            self.next_player()
            
            return True
        
        return False
    
    def get_ai_move(self, player_idx: int) -> Dict[str, Any]:
        """Get a move for an AI player"""
        if player_idx != self.current_player_idx:
            return {"error": "Not this player's turn"}
            
        player = self.players[player_idx]
        
        if not isinstance(player, GoAIPlayer):
            return {"error": "Not an AI player"}
            
        # Get valid moves
        valid_moves = self.get_valid_moves()
        
        # Let the AI choose a move
        return player.choose_move(self.board, valid_moves)
    
    def _score_game(self):
        """Score the game at the end"""
        # Based on ruleset
        if self.ruleset == "chinese":
            self._score_chinese()
        else:  # Default to Japanese or other area scoring
            self._score_japanese()
    
    def _score_chinese(self):
        """Chinese scoring: territory + stones on the board"""
        black_score = 0
        white_score = 0
        
        # Count stones on the board
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == Stone.BLACK:
                    black_score += 1
                elif self.board[y][x] == Stone.WHITE:
                    white_score += 1
        
        # Calculate territory
        territories = self._calculate_territories()
        black_score += territories["black"]
        white_score += territories["white"]
        
        # Add komi
        white_score += self.komi
        
        # Store territory counts
        self.territories = territories
        
        # Set player scores
        self.players[0].score = black_score
        self.players[1].score = white_score
        
        # Determine the winner
        if black_score > white_score:
            self.winner = self.players[0]  # Black
        elif white_score > black_score:
            self.winner = self.players[1]  # White
        else:
            # In case of a tie (very rare with fractional komi)
            self.winner = None
    
    """
Games Kernel - An advanced framework for implementing traditional and AI-enhanced board and tile games

This module provides a robust, extensible architecture for various game types, supporting:
- Traditional games like Chess, Go, and Mahjong
- Advanced AI opponent capabilities with adaptive difficulty
- Deep strategy analysis and pattern recognition
- Integration with cognitive systems for enhanced gameplay
- Game state persistence and analysis
- Metrics tracking for player improvement
"""

from abc import ABC, abstractmethod
import random
import hashlib
import pickle
import logging
import os
import math
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import copy
from collections import Counter, defaultdict, deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("games.kernel")

# Optional dependencies (with graceful fallbacks)
try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False
    logger.warning("NumPy not available; using fallback implementations for numerical operations")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    ml_available = True
except ImportError:
    ml_available = False
    logger.warning("scikit-learn not available; advanced pattern recognition disabled")

# Constants
DEFAULT_SAVE_DIR = os.path.join(os.path.expanduser("~"), ".sully", "games")
os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

class GameDifficulty(Enum):
    """Enum for game difficulty levels"""
    BEGINNER = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4
    MASTER = 5
    ADAPTIVE = 6

class Player:
    """Enhanced base class for all game players"""
    def __init__(self, name: str, id: str = None):
        self.name = name
        self.id = id or self._generate_id(name)
        self.score = 0
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.created_at = datetime.now().isoformat()
        self.last_active = datetime.now().isoformat()
        self.metadata = {}
        
    def _generate_id(self, name: str) -> str:
        """Generate a unique ID based on name and timestamp"""
        hash_base = f"{name}_{time.time()}"
        return hashlib.md5(hash_base.encode()).hexdigest()[:12]
    
    def update_stats(self, won: bool = False, lost: bool = False, draw: bool = False):
        """Update player statistics after a game"""
        self.total_games += 1
        self.last_active = datetime.now().isoformat()
        
        if won:
            self.wins += 1
        elif lost:
            self.losses += 1
        elif draw:
            self.draws += 1
    
    def get_win_rate(self) -> float:
        """Get player's win rate"""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata about the player"""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert player to dictionary for serialization"""
        return {
            "name": self.name,
            "id": self.id,
            "score": self.score,
            "stats": {
                "total_games": self.total_games,
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "win_rate": self.get_win_rate()
            },
            "created_at": self.created_at,
            "last_active": self.last_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        """Create player from dictionary"""
        player = cls(data["name"], data.get("id"))
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        return player
    
    def __str__(self):
        return f"Player: {self.name} (ID: {self.id}, Score: {self.score}, W/L/D: {self.wins}/{self.losses}/{self.draws})"


class AIPlayer(Player):
    """AI player with configurable difficulty and learning capability"""
    
    def __init__(self, name: str, difficulty: GameDifficulty = GameDifficulty.MEDIUM, 
                 adaptive: bool = True, learning_rate: float = 0.05):
        super().__init__(name)
        self.difficulty = difficulty
        self.adaptive = adaptive
        self.learning_rate = learning_rate
        self.strategy_patterns = {}
        self.opponent_models = {}
        self.performance_history = []
        
    def record_game(self, game_type: str, opponent_id: str, moves: List[Any], outcome: str):
        """Record game for learning"""
        self.performance_history.append({
            "game_type": game_type,
            "opponent_id": opponent_id,
            "moves_count": len(moves),
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update opponent model
        if opponent_id not in self.opponent_models:
            self.opponent_models[opponent_id] = {"games": [], "patterns": {}}
        
        self.opponent_models[opponent_id]["games"].append({
            "moves": moves,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })
        
        # If we have enough games, update strategy
        if len(self.opponent_models[opponent_id]["games"]) >= 3:
            self._update_opponent_patterns(opponent_id, game_type)
            
        # Adapt difficulty if needed
        if self.adaptive:
            self._adapt_difficulty(opponent_id)
    
    def _update_opponent_patterns(self, opponent_id: str, game_type: str):
        """Analyze and update opponent patterns"""
        games = self.opponent_models[opponent_id]["games"]
        
        # Simple pattern detection (would be more sophisticated in real implementation)
        opening_moves = []
        for game in games[-5:]:  # Look at last 5 games
            if len(game["moves"]) >= 3:
                opening_moves.append(str(game["moves"][:3]))
        
        # Count occurrences
        if opening_moves:
            counter = Counter(opening_moves)
            most_common = counter.most_common(3)
            
            # Store patterns
            self.opponent_models[opponent_id]["patterns"]["openings"] = [
                {"pattern": pattern, "frequency": count / len(opening_moves)}
                for pattern, count in most_common
            ]
    
    def _adapt_difficulty(self, opponent_id: str):
        """Adapt difficulty based on performance against opponent"""
        games = self.opponent_models[opponent_id]["games"]
        if len(games) < 3:
            return
            
        # Check recent outcomes
        recent_games = games[-3:]
        wins = sum(1 for game in recent_games if game["outcome"] == "win")
        losses = sum(1 for game in recent_games if game["outcome"] == "loss")
        
        current_level = self.difficulty.value
        
        # Adjust difficulty
        if wins == 3 and current_level < GameDifficulty.MASTER.value:
            # Increase difficulty if winning consistently
            self.difficulty = GameDifficulty(min(current_level + 1, GameDifficulty.MASTER.value))
            logger.info(f"AI {self.name} increased difficulty to {self.difficulty.name} against {opponent_id}")
        elif losses == 3 and current_level > GameDifficulty.BEGINNER.value:
            # Decrease difficulty if losing consistently
            self.difficulty = GameDifficulty(max(current_level - 1, GameDifficulty.BEGINNER.value))
            logger.info(f"AI {self.name} decreased difficulty to {self.difficulty.name} against {opponent_id}")
    
    def get_predicted_move(self, game_type: str, opponent_id: str, game_state: Any) -> Optional[Any]:
        """Predict opponent's move based on patterns"""
        if opponent_id not in self.opponent_models or "patterns" not in self.opponent_models[opponent_id]:
            return None
            
        patterns = self.opponent_models[opponent_id]["patterns"]
        # This would use the patterns to predict moves in a real implementation
        return None  # Placeholder
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert AI player to dictionary"""
        data = super().to_dict()
        data.update({
            "ai_settings": {
                "difficulty": self.difficulty.name,
                "adaptive": self.adaptive,
                "learning_rate": self.learning_rate
            },
            "performance_history_length": len(self.performance_history),
            "known_opponents": list(self.opponent_models.keys())
        })
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIPlayer':
        """Create AI player from dictionary"""
        ai_settings = data.get("ai_settings", {})
        difficulty_name = ai_settings.get("difficulty", "MEDIUM")
        
        try:
            difficulty = GameDifficulty[difficulty_name]
        except KeyError:
            difficulty = GameDifficulty.MEDIUM
            
        player = cls(
            data["name"],
            difficulty=difficulty,
            adaptive=ai_settings.get("adaptive", True),
            learning_rate=ai_settings.get("learning_rate", 0.05)
        )
        
        # Set base player attributes
        player.id = data.get("id", player.id)
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player


class GameAnalytics:
    """Analytics engine for game analysis"""
    
    def __init__(self):
        self.game_records = defaultdict(list)
        self.player_stats = {}
        self.pattern_library = {}
        
    def record_game(self, game_type: str, game_data: Dict[str, Any]):
        """Record a completed game"""
        self.game_records[game_type].append({
            "timestamp": datetime.now().isoformat(),
            "data": game_data
        })
        
        # Update player stats
        for player_id, outcome in game_data.get("player_outcomes", {}).items():
            if player_id not in self.player_stats:
                self.player_stats[player_id] = {
                    "games": defaultdict(int),
                    "wins": defaultdict(int),
                    "losses": defaultdict(int),
                    "draws": defaultdict(int),
                    "avg_moves": defaultdict(list)
                }
            
            stats = self.player_stats[player_id]
            stats["games"][game_type] += 1
            
            if outcome == "win":
                stats["wins"][game_type] += 1
            elif outcome == "loss":
                stats["losses"][game_type] += 1
            elif outcome == "draw":
                stats["draws"][game_type] += 1
                
            stats["avg_moves"][game_type].append(game_data.get("move_count", 0))
            
        # Analyze for patterns every 10 games
        if len(self.game_records[game_type]) % 10 == 0:
            self._analyze_patterns(game_type)
    
    def _analyze_patterns(self, game_type: str):
        """Analyze game records for patterns"""
        recent_games = self.game_records[game_type][-20:]  # Last 20 games
        
        if not recent_games:
            return
            
        # Extract features for analysis
        moves_per_game = [g["data"].get("move_count", 0) for g in recent_games]
        game_durations = [g["data"].get("duration_seconds", 0) for g in recent_games]
        
        # Store basic statistics
        if game_type not in self.pattern_library:
            self.pattern_library[game_type] = {}
            
        self.pattern_library[game_type]["avg_moves"] = sum(moves_per_game) / len(moves_per_game)
        self.pattern_library[game_type]["avg_duration"] = sum(game_durations) / len(game_durations)
        
        # If ML is available, do more sophisticated analysis
        if ml_available and numpy_available and len(recent_games) >= 5:
            try:
                # Create feature matrix
                features = np.array([
                    [g["data"].get("move_count", 0), 
                     g["data"].get("duration_seconds", 0),
                     len(g["data"].get("moves", [])),
                     g["data"].get("outcome_delay", 0)]
                    for g in recent_games if "data" in g
                ])
                
                # Normalize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                # Cluster games
                kmeans = KMeans(n_clusters=min(3, len(scaled_features)))
                clusters = kmeans.fit_predict(scaled_features)
                
                # Store cluster centers
                self.pattern_library[game_type]["clusters"] = {
                    "centers": kmeans.cluster_centers_.tolist(),
                    "feature_names": ["move_count", "duration", "moves_length", "outcome_delay"],
                    "counts": np.bincount(clusters).tolist()
                }
            except Exception as e:
                logger.warning(f"ML analysis error for {game_type}: {e}")
    
    def get_player_stats(self, player_id: str) -> Dict[str, Any]:
        """Get statistics for a player"""
        if player_id not in self.player_stats:
            return {"message": "No data for player"}
            
        stats = self.player_stats[player_id]
        result = {"games_by_type": {}}
        
        for game_type in stats["games"]:
            games_played = stats["games"][game_type]
            wins = stats["wins"][game_type]
            losses = stats["losses"][game_type]
            draws = stats["draws"][game_type]
            
            avg_moves = 0
            if stats["avg_moves"][game_type]:
                avg_moves = sum(stats["avg_moves"][game_type]) / len(stats["avg_moves"][game_type])
                
            result["games_by_type"][game_type] = {
                "played": games_played,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / games_played if games_played > 0 else 0,
                "avg_moves": avg_moves
            }
            
        # Calculate overall stats
        total_played = sum(stats["games"].values())
        total_wins = sum(stats["wins"].values())
        total_losses = sum(stats["losses"].values())
        total_draws = sum(stats["draws"].values())
        
        result["overall"] = {
            "played": total_played,
            "wins": total_wins,
            "losses": total_losses,
            "draws": total_draws,
            "win_rate": total_wins / total_played if total_played > 0 else 0
        }
        
        return result
        
    def get_game_patterns(self, game_type: str) -> Dict[str, Any]:
        """Get detected patterns for a game type"""
        if game_type not in self.pattern_library:
            return {"message": "No pattern data for game type"}
            
        return self.pattern_library[game_type]
        
    def export_data(self) -> Dict[str, Any]:
        """Export analytics data"""
        return {
            "game_record_counts": {game: len(records) for game, records in self.game_records.items()},
            "player_count": len(self.player_stats),
            "pattern_library": self.pattern_library
        }
        
    def import_data(self, data: Dict[str, Any]):
        """Import analytics data"""
        if "pattern_library" in data:
            self.pattern_library.update(data["pattern_library"])


class Game(ABC):
    """Enhanced abstract base class for all games"""
    
    def __init__(self, name: str, players: List[Player], game_id: Optional[str] = None):
        self.name = name
        self.players = players
        self.game_id = game_id or str(uuid.uuid4())
        self.current_player_idx = 0
        self.is_game_over = False
        self.winner = None
        self.start_time = datetime.now()
        self.end_time = None
        self.move_history = []
        self.game_state_history = []
        self.metadata = {}
        self.analytics = None  # Will be set by GameEngine
        
    @property
    def current_player(self) -> Player:
        """Returns the player whose turn it is"""
        return self.players[self.current_player_idx]
    
    @property
    def duration(self) -> float:
        """Returns the game duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def next_player(self):
        """Advances to the next player"""
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
    
    @abstractmethod
    def initialize_game(self):
        """Set up the game state"""
        pass
    
    @abstractmethod
    def is_valid_move(self, move: Any) -> bool:
        """Check if a move is valid"""
        pass
    
    @abstractmethod
    def make_move(self, move: Any) -> bool:
        """Execute a player's move and return True if successful"""
        pass
    
    @abstractmethod
    def check_game_over(self) -> bool:
        """Check if the game has ended"""
        pass
    
    @abstractmethod
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        pass
    
    @abstractmethod
    def display(self):
        """Display the current game state"""
        pass
    
    def get_valid_moves(self) -> List[Any]:
        """Return a list of all valid moves for the current player"""
        # Default implementation - should be overridden for efficiency
        return []
        
    def record_move(self, move: Any, player_idx: int):
        """Record a move in the game history"""
        self.move_history.append({
            "move": move,
            "player_idx": player_idx,
            "player_name": self.players[player_idx].name if player_idx < len(self.players) else "Unknown",
            "timestamp": datetime.now().isoformat()
        })
        
        # Optionally record game state (can be memory intensive)
        if hasattr(self, "record_states") and self.record_states:
            self.game_state_history.append(self.get_game_state())
    
    def end_game(self, winner: Optional[Player] = None):
        """End the game and update player statistics"""
        self.is_game_over = True
        self.winner = winner
        self.end_time = datetime.now()
        
        # Update player statistics
        for player in self.players:
            if winner is None:
                # Draw
                player.update_stats(draw=True)
            elif player == winner:
                # Win
                player.update_stats(won=True)
            else:
                # Loss
                player.update_stats(lost=True)
        
        # Record game in analytics if available
        if self.analytics:
            outcomes = {}
            for i, player in enumerate(self.players):
                if winner is None:
                    outcomes[player.id] = "draw"
                elif player == winner:
                    outcomes[player.id] = "win"
                else:
                    outcomes[player.id] = "loss"
            
            self.analytics.record_game(self.name, {
                "game_id": self.game_id,
                "player_outcomes": outcomes,
                "move_count": len(self.move_history),
                "duration_seconds": self.duration,
                "winner_id": winner.id if winner else None,
                "moves": self.move_history
            })
    
    def save_game(self, filepath: Optional[str] = None) -> str:
        """Save the game to a file"""
        if not filepath:
            filepath = os.path.join(DEFAULT_SAVE_DIR, f"{self.name}_{self.game_id}.game")
            
        try:
            game_data = {
                "name": self.name,
                "game_id": self.game_id,
                "players": [player.to_dict() for player in self.players],
                "current_player_idx": self.current_player_idx,
                "is_game_over": self.is_game_over,
                "winner": self.winner.to_dict() if self.winner else None,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "move_history": self.move_history,
                "metadata": self.metadata,
                "game_state": self.get_game_state()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(game_data, f)
                
            return filepath
        except Exception as e:
            logger.error(f"Error saving game: {e}")
            return ""
    
    @classmethod
    def load_game(cls, filepath: str) -> Optional['Game']:
        """Load a game from a file"""
        try:
            with open(filepath, 'rb') as f:
                game_data = pickle.load(f)
                
            # Create players
            players = [Player.from_dict(p_data) for p_data in game_data.get("players", [])]
            
            # Create game instance
            game = cls(game_data["name"], players, game_data["game_id"])
            game.current_player_idx = game_data.get("current_player_idx", 0)
            game.is_game_over = game_data.get("is_game_over", False)
            
            # Set winner if exists
            if game_data.get("winner"):
                winner_id = game_data["winner"].get("id")
                game.winner = next((p for p in players if p.id == winner_id), None)
                
            # Set timestamps
            if "start_time" in game_data:
                game.start_time = datetime.fromisoformat(game_data["start_time"])
            if "end_time" in game_data and game_data["end_time"]:
                game.end_time = datetime.fromisoformat(game_data["end_time"])
                
            # Set history and metadata
            game.move_history = game_data.get("move_history", [])
            game.metadata = game_data.get("metadata", {})
            
            # Initialize game with saved state
            game.initialize_game()
            
            # Additional game-specific loading logic would go here
                
            return game
        except Exception as e:
            logger.error(f"Error loading game from {filepath}: {e}")
            return None
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics for this game"""
        result = {
            "game_id": self.game_id,
            "name": self.name,
            "players": [p.name for p in self.players],
            "moves_count": len(self.move_history),
            "duration": self.duration,
            "is_complete": self.is_game_over,
            "winner": self.winner.name if self.winner else None
        }
        
        # Add move frequency analysis
        if self.move_history:
            move_types = {}
            for move_record in self.move_history:
                move = move_record.get("move", {})
                move_type = str(type(move)).split("'")[1] if isinstance(move, object) else "unknown"
                
                if move_type not in move_types:
                    move_types[move_type] = 0
                move_types[move_type] += 1
                
            result["move_type_frequency"] = move_types
            
            # Calculate average time between moves
            if len(self.move_history) >= 2:
                timestamps = []
                for move in self.move_history:
                    if "timestamp" in move:
                        try:
                            timestamps.append(datetime.fromisoformat(move["timestamp"]))
                        except:
                            pass
                
                if len(timestamps) >= 2:
                    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                                 for i in range(len(timestamps)-1)]
                    result["avg_move_time"] = sum(time_diffs) / len(time_diffs)
        
        return result


class GameEngine:
    """Enhanced central engine to manage games"""
    
    def __init__(self):
        self.available_games = {}
        self.active_games = {}
        self.analytics = GameAnalytics()
        self.ai_profiles = {}
        self.game_variants = {}
        self.event_listeners = defaultdict(list)
        
    def register_game(self, game_class):
        """Register a game with the engine"""
        self.available_games[game_class.__name__] = game_class
        logger.info(f"Game registered: {game_class.__name__}")
    
    def register_game_variant(self, base_game_name: str, variant_name: str, config: Dict[str, Any]):
        """Register a variant of an existing game"""
        if base_game_name not in self.available_games:
            logger.warning(f"Cannot register variant for unknown game: {base_game_name}")
            return False
            
        variant_id = f"{base_game_name}.{variant_name}"
        self.game_variants[variant_id] = {
            "base_game": base_game_name,
            "name": variant_name,
            "config": config
        }
        logger.info(f"Game variant registered: {variant_id}")
        return True
    
    def register_ai_profile(self, profile_name: str, ai_class: Any = AIPlayer, settings: Dict[str, Any] = None):
        """Register an AI player profile"""
        settings = settings or {
            "difficulty": GameDifficulty.MEDIUM,
            "adaptive": True,
            "learning_rate": 0.05
        }
        
        self.ai_profiles[profile_name] = {
            "class": ai_class,
            "settings": settings
        }
        logger.info(f"AI profile registered: {profile_name}")
    
    def create_ai_player(self, profile_name: str, player_name: str = None) -> AIPlayer:
        """Create an AI player from a registered profile"""
        if profile_name not in self.ai_profiles:
            logger.warning(f"Unknown AI profile: {profile_name}, using default AI settings")
            return AIPlayer(player_name or f"AI_{profile_name}")
            
        profile = self.ai_profiles[profile_name]
        ai_class = profile["class"]
        settings = profile["settings"]
        
        # Extract settings with proper type conversion
        difficulty = settings.get("difficulty", GameDifficulty.MEDIUM)
        if isinstance(difficulty, str):
            try:
                difficulty = GameDifficulty[difficulty.upper()]
            except KeyError:
                difficulty = GameDifficulty.MEDIUM
                
        adaptive = settings.get("adaptive", True)
        learning_rate = settings.get("learning_rate", 0.05)
        
        # Create AI player instance
        return ai_class(
            player_name or f"AI_{profile_name}",
            difficulty=difficulty,
            adaptive=adaptive,
            learning_rate=learning_rate
        )
    
    def create_game(self, game_name: str, players: List[Player], variant: str = None, 
                   game_id: str = None, settings: Dict[str, Any] = None) -> Optional[Game]:
        """Create a new game instance"""
        settings = settings or {}
        
        # Check if this is a variant
        if variant and f"{game_name}.{variant}" in self.game_variants:
            variant_info = self.game_variants[f"{game_name}.{variant}"]
            game_name = variant_info["base_game"]
            # Merge variant config with provided settings
            merged_settings = variant_info["config"].copy()
            merged_settings.update(settings)
            settings = merged_settings
        
        # Create the game
        if game_name in self.available_games:
            try:
                # Different games may require different parameters
                game_class = self.available_games[game_name]
                signature_params = list(game_class.__init__.__code__.co_varnames)
                
                # Basic initialization with players
                if "settings" in signature_params:
                    game = game_class(players, settings=settings)
                else:
                    # Pass any matching settings as kwargs
                    kwargs = {k: v for k, v in settings.items() if k in signature_params}
                    game = game_class(players, **kwargs)
                
                # Set game_id if provided
                if game_id:
                    game.game_id = game_id
                    
                # Set analytics
                game.analytics = self.analytics
                
                # Initialize game
                game.initialize_game()
                
                # Store in active games
                self.active_games[game.game_id] = game
                
                # Trigger event
                self._trigger_event("game_created", {
                    "game_id": game.game_id,
                    "game_name": game_name,
                    "variant": variant,
                    "player_count": len(players)
                })
                
                return game
            except Exception as e:
                logger.error(f"Error creating game {game_name}: {e}")
                return None
        return None
    
    def get_game(self, game_id: str) -> Optional[Game]:
        """Get an active game by ID"""
        return self.active_games.get(game_id)
    
    def end_game(self, game_id: str, winner_idx: Optional[int] = None) -> bool:
        """End a game and update statistics"""
        if game_id not in self.active_games:
            return False
            
        game = self.active_games[game_id]
        winner = None
        if winner_idx is not None and 0 <= winner_idx < len(game.players):
            winner = game.players[winner_idx]
            
        game.end_game(winner)
        
        # Trigger event
        self._trigger_event("game_ended", {
            "game_id": game_id,
            "game_name": game.name,
            "winner": winner.name if winner else None,
            "duration": game.duration,
            "moves": len(game.move_history)
        })
        
        return True
    
    def get_available_games(self) -> Dict[str, Any]:
        """List all available games with details"""
        result = {}
        
        # Basic games
        for game_name, game_class in self.available_games.items():
            result[game_name] = {
                "name": game_name,
                "description": game_class.__doc__ or "No description available",
                "variants": []
            }
        
        # Add variants
        for variant_id, variant_info in self.game_variants.items():
            base_game = variant_info["base_game"]
            if base_game in result:
                result[base_game]["variants"].append({
                    "name": variant_info["name"],
                    "id": variant_id,
                    "config": variant_info["config"]
                })
        
        return result
    
    def get_game_stats(self, game_name: str) -> Dict[str, Any]:
        """Get statistics for a specific game type"""
        # Get patterns from analytics
        patterns = self.analytics.get_game_patterns(game_name)
        
        # Count active games of this type
        active_count = sum(1 for game in self.active_games.values() if game.name == game_name)
        
        # Get completed games count
        completed = 0
        durations = []
        move_counts = []
        
        for game_record in self.analytics.game_records.get(game_name, []):
            completed += 1
            data = game_record.get("data", {})
            
            if "duration_seconds" in data:
                durations.append(data["duration_seconds"])
                
            if "move_count" in data:
                move_counts.append(data["move_count"])
        
        stats = {
            "name": game_name,
            "active_games": active_count,
            "completed_games": completed,
            "patterns": patterns
        }
        
        if durations:
            stats["avg_duration"] = sum(durations) / len(durations)
            stats["max_duration"] = max(durations)
            stats["min_duration"] = min(durations)
            
        if move_counts:
            stats["avg_moves"] = sum(move_counts) / len(move_counts)
            stats["max_moves"] = max(move_counts)
            stats["min_moves"] = min(move_counts)
            
        return stats
    
    def subscribe_to_event(self, event_type: str, callback: Callable):
        """Subscribe to game engine events"""
        self.event_listeners[event_type].append(callback)
        
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger an event for all subscribers"""
        for callback in self.event_listeners.get(event_type, []):
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type}: {e}")

    def analyze_gameplay_patterns(self, player_id: str = None, game_type: str = None, threshold: int = 10):
        """Analyze gameplay patterns for insights"""
        # Filter analytics data
        if player_id:
            player_stats = self.analytics.get_player_stats(player_id)
            if not player_stats or "games_by_type" not in player_stats:
                return {"message": "Insufficient data for player"}
                
            # Check if we have enough games
            has_enough_data = False
            for game, stats in player_stats.get("games_by_type", {}).items():
                if stats.get("played", 0) >= threshold:
                    has_enough_data = True
                    break
                    
            if not has_enough_data:
                return {"message": f"Need at least {threshold} games to analyze patterns"}
                
            # Analyze player's style
            styles = {}
            for game, stats in player_stats.get("games_by_type", {}).items():
                avg_moves = stats.get("avg_moves", 0)
                win_rate = stats.get("win_rate", 0)
                
                # Simple analysis - can be much more sophisticated
                if avg_moves > 0:
                    if avg_moves < 20:
                        styles[game] = "Aggressive"
                    elif avg_moves < 40:
                        styles[game] = "Balanced"
                    else:
                        styles[game] = "Defensive"
                        
            return {
                "player_id": player_id,
                "play_styles": styles,
                "total_games": player_stats.get("overall", {}).get("played", 0),
                "overall_win_rate": player_stats.get("overall", {}).get("win_rate", 0)
            }
            
        elif game_type:
            patterns = self.analytics.get_game_patterns(game_type)
            if not patterns:
                return {"message": "Insufficient data for game type"}
                
            return {
                "game_type": game_type,
                "patterns": patterns
            }
            
        return {"message": "Must specify either player_id or game_type"}
    
    def save_state(self, filepath: str = None) -> str:
        """Save engine state to file"""
        if not filepath:
            filepath = os.path.join(DEFAULT_SAVE_DIR, f"game_engine_state_{int(time.time())}.state")
            
        try:
            # Export analytics
            analytics_data = self.analytics.export_data()
            
            # Gather AI profiles (exclude class instances)
            ai_profiles_data = {}
            for name, profile in self.ai_profiles.items():
                ai_profiles_data[name] = {
                    "settings": profile.get("settings", {})
                }
            
            # Gather game variants
            state = {
                "analytics": analytics_data,
                "ai_profiles": ai_profiles_data,
                "game_variants": self.game_variants,
                "timestamp": datetime.now().isoformat(),
                "active_games_count": len(self.active_games),
                "available_games": list(self.available_games.keys())
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            return filepath
        except Exception as e:
            logger.error(f"Error saving engine state: {e}")
            return ""
    
    def load_state(self, filepath: str) -> bool:
        """Load engine state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Load analytics
            if "analytics" in state:
                self.analytics.import_data(state["analytics"])
                
            # Load game variants
            if "game_variants" in state:
                self.game_variants.update(state["game_variants"])
                
            # Load AI profiles (just settings, not class instances)
            if "ai_profiles" in state:
                for name, profile_data in state["ai_profiles"].items():
                    if name not in self.ai_profiles:
                        self.ai_profiles[name] = {
                            "class": AIPlayer,  # Default class
                            "settings": profile_data.get("settings", {})
                        }
                    else:
                        # Update settings for existing profiles
                        self.ai_profiles[name]["settings"].update(profile_data.get("settings", {}))
                        
            logger.info(f"Loaded engine state from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading engine state: {e}")
            return False

#===============================================================================
# Game Implementations
#===============================================================================

"""
Traditional Mahjong Game Implementation
"""

class TileType(Enum):
    """Enumeration of Mahjong tile types"""
    DOTS = "Dots"  # 筒子
    BAMBOO = "Bamboo"  # 索子
    CHARACTERS = "Characters"  # 萬子
    WIND = "Wind"  # 風牌
    DRAGON = "Dragon"  # 三元牌
    FLOWER = "Flower"  # 花牌
    SEASON = "Season"  # 季節牌

class Wind(Enum):
    """Enumeration of wind directions"""
    EAST = "East"
    SOUTH = "South"
    WEST = "West"
    NORTH = "North"

class Dragon(Enum):
    """Enumeration of dragon types"""
    RED = "Red"  # 中
    GREEN = "Green"  # 發
    WHITE = "White"  # 白

@dataclass
class Tile:
    """Represents a single Mahjong tile"""
    type: TileType
    value: Any  # Number for suits, Wind/Dragon enum for winds/dragons
    
    def __str__(self):
        if self.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            return f"{self.value} {self.type.value}"
        elif self.type == TileType.WIND:
            return f"{self.value.value} Wind"
        elif self.type == TileType.DRAGON:
            return f"{self.value.value} Dragon"
        elif self.type == TileType.FLOWER:
            return f"Flower {self.value}"
        elif self.type == TileType.SEASON:
            return f"Season {self.value}"
        return "Unknown Tile"
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.type == other.type and self.value == other.value
    
    def __hash__(self):
        return hash((self.type, self.value))

class MahjongHand:
    """Represents a player's hand in Mahjong"""
    
    def __init__(self):
        self.tiles = []  # Concealed tiles
        self.revealed_sets = []  # Sets revealed through calls (chi, pon, kan)
        self.discards = []  # Tiles discarded by this player
    
    def add_tile(self, tile: Tile):
        """Add a tile to the hand"""
        self.tiles.append(tile)
        # Typically in Mahjong, hands are sorted for easier viewing
        self.sort_hand()
    
    def discard_tile(self, index: int) -> Optional[Tile]:
        """Discard a tile from the hand by index"""
        if 0 <= index < len(self.tiles):
            tile = self.tiles.pop(index)
            self.discards.append(tile)
            return tile
        return None
    
    def sort_hand(self):
        """Sort the tiles in the hand for easier viewing"""
        # First by type, then by value
        def sort_key(tile):
            type_order = {
                TileType.CHARACTERS: 0,
                TileType.DOTS: 1,
                TileType.BAMBOO: 2,
                TileType.WIND: 3,
                TileType.DRAGON: 4,
                TileType.FLOWER: 5,
                TileType.SEASON: 6
            }
            
            # For winds, define a specific order
            wind_order = {
                Wind.EAST: 0,
                Wind.SOUTH: 1,
                Wind.WEST: 2,
                Wind.NORTH: 3
            }
            
            # For dragons, define a specific order
            dragon_order = {
                Dragon.WHITE: 0,
                Dragon.GREEN: 1,
                Dragon.RED: 2
            }
            
            type_val = type_order.get(tile.type, 7)
            
            if tile.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
                return (type_val, tile.value)
            elif tile.type == TileType.WIND:
                return (type_val, wind_order.get(tile.value, 4))
            elif tile.type == TileType.DRAGON:
                return (type_val, dragon_order.get(tile.value, 3))
            else:
                return (type_val, tile.value)
        
        self.tiles.sort(key=sort_key)
    
    def can_win(self, tile: Optional[Tile] = None) -> bool:
        """
        Check if the current hand can win with the given tile.
        If tile is None, check if the current hand is a winning hand.
        """
        check_tiles = self.tiles.copy()
        if tile:
            check_tiles.append(tile)
        
        # A winning hand typically has 4 sets and a pair
        # In this simplified version, we'll just check if all tiles can form valid sets
        # A full implementation would be much more complex, checking all possible combinations
        
        # First, ensure we have the right number of tiles
        if len(check_tiles) != 14:
            return False
        
        # Create a counting dictionary
        tile_count = Counter(check_tiles)
        
        # Try to find a pair and see if rest can be divided into sets of 3
        for pair_tile, count in tile_count.items():
            if count >= 2:
                # Remove the pair
                remaining = tile_count.copy()
                remaining[pair_tile] -= 2
                
                # Check if remaining tiles can be grouped into sets
                while sum(remaining.values()) > 0:
                    found_set = False
                    
                    # Try to find a triplet
                    for t, c in remaining.items():
                        if c >= 3:
                            remaining[t] -= 3
                            found_set = True
                            break
                    
                    if not found_set:
                        # Try to find a sequence (for suited tiles only)
                        for t in remaining:
                            if t.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS] and remaining[t] > 0:
                                next_tile1 = Tile(t.type, t.value + 1)
                                next_tile2 = Tile(t.type, t.value + 2)
                                
                                if (next_tile1 in remaining and remaining[next_tile1] > 0 and 
                                    next_tile2 in remaining and remaining[next_tile2] > 0):
                                    remaining[t] -= 1
                                    remaining[next_tile1] -= 1
                                    remaining[next_tile2] -= 1
                                    found_set = True
                                    break
                    
                    if not found_set:
                        break
                
                if sum(remaining.values()) == 0:
                    return True
        
        return False
    
    def can_chi(self, tile: Tile) -> bool:
        """Check if the player can call chi (sequence) with the given tile"""
        # Chi is only valid for suited tiles (not honors) and from the player to the left
        if tile.type not in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            return False
        
        # Check for possible sequences
        if (Tile(tile.type, tile.value + 1) in self.tiles and 
            Tile(tile.type, tile.value + 2) in self.tiles):
            return True
        
        if (Tile(tile.type, tile.value - 1) in self.tiles and 
            Tile(tile.type, tile.value + 1) in self.tiles):
            return True
        
        if (Tile(tile.type, tile.value - 2) in self.tiles and 
            Tile(tile.type, tile.value - 1) in self.tiles):
            return True
        
        return False
    
    def can_pon(self, tile: Tile) -> bool:
        """Check if the player can call pon (triplet) with the given tile"""
        tile_count = Counter(self.tiles)
        return tile_count[tile] >= 2  # Need at least 2 matching tiles in hand
    
    def can_kan(self, tile: Tile, is_closed: bool = False) -> bool:
        """Check if the player can call kan (quad) with the given tile"""
        if is_closed:
            # For closed kan, need all 4 tiles in hand
            tile_count = Counter(self.tiles)
            return tile_count[tile] >= 4
        else:
            # For open kan from discard
            tile_count = Counter(self.tiles)
            return tile_count[tile] >= 3
    
    def __str__(self):
        hand_str = "Hand: " + ", ".join(str(tile) for tile in self.tiles)
        if self.revealed_sets:
            revealed_str = "Revealed: " + ", ".join(str(s) for s in self.revealed_sets)
            hand_str += "\n" + revealed_str
        return hand_str

class MahjongPlayer(Player):
    """Player in a Mahjong game"""
    
    def __init__(self, name: str, wind: Wind, id: str = None):
        super().__init__(name, id)
        self.hand = MahjongHand()
        self.wind = wind  # Player's seat wind
    
    def __str__(self):
        return f"{self.name} ({self.wind.value}) - {self.hand}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = super().to_dict()
        data["wind"] = self.wind.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MahjongPlayer':
        """Create from dictionary"""
        wind_name = data.get("wind", "EAST")
        try:
            wind = Wind[wind_name]
        except KeyError:
            wind = Wind.EAST
            
        player = cls(data["name"], wind, data.get("id"))
        
        # Set other attributes
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player

class MahjongAIPlayer(MahjongPlayer, AIPlayer):
    """AI player for Mahjong games"""
    
    def __init__(self, name: str, wind: Wind, difficulty: GameDifficulty = GameDifficulty.MEDIUM,
                 adaptive: bool = True, learning_rate: float = 0.05, id: str = None):
        MahjongPlayer.__init__(self, name, wind, id)
        AIPlayer.__init__(self, name, difficulty, adaptive, learning_rate)
        self.hand_evaluation_cache = {}
        
    def evaluate_tile_value(self, tile: Tile, hand: MahjongHand) -> float:
        """Evaluate the value of a tile for discard decision"""
        # Caching for performance
        cache_key = (tile, tuple(hand.tiles))
        if cache_key in self.hand_evaluation_cache:
            return self.hand_evaluation_cache[cache_key]
            
        value = 0.0
        
        # Check if part of a pair
        tile_count = Counter(hand.tiles)
        if tile_count[tile] >= 2:
            value += 3.0  # Pairs are valuable
            
        # Check if part of a potential sequence (for suited tiles)
        if tile.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            # Check tiles before
            if Tile(tile.type, tile.value - 1) in hand.tiles:
                value += 1.0
            if Tile(tile.type, tile.value - 2) in hand.tiles:
                value += 0.5
                
            # Check tiles after
            if Tile(tile.type, tile.value + 1) in hand.tiles:
                value += 1.0
            if Tile(tile.type, tile.value + 2) in hand.tiles:
                value += 0.5
                
        # Honor tiles (winds and dragons) value
        if tile.type == TileType.WIND:
            # Player's seat wind and round wind are more valuable
            if tile.value == self.wind:
                value += 2.0
                
        if tile.type == TileType.DRAGON:
            value += 1.5  # Dragons are generally valuable
            
        # Cache and return
        self.hand_evaluation_cache[cache_key] = value
        return value
    
    def choose_tile_to_discard(self, hand: MahjongHand) -> int:
        """Choose which tile to discard based on AI difficulty"""
        if len(hand.tiles) == 0:
            return -1
            
        # For very easy difficulty, just discard randomly
        if self.difficulty == GameDifficulty.BEGINNER:
            return random.randint(0, len(hand.tiles) - 1)
            
        # For higher difficulties, evaluate tiles
        tile_values = []
        for i, tile in enumerate(hand.tiles):
            value = self.evaluate_tile_value(tile, hand)
            tile_values.append((i, value))
            
        # Sort by value (ascending - lower is better to discard)
        tile_values.sort(key=lambda x: x[1])
        
        # Add randomness based on difficulty
        if self.difficulty == GameDifficulty.EASY:
            # Might not pick the absolute worst tile
            candidates = tile_values[:min(3, len(tile_values))]
            return random.choice(candidates)[0]
        elif self.difficulty == GameDifficulty.MEDIUM:
            # More focused on optimal play
            candidates = tile_values[:min(2, len(tile_values))]
            return random.choice(candidates)[0]
        else:
            # Expert+ always picks the optimal tile
            return tile_values[0][0]
    
    def decide_call(self, current_discard: Tile, can_chi: bool, can_pon: bool, can_kan: bool, can_win: bool) -> str:
        """Decide whether to call Chi, Pon, Kan, Win or Pass based on AI difficulty"""
        # Always win if possible
        if can_win:
            return "win"
            
        # Higher difficulty levels make better decisions
        if self.difficulty in [GameDifficulty.BEGINNER, GameDifficulty.EASY]:
            # Lower difficulties make more random calls
            options = []
            if can_chi:
                options.append("chi")
            if can_pon:
                options.append("pon")
            if can_kan:
                options.append("kan")
                
            # Add multiple "pass" options to weight randomness
            # Easier difficulties pass more often
            passes = 5 if self.difficulty == GameDifficulty.BEGINNER else 3
            for _ in range(passes):
                options.append("pass")
                
            return random.choice(options) if options else "pass"
            
        elif self.difficulty == GameDifficulty.MEDIUM:
            # Medium makes decent decisions
            # Prioritize Kan > Pon > Chi
            if can_kan:
                return "kan" if random.random() < 0.8 else "pass"
            if can_pon:
                return "pon" if random.random() < 0.7 else "pass"
            if can_chi:
                return "chi" if random.random() < 0.6 else "pass"
            return "pass"
            
        else:
            # Expert+ makes optimal calls
            # This would involve complex hand analysis, but simplified here
            if can_kan:
                return "kan"
            if can_pon:
                return "pon"
            if can_chi and self.evaluate_tile_value(current_discard, self.hand) > 1.5:
                return "chi"
            return "pass"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = MahjongPlayer.to_dict(self)
        
        # Add AI-specific data
        data["ai_settings"] = {
            "difficulty": self.difficulty.name,
            "adaptive": self.adaptive,
            "learning_rate": self.learning_rate
        }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MahjongAIPlayer':
        """Create from dictionary"""
        wind_name = data.get("wind", "EAST")
        try:
            wind = Wind[wind_name]
        except KeyError:
            wind = Wind.EAST
            
        # Extract AI settings
        ai_settings = data.get("ai_settings", {})
        difficulty_name = ai_settings.get("difficulty", "MEDIUM")
        try:
            difficulty = GameDifficulty[difficulty_name]
        except KeyError:
            difficulty = GameDifficulty.MEDIUM
            
        adaptive = ai_settings.get("adaptive", True)
        learning_rate = ai_settings.get("learning_rate", 0.05)
        
        # Create instance
        player = cls(
            data["name"], 
            wind, 
            difficulty=difficulty,
            adaptive=adaptive,
            learning_rate=learning_rate,
            id=data.get("id")
        )
        
        # Set other attributes
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player

class MahjongGame(Game):
    """Enhanced implementation of traditional Mahjong"""
    
    def __init__(self, players: List[Player], settings: Dict[str, Any] = None, game_id: str = None):
        """
        Initialize a Mahjong game
        
        Args:
            players: List of players (must be exactly 4)
            settings: Dictionary of game settings
            game_id: Optional game ID
        """
        if len(players) != 4:
            raise ValueError("Mahjong requires exactly 4 players")
        
        # Process settings
        self.settings = settings or {}
        self.ruleset = self.settings.get("ruleset", "japanese")  # japanese, chinese, etc.
        self.use_red_fives = self.settings.get("use_red_fives", True)
        self.include_flowers = self.settings.get("include_flowers", True)
        self.include_seasons = self.settings.get("include_seasons", True)
        self.auto_win = self.settings.get("auto_win", True)  # Automatically win when possible
        
        # Assign winds to players if they don't already have them
        winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
        for i, player in enumerate(players):
            if not isinstance(player, MahjongPlayer):
                # Convert regular Player to MahjongPlayer
                players[i] = MahjongPlayer(player.name, winds[i], player.id if hasattr(player, 'id') else None)
            elif not hasattr(player, 'wind'):
                # Assign wind if missing
                player.wind = winds[i]
        
        super().__init__("Mahjong", players, game_id)
        
        # Mahjong-specific game state
        self.wall = []  # The tile wall
        self.dead_wall = []  # Tiles for replacement draws (dora indicators, etc.)
        self.discards = []  # Discarded tiles
        self.dora_indicators = []  # Tiles that indicate the dora
        self.round_wind = Wind.EAST  # Current round wind
        self.current_discard = None  # The current tile in the discard pool (for calls)
        self.last_action = None  # Last action performed
        
        # Game scoring
        self.scores = {player.id: 0 for player in players}
        self.uma_values = [15, 5, -5, -15]  # Points adjustment based on final position
        
        # Statistics tracking
        self.stats = {
            "calls": {"chi": 0, "pon": 0, "kan": 0, "win": 0},
            "discards_by_player": {player.id: 0 for player in players},
            "tile_waits": {},
            "winning_hands": []
        }
    
    def initialize_game(self):
        """Set up the game state"""
        logger.info(f"Initializing Mahjong game {self.game_id}")
        self.is_game_over = False
        self.winner = None
        self.wall = self._create_tile_set()
        random.shuffle(self.wall)
        
        # Separate the dead wall (14 tiles from the end)
        self.dead_wall = self.wall[-14:]
        self.wall = self.wall[:-14]
        
        # Set the first dora indicator
        self.dora_indicators = [self.dead_wall[4]]
        
        # Deal tiles to players (13 tiles each)
        for player in self.players:
            player.hand = MahjongHand()
            for _ in range(13):
                player.hand.add_tile(self.draw_tile())
        
        # East player (dealer) draws an extra tile to start
        self.current_player_idx = 0  # East player starts
        first_player = self.players[self.current_player_idx]
        first_player.hand.add_tile(self.draw_tile())
        
        # Now East player needs to discard
        self.last_action = "initialize"
        logger.info(f"Mahjong game {self.game_id} initialized")
    
    def

def _score_japanese(self):
        """Japanese scoring: territory + captures"""
        black_score = self.players[0].captures
        white_score = self.players[1].captures
        
        # Calculate territory
        territories = self._calculate_territories()
        black_score += territories["black"]
        white_score += territories["white"]
        
        # Add komi
        white_score += self.komi
        
        # Store territory counts
        self.territories = territories
        
        # Set player scores
        self.players[0].score = black_score
        self.players[1].score = white_score
        
        # Determine the winner
        if black_score > white_score:
            self.winner = self.players[0]  # Black
        elif white_score > black_score:
            self.winner = self.players[1]  # White
        else:
            # In case of a tie (very rare with fractional komi)
            self.winner = None
    
    def _calculate_territories(self) -> Dict[str, int]:
        """Calculate territory control"""
        territories = {"black": 0, "white": 0}
        
        # Create a matrix to track visited intersections
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # For each empty intersection, determine which player controls it
        for y in range(self.board_size):
            for x in

abs_rect = self.get_absolute_rect()
        
        # Draw the panel background
        pygame.draw.rect(
            surface, 
            self.color or Theme.get(self.theme, "panel"), 
            abs_rect,
            border_radius=self.border_radius
        )
        
        # Draw the border if needed
        if self.border_width > 0:
            pygame.draw.rect(
                surface, 
                self.border_color or Theme.get(self.theme, "border"), 
                abs_rect,
                self.border_width,
                border_radius=self.border_radius
            )
        
        # Set up clipping for scrollable content
        if self.scrollable:
            old_clip = surface.get_clip()
            surface.set_clip(abs_rect)
        
        # Draw children with scroll offset
        for child in self.children:
            if self.scrollable:
                # Adjust child position for scrolling
                original_y = child.rect.y
                child.rect.y -= self.scroll_y
                
                # Only draw if visible in the panel
                child_abs = child.get_absolute_rect()
                if child_abs.bottom > abs_rect.top and child_abs.top < abs_rect.bottom:
                    child.draw(surface)
                
                # Restore original position
                child.rect.y = original_y
            else:
                child.draw(surface)
        
        # Restore original clipping rect
        if self.scrollable:
            surface.set_clip(old_clip)
            
            # Draw scroll indicators if needed
            if self.max_scroll_y > 0:
                # Draw scroll track
                track_rect = pygame.Rect(
                    abs_rect.right - 10, 
                    abs_rect.top + 5, 
                    5, 
                    abs_rect.height - 10
                )
                pygame.draw.rect(surface, Theme.get(self.theme, "border"), track_rect, border_radius=2)
                
                # Draw scroll thumb
                thumb_height = max(20, abs_rect.height * abs_rect.height / (abs_rect.height + self.max_scroll_y))
                thumb_pos = abs_rect.top + 5 + (track_rect.height - thumb_height) * self.scroll_y / self.max_scroll_y
                thumb_rect = pygame.Rect(
                    abs_rect.right - 10,
                    thumb_pos,
                    5,
                    thumb_height
                )
                pygame.draw.rect(surface, Theme.get(self.theme, "button"), thumb_rect, border_radius=2)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if self.scrollable:
            # Handle scrolling events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if abs_rect.collidepoint(event.pos):
                    if event.button == 4:  # Scroll up
                        self.scroll_y = max(0, self.scroll_y - 20)
                        return True
                    elif event.button == 5:  # Scroll down
                        self.scroll_y = min(self.max_scroll_y, self.scroll_y + 20)
                        return True
                    elif event.button == 1:  # Left click
                        # Check if clicking on scroll track
                        track_rect = pygame.Rect(
                            abs_rect.right - 10, 
                            abs_rect.top + 5, 
                            5, 
                            abs_rect.height - 10
                        )
                        if track_rect.collidepoint(event.pos):
                            # Start dragging
                            self.dragging = True
                            self.drag_start_y = event.pos[1]
                            self.drag_start_scroll = self.scroll_y
                            return True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.dragging:
                    self.dragging = False
                    return True
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    # Calculate new scroll position
                    drag_offset = event.pos[1] - self.drag_start_y
                    track_height = abs_rect.height - 10
                    scroll_ratio = drag_offset / track_height
                    new_scroll = self.drag_start_scroll + scroll_ratio * self.max_scroll_y
                    self.scroll_y = max(0, min(self.max_scroll_y, new_scroll))
                    return True
        
        # Pass event to children, adjusting for scroll offset
        if self.scrollable:
            pos = event.pos
            if hasattr(event, 'pos') and abs_rect.collidepoint(pos):
                for child in reversed(self.children):
                    # Adjust child position for scrolling
                    original_y = child.rect.y
                    child.rect.y -= self.scroll_y
                    
                    # Clone event with adjusted position
                    adjusted_event = pygame.event.Event(event.type, {**event.__dict__})
                    if hasattr(adjusted_event, 'pos'):
                        adjusted_event.pos = pos
                    
                    # Pass event to child
                    result = child.handle_event(adjusted_event)
                    
                    # Restore original position
                    child.rect.y = original_y
                    
                    if result:
                        return True
        
        return False
    
    def add_child(self, child: UIElement):
        """Add a child UI element"""
        super().add_child(child)
        self._recalculate_scroll()
    
    def _recalculate_scroll(self):
        """Recalculate max scroll value based on content height"""
        if not self.scrollable:
            return
            
        max_y = 0
        for child in self.children:
            child_bottom = child.rect.y + child.rect.height
            max_y = max(max_y, child_bottom)
        
        content_height = max_y + self.padding
        visible_height = self.rect.height
        
        self.max_scroll_y = max(0, content_height - visible_height)
        
        # Adjust current scroll if needed
        if self.scroll_y > self.max_scroll_y:
            self.scroll_y = self.max_scroll_y

class Dropdown(UIElement):
    """A dropdown selection menu"""
    
    def __init__(self, x: int, y: int, width: int, height: int, options: List[str], 
                 callback: Optional[Callable[[str], None]] = None, parent=None,
                 color: Optional[Tuple[int, int, int]] = None,
                 text_color: Optional[Tuple[int, int, int]] = None,
                 selected_index: int = 0):
        """
        Initialize a dropdown.
        
        Args:
            x: X position
            y: Y position
            width: Dropdown width
            height: Dropdown height
            options: List of options
            callback: Function to call when selection changes
            parent: Parent UI element
            color: Dropdown color (or None for theme default)
            text_color: Text color (or None for theme default)
            selected_index: Initially selected option index
        """
        super().__init__(x, y, width, height, parent)
        self.options = options
        self.callback = callback
        self.color = color
        self.text_color = text_color
        self.selected_index = min(selected_index, len(options) - 1) if options else 0
        self.expanded = False
        self.max_visible = 5  # Maximum number of visible options when expanded
        self.font = None
        self.hover_option = -1
        
        # Load font if pygame is available
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont("Arial", 16)
    
    def draw(self, surface: pygame.Surface):
        """Draw the dropdown"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        # Draw the dropdown box
        bg_color = self.color or Theme.get(self.theme, "button")
        text_color = self.text_color or Theme.get(self.theme, "button_text")
        
        pygame.draw.rect(surface, bg_color, abs_rect, border_radius=4)
        pygame.draw.rect(surface, Theme.get(self.theme, "border"), abs_rect, 1, border_radius=4)
        
        # Draw selected option
        if self.options and 0 <= self.selected_index < len(self.options):
            selected_text = self.options[self.selected_index]
            text_surface = self.font.render(selected_text, True, text_color)
            text_rect = text_surface.get_rect(midleft=(abs_rect.left + 10, abs_rect.centery))
            surface.blit(text_surface, text_rect)
        
        # Draw dropdown arrow
        arrow_points = [
            (abs_rect.right - 15, abs_rect.centery - 3),
            (abs_rect.right - 8, abs_rect.centery - 3),
            (abs_rect.right - 11.5, abs_rect.centery + 3)
        ]
        pygame.draw.polygon(surface, text_color, arrow_points)
        
        # Draw expanded options
        if self.expanded:
            options_to_show = min(len(self.options), self.max_visible)
            dropdown_height = options_to_show * self.rect.height
            dropdown_rect = pygame.Rect(abs_rect.left, abs_rect.bottom, abs_rect.width, dropdown_height)
            
            # Draw dropdown background
            pygame.draw.rect(surface, bg_color, dropdown_rect, border_radius=4)
            pygame.draw.rect(surface, Theme.get(self.theme, "border"), dropdown_rect, 1, border_radius=4)
            
            # Draw options
            for i in range(min(len(self.options), self.max_visible)):
                option_rect = pygame.Rect(
                    dropdown_rect.left, 
                    dropdown_rect.top + i * self.rect.height, 
                    dropdown_rect.width, 
                    self.rect.height
                )
                
                # Highlight hovered option
                if i == self.hover_option:
                    pygame.draw.rect(
                        surface, 
                        Theme.get(self.theme, "button_hover"), 
                        option_rect
                    )
                
                # Draw option text
                option_text = self.options[i]
                text_surface = self.font.render(option_text, True, text_color)
                text_rect = text_surface.get_rect(midleft=(option_rect.left + 10, option_rect.centery))
                surface.blit(text_surface, text_rect)
        
        # Draw children
        super().draw(surface)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Toggle dropdown when clicking the main area
            if abs_rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return True
            
            # Check if clicking on an expanded option
            if self.expanded:
                options_to_show = min(len(self.options), self.max_visible)
                dropdown_rect = pygame.Rect(abs_rect.left, abs_rect.bottom, abs_rect.width, options_to_show * self.rect.height)
                
                if dropdown_rect.collidepoint(event.pos):
                    # Calculate which option was clicked
                    option_index = (event.pos[1] - dropdown_rect.top) // self.rect.height
                    if 0 <= option_index < len(self.options):
                        self.selected_index = option_index
                        self.expanded = False
                        
                        if self.callback:
                            self.callback(self.options[self.selected_index])
                        
                        return True
                else:
                    # Close dropdown when clicking outside
                    self.expanded = False
                    return False
        
        elif event.type == pygame.MOUSEMOTION and self.expanded:
            # Track hovered option
            options_to_show = min(len(self.options), self.max_visible)
            dropdown_rect = pygame.Rect(abs_rect.left, abs_rect.bottom, abs_rect.width, options_to_show * self.rect.height)
            
            if dropdown_rect.collidepoint(event.pos):
                self.hover_option = (event.pos[1] - dropdown_rect.top) // self.rect.height
            else:
                self.hover_option = -1
        
        return False
    
    def get_selected(self) -> str:
        """Get the selected option"""
        if self.options and 0 <= self.selected_index < len(self.options):
            return self.options[self.selected_index]
        return ""
    
    def set_selected(self, option: str) -> bool:
        """Set the selected option"""
        if option in self.options:
            self.selected_index = self.options.index(option)
            return True
        return False
    
    def set_selected_index(self, index: int) -> bool:
        """Set the selected option by index"""
        if 0 <= index < len(self.options):
            self.selected_index = index
            return True
        return False

class Slider(UIElement):
    """A slider control for numeric values"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 min_value: float, max_value: float, value: float,
                 callback: Optional[Callable[[float], None]] = None, parent=None,
                 color: Optional[Tuple[int, int, int]] = None,
                 handle_color: Optional[Tuple[int, int, int]] = None,
                 show_value: bool = True,
                 step: Optional[float] = None,
                 format_str: str = "{:.1f}"):
        """
        Initialize a slider.
        
        Args:
            x: X position
            y: Y position
            width: Slider width
            height: Slider height
            min_value: Minimum value
            max_value: Maximum value
            value: Current value
            callback: Function to call when value changes
            parent: Parent UI element
            color: Slider color (or None for theme default)
            handle_color: Handle color (or None for theme default)
            show_value: Whether to show the current value
            step: Step size (or None for continuous)
            format_str: Format string for value display
        """
        super().__init__(x, y, width, height, parent)
        self.min_value = min_value
        self.max_value = max_value
        self.value = max(min_value, min(max_value, value))
        self.callback = callback
        self.color = color
        self.handle_color = handle_color
        self.show_value = show_value
        self.step = step
        self.format_str = format_str
        self.font = None
        self.dragging = False
        
        # Load font if pygame is available
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont("Arial", 14)
    
    def draw(self, surface: pygame.Surface):
        """Draw the slider"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        # Draw the track
        track_rect = pygame.Rect(
            abs_rect.left, 
            abs_rect.centery - 2,
            abs_rect.width,
            4
        )
        pygame.draw.rect(surface, self.color or Theme.get(self.theme, "button"), track_rect, border_radius=2)
        
        # Calculate handle position
        value_ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        handle_x = abs_rect.left + int(value_ratio * abs_rect.width)
        handle_rect = pygame.Rect(
            handle_x - 6,
            abs_rect.centery - 8,
            12,
            16
        )
        
        # Draw the handle
        pygame.draw.rect(
            surface, 
            self.handle_color or Theme.get(self.theme, "accent"),
            handle_rect,
            border_radius=6
        )
        pygame.draw.rect(
            surface, 
            Theme.get(self.theme, "border"),
            handle_rect,
            1,
            border_radius=6
        )
        
        # Draw the value
        if self.show_value:
            value_text = self.format_str.format(self.value)
            text_surface = self.font.render(value_text, True, Theme.get(self.theme, "text"))
            text_rect = text_surface.get_rect(midright=(abs_rect.right, abs_rect.centery - 12))
            surface.blit(text_surface, text_rect)
        
        # Draw children
        super().draw(surface)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Start dragging when clicking on the handle or track
            value_ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
            handle_x = abs_rect.left + int(value_ratio * abs_rect.width)
            handle_rect = pygame.Rect(
                handle_x - 6,
                abs_rect.centery - 8,
                12,
                16
            )
            
            track_rect = pygame.Rect(
                abs_rect.left, 
                abs_rect.centery - 10,  # Make the track easier to click
                abs_rect.width,
                20
            )
            
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
                return True
            elif track_rect.collidepoint(event.pos):
                # Jump to clicked position
                self._update_value_from_pos(event.pos[0])
                self.dragging = True
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_value_from_pos(event.pos[0])
                return True
        
        return False
    
    def _update_value_from_pos(self, x_pos: int):
        """Update value based on mouse position"""
        abs_rect = self.get_absolute_rect()
        
        # Calculate value from position
        track_width = abs_rect.width
        pos_ratio = max(0, min(1, (x_pos - abs_rect.left) / track_width))
        new_value = self.min_value + pos_ratio * (self.max_value - self.min_value)
        
        # Apply step if needed
        if self.step:
            new_value = round(new_value / self.step) * self.step
        
        # Enforce bounds
        new_value = max(self.min_value, min(self.max_value, new_value))
        
        # Update value
        if new_value != self.value:
            self.value = new_value
            
            if self.callback:
                self.callback(self.value)
    
    def set_value(self, value: float, trigger_callback: bool = True):
        """Set the slider value"""
        new_value = max(self.min_value, min(self.max_value, value))
        
        if new_value != self.value:
            self.value = new_value
            
            if trigger_callback and self.callback:
                self.callback(self.value)

class Checkbox(UIElement):
    """A checkbox control for boolean values"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, checked: bool = False,
                 callback: Optional[Callable[[bool], None]] = None, parent=None,
                 color: Optional[Tuple[int, int, int]] = None,
                 text_color: Optional[Tuple[int, int, int]] = None):
        """
        Initialize a checkbox.
        
        Args:
            x: X position
            y: Y position
            width: Checkbox width
            height: Checkbox height
            text: Checkbox label text
            checked: Whether the checkbox is checked
            callback: Function to call when state changes
            parent: Parent UI element
            color: Checkbox color (or None for theme default)
            text_color: Text color (or None for theme default)
        """
        super().__init__(x, y, width, height, parent)
        self.text = text
        self.checked = checked
        self.callback = callback
        self.color = color
        self.text_color = text_color
        self.font = None
        
        # Load font if pygame is available
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont("Arial", 16)
    
    def draw(self, surface: pygame.Surface):
        """Draw the checkbox"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        # Determine colors
        if self.enabled:
            bg_color = self.color or Theme.get(self.theme, "button")
            text_color = self.text_color or Theme.get(self.theme, "text")
        else:
            bg_color = tuple(max(0, c - 40) for c in (self.color or Theme.get(self.theme, "button")))
            text_color = tuple(max(0, c - 40) for c in (self.text_color or Theme.get(self.theme, "text")))
        
        # Draw the box
        box_size = min(abs_rect.height, 20)
        box_rect = pygame.Rect(
            abs_rect.left,
            abs_rect.centery - box_size // 2,
            box_size,
            box_size
        )
        pygame.draw.rect(surface, bg_color, box_rect, border_radius=3)
        pygame.draw.rect(surface, Theme.get(self.theme, "border"), box_rect, 1, border_radius=3)
        
        # Draw check mark if checked
        if self.checked:
            check_color = Theme.get(self.theme, "accent")
            # Draw checkmark
            pygame.draw.rect(surface, check_color, box_rect.inflate(-6, -6), border_radius=1)
        
        # Draw text
        if self.text:
            text_surface = self.font.render(self.text, True, text_color)
            text_rect = text_surface.get_rect(midleft=(abs_rect.left + box_size + 5, abs_rect.centery))
            surface.blit(text_surface, text_rect)
        
        # Draw children
        super().draw(surface)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if abs_rect.collidepoint(event.pos):
                self.checked = not self.checked
                
                if self.callback:
                    self.callback(self.checked)
                
                return True
        
        return False
    
    def set_checked(self, checked: bool, trigger_callback: bool = True):
        """Set the checkbox state"""
        if self.checked != checked:
            self.checked = checked
            
            if trigger_callback and self.callback:
                self.callback(self.checked)

class Dialog(UIElement):
    """A modal dialog box"""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str,
                 parent=None, closable: bool = True, draggable: bool = True,
                 modal: bool = True):
        """
        Initialize a dialog box.
        
        Args:
            x: X position
            y: Y position
            width: Dialog width
            height: Dialog height
            title: Dialog title
            parent: Parent UI element
            closable: Whether the dialog can be closed
            draggable: Whether the dialog can be dragged
            modal: Whether the dialog blocks interaction with elements behind it
        """
        super().__init__(x, y, width, height, parent)
        self.title = title
        self.closable = closable
        self.draggable = draggable
        self.modal = modal
        self.dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_rect = None
        self.on_close = None
        
        # Create title bar
        self.title_height = 30
        self.content_panel = Panel(0, self.title_height, width, height - self.title_height, self, padding=10)
        
        # Create close button if closable
        if self.closable:
            self.close_button = Button(width - 30, 0, 30, self.title_height, "✕", self.close, self)
        
        # Fonts
        if PYGAME_AVAILABLE:
            self.title_font = pygame.font.SysFont("Arial", 16, bold=True)
    
    def draw(self, surface: pygame.Surface):
        """Draw the dialog"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        # Draw dialog background with shadow
        shadow_rect = abs_rect.inflate(12, 12)
        shadow_rect.topleft = (abs_rect.left + 6, abs_rect.top + 6)
        
        # Semi-transparent shadow
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        shadow_surface.fill((0, 0, 0, 64))
        surface.blit(shadow_surface, shadow_rect.topleft)
        
        # Dialog background
        pygame.draw.rect(surface, Theme.get(self.theme, "panel"), abs_rect, border_radius=4)
        pygame.draw.rect(surface, Theme.get(self.theme, "border"), abs_rect, 1, border_radius=4)
        
        # Draw title bar
        title_rect = pygame.Rect(abs_rect.left, abs_rect.top, abs_rect.width, self.title_height)
        pygame.draw.rect(surface, Theme.get(self.theme, "accent"), title_rect, border_top_left_radius=4, border_top_right_radius=4)
        
        # Draw title text
        if self.title:
            title_surface = self.title_font.render(self.title, True, Theme.get(self.theme, "button_text"))
            title_x = abs_rect.left + 10
            title_y = abs_rect.top + (self.title_height - title_surface.get_height()) // 2
            surface.blit(title_surface, (title_x, title_y))
        
        # Draw children (content panel and close button)
        super().draw(surface)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if self.draggable:
            # Handle dragging the dialog
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                title_rect = pygame.Rect(abs_rect.left, abs_rect.top, abs_rect.width, self.title_height)
                if title_rect.collidepoint(event.pos):
                    self.dragging = True
                    self.drag_start_pos = event.pos
                    self.drag_start_rect = pygame.Rect(self.rect)
                    return True
            
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragging:
                    self.dragging = False
                    return True
            
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                delta_x = event.pos[0] - self.drag_start_pos[0]
                delta_y = event.pos[1] - self.drag_start_pos[1]
                self.rect.x = self.drag_start_rect.x + delta_x
                self.rect.y = self.drag_start_rect.y + delta_y
                return True
        
        # For modal dialogs, capture all events within the dialog
        if self.modal and abs_rect.collidepoint(event.pos):
            return True
        
        return False
    
    def close(self):
        """Close the dialog"""
        self.visible = False
        
        if self.on_close:
            self.on_close()
    
    def set_content(self, element: UIElement):
        """Set the dialog content"""
        self.content_panel.children = []
        self.content_panel.add_child(element)
    
    def center_on_parent(self):
        """Center the dialog on its parent"""
        if self.parent:
            parent_rect = self.parent.get_absolute_rect()
            self.rect.center = parent_rect.center
            
            # Adjust for offset in parent
            if hasattr(self.parent, 'rect'):
                self.rect.x -= self.parent.rect.x
                self.rect.y -= self.parent.rect.y

class MessageDialog(Dialog):
    """A simple message dialog with buttons"""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str, message: str,
                 buttons: List[str] = ["OK"], callback: Optional[Callable[[str], None]] = None,
                 parent=None):
        """
        Initialize a message dialog.
        
        Args:
            x: X position
            y: Y position
            width: Dialog width
            height: Dialog height
            title: Dialog title
            message: Dialog message
            buttons: List of button labels
            callback: Function to call when a button is clicked
            parent: Parent UI element
        """
        super().__init__(x, y, width, height, title, parent)
        self.message = message
        self.buttons = buttons
        self.callback = callback
        
        # Create message label
        self.message_label = Label(10, 10, width - 20, height - 60, message, self.content_panel, wrap=True)
        
        # Create buttons
        self._create_buttons()
    
    def _create_buttons(self):
        """Create the dialog buttons"""
        button_width = 100
        button_height = 30
        button_spacing = 10
        
        total_width = len(self.buttons) * button_width + (len(self.buttons) - 1) * button_spacing
        start_x = (self.rect.width - total_width) // 2
        
        for i, button_text in enumerate(self.buttons):
            button_x = start_x + i * (button_width + button_spacing)
            button_y = self.rect.height - 50
            
            button = Button(
                button_x, 
                button_y, 
                button_width, 
                button_height, 
                button_text,
                lambda btn=button_text: self._button_clicked(btn),
                self
            )
    
    def _button_clicked(self, button_text: str):
        """Handle button click"""
        if self.callback:
            self.callback(button_text)
        
        self.close()
    
    def set_message(self, message: str):
        """Set the dialog message"""
        self.message = message
        self.message_label.set_text(message)

class ConfirmDialog(MessageDialog):
    """A confirmation dialog with Yes/No buttons"""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str, message: str,
                 callback: Optional[Callable[[bool], None]] = None, parent=None,
                 yes_text: str = "Yes", no_text: str = "No"):
        """
        Initialize a confirmation dialog.
        
        Args:
            x: X position
            y: Y position
            width: Dialog width
            height: Dialog height
            title: Dialog title
            message: Dialog message
            callback: Function to call with the result (True for yes, False for no)
            parent: Parent UI element
            yes_text: Text for the 'yes' button
            no_text: Text for the 'no' button
        """
        super().__init__(x, y, width, height, title, message, [yes_text, no_text], 
                        lambda btn: self._confirm_callback(btn == yes_text), parent)
    
    def _confirm_callback(self, result: bool):
        """Handle confirmation result"""
        if self.callback:
            self.callback(result)

# Game-specific UI classes

class GameBoard(UIElement):
    """Base class for game boards (Chess, Go, etc.)"""
    
    def __init__(self, x: int, y: int, width: int, height: int, game, parent=None):
        """
        Initialize a game board.
        
        Args:
            x: X position
            y: Y position
            width: Board width
            height: Board height
            game: Game instance
            parent: Parent UI element
        """
        super().__init__(x, y, width, height, parent)
        self.game = game
        self.board_size = min(width, height)
        self.cell_size = 0
        self.board_offset_x = 0
        self.board_offset_y = 0
        self.selected_cell = None
        self.last_move = None
        self.valid_moves = []
        self.show_valid_moves = config.show_valid_moves
        self.show_last_move = config.show_last_move
        self.flip_board = False
        self.piece_images = {}
        self.animation_manager = AnimationManager()
        self.animated_pieces = {}
        
        # Calculate board dimensions
        self._update_dimensions()
    
    def _update_dimensions(self):
        """Update board dimensions based on available space"""
        self.board_size = min(self.rect.width, self.rect.height)
        self.board_offset_x = (self.rect.width - self.board_size) // 2
        self.board_offset_y = (self.rect.height - self.board_size) // 2
    
    def screen_to_board(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to board coordinates"""
        # Implement in subclasses
        return (0, 0)
    
    def board_to_screen(self, board_x: int, board_y: int) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates"""
        # Implement in subclasses
        return (0, 0)
    
    def draw(self, surface: pygame.Surface):
        """Draw the game board"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        # Update animations
        self.animation_manager.update()
        
        # Draw the board and pieces
        self._draw_board(surface, abs_rect)
        self._draw_pieces(surface, abs_rect)
        
        # Draw selection, valid moves, and last move highlights
        self._draw_highlights(surface, abs_rect)
        
        # Draw children
        super().draw(surface)
    
    def _draw_board(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the game board background"""
        # Implement in subclasses
        pass
    
    def _draw_pieces(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the game pieces"""
        # Implement in subclasses
        pass
    
    def _draw_highlights(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw highlights for selection, valid moves, etc."""
        # Implement in subclasses
        pass
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if clicking on the board
            if abs_rect.collidepoint(event.pos):
                pos_x, pos_y = event.pos[0] - abs_rect.left, event.pos[1] - abs_rect.top
                board_x, board_y = self.screen_to_board(pos_x, pos_y)
                
                # Handle click on board
                return self._handle_board_click(board_x, board_y, event)
        
        return False
    
    def _handle_board_click(self, board_x: int, board_y: int, event: pygame.event.Event) -> bool:
        """Handle a click on the board"""
        # Implement in subclasses
        return False
    
    def update_from_game(self):
        """Update the board state from the game"""
        # Implement in subclasses
        pass

class ChessBoard(GameBoard):
    """Chess board UI for the Chess game"""
    
    def __init__(self, x: int, y: int, width: int, height: int, game: ChessGame, parent=None):
        """
        Initialize a chess board.
        
        Args:
            x: X position
            y: Y position
            width: Board width
            height: Board height
            game: ChessGame instance
            parent: Parent UI element
        """
        super().__init__(x, y, width, height, game, parent)
        self.cell_size = self.board_size // 8
        self.possible_moves = []
        self.board_style = config.board_style
        
        # Load piece images
        self._load_piece_images()
    
    def _load_piece_images(self):
        """Load chess piece images"""
        piece_set = config.piece_set
        piece_dir = IMAGE_DIR / "chess" / piece_set
        
        if not piece_dir.exists():
            # Fall back to default piece set
            piece_set = "default"
            piece_dir = IMAGE_DIR / "chess" / piece_set
            
            if not piece_dir.exists():
                # Create directory if it doesn't exist
                piece_dir.mkdir(parents=True, exist_ok=True)
        
        self.piece_images = {}
        
        # Try to load piece images
        for color in ["white", "black"]:
            for piece in ["pawn", "rook", "knight", "bishop", "queen", "king"]:
                image_path = piece_dir / f"{color}_{piece}.png"
                
                try:
                    if image_path.exists():
                        image = pygame.image.load(str(image_path))
                        self.piece_images[f"{color}_{piece}"] = image
                    else:
                        # Create placeholder images
                        self._create_placeholder_piece_image(color, piece)
                except Exception as e:
                    logger.error(f"Failed to load piece image {image_path}: {e}")
                    # Create placeholder images
                    self._create_placeholder_piece_image(color, piece)
        
        # Resize piece images
        self._resize_piece_images()
    
    def _create_placeholder_piece_image(self, color: str, piece: str):
        """Create a placeholder piece image"""
        size = 64  # Placeholder size
        image = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Set colors
        bg_color = (255, 255, 255) if color == "white" else (0, 0, 0)
        fg_color = (0, 0, 0) if color == "white" else (255, 255, 255)
        
        # Draw piece shape
        pygame.draw.circle(image, bg_color, (size // 2, size // 2), size // 2 - 2)
        pygame.draw.circle(image, fg_color, (size // 2, size // 2), size // 2 - 2, 2)
        
        # Draw piece symbol
        font = pygame.font.SysFont("Arial", size // 2, bold=True)
        
        symbols = {
            "pawn": "P",
            "rook": "R",
            "knight": "N",
            "bishop": "B",
            "queen": "Q",
            "king": "K"
        }
        
        symbol = symbols.get(piece, "?")
        text = font.render(symbol, True, fg_color)
        text_rect = text.get_rect(center=(size // 2, size // 2))
        image.blit(text, text_rect)
        
        # Store the image
        self.piece_images[f"{color}_{piece}"] = image
    
    def _resize_piece_images(self):
        """Resize piece images to fit the current cell size"""
        piece_size = int(self.cell_size * 0.85)  # Slightly smaller than the cell
        
        for key, image in self.piece_images.items():
            self.piece_images[key] = pygame.transform.scale(image, (piece_size, piece_size))
    
    def screen_to_board(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to board coordinates (0-7, 0-7)"""
        if not (self.board_offset_x <= screen_x < self.board_offset_x + self.board_size and
                self.board_offset_y <= screen_y < self.board_offset_y + self.board_size):
            return (-1, -1)
        
        file = (screen_x - self.board_offset_x) // self.cell_size
        rank = (screen_y - self.board_offset_y) // self.cell_size
        
        # Flip coordinates if board is flipped
        if self.flip_board:
            file = 7 - file
            rank = 7 - rank
            
        return (file, rank)
    
    def board_to_screen(self, file: int, rank: int) -> Tuple[int, int]:
        """Convert board coordinates (0-7, 0-7) to screen coordinates"""
        # Flip coordinates if board is flipped
        if self.flip_board:
            file = 7 - file
            rank = 7 - rank
            
        x = self.board_offset_x + file * self.cell_size
        y = self.board_offset_y + rank * self.cell_size
        
        return (x, y)
    
    def _draw_board(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the chess board background"""
        # Draw the outer border
        border_rect = pygame.Rect(
            abs_rect.left + self.board_offset_x - 10,
            abs_rect.top + self.board_offset_y - 10,
            self.board_size + 20,
            self.board_size + 20
        )
        pygame.draw.rect(surface, Theme.get(self.theme, "border"), border_rect, border_radius=4)
        
        # Draw the cells
        for rank in range(8):
            for file in range(8):
                x, y = self.board_to_screen(file, rank)
                cell_rect = pygame.Rect(
                    abs_rect.left + x,
                    abs_rect.top + y,
                    self.cell_size,
                    self.cell_size
                )
                
                # Cell color (alternating light/dark)
                color = Theme.get(self.theme, "board_light") if (file + rank) % 2 == 0 else Theme.get(self.theme, "board_dark")
                pygame.draw.rect(surface, color, cell_rect)
        
        # Draw coordinates
        font = pygame.font.SysFont("Arial", 12)
        
        for i in range(8):
            # Ranks (1-8)
            rank_text = str(8 - i) if not self.flip_board else str(i + 1)
            rank_surface = font.render(rank_text, True, Theme.get(self.theme, "text"))
            rank_x = abs_rect.left + self.board_offset_x - 20
            rank_y = abs_rect.top + self.board_offset_y + i * self.cell_size + self.cell_size // 2 - rank_surface.get_height() // 2
            surface.blit(rank_surface, (rank_x, rank_y))
            
            # Files (a-h)
            file_text = chr(97 + i) if not self.flip_board else chr(97 + (7 - i))
            file_surface = font.render(file_text, True, Theme.get(self.theme, "text"))
            file_x = abs_rect.left + self.board_offset_x + i * self.cell_size + self.cell_size // 2 - file_surface.get_width() // 2
            file_y = abs_rect.top + self.board_offset_y + self.board_size + 10
            surface.blit(file_surface, (file_x, file_y))
    
    def _draw_pieces(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the chess pieces"""
        # Draw pieces on the board
        for rank in range(8):
            for file in range(8):
                piece = self.game.board[rank][file]
                if piece.type != PieceType.EMPTY and (file, rank) not in self.animated_pieces:
                    self._draw_piece(surface, abs_rect, piece, file, rank)
        
        # Draw animated pieces
        for (file, rank), piece_info in list(self.animated_pieces.items()):
            piece, start_pos, end_pos, progress = piece_info
            
            if progress >= 1.0:
                # Animation complete, remove from animated pieces
                del self.animated_pieces[(file, rank)]
                
                # Draw the piece at its final position
                if piece.type != PieceType.EMPTY:
                    self._draw_piece(surface, abs_rect, piece, file, rank)
            else:
                # Interpolate position
                start_x, start_y = self.board_to_screen(start_pos[0], start_pos[1])
                end_x, end_y = self.board_to_screen(end_pos[0], end_pos[1])
                
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress
                
                self._draw_piece_at_pos(surface, abs_rect, piece, current_x, current_y)
    
    def _draw_piece(self, surface: pygame.Surface, abs_rect: pygame.Rect, piece: ChessPiece, file: int, rank: int):
        """Draw a chess piece at board coordinates"""
        x, y = self.board_to_screen(file, rank)
        self._draw_piece_at_pos(surface, abs_rect, piece, x, y)
    
    def _draw_piece_at_pos(self, surface: pygame.Surface, abs_rect: pygame.Rect, piece: ChessPiece, x: float, y: float):
        """Draw a chess piece at specific screen coordinates"""
        piece_type = piece.type.value.lower()
        color = "white" if piece.color == PieceColor.WHITE else "black"
        
        image_key = f"{color}_{piece_type}"
        if image_key in self.piece_images:
            image = self.piece_images[image_key]
            image_rect = image.get_rect(center=(
                abs_rect.left + x + self.cell_size // 2,
                abs_rect.top + y + self.cell_size // 2
            ))
            surface.blit(image, image_rect)
    
    def _draw_highlights(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw highlights for selection, valid moves, etc."""
        # Draw last move highlight
        if self.show_last_move and self.last_move:
            from_file, from_rank = self.last_move["from_x"], self.last_move["from_y"]
            to_file, to_rank = self.last_move["to_x"], self.last_move["to_y"]
            
            # Source square
            x, y = self.board_to_screen(from_file, from_rank)
            from_rect = pygame.Rect(
                abs_rect.left + x,
                abs_rect.top + y,
                self.cell_size,
                self.cell_size
            )
            
            # Destination square
            x, y = self.board_to_screen(to_file, to_rank)
            to_rect = pygame.Rect(
                abs_rect.left + x,
                abs_rect.top + y,
                self.cell_size,
                self.cell_size
            )
            
            # Semi-transparent yellow highlight
            for rect in [from_rect, to_rect]:
                highlight_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                highlight_surface.fill((255, 255, 0, 64))  # Yellow with alpha
                surface.blit(highlight_surface, rect.topleft)
        
        # Draw selected cell
        if self.selected_cell:
            file, rank = self.selected_cell
            x, y = self.board_to_screen(file, rank)
            
            selected_rect = pygame.Rect(
                abs_rect.left + x,
                abs_rect.top + y,
                self.cell_size,
                self.cell_size
            )
            
            # Draw selected highlight
            pygame.draw.rect(surface, (0, 255, 0, 128), selected_rect, 3)
            
            # Draw valid moves
            if self.show_valid_moves and self.possible_moves:
                for move in self.possible_moves:
                    if move.get("from_x") == file and move.get("from_y") == rank:
                        to_file, to_rank = move.get("to_x"), move.get("to_y")
                        x, y = self.board_to_screen(to_file, to_rank)
                        
                        move_rect = pygame.Rect(
                            abs_rect.left + x,
                            abs_rect.top + y,
                            self.cell_size,
                            self.cell_size
                        )
                        
                        # Different highlight for capture vs. normal move
                        target_piece = self.game.board[to_rank][to_file]
                        if target_piece.type != PieceType.EMPTY:
                            # Capture - red circle
                            pygame.draw.circle(
                                surface,
                                (255, 0, 0, 128),
                                move_rect.center,
                                self.cell_size // 3,
                                3
                            )
                        else:
                            # Normal move - green dot
                            pygame.draw.circle(
                                surface,
                                (0, 255, 0, 128),
                                move_rect.center,
                                self.cell_size // 6
                            )
    
    def _handle_board_click(self, file: int, rank: int, event: pygame.event.Event) -> bool:
        """Handle a click on the board"""
        if not 0 <= file < 8 or not 0 <= rank < 8:
            return False
            
        # Get the piece at the clicked cell
        piece = self.game.board[rank][file]
        
        # If no piece is selected, select this one if it belongs to the current player
        if not self.selected_cell:
            if piece.type != PieceType.EMPTY and piece.color == self.game.current_player.color:
                self.selected_cell = (file, rank)
                self.update_possible_moves()
                return True
        else:
            # If a piece is already selected
            selected_file, selected_rank = self.selected_cell
            selected_piece = self.game.board[selected_rank][selected_file]
            
            # If clicking on own piece, select that piece instead
            if piece.type != PieceType.EMPTY and piece.color == self.game.current_player.color:
                self.selected_cell = (file, rank)
                self.update_possible_moves()
                return True
            
            # Try to move the selected piece to the clicked cell
            move = {
                "from_x": selected_file,
                "from_y": selected_rank,
                "to_x": file,
                "to_y": rank
            }
            
            # Check for special moves (promotion, castling, en passant)
            if selected_piece.type == PieceType.PAWN and (rank == 0 or rank == 7):
                # Promotion
                move["promotion"] = "Q"  # Queen by default
            
            # Check if the move is valid
            if self.game.is_valid_move(move):
                # Store the last move before making it
                self.last_move = move
                
                # Animate the piece movement
                self._animate_move(selected_file, selected_rank, file, rank, selected_piece)
                
                # Make the move
                self.game.make_move(move)
                
                # Clear selection
                self.selected_cell = None
                self.possible_moves = []
                
                # Update the board from the game
                self.update_from_game()
                
                return True
            
        return False
    
    def _animate_move(self, from_file: int, from_rank: int, to_file: int, to_rank: int, piece: ChessPiece):
        """Animate a piece movement"""
        # Create a copy of the piece for animation
        piece_copy = ChessPiece(piece.type, piece.color)
        piece_copy.has_moved = piece.has_moved
        
        # Add to animated pieces
        self.animated_pieces[(to_file, to_rank)] = (piece_copy, (from_file, from_rank), (to_file, to_rank), 0.0)
        
        # Start animation
        animation_key = f"piece_{to_file}_{to_rank}"
        self.animation_manager.start_animation(
            animation_key,
            0.0,
            1.0,
            0.3,  # Duration
            "ease_out",
            lambda _: self._animation_complete(to_file, to_rank)
        )
    
    def _animation_complete(self, file: int, rank: int):
        """Called when a piece animation is complete"""
        pass
    
    def update_from_game(self):
        """Update the board state from the game"""
        # Update possible moves if a piece is selected
        if self.selected_cell:
            self.update_possible_moves()
    
    def update_possible_moves(self):
        """Update the list of possible moves for the selected piece"""
        if not self.selected_cell:
            self.possible_moves = []
            return
            
        file, rank = self.selected_cell
        
        # Get all valid moves
        all_moves = self.game.get_valid_moves()
        
        # Filter moves for the selected piece
        self.possible_moves = [move for move in all_moves 
                              if move.get("from_x") == file and move.get("from_y") == rank]
    
    def set_flip_board(self, flip: bool):
        """Set whether to flip the board perspective"""
        self.flip_board = flip

class GoBoard(GameBoard):
    """Go board UI for the Go game"""
    
    def __init__(self, x: int, y: int, width: int, height: int, game: GoGame, parent=None):
        """
        Initialize a Go board.
        
        Args:
            x: X position
            y: Y position
            width: Board width
            height: Board height
            game: GoGame instance
            parent: Parent UI element
        """
        super().__init__(x, y, width, height, game, parent)
        self.board_style = config.board_style
        self.stone_images = {}
        self.hover_pos = None
        
        # Load stone images
        self._load_stone_images()
        
        # Calculate cell size based on board size
        self._update_dimensions()
    
    def _update_dimensions(self):
        """Update board dimensions based on available space and game board size"""
        self.board_size = min(self.rect.width, self.rect.height)
        
        # Calculate cell size - make room for the grid lines extending to the edges
        grid_size = self.game.board_size - 1
        self.cell_size = self.board_size / grid_size
        
        # Calculate board offset - center the board
        self.board_offset_x = (self.rect.width - self.board_size) // 2
        self.board_offset_y = (self.rect.height - self.board_size) // 2
        
        # Resize stone images
        self._resize_stone_images()
    
    def _load_stone_images(self):
        """Load Go stone images"""
        stone_set = config.piece_set
        stone_dir = IMAGE_DIR / "go" / stone_set
        
        if not stone_dir.exists():
            # Fall back to default stone set
            stone_set = "default"
            stone_dir = IMAGE_DIR / "go" / stone_set
            
            if not stone_dir.exists():
                # Create directory if it doesn't exist
                stone_dir.mkdir(parents=True, exist_ok=True)
        
        self.stone_images = {}
        
        # Try to load stone images
        for color in ["black", "white"]:
            image_path = stone_dir / f"{color}_stone.png"
            
            try:
                if image_path.exists():
                    image = pygame.image.load(str(image_path))
                    self.stone_images[color] = image
                else:
                    # Create placeholder images
                    self._create_placeholder_stone_image(color)
            except Exception as e:
                logger.error(f"Failed to load stone image {image_path}: {e}")
                # Create placeholder images
                self._create_placeholder_stone_image(color)
    
    def _create_placeholder_stone_image(self, color: str):
        """Create a placeholder stone image"""
        size = 64  # Placeholder size
        image = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Set colors
        stone_color = (0, 0, 0) if color == "black" else (255, 255, 255)
        border_color = (100, 100, 100)
        
        # Draw stone circle
        pygame.draw.circle(image, stone_color, (size // 2, size // 2), size // 2 - 2)
        pygame.draw.circle(image, border_color, (size // 2, size // 2), size // 2 - 2, 1)
        
        # Add highlight for white stones
        if color == "white":
            highlight_pos = (size // 2 - size // 6, size // 2 - size // 6)
            highlight_radius = size // 8
            pygame.draw.circle(image, (220, 220, 220), highlight_pos, highlight_radius)
        
        # Store the image
        self.stone_images[color] = image
    
    def _resize_stone_images(self):
        """Resize stone images to fit the current cell size"""
        stone_size = int(self.cell_size * 0.85)  # Slightly smaller than the cell
        
       """
Advanced Pygame UI for the Games Kernel

This module provides a comprehensive graphical user interface for various board games
using Pygame. It supports Chess, Go, and Mahjong games with customizable settings, 
responsive design, AI opponents, game analytics, and Sully cognitive integration.
"""

import pygame
import sys
import os
import logging
import json
import time
import math
import random
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='games_visualizer.log'
)
logger = logging.getLogger("games.visualizer")

# Try to import game modules, with graceful error handling
try:
    # Import the games kernel
    from games import (
        GameEngine, Player, Game, AIPlayer, GameDifficulty, GameAnalytics,
        MahjongGame, MahjongPlayer, MahjongAIPlayer, Wind, TileType, Tile,
        ChessGame, ChessPlayer, ChessAIPlayer, PieceColor, PieceType, ChessPiece,
        GoGame, GoPlayer, GoAIPlayer, Stone
    )
    GAMES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import game modules: {e}")
    print(f"Error: Could not import required game modules. {e}")
    print("Please ensure the games.py module is available.")
    GAMES_AVAILABLE = False

# Try to import integration with Sully AI
try:
    from sully import Sully
    SULLY_AVAILABLE = True
except ImportError:
    logger.warning("Sully integration not available.")
    SULLY_AVAILABLE = False

# Initialize pygame if available
try:
    pygame.init()
    PYGAME_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to initialize pygame: {e}")
    PYGAME_AVAILABLE = False

# Get screen info for responsive design
if PYGAME_AVAILABLE:
    try:
        display_info = pygame.display.Info()
        DEFAULT_SCREEN_WIDTH = min(1280, display_info.current_w - 100)
        DEFAULT_SCREEN_HEIGHT = min(900, display_info.current_h - 100)
    except:
        DEFAULT_SCREEN_WIDTH = 1280
        DEFAULT_SCREEN_HEIGHT = 900
else:
    DEFAULT_SCREEN_WIDTH = 1280
    DEFAULT_SCREEN_HEIGHT = 900

# Default asset paths
ASSET_DIR = Path("assets")
FONT_DIR = ASSET_DIR / "fonts"
IMAGE_DIR = ASSET_DIR / "images"
SOUND_DIR = ASSET_DIR / "sounds"
SAVE_DIR = Path("saves")

# Ensure directories exist
for directory in [ASSET_DIR, FONT_DIR, IMAGE_DIR, SOUND_DIR, SAVE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Theme definitions
class Theme:
    """Theme colors and settings"""
    
    THEMES = {
        "light": {
            "background": (240, 240, 240),
            "panel": (225, 225, 225),
            "text": (10, 10, 10),
            "text_secondary": (60, 60, 60),
            "accent": (66, 135, 245),
            "accent_hover": (56, 115, 225),
            "button": (210, 210, 210),
            "button_hover": (190, 190, 190),
            "button_text": (10, 10, 10),
            "board_light": (240, 230, 210),
            "board_dark": (180, 140, 100),
            "highlight": (100, 200, 100, 180),
            "warning": (245, 66, 66),
            "success": (66, 245, 132),
            "border": (200, 200, 200)
        },
        "dark": {
            "background": (40, 40, 40),
            "panel": (60, 60, 60),
            "text": (230, 230, 230),
            "text_secondary": (180, 180, 180),
            "accent": (66, 135, 245),
            "accent_hover": (96, 165, 255),
            "button": (80, 80, 80),
            "button_hover": (100, 100, 100),
            "button_text": (230, 230, 230),
            "board_light": (180, 170, 150),
            "board_dark": (90, 70, 50),
            "highlight": (100, 200, 100, 180),
            "warning": (245, 66, 66),
            "success": (66, 245, 132),
            "border": (100, 100, 100)
        },
        "blue": {
            "background": (225, 235, 245),
            "panel": (210, 225, 240),
            "text": (20, 40, 80),
            "text_secondary": (60, 80, 120),
            "accent": (30, 90, 180),
            "accent_hover": (40, 100, 200),
            "button": (190, 210, 230),
            "button_hover": (170, 190, 220),
            "button_text": (20, 40, 80),
            "board_light": (220, 230, 240),
            "board_dark": (90, 130, 180),
            "highlight": (100, 200, 100, 180),
            "warning": (245, 66, 66),
            "success": (66, 245, 132),
            "border": (170, 190, 210)
        },
        "wood": {
            "background": (235, 225, 205),
            "panel": (220, 210, 190),
            "text": (60, 40, 20),
            "text_secondary": (100, 80, 60),
            "accent": (150, 100, 50),
            "accent_hover": (170, 120, 70),
            "button": (200, 180, 160),
            "button_hover": (180, 160, 140),
            "button_text": (60, 40, 20),
            "board_light": (230, 200, 170),
            "board_dark": (160, 120, 80),
            "highlight": (100, 200, 100, 180),
            "warning": (245, 66, 66),
            "success": (66, 245, 132),
            "border": (180, 160, 140)
        }
    }
    
    @staticmethod
    def get(theme_name: str, key: str, default: Any = None) -> Any:
        """Get a theme color or setting"""
        theme = Theme.THEMES.get(theme_name, Theme.THEMES["light"])
        return theme.get(key, default)

# Game configuration
@dataclass
class GameConfig:
    """Configuration settings for the game visualizer."""
    screen_width: int = DEFAULT_SCREEN_WIDTH
    screen_height: int = DEFAULT_SCREEN_HEIGHT
    theme: str = "light"
    fullscreen: bool = False
    music_enabled: bool = True
    sound_enabled: bool = True
    fps: int = 60
    animation_speed: float = 1.0
    show_valid_moves: bool = True
    show_last_move: bool = True
    auto_save: bool = True
    auto_save_interval: int = 5  # minutes
    piece_set: str = "default"
    board_style: str = "default"
    debug: bool = False
    sully_integration: bool = True
    ai_thinking_time: float = 1.0
    locale: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        """Create from dictionary"""
        # Filter out unknown attributes
        known_fields = set(cls.__annotations__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

# Try to load config from file
CONFIG_PATH = Path("config.json")
config = GameConfig()

try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config_data = json.load(f)
            config = GameConfig.from_dict(config_data)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
except Exception as e:
    logger.warning(f"Failed to load configuration: {e}")
    logger.info("Using default configuration")

# Save config function
def save_config():
    """Save the current configuration to a file"""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")

# Constants
SCREEN_WIDTH = config.screen_width
SCREEN_HEIGHT = config.screen_height
FPS = config.fps

# Helper classes for the UI

class AnimationManager:
    """Manages animations for UI elements"""
    
    def __init__(self):
        self.animations = {}
        self.animation_speed = config.animation_speed
    
    def start_animation(self, key: str, start_value: Any, end_value: Any, 
                       duration: float, easing: str = "linear", 
                       callback: Optional[Callable] = None):
        """Start a new animation"""
        self.animations[key] = {
            "start_value": start_value,
            "end_value": end_value,
            "start_time": time.time(),
            "duration": duration,
            "easing": easing,
            "callback": callback
        }
    
    def update(self):
        """Update all animations"""
        current_time = time.time()
        completed = []
        
        for key, anim in self.animations.items():
            elapsed = (current_time - anim["start_time"]) * self.animation_speed
            progress = min(1.0, elapsed / anim["duration"])
            
            # Apply easing function
            if anim["easing"] == "ease_in":
                t = progress * progress
            elif anim["easing"] == "ease_out":
                t = 1 - (1 - progress) * (1 - progress)
            elif anim["easing"] == "ease_in_out":
                t = 0.5 - 0.5 * math.cos(math.pi * progress)
            else:  # linear
                t = progress
            
            # Mark as completed if done
            if progress >= 1.0:
                completed.append(key)
                # Call callback if provided
                if anim["callback"]:
                    anim["callback"](anim["end_value"])
        
        # Remove completed animations
        for key in completed:
            self.animations.pop(key)
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get the current value of an animation"""
        if key not in self.animations:
            return default
            
        anim = self.animations[key]
        elapsed = (time.time() - anim["start_time"]) * self.animation_speed
        progress = min(1.0, elapsed / anim["duration"])
        
        # Apply easing function
        if anim["easing"] == "ease_in":
            t = progress * progress
        elif anim["easing"] == "ease_out":
            t = 1 - (1 - progress) * (1 - progress)
        elif anim["easing"] == "ease_in_out":
            t = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:  # linear
            t = progress
        
        # Interpolate value
        if isinstance(anim["start_value"], (int, float)) and isinstance(anim["end_value"], (int, float)):
            return anim["start_value"] + (anim["end_value"] - anim["start_value"]) * t
        elif isinstance(anim["start_value"], tuple) and isinstance(anim["end_value"], tuple):
            # For colors or other tuples
            return tuple(s + (e - s) * t for s, e in zip(anim["start_value"], anim["end_value"]))
        else:
            # For other types, just return the end value when animation completes
            return anim["end_value"] if progress >= 1.0 else anim["start_value"]
    
    def is_animating(self, key: str) -> bool:
        """Check if an animation is in progress"""
        return key in self.animations

class SoundManager:
    """Manages sound effects and music"""
    
    def __init__(self):
        self.sounds = {}
        self.music = None
        self.current_music = None
        self.sound_enabled = config.sound_enabled
        self.music_enabled = config.music_enabled
        
        # Initialize if pygame is available
        if not PYGAME_AVAILABLE:
            return
            
        # Try to initialize the mixer
        try:
            pygame.mixer.init()
            
            # Load common sounds
            self.load_sound("click", "click.wav")
            self.load_sound("move", "move.wav")
            self.load_sound("capture", "capture.wav")
            self.load_sound("check", "check.wav")
            self.load_sound("win", "win.wav")
            self.load_sound("lose", "lose.wav")
            self.load_sound("draw", "draw.wav")
            self.load_sound("error", "error.wav")
            
            # Load background music
            self.load_music("menu", "menu.mp3")
            self.load_music("gameplay", "gameplay.mp3")
        except Exception as e:
            logger.error(f"Failed to initialize sound: {e}")
            self.sound_enabled = False
            self.music_enabled = False
    
    def load_sound(self, name: str, filename: str):
        """Load a sound effect"""
        if not self.sound_enabled or not PYGAME_AVAILABLE:
            return
            
        try:
            file_path = SOUND_DIR / filename
            if file_path.exists():
                self.sounds[name] = pygame.mixer.Sound(str(file_path))
            else:
                logger.warning(f"Sound file not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load sound {name}: {e}")
    
    def load_music(self, name: str, filename: str):
        """Load background music"""
        if not self.music_enabled or not PYGAME_AVAILABLE:
            return
            
        try:
            file_path = SOUND_DIR / filename
            if file_path.exists():
                # We don't actually load the music yet, just store the path
                self.music = {name: str(file_path)}
            else:
                logger.warning(f"Music file not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load music {name}: {e}")
    
    def play_sound(self, name: str):
        """Play a sound effect"""
        if not self.sound_enabled or not PYGAME_AVAILABLE:
            return
            
        if name in self.sounds:
            try:
                self.sounds[name].play()
            except Exception as e:
                logger.error(f"Failed to play sound {name}: {e}")
    
    def play_music(self, name: str, loops: int = -1):
        """Play background music"""
        if not self.music_enabled or not PYGAME_AVAILABLE:
            return
            
        if name in self.music:
            try:
                if self.current_music != name:
                    pygame.mixer.music.load(self.music[name])
                    pygame.mixer.music.play(loops)
                    self.current_music = name
            except Exception as e:
                logger.error(f"Failed to play music {name}: {e}")
    
    def stop_music(self):
        """Stop background music"""
        if not PYGAME_AVAILABLE:
            return
            
        try:
            pygame.mixer.music.stop()
            self.current_music = None
        except Exception as e:
            logger.error(f"Failed to stop music: {e}")
    
    def set_sound_enabled(self, enabled: bool):
        """Enable or disable sound effects"""
        self.sound_enabled = enabled
    
    def set_music_enabled(self, enabled: bool):
        """Enable or disable background music"""
        self.music_enabled = enabled
        if not enabled and self.current_music:
            self.stop_music()

class UIElement:
    """Base class for UI elements"""
    
    def __init__(self, x: int, y: int, width: int, height: int, parent=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.parent = parent
        self.visible = True
        self.enabled = True
        self.children = []
        self.hover = False
        self.theme = config.theme
    
    def get_absolute_rect(self) -> pygame.Rect:
        """Get the absolute rectangle coordinates"""
        if self.parent:
            parent_rect = self.parent.get_absolute_rect()
            return pygame.Rect(
                parent_rect.x + self.rect.x,
                parent_rect.y + self.rect.y,
                self.rect.width,
                self.rect.height
            )
        return self.rect
    
    def add_child(self, child: 'UIElement'):
        """Add a child UI element"""
        self.children.append(child)
        child.parent = self
    
    def update(self, mouse_pos: Tuple[int, int], mouse_buttons: Tuple[bool, bool, bool]):
        """Update the UI element"""
        if not self.visible or not self.enabled:
            return
            
        absolute_rect = self.get_absolute_rect()
        self.hover = absolute_rect.collidepoint(mouse_pos)
        
        for child in self.children:
            child.update(mouse_pos, mouse_buttons)
    
    def draw(self, surface: pygame.Surface):
        """Draw the UI element"""
        if not self.visible:
            return
            
        for child in self.children:
            child.draw(surface)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if not self.visible or not self.enabled:
            return False
            
        for child in reversed(self.children):  # Reverse so later children (drawn on top) get events first
            if child.handle_event(event):
                return True
                
        return False

class Button(UIElement):
    """A clickable button for the GUI"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 callback: Optional[Callable] = None, parent=None,
                 color: Optional[Tuple[int, int, int]] = None, 
                 text_color: Optional[Tuple[int, int, int]] = None,
                 hover_color: Optional[Tuple[int, int, int]] = None,
                 border_radius: int = 4,
                 icon: Optional[str] = None,
                 tooltip: Optional[str] = None):
        """
        Initialize a button.
        
        Args:
            x: X position
            y: Y position
            width: Button width
            height: Button height
            text: Button text
            callback: Function to call when clicked
            parent: Parent UI element
            color: Button color (or None for theme default)
            text_color: Text color (or None for theme default)
            hover_color: Color when mouse hovers over button (or None for theme default)
            border_radius: Radius for rounded corners
            icon: Optional icon name
            tooltip: Optional tooltip text
        """
        super().__init__(x, y, width, height, parent)
        self.text = text
        self.callback = callback
        self.color = color
        self.text_color = text_color
        self.hover_color = hover_color
        self.border_radius = border_radius
        self.icon = icon
        self.tooltip = tooltip
        self.pressed = False
        self.font = None
        self.icon_surface = None
        self.disabled_color = None
        self.disabled_text_color = None
        
        # Load font if pygame is available
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont("Arial", 16)
            
            # Load icon if specified
            if self.icon:
                self._load_icon()
    
    def _load_icon(self):
        """Load icon image"""
        icon_path = IMAGE_DIR / f"{self.icon}.png"
        if icon_path.exists():
            try:
                self.icon_surface = pygame.image.load(str(icon_path))
                # Scale the icon to fit the button
                icon_size = min(self.rect.height - 8, 24)
                self.icon_surface = pygame.transform.scale(self.icon_surface, (icon_size, icon_size))
            except Exception as e:
                logger.error(f"Failed to load icon {self.icon}: {e}")
                self.icon_surface = None
    
    def update(self, mouse_pos: Tuple[int, int], mouse_buttons: Tuple[bool, bool, bool]):
        """Update button state"""
        super().update(mouse_pos, mouse_buttons)
        
        if not self.enabled:
            return
            
        if self.hover and mouse_buttons[0]:  # Left mouse button pressed
            self.pressed = True
        elif not mouse_buttons[0] and self.pressed:
            if self.hover and self.callback:
                self.callback()
            self.pressed = False
    
    def draw(self, surface: pygame.Surface):
        """Draw the button"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        # Determine colors based on theme and state
        if not self.enabled:
            bg_color = self.disabled_color or tuple(max(0, c - 40) for c in Theme.get(self.theme, "button"))
            text_color = self.disabled_text_color or tuple(max(0, c - 40) for c in Theme.get(self.theme, "button_text"))
        elif self.pressed:
            bg_color = self.hover_color or Theme.get(self.theme, "button_hover")
            text_color = self.text_color or Theme.get(self.theme, "button_text")
        elif self.hover:
            bg_color = self.hover_color or Theme.get(self.theme, "button_hover")
            text_color = self.text_color or Theme.get(self.theme, "button_text")
        else:
            bg_color = self.color or Theme.get(self.theme, "button")
            text_color = self.text_color or Theme.get(self.theme, "button_text")
        
        # Draw the button
        pygame.draw.rect(surface, bg_color, abs_rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, Theme.get(self.theme, "border"), abs_rect, 1, border_radius=self.border_radius)
        
        # Draw the icon if available
        icon_padding = 0
        if self.icon_surface:
            icon_x = abs_rect.left + 5
            icon_y = abs_rect.top + (abs_rect.height - self.icon_surface.get_height()) // 2
            surface.blit(self.icon_surface, (icon_x, icon_y))
            icon_padding = self.icon_surface.get_width() + 5
        
        # Draw the text
        if self.text:
            text_surface = self.font.render(self.text, True, text_color)
            text_rect = text_surface.get_rect()
            
            if self.icon_surface:
                # Center the text to the right of the icon
                text_rect.midleft = (abs_rect.left + icon_padding, abs_rect.centery)
            else:
                # Center the text
                text_rect.center = abs_rect.center
                
            surface.blit(text_surface, text_rect)
        
        # Draw children
        super().draw(surface)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if abs_rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and abs_rect.collidepoint(event.pos) and self.callback:
                self.callback()
                return True
        
        return False
    
    def set_text(self, text: str):
        """Set the button text"""
        self.text = text
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the button"""
        self.enabled = enabled
    
    def set_icon(self, icon: str):
        """Set the button icon"""
        self.icon = icon
        self._load_icon()

class Label(UIElement):
    """A text label for the GUI"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 parent=None, font_size: int = 16, color: Optional[Tuple[int, int, int]] = None,
                 align: str = "left", bold: bool = False, wrap: bool = False):
        """
        Initialize a label.
        
        Args:
            x: X position
            y: Y position
            width: Label width
            height: Label height
            text: Label text
            parent: Parent UI element
            font_size: Font size
            color: Text color (or None for theme default)
            align: Text alignment ("left", "center", "right")
            bold: Whether the text should be bold
            wrap: Whether to wrap the text
        """
        super().__init__(x, y, width, height, parent)
        self.text = text
        self.font_size = font_size
        self.color = color
        self.align = align
        self.bold = bold
        self.wrap = wrap
        self.font = None
        self.text_surfaces = []
        
        # Load font if pygame is available
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont("Arial", font_size, bold=bold)
            self._prepare_text()
    
    def _prepare_text(self):
        """Prepare the text for rendering"""
        if not self.font:
            return
            
        self.text_surfaces = []
        
        if not self.wrap:
            text_surface = self.font.render(self.text, True, self.color or Theme.get(self.theme, "text"))
            self.text_surfaces.append(text_surface)
        else:
            # Wrap text to fit the width
            words = self.text.split(' ')
            current_line = ""
            
            for word in words:
                test_line = current_line + word + " "
                test_surface = self.font.render(test_line, True, self.color or Theme.get(self.theme, "text"))
                
                if test_surface.get_width() <= self.rect.width:
                    current_line = test_line
                else:
                    # Line is too long, add current line and start a new one
                    if current_line:
                        line_surface = self.font.render(current_line, True, self.color or Theme.get(self.theme, "text"))
                        self.text_surfaces.append(line_surface)
                    current_line = word + " "
            
            # Add the last line
            if current_line:
                line_surface = self.font.render(current_line, True, self.color or Theme.get(self.theme, "text"))
                self.text_surfaces.append(line_surface)
    
    def draw(self, surface: pygame.Surface):
        """Draw the label"""
        if not self.visible:
            return
            
        abs_rect = self.get_absolute_rect()
        
        if not self.text_surfaces:
            self._prepare_text()
        
        if not self.wrap:
            # Draw single line text
            if self.text_surfaces:
                text_surface = self.text_surfaces[0]
                text_rect = text_surface.get_rect()
                
                if self.align == "center":
                    text_rect.midtop = (abs_rect.centerx, abs_rect.top)
                elif self.align == "right":
                    text_rect.topright = (abs_rect.right, abs_rect.top)
                else:  # left
                    text_rect.topleft = abs_rect.topleft
                    
                surface.blit(text_surface, text_rect)
        else:
            # Draw wrapped text
            y_offset = 0
            for text_surface in self.text_surfaces:
                text_rect = text_surface.get_rect()
                
                if self.align == "center":
                    text_rect.midtop = (abs_rect.centerx, abs_rect.top + y_offset)
                elif self.align == "right":
                    text_rect.topright = (abs_rect.right, abs_rect.top + y_offset)
                else:  # left
                    text_rect.topleft = (abs_rect.left, abs_rect.top + y_offset)
                    
                surface.blit(text_surface, text_rect)
                y_offset += text_rect.height
        
        # Draw children
        super().draw(surface)
    
    def set_text(self, text: str):
        """Set the label text"""
        if self.text != text:
            self.text = text
            self._prepare_text()
    
    def set_color(self, color: Tuple[int, int, int]):
        """Set the text color"""
        if self.color != color:
            self.color = color
            self._prepare_text()

class Panel(UIElement):
    """A panel container for other UI elements"""
    
    
self.en_passant_target = previous_state["en_passant_target"]
        self.halfmove_clock = previous_state["halfmove_clock"]
        self.fullmove_number = previous_state["fullmove_number"]
        
        # Update player check status
        self.current_player.in_check = self.check_status
        
        # Remove last move from history
        if self.move_history:
            self.move_history.pop()
        
        # Revert the time control if applicable
        if self.timed_mode:
            # This is a simplification; accurate time tracking during takebacks is complex
            # Just reset the move timer
            self.last_move_time = datetime.now()
        
        return True
    
    def check_game_over(self) -> bool:
        """Check if the game has ended (checkmate, stalemate, or draw)"""
        # If the current player has no valid moves
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            if self.check_status:
                # Checkmate
                self.checkmate_status = True
                return True
            else:
                # Stalemate
                self.stalemate_status = True
                return True
        
        # Check for threefold repetition
        for position in set(self.position_history):
            if self.position_history.count(position) >= 3:
                # Threefold repetition
                self.stalemate_status = True
                return True
        
        # Check for 50-move rule
        if self.halfmove_clock >= 100:  # 50 full moves (each player moves once)
            self.stalemate_status = True
            return True
        
        # Check for insufficient material
        if self._has_insufficient_material():
            self.stalemate_status = True
            return True
        
        return False
    
    def _has_insufficient_material(self) -> bool:
        """Check if there is insufficient material to checkmate"""
        piece_counts = {
            PieceColor.WHITE: {"total": 0, "knights": 0, "bishops": 0, "light_bishops": 0, "dark_bishops": 0},
            PieceColor.BLACK: {"total": 0, "knights": 0, "bishops": 0, "light_bishops": 0, "dark_bishops": 0}
        }
        
        # Count pieces
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type in [PieceType.PAWN, PieceType.ROOK, PieceType.QUEEN]:
                    # If any of these pieces exist, there's sufficient material
                    return False
                
                if piece.type == PieceType.KNIGHT:
                    piece_counts[piece.color]["knights"] += 1
                    piece_counts[piece.color]["total"] += 1
                elif piece.type == PieceType.BISHOP:
                    piece_counts[piece.color]["bishops"] += 1
                    piece_counts[piece.color]["total"] += 1
                    # Check if bishop is on light or dark square
                    if (x + y) % 2 == 0:
                        piece_counts[piece.color]["light_bishops"] += 1
                    else:
                        piece_counts[piece.color]["dark_bishops"] += 1
        
        # King vs. King
        if piece_counts[PieceColor.WHITE]["total"] == 0 and piece_counts[PieceColor.BLACK]["total"] == 0:
            return True
        
        # King and Bishop vs. King
        if (piece_counts[PieceColor.WHITE]["total"] == 1 and piece_counts[PieceColor.WHITE]["bishops"] == 1 and
            piece_counts[PieceColor.BLACK]["total"] == 0) or (
            piece_counts[PieceColor.BLACK]["total"] == 1 and piece_counts[PieceColor.BLACK]["bishops"] == 1 and
            piece_counts[PieceColor.WHITE]["total"] == 0):
            return True
        
        # King and Knight vs. King
        if (piece_counts[PieceColor.WHITE]["total"] == 1 and piece_counts[PieceColor.WHITE]["knights"] == 1 and
            piece_counts[PieceColor.BLACK]["total"] == 0) or (
            piece_counts[PieceColor.BLACK]["total"] == 1 and piece_counts[PieceColor.BLACK]["knights"] == 1 and
            piece_counts[PieceColor.WHITE]["total"] == 0):
            return True
        
        # King and Bishop vs. King and Bishop (same color bishops)
        if (piece_counts[PieceColor.WHITE]["total"] == 1 and piece_counts[PieceColor.WHITE]["bishops"] == 1 and
            piece_counts[PieceColor.BLACK]["total"] == 1 and piece_counts[PieceColor.BLACK]["bishops"] == 1):
            white_bishop_color = "light" if piece_counts[PieceColor.WHITE]["light_bishops"] > 0 else "dark"
            black_bishop_color = "light" if piece_counts[PieceColor.BLACK]["light_bishops"] > 0 else "dark"
            return white_bishop_color == black_bishop_color
        
        return False
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        return {
            "current_player": self.current_player_idx,
            "board": [[str(piece) for piece in row] for row in self.board],
            "check": self.check_status,
            "checkmate": self.checkmate_status,
            "stalemate": self.stalemate_status,
            "en_passant_target": self.en_passant_target,
            "move_history": self.move_history,
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
            "white_captured": [str(p) for p in self.players[0].captured_pieces],
            "black_captured": [str(p) for p in self.players[1].captured_pieces],
            "player_times": self.player_time if self.timed_mode else {},
            "stats": self.stats
        }
    
    def get_player_view(self, player_idx: int) -> Dict[str, Any]:
        """Get a view of the game state for a specific player"""
        game_state = self.get_game_state()
        
        # Add valid moves if it's this player's turn
        if player_idx == self.current_player_idx:
            game_state["valid_moves"] = self.get_valid_moves()
            
        # Add any player-specific information
        game_state["is_your_turn"] = (player_idx == self.current_player_idx)
        game_state["your_color"] = self.players[player_idx].color.value
        game_state["in_check"] = self.players[player_idx].in_check
        game_state["remaining_time"] = self.player_time.get(self.players[player_idx].id) if self.timed_mode else None
        
        return game_state
    
    def get_board_analysis(self) -> Dict[str, Any]:
        """Get an analysis of the current board state"""
        # Count pieces by type
        piece_counts = {
            "white": {piece_type.name: 0 for piece_type in PieceType if piece_type != PieceType.EMPTY},
            "black": {piece_type.name: 0 for piece_type in PieceType if piece_type != PieceType.EMPTY}
        }
        
        # Count controlled squares
        controlled_squares = {
            "white": set(),
            "black": set()
        }
        
        # Find all pieces and calculate control
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type != PieceType.EMPTY:
                    # Count piece
                    color_key = "white" if piece.color == PieceColor.WHITE else "black"
                    piece_counts[color_key][piece.type.name] += 1
                    
                    # Calculate controlled squares
                    for ty in range(8):
                        for tx in range(8):
                            if (tx, ty) != (x, y) and self._can_piece_move_to(piece, x, y, tx, ty, self.board):
                                controlled_squares[color_key].add((tx, ty))
        
        # Calculate center control (e4, d4, e5, d5)
        center_squares = [(3, 3), (4, 3), (3, 4), (4, 4)]
        center_control = {
            "white": sum(1 for sq in center_squares if sq in controlled_squares["white"]),
            "black": sum(1 for sq in center_squares if sq in controlled_squares["black"])
        }
        
        # Calculate piece mobility
        mobility = {
            "white": len(controlled_squares["white"]),
            "black": len(controlled_squares["black"])
        }
        
        # Get pawn structure information
        pawn_structure = self._analyze_pawn_structure()
        
        # Calculate piece coordination
        piece_coordination = {
            "white": self._calculate_piece_coordination(PieceColor.WHITE),
            "black": self._calculate_piece_coordination(PieceColor.BLACK)
        }
        
        # Calculate king safety
        king_safety = {
            "white": self._calculate_king_safety(PieceColor.WHITE),
            "black": self._calculate_king_safety(PieceColor.BLACK)
        }
        
        return {
            "piece_counts": piece_counts,
            "controlled_squares": {
                "white": len(controlled_squares["white"]),
                "black": len(controlled_squares["black"])
            },
            "center_control": center_control,
            "mobility": mobility,
            "pawn_structure": pawn_structure,
            "piece_coordination": piece_coordination,
            "king_safety": king_safety,
            "material_advantage": self.stats["material_advantage"][-1]["advantage"] if self.stats["material_advantage"] else 0
        }
    
    def _analyze_pawn_structure(self) -> Dict[str, Any]:
        """Analyze the pawn structure"""
        result = {
            "white": {"isolated": 0, "doubled": 0, "backward": 0, "advanced": 0},
            "black": {"isolated": 0, "doubled": 0, "backward": 0, "advanced": 0}
        }
        
        # Count pawns in each file
        white_pawn_files = [0] * 8
        black_pawn_files = [0] * 8
        
        # Record the most advanced pawn in each file
        white_pawn_ranks = [-1] * 8
        black_pawn_ranks = [-1] * 8
        
        # Find all pawns
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type == PieceType.PAWN:
                    if piece.color == PieceColor.WHITE:
                        white_pawn_files[x] += 1
                        white_pawn_ranks[x] = max(white_pawn_ranks[x], y)
                    else:
                        black_pawn_files[x] += 1
                        black_pawn_ranks[x] = max(black_pawn_ranks[x], 7 - y)  # Flip for black
        
        # Check for isolated pawns
        for x in range(8):
            # White pawns
            if white_pawn_files[x] > 0:
                has_neighbor = (x > 0 and white_pawn_files[x-1] > 0) or (x < 7 and white_pawn_files[x+1] > 0)
                if not has_neighbor:
                    result["white"]["isolated"] += white_pawn_files[x]
            
            # Black pawns
            if black_pawn_files[x] > 0:
                has_neighbor = (x > 0 and black_pawn_files[x-1] > 0) or (x < 7 and black_pawn_files[x+1] > 0)
                if not has_neighbor:
                    result["black"]["isolated"] += black_pawn_files[x]
            
            # Doubled pawns
            if white_pawn_files[x] > 1:
                result["white"]["doubled"] += white_pawn_files[x] - 1
            if black_pawn_files[x] > 1:
                result["black"]["doubled"] += black_pawn_files[x] - 1
            
            # Backward pawns (simplified)
            if white_pawn_files[x] > 0:
                is_backward = False
                if x > 0 and white_pawn_ranks[x-1] > white_pawn_ranks[x]:
                    is_backward = True
                if x < 7 and white_pawn_ranks[x+1] > white_pawn_ranks[x]:
                    is_backward = True
                if is_backward:
                    result["white"]["backward"] += 1
            
            if black_pawn_files[x] > 0:
                is_backward = False
                if x > 0 and black_pawn_ranks[x-1] > black_pawn_ranks[x]:
                    is_backward = True
                if x < 7 and black_pawn_ranks[x+1] > black_pawn_ranks[x]:
                    is_backward = True
                if is_backward:
                    result["black"]["backward"] += 1
            
            # Advanced pawns
            if white_pawn_files[x] > 0 and white_pawn_ranks[x] >= 5:
                result["white"]["advanced"] += 1
            if black_pawn_files[x] > 0 and black_pawn_ranks[x] >= 5:
                result["black"]["advanced"] += 1
        
        return result
    
    def _calculate_piece_coordination(self, color: PieceColor) -> float:
        """Calculate a measure of piece coordination"""
        # Count how many pieces are defended by other pieces
        defended_pieces = 0
        total_pieces = 0
        
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type != PieceType.EMPTY and piece.type != PieceType.KING and piece.color == color:
                    total_pieces += 1
                    
                    # Check if this piece is defended
                    is_defended = False
                    for dy in range(8):
                        for dx in range(8):
                            defending_piece = self.board[dy][dx]
                            if defending_piece.type != PieceType.EMPTY and defending_piece.color == color and (dx, dy) != (x, y):
                                if self._can_piece_move_to(defending_piece, dx, dy, x, y, self.board):
                                    is_defended = True
                                    break
                        if is_defended:
                            break
                            
                    if is_defended:
                        defended_pieces += 1
        
        if total_pieces == 0:
            return 0
            
        return defended_pieces / total_pieces
    
    def _calculate_king_safety(self, color: PieceColor) -> float:
        """Calculate a measure of king safety"""
        # Find the king
        king_pos = None
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type == PieceType.KING and piece.color == color:
                    king_pos = (x, y)
                    break
            if king_pos:
                break
                
        if not king_pos:
            return 0  # Should not happen
            
        kx, ky = king_pos
        
        # Check pieces defending the king
        defenders = 0
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type != PieceType.EMPTY and piece.color == color and (x, y) != (kx, ky):
                    # Check if piece is near the king
                    if abs(x - kx) <= 1 and abs(y - ky) <= 1:
                        defenders += 1
        
        # Check pawn shield (different for white and black)
        pawn_shield = 0
        if color == PieceColor.WHITE:
            # White king is typically on the bottom
            if ky <= 2:  # King is on ranks 1-3
                # Check pawns in front of king
                for dx in [-1, 0, 1]:
                    x = kx + dx
                    if 0 <= x < 8 and ky + 1 < 8:
                        if self.board[ky + 1][x].type == PieceType.PAWN and self.board[ky + 1][x].color == color:
                            pawn_shield += 1
            else:
                pawn_shield = 1  # Assume some safety for advanced king
        else:
            # Black king is typically on the top
            if ky >= 5:  # King is on ranks 6-8
                # Check pawns in front of king
                for dx in [-1, 0, 1]:
                    x = kx + dx
                    if 0 <= x < 8 and ky - 1 >= 0:
                        if self.board[ky - 1][x].type == PieceType.PAWN and self.board[ky - 1][x].color == color:
                            pawn_shield += 1
            else:
                pawn_shield = 1  # Assume some safety for advanced king
        
        # Calculate safety score (0-10 scale)
        safety = 3 + defenders + pawn_shield * 2  # Base + defenders + pawn shield bonus
        
        # If in check, reduce safety
        if (color == PieceColor.WHITE and self.players[0].in_check) or (color == PieceColor.BLACK and self.players[1].in_check):
            safety -= 3
            
        # Normalize to 0-10 scale
        return min(10, max(0, safety)) / 10
    
    def get_fen(self) -> str:
        """Get the Forsyth-Edwards Notation (FEN) string for the current position"""
        fen = ""
        
        # Board position
        for y in range(7, -1, -1):  # FEN starts from rank 8
            empty_count = 0
            for x in range(8):
                piece = self.board[y][x]
                if piece.type == PieceType.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    symbol = piece.type.value
                    fen += symbol.upper() if piece.color == PieceColor.WHITE else symbol.lower()
            if empty_count > 0:
                fen += str(empty_count)
            if y > 0:
                fen += "/"
        
        # Active color
        fen += " w" if self.current_player.color == PieceColor.WHITE else " b"
        
        # Castling availability
        castling = ""
        # Kingside white
        if (not self.board[0][4].has_moved and self.board[0][4].type == PieceType.KING and 
            not self.board[0][7].has_moved and self.board[0][7].type == PieceType.ROOK):
            castling += "K"
        # Queenside white
        if (not self.board[0][4].has_moved and self.board[0][4].type == PieceType.KING and 
            not self.board[0][0].has_moved and self.board[0][0].type == PieceType.ROOK):
            castling += "Q"
        # Kingside black
        if (not self.board[7][4].has_moved and self.board[7][4].type == PieceType.KING and 
            not self.board[7][7].has_moved and self.board[7][7].type == PieceType.ROOK):
            castling += "k"
        # Queenside black
        if (not self.board[7][4].has_moved and self.board[7][4].type == PieceType.KING and 
            not self.board[7][0].has_moved and self.board[7][0].type == PieceType.ROOK):
            castling += "q"
        
        fen += " " + (castling if castling else "-")
        
        # En passant target square
        if self.en_passant_target:
            x, y = self.en_passant_target
            file = chr(97 + x)  # Convert to a-h
            rank = str(y + 1)   # Convert to 1-8
            fen += " " + file + rank
        else:
            fen += " -"
        
        # Halfmove clock and fullmove number
        fen += f" {self.halfmove_clock} {self.fullmove_number}"
        
        return fen
    
    def load_from_fen(self, fen: str) -> bool:
        """Load a position from a Forsyth-Edwards Notation (FEN) string"""
        try:
            parts = fen.split()
            if len(parts) != 6:
                return False
                
            board_str, active_color, castling, en_passant, halfmove, fullmove = parts
            
            # Parse board position
            self.board = [[ChessPiece(PieceType.EMPTY, PieceColor.NONE) for _ in range(8)] for _ in range(8)]
            ranks = board_str.split("/")
            if len(ranks) != 8:
                return False
                
            for y, rank in enumerate(ranks):
                real_y = 7 - y  # FEN starts from the 8th rank
                x = 0
                for char in rank:
                    if char.isdigit():
                        x += int(char)
                    else:
                        piece_type = None
                        piece_color = PieceColor.WHITE if char.isupper() else PieceColor.BLACK
                        char = char.upper()
                        
                        if char == "P":
                            piece_type = PieceType.PAWN
                        elif char == "N":
                            piece_type = PieceType.KNIGHT
                        elif char == "B":
                            piece_type = PieceType.BISHOP
                        elif char == "R":
                            piece_type = PieceType.ROOK
                        elif char == "Q":
                            piece_type = PieceType.QUEEN
                        elif char == "K":
                            piece_type = PieceType.KING
                        else:
                            return False
                            
                        self.board[real_y][x] = ChessPiece(piece_type, piece_color)
                        x += 1
                
                if x != 8:
                    return False
            
            # Set active color
            if active_color == "w":
                self.current_player_idx = 0
            elif active_color == "b":
                self.current_player_idx = 1
            else:
                return False
            
            # Set castling rights
            for y in range(8):
                for x in range(8):
                    if self.board[y][x].type in [PieceType.KING, PieceType.ROOK]:
                        self.board[y][x].has_moved = True
            
            # White king
            if "K" in castling:
                self.board[0][4].has_moved = False
                self.board[0][7].has_moved = False
            if "Q" in castling:
                self.board[0][4].has_moved = False
                self.board[0][0].has_moved = False
            
            # Black king
            if "k" in castling:
                self.board[7][4].has_moved = False
                self.board[7][7].has_moved = False
            if "q" in castling:
                self.board[7][4].has_moved = False
                self.board[7][0].has_moved = False
            
            # Set en passant target
            if en_passant == "-":
                self.en_passant_target = None
            else:
                if len(en_passant) != 2:
                    return False
                file, rank = en_passant
                if not (file in "abcdefgh" and rank in "12345678"):
                    return False
                x = ord(file) - ord("a")
                y = int(rank) - 1
                self.en_passant_target = (x, y)
            
            # Set halfmove clock and fullmove number
            try:
                self.halfmove_clock = int(halfmove)
                self.fullmove_number = int(fullmove)
            except ValueError:
                return False
            
            # Update check status
            self.check_status = self._is_in_check(self.current_player.color)
            self.players[self.current_player_idx].in_check = self.check_status
            
            # Reset game state
            self.position_history = [self._get_position_hash()]
            self.board_history = [copy.deepcopy(self.board)]
            self.state_history = [self._get_game_state_minimal()]
            self.is_game_over = False
            self.winner = None
            self.checkmate_status = False
            self.stalemate_status = False
            
            # Check if game is already over
            self.check_game_over()
            
            return True
        except Exception as e:
            logger.error(f"Error loading FEN: {e}")
            return False
    
    def display(self):
        """Display the current game state"""
        print(f"=== Chess Game ===")
        print(f"Current Player: {self.current_player.name} ({self.current_player.color.name})")
        
        # Display the board
        print("  a b c d e f g h")
        print(" +-----------------+")
        for y in range(7, -1, -1):
            row_str = f"{y+1}| "
            for x in range(8):
                row_str += f"{str(self.board[y][x])} "
            row_str += f"|{y+1}"
            print(row_str)
        print(" +-----------------+")
        print("  a b c d e f g h")
        
        # Display additional game state
        if self.check_status:
            print("CHECK!")
        if self.checkmate_status:
            print("CHECKMATE!")
        if self.stalemate_status:
            print("STALEMATE!")
        
        if self.is_game_over:
            if self.winner:
                print(f"Game Over! Winner: {self.winner.name}")
            else:
                print("Game Over! Draw")
        
        print(f"White captured: {', '.join(str(p) for p in self.players[0].captured_pieces) or 'None'}")
        print(f"Black captured: {', '.join(str(p) for p in self.players[1].captured_pieces) or 'None'}")
        
        if self.timed_mode:
            print(f"White time: {self.player_time.get(self.players[0].id, 0):.1f}s")
            print(f"Black time: {self.player_time.get(self.players[1].id, 0):.1f}s")


# Main entry point to demonstrate usage
if __name__ == "__main__":
    engine = GameEngine()
    
    # Register games
    engine.register_game(MahjongGame)
    engine.register_game(ChessGame)
    engine.register_game(GoGame)
    
    for x in range(self.board_size):
                    if not visited[y][x] and self.board[y][x] == Stone.EMPTY:
                        # We found an empty intersection, determine territory
                        connected_empty = []
                        bordering_black = False
                        bordering_white = False
                        
                        # Find all connected empty intersections
                        self._flood_fill(x, y, connected_empty, visited)
                        
                        # Check the borders of the connected empties
                        for ex, ey in connected_empty:
                            # Check adjacent intersections for stones
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nx, ny = ex + dx, ey + dy
                                
                                # Check if the new coordinates are valid
                                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                                    continue
                                
                                # Check if there's a stone
                                if self.board[ny][nx] == Stone.BLACK:
                                    bordering_black = True
                                elif self.board[ny][nx] == Stone.WHITE:
                                    bordering_white = True
                        
                        # If only one player's stones border this area, it's their territory
                        territory_size = len(connected_empty)
                        if bordering_black and not bordering_white:
                            territories["black"] += territory_size
                        elif bordering_white and not bordering_black:
                            territories["white"] += territory_size
                        # Otherwise, it's neutral territory
        
        return territories
    
    def _flood_fill(self, x: int, y: int, connected: List[Tuple[int, int]], visited: List[List[bool]]):
        """Recursively find all connected empty intersections"""
        if visited[y][x] or self.board[y][x] != Stone.EMPTY:
            return
        
        visited[y][x] = True
        connected.append((x, y))
        
        # Check adjacent intersections
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if the new coordinates are valid
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                self._flood_fill(nx, ny, connected, visited)
    
    def check_game_over(self) -> bool:
        """Check if the game has ended"""
        return self.is_game_over
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        return {
            "current_player": self.current_player_idx,
            "board": [[stone.value for stone in row] for row in self.board],
            "consecutive_passes": self.consecutive_passes,
            "move_history": self.move_history,
            "black_captures": self.players[0].captures,
            "white_captures": self.players[1].captures,
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None,
            "black_score": self.players[0].score,
            "white_score": self.players[1].score,
            "komi": self.komi,
            "ruleset": self.ruleset,
            "territories": self.territories,
            "stats": self.stats
        }
    
    def get_player_view(self, player_idx: int) -> Dict[str, Any]:
        """Get a view of the game state for a specific player"""
        game_state = self.get_game_state()
        
        # Add valid moves if it's this player's turn
        if player_idx == self.current_player_idx:
            game_state["valid_moves"] = self.get_valid_moves()
            
        # Add any player-specific information
        game_state["is_your_turn"] = (player_idx == self.current_player_idx)
        game_state["your_color"] = self.players[player_idx].stone.value
        
        return game_state
    
    def get_board_analysis(self) -> Dict[str, Any]:
        """Get an analysis of the current board state"""
        # Calculate territories
        territories = self._calculate_territories()
        
        # Calculate influence map (simplified)
        influence = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == Stone.BLACK:
                    # Add positive influence for black
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.board_size and 0 <= nx < self.board_size:
                                # Influence decreases with distance
                                dist = abs(dx) + abs(dy)
                                if dist > 0:
                                    influence[ny][nx] += 3 / dist
                elif self.board[y][x] == Stone.WHITE:
                    # Add negative influence for white
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.board_size and 0 <= nx < self.board_size:
                                # Influence decreases with distance
                                dist = abs(dx) + abs(dy)
                                if dist > 0:
                                    influence[ny][nx] -= 3 / dist
        
        # Identify key strategic points
        strategic_points = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == Stone.EMPTY:
                    # Check if this is a liberty of a group with few liberties
                    for color in [Stone.BLACK, Stone.WHITE]:
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < self.board_size and 0 <= nx < self.board_size and 
                                    self.board[ny][nx] == color):
                                    # Check how many liberties this group has
                                    group = self._find_group(self.board, nx, ny)
                                    liberties = set()
                                    for gx, gy in group:
                                        for gdx, gdy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                            gnx, gny = gx + gdx, gy + gdy
                                            if (0 <= gnx < self.board_size and 0 <= gny < self.board_size and 
                                                self.board[gny][gnx] == Stone.EMPTY):
                                                liberties.add((gnx, gny))
                                    
                                    if len(liberties) <= 2:
                                        strategic_points.append({
                                            "x": x,
                                            "y": y,
                                            "type": "liberty",
                                            "group_color": "black" if color == Stone.BLACK else "white",
                                            "liberty_count": len(liberties)
                                        })
        
        return {
            "territories": territories,
            "influence": influence,
            "strategic_points": strategic_points,
            "black_groups": sum(1 for y in range(self.board_size) for x in range(self.board_size) 
                              if self.board[y][x] == Stone.BLACK and 
                              all(self.board[y-1][x] != Stone.BLACK if y > 0 else True) and
                              all(self.board[y][x-1] != Stone.BLACK if x > 0 else True)),
            "white_groups": sum(1 for y in range(self.board_size) for x in range(self.board_size) 
                              if self.board[y][x] == Stone.WHITE and 
                              all(self.board[y-1][x] != Stone.WHITE if y > 0 else True) and
                              all(self.board[y][x-1] != Stone.WHITE if x > 0 else True))
        }
    
    def display(self):
        """Display the current game state"""
        print(f"=== Go Game ({self.board_size}x{self.board_size}) ===")
        print(f"Current Player: {self.current_player.name} ({self.current_player.stone.value})")
        
        # Display the board
        board_str = "  "
        for i in range(self.board_size):
            board_str += f"{i:2d}"
        board_str += "\n"
        
        for y in range(self.board_size):
            board_str += f"{y:2d}"
            for x in range(self.board_size):
                board_str += f" {self.board[y][x].value}"
            board_str += "\n"
        
        print(board_str)
        
        print(f"Black Captures: {self.players[0].captures}")
        print(f"White Captures: {self.players[1].captures}")
        
        if self.is_game_over:
            print(f"\nGame Over!")
            print(f"Black Score: {self.players[0].score}")
            print(f"White Score: {self.players[1].score}")
            print(f"Winner: {self.winner.name if self.winner else 'Draw'}")


"""
Chess Game Implementation
"""

class PieceType(Enum):
    """Enum representing chess piece types"""
    PAWN = "P"
    KNIGHT = "N"
    BISHOP = "B"
    ROOK = "R"
    QUEEN = "Q"
    KING = "K"
    EMPTY = " "

class PieceColor(Enum):
    """Enum representing chess piece colors"""
    WHITE = "W"
    BLACK = "B"
    NONE = "N"

class ChessPiece:
    """Represents a chess piece"""
    
    def __init__(self, piece_type: PieceType, color: PieceColor):
        self.type = piece_type
        self.color = color
        self.has_moved = False  # Useful for castling and initial pawn movement
    
    def __str__(self):
        if self.type == PieceType.EMPTY:
            return " "
        symbol = self.type.value
        return symbol.upper() if self.color == PieceColor.WHITE else symbol.lower()
    
    def __eq__(self, other):
        if not isinstance(other, ChessPiece):
            return False
        return self.type == other.type and self.color == other.color

class ChessPlayer(Player):
    """Player in a chess game"""
    
    def __init__(self, name: str, color: PieceColor, id: str = None):
        super().__init__(name, id)
        self.color = color
        self.captured_pieces = []
        self.in_check = False
    
    def __str__(self):
        captured = ", ".join(str(p) for p in self.captured_pieces) if self.captured_pieces else "None"
        check_status = " (IN CHECK)" if self.in_check else ""
        return f"{self.name} ({self.color.name}{check_status}, Captured: {captured})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = super().to_dict()
        data["color"] = self.color.value
        data["captured_pieces"] = [str(p) for p in self.captured_pieces]
        data["in_check"] = self.in_check
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChessPlayer':
        """Create from dictionary"""
        color_value = data.get("color", "W")
        color = PieceColor.WHITE if color_value == "W" else PieceColor.BLACK
        
        player = cls(data["name"], color, data.get("id"))
        player.in_check = data.get("in_check", False)
        
        # Set base player attributes
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player

class ChessAIPlayer(ChessPlayer, AIPlayer):
    """AI player for chess games"""
    
    def __init__(self, name: str, color: PieceColor, difficulty: GameDifficulty = GameDifficulty.MEDIUM,
                 adaptive: bool = True, learning_rate: float = 0.05, id: str = None):
        ChessPlayer.__init__(self, name, color, id)
        AIPlayer.__init__(self, name, difficulty, adaptive, learning_rate)
        
        # AI-specific attributes for chess
        self.opening_book = {}  # Map of board states to good moves
        self.position_evaluations = {}  # Cache of position evaluations
        self.preferred_pieces = []  # Pieces the AI prefers to move
        self.aggressive_factor = 0.7  # How aggressively the AI plays (0-1)
        
        # Set up difficulty-specific parameters
        self._configure_difficulty()
    
    def _configure_difficulty(self):
        """Configure AI parameters based on difficulty level"""
        if self.difficulty == GameDifficulty.BEGINNER:
            self.evaluation_depth = 1
            self.aggressive_factor = 0.5
            self.use_opening_book = False
            self.randomness = 0.5  # High randomness for beginners
        elif self.difficulty == GameDifficulty.EASY:
            self.evaluation_depth = 2
            self.aggressive_factor = 0.6
            self.use_opening_book = True
            self.randomness = 0.3
        elif self.difficulty == GameDifficulty.MEDIUM:
            self.evaluation_depth = 3
            self.aggressive_factor = 0.7
            self.use_opening_book = True
            self.randomness = 0.15
        elif self.difficulty == GameDifficulty.HARD:
            self.evaluation_depth = 4
            self.aggressive_factor = 0.8
            self.use_opening_book = True
            self.randomness = 0.05
        elif self.difficulty == GameDifficulty.EXPERT:
            self.evaluation_depth = 5
            self.aggressive_factor = 0.9
            self.use_opening_book = True
            self.randomness = 0.02
        else:  # MASTER
            self.evaluation_depth = 6
            self.aggressive_factor = 0.95
            self.use_opening_book = True
            self.randomness = 0.01
    
    def evaluate_position(self, board, is_endgame=False):
        """Evaluate a chess position for the current player"""
        # This would be a complex evaluation in a real chess engine
        # For simplicity, we'll use a material-based evaluation with simple positional factors
        
        # Piece values
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3.25,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0  # King's value isn't counted in material
        }
        
        # Simple positional bonuses - center control, development
        positional_bonus = {
            PieceType.PAWN: [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
                [0.05, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.05],
                [0, 0, 0, 0.2, 0.2, 0, 0, 0],
                [0.05, -0.05, -0.1, 0, 0, -0.1, -0.05, 0.05],
                [0.05, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.05],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            PieceType.KNIGHT: [
                [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5],
                [-0.4, -0.2, 0, 0, 0, 0, -0.2, -0.4],
                [-0.3, 0, 0.1, 0.15, 0.15, 0.1, 0, -0.3],
                [-0.3, 0.05, 0.15, 0.2, 0.2, 0.15, 0.05, -0.3],
                [-0.3, 0, 0.15, 0.2, 0.2, 0.15, 0, -0.3],
                [-0.3, 0.05, 0.1, 0.15, 0.15, 0.1, 0.05, -0.3],
                [-0.4, -0.2, 0, 0.05, 0.05, 0, -0.2, -0.4],
                [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5]
            ],
            # Other piece type tables would go here
        }
        
        # Count material
        white_material = 0
        black_material = 0
        
        # Add positional evaluation
        white_position = 0
        black_position = 0
        
        for y in range(8):
            for x in range(8):
                piece = board[y][x]
                if piece.type != PieceType.EMPTY:
                    value = piece_values.get(piece.type, 0)
                    
                    # Add position bonus if available
                    if piece.type in positional_bonus:
                        pos_y = y if piece.color == PieceColor.WHITE else 7 - y
                        pos_bonus = positional_bonus[piece.type][pos_y][x]
                        value += pos_bonus
                    
                    if piece.color == PieceColor.WHITE:
                        white_material += value
                        white_position += 0.1  # Simple bonus for developed pieces
                    else:
                        black_material += value
                        black_position += 0.1
        
        # Calculate advantage from perspective of current player
        material_advantage = 0
        positional_advantage = 0
        
        if self.color == PieceColor.WHITE:
            material_advantage = white_material - black_material
            positional_advantage = white_position - black_position
        else:
            material_advantage = black_material - white_material
            positional_advantage = black_position - white_position
        
        # Weight material more in endgame
        if is_endgame:
            return material_advantage + (positional_advantage * 0.5)
        else:
            return material_advantage + positional_advantage
    
    def choose_move(self, board, valid_moves):
        """Choose a move based on AI difficulty"""
        if not valid_moves:
            return None
            
        # For very low difficulty, just choose a random move
        if self.difficulty == GameDifficulty.BEGINNER and random.random() < 0.6:
            return random.choice(valid_moves)
        
        # Check opening book for early game
        if self.use_opening_book and len(self.opening_book) > 0:
            board_hash = str(board)  # Simple hashing
            if board_hash in self.opening_book:
                book_moves = self.opening_book[board_hash]
                if book_moves:
                    # Look for book moves that are in valid moves
                    for book_move in book_moves:
                        for valid_move in valid_moves:
                            if (valid_move.get("from_x") == book_move.get("from_x") and 
                                valid_move.get("from_y") == book_move.get("from_y") and
                                valid_move.get("to_x") == book_move.get("to_x") and
                                valid_move.get("to_y") == book_move.get("to_y")):
                                if random.random() < 0.9:  # 90% chance to use book move
                                    return valid_move
        
        # Evaluate all valid moves
        move_scores = []
        
        for move in valid_moves:
            # Create a copy of the board to simulate move
            board_copy = copy.deepcopy(board)
            
            # Apply the move to the board copy
            from_x, from_y = move.get("from_x"), move.get("from_y")
            to_x, to_y = move.get("to_x"), move.get("to_y")
            
            # Check if this is a capture
            is_capture = board_copy[to_y][to_x].type != PieceType.EMPTY
            
            # Check promotion
            is_promotion = (board_copy[from_y][from_x].type == PieceType.PAWN and 
                           (to_y == 0 or to_y == 7))
            
            # Make the move
            board_copy[to_y][to_x] = board_copy[from_y][from_x]
            board_copy[from_y][from_x] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
            
            # Handle promotion
            if is_promotion:
                promotion = move.get("promotion", "Q")
                promotion_type = {"Q": PieceType.QUEEN, "R": PieceType.ROOK, 
                                 "B": PieceType.BISHOP, "N": PieceType.KNIGHT}.get(promotion, PieceType.QUEEN)
                board_copy[to_y][to_x].type = promotion_type
            
            # Evaluate the resulting position
            score = self.evaluate_position(board_copy)
            
            # Add bonus for captures and promotions based on aggressiveness
            if is_capture:
                captured_value = {
                    PieceType.PAWN: 1,
                    PieceType.KNIGHT: 3,
                    PieceType.BISHOP: 3,
                    PieceType.ROOK: 5,
                    PieceType.QUEEN: 9,
                    PieceType.KING: 0,
                    PieceType.EMPTY: 0
                }.get(board[to_y][to_x].type, 0)
                
                # Bonus for captures, weighted by aggressiveness
                score += captured_value * self.aggressive_factor
            
            if is_promotion:
                # Bonus for promotion
                score += 7  # Almost as good as getting a queen
            
            # Add some randomness based on difficulty
            score += random.uniform(-self.randomness, self.randomness)
            
            move_scores.append((move, score))
        
        # Choose the move with the highest score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[0][0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = ChessPlayer.to_dict(self)
        
        # Add AI-specific data
        data["ai_settings"] = {
            "difficulty": self.difficulty.name,
            "adaptive": self.adaptive,
            "learning_rate": self.learning_rate,
            "aggressive_factor": self.aggressive_factor,
            "evaluation_depth": self.evaluation_depth
        }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChessAIPlayer':
        """Create from dictionary"""
        color_value = data.get("color", "W")
        color = PieceColor.WHITE if color_value == "W" else PieceColor.BLACK
        
        # Extract AI settings
        ai_settings = data.get("ai_settings", {})
        difficulty_name = ai_settings.get("difficulty", "MEDIUM")
        try:
            difficulty = GameDifficulty[difficulty_name]
        except KeyError:
            difficulty = GameDifficulty.MEDIUM
            
        adaptive = ai_settings.get("adaptive", True)
        learning_rate = ai_settings.get("learning_rate", 0.05)
        
        # Create instance
        player = cls(
            data["name"], 
            color, 
            difficulty=difficulty,
            adaptive=adaptive,
            learning_rate=learning_rate,
            id=data.get("id")
        )
        
        # Set additional AI parameters
        player.aggressive_factor = ai_settings.get("aggressive_factor", 0.7)
        player.in_check = data.get("in_check", False)
        
        # Set base player attributes
        player.score = data.get("score", 0)
        player.total_games = data.get("stats", {}).get("total_games", 0)
        player.wins = data.get("stats", {}).get("wins", 0)
        player.losses = data.get("stats", {}).get("losses", 0)
        player.draws = data.get("stats", {}).get("draws", 0)
        player.created_at = data.get("created_at", player.created_at)
        player.last_active = data.get("last_active", player.last_active)
        player.metadata = data.get("metadata", {})
        
        return player

class ChessGame(Game):
    """Enhanced implementation of the chess game"""
    
    def __init__(self, players: List[Player], settings: Dict[str, Any] = None, game_id: str = None):
        """
        Initialize a chess game
        
        Args:
            players: List of players (must be exactly 2)
            settings: Dictionary of game settings
            game_id: Optional game ID
        """
        if len(players) != 2:
            raise ValueError("Chess requires exactly 2 players")
        
        # Process settings
        self.settings = settings or {}
        self.timed_mode = self.settings.get("timed_mode", False)
        self.time_control = self.settings.get("time_control", {"minutes": 10, "increment": 0})
        self.allow_takeback = self.settings.get("allow_takeback", True)
        self.auto_queen = self.settings.get("auto_queen", True)  # Auto-promote to queen
        
        # Assign colors to players if needed
        colors = [PieceColor.WHITE, PieceColor.BLACK]
        for i, player in enumerate(players):
            if not isinstance(player, ChessPlayer):
                # Convert regular Player to ChessPlayer
                players[i] = ChessPlayer(player.name, colors[i], player.id if hasattr(player, 'id') else None)
            elif not hasattr(player, 'color'):
                # Assign color if missing
                player.color = colors[i]
        
        super().__init__("Chess", players, game_id)
        
        # Chess-specific game state
        self.board = None
        self.check_status = False
        self.checkmate_status = False
        self.stalemate_status = False
        self.en_passant_target = None  # Coordinates for potential en passant capture
        self.halfmove_clock = 0  # For 50-move rule
        self.fullmove_number = 1  # Incremented after Black's move
        
        # Time control
        self.player_time = {player.id: self.time_control.get("minutes", 10) * 60 for player in players}
        self.last_move_time = None
        
        # Game history for threefold repetition check
        self.position_history = []
        
        # For takeback support
        self.board_history = []
        self.state_history = []
        
        # Statistics tracking
        self.stats = {
            "captures": {player.id: [] for player in players},
            "checks": {player.id: 0 for player in players},
            "castling": {player.id: False for player in players},
            "piece_moves": {piece_type.name: 0 for piece_type in PieceType if piece_type != PieceType.EMPTY},
            "squares_visited": [[0 for _ in range(8)] for _ in range(8)],
            "material_advantage": []
        }
    
    def initialize_game(self):
        """Set up the game state"""
        logger.info(f"Initializing Chess game {self.game_id}")
        self.is_game_over = False
        self.winner = None
        self.check_status = False
        self.checkmate_status = False
        self.stalemate_status = False
        self.en_passant_target = None
        self.move_history = []
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # Create empty board (8x8)
        self.board = [[ChessPiece(PieceType.EMPTY, PieceColor.NONE) for _ in range(8)] for _ in range(8)]
        
        # Set up pawns
        for col in range(8):
            self.board[1][col] = ChessPiece(PieceType.PAWN, PieceColor.WHITE)
            self.board[6][col] = ChessPiece(PieceType.PAWN, PieceColor.BLACK)
        
        # Set up other pieces
        # White pieces
        self.board[0][0] = ChessPiece(PieceType.ROOK, PieceColor.WHITE)
        self.board[0][1] = ChessPiece(PieceType.KNIGHT, PieceColor.WHITE)
        self.board[0][2] = ChessPiece(PieceType.BISHOP, PieceColor.WHITE)
        self.board[0][3] = ChessPiece(PieceType.QUEEN, PieceColor.WHITE)
        self.board[0][4] = ChessPiece(PieceType.KING, PieceColor.WHITE)
        self.board[0][5] = ChessPiece(PieceType.BISHOP, PieceColor.WHITE)
        self.board[0][6] = ChessPiece(PieceType.KNIGHT, PieceColor.WHITE)
        self.board[0][7] = ChessPiece(PieceType.ROOK, PieceColor.WHITE)
        
        # Black pieces
        self.board[7][0] = ChessPiece(PieceType.ROOK, PieceColor.BLACK)
        self.board[7][1] = ChessPiece(PieceType.KNIGHT, PieceColor.BLACK)
        self.board[7][2] = ChessPiece(PieceType.BISHOP, PieceColor.BLACK)
        self.board[7][3] = ChessPiece(PieceType.QUEEN, PieceColor.BLACK)
        self.board[7][4] = ChessPiece(PieceType.KING, PieceColor.BLACK)
        self.board[7][5] = ChessPiece(PieceType.BISHOP, PieceColor.BLACK)
        self.board[7][6] = ChessPiece(PieceType.KNIGHT, PieceColor.BLACK)
        self.board[7][7] = ChessPiece(PieceType.ROOK, PieceColor.BLACK)
        
        # Reset player states
        for player in self.players:
            player.captured_pieces = []
            player.in_check = False
            
        # Reset position history
        self.position_history = [self._get_position_hash()]
        
        # Reset board history
        self.board_history = [copy.deepcopy(self.board)]
        self.state_history = [self._get_game_state_minimal()]
        
        # White starts in chess
        self.current_player_idx = 0
        
        # Reset time control
        if self.timed_mode:
            self.player_time = {player.id: self.time_control.get("minutes", 10) * 60 for player in players}
            self.last_move_time = datetime.now()
        
        # Reset stats
        self.stats = {
            "captures": {player.id: [] for player in players},
            "checks": {player.id: 0 for player in players},
            "castling": {player.id: False for player in players},
            "piece_moves": {piece_type.name: 0 for piece_type in PieceType if piece_type != PieceType.EMPTY},
            "squares_visited": [[0 for _ in range(8)] for _ in range(8)],
            "material_advantage": []
        }
        
        logger.info(f"Chess game {self.game_id} initialized")
    
    def is_valid_move(self, move: Dict[str, Any]) -> bool:
        """Check if a move is valid"""
        from_x, from_y = move.get("from_x"), move.get("from_y")
        to_x, to_y = move.get("to_x"), move.get("to_y")
        
        # Check if coordinates are valid
        if not (0 <= from_x < 8 and 0 <= from_y < 8 and 0 <= to_x < 8 and 0 <= to_y < 8):
            return False
        
        # Get the piece to move
        piece = self.board[from_y][from_x]
        
        # Check if the piece belongs to the current player
        if piece.color != self.current_player.color:
            return False
        
        # Check if the destination is empty or contains an opponent's piece
        dest_piece = self.board[to_y][to_x]
        if dest_piece.color == piece.color:
            return False
        
        # Check if the move is valid for the specific piece type
        if not self._is_piece_move_valid(piece, from_x, from_y, to_x, to_y):
            return False
        
        # Check for special moves
        promotion = move.get("promotion")
        if promotion and not self._is_promotion_valid(piece, to_y, promotion):
            return False
        
        castling = move.get("castling", False)
        if castling and not self._is_castling_valid(piece, from_x, from_y, to_x, to_y):
            return False
        
        en_passant = move.get("en_passant", False)
        if en_passant and not self._is_en_passant_valid(piece, from_x, from_y, to_x, to_y):
            return False
        
        # Check if the move would put/leave the player in check
        if self._would_be_in_check(from_x, from_y, to_x, to_y):
            return False
        
        return True
    
    def get_valid_moves(self) -> List[Dict[str, Any]]:
        """Return a list of all valid moves for the current player"""
        valid_moves = []
        
        # Check all possible piece movements
        for from_y in range(8):
            for from_x in range(8):
                piece = self.board[from_y][from_x]
                
                # Skip empty squares and opponent's pieces
                if piece.type == PieceType.EMPTY or piece.color != self.current_player.color:
                    continue
                
                # Check all possible destinations
                for to_y in range(8):
                    for to_x in range(8):
                        # Skip the same square
                        if from_x == to_x and from_y == to_y:
                            continue
                        
                        # Basic move
                        move = {
                            "from_x": from_x,
                            "from_y": from_y,
                            "to_x": to_x,
                            "to_y": to_y
                        }
                        
                        # Check for special moves
                        # Pawn promotion
                        if (piece.type == PieceType.PAWN and 
                            ((piece.color == PieceColor.WHITE and to_y == 7) or 
                             (piece.color == PieceColor.BLACK and to_y == 0))):
                            if self.auto_queen:
                                move["promotion"] = "Q"
                            else:
                                # Create variants for different promotion options
                                for promotion in ["Q", "R", "B", "N"]:
                                    promotion_move = move.copy()
                                    promotion_move["promotion"] = promotion
                                    if self.is_valid_move(promotion_move):
                                        valid_moves.append(promotion_move)
                                continue  # Skip adding the basic move
                        
                        # Castling
                        if (piece.type == PieceType.KING and abs(to_x - from_x) == 2):
                            move["castling"] = True
                        
                        # En passant
                        if (piece.type == PieceType.PAWN and 
                            abs(to_x - from_x) == 1 and 
                            self.en_passant_target == (to_x, to_y)):
                            move["en_passant"] = True
                        
                        # Add the move if it's valid
                        if self.is_valid_move(move):
                            valid_moves.append(move)
        
        return valid_moves
    
    def _is_piece_move_valid(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if a move is valid for the specific piece type"""
        # Calculate deltas
        dx = to_x - from_x
        dy = to_y - from_y
        
        if piece.type == PieceType.PAWN:
            # Pawns move differently depending on color
            direction = 1 if piece.color == PieceColor.WHITE else -1
            
            # Normal forward movement (one square)
            if dx == 0 and dy == direction and self.board[to_y][to_x].type == PieceType.EMPTY:
                return True
            
            # Initial two-square movement
            if (dx == 0 and dy == 2 * direction and 
                not piece.has_moved and 
                self.board[from_y + direction][from_x].type == PieceType.EMPTY and 
                self.board[to_y][to_x].type == PieceType.EMPTY):
                return True
            
            # Capturing diagonally
            if abs(dx) == 1 and dy == direction:
                # Regular capture
                if self.board[to_y][to_x].type != PieceType.EMPTY:
                    return True
                
                # En passant capture
                if self.en_passant_target == (to_x, to_y):
                    return True
            
            return False
        
        elif piece.type == PieceType.KNIGHT:
            # Knights move in an L-shape
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        
        elif piece.type == PieceType.BISHOP:
            # Bishops move diagonally
            if abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear(from_x, from_y, to_x, to_y)
        
        elif piece.type == PieceType.ROOK:
            # Rooks move horizontally or vertically
            if dx != 0 and dy != 0:
                return False
            
            # Check if the path is clear
            return self._is_path_clear(from_x, from_y, to_x, to_y)
        
        elif piece.type == PieceType.QUEEN:
            # Queens move like a rook or bishop
            if dx != 0 and dy != 0 and abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear(from_x, from_y, to_x, to_y)
        
        elif piece.type == PieceType.KING:
            # Kings move one square in any direction
            if abs(dx) <= 1 and abs(dy) <= 1:
                return True
            
            # Castling (handled separately in _is_castling_valid)
            if abs(dx) == 2 and dy == 0:
                return self._is_castling_valid(piece, from_x, from_y, to_x, to_y)
            
            return False
        
        return False
    
    def _is_path_clear(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if the path between two positions is clear of pieces"""
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Determine step direction
        x_step = 0 if dx == 0 else (1 if dx > 0 else -1)
        y_step = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Start from the square after the origin
        x, y = from_x + x_step, from_y + y_step
        
        # Check each square along the path
        while (x, y) != (to_x, to_y):
            if self.board[y][x].type != PieceType.EMPTY:
                return False
            x += x_step
            y += y_step
        
        return True
    
    def _is_promotion_valid(self, piece: ChessPiece, to_y: int, promotion: str) -> bool:
        """Check if a pawn promotion is valid"""
        # Promotion is only valid for pawns reaching the opposite edge
        if piece.type != PieceType.PAWN:
            return False
        
        # Check if the pawn is reaching the opposite edge
        if (piece.color == PieceColor.WHITE and to_y != 7) or (piece.color == PieceColor.BLACK and to_y != 0):
            return False
        
        # Check if the promotion type is valid
        valid_promotions = ["Q", "N", "R", "B"]
        return promotion.upper() in valid_promotions
    
    def _is_castling_valid(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if castling is valid"""
        # Castling is only valid for kings
        if piece.type != PieceType.KING:
            return False
        
        # King must not have moved
        if piece.has_moved:
            return False
        
        # King must be on the correct starting position
        if from_y != 0 and from_y != 7:
            return False
        if from_x != 4:
            return False
        
        # Must be a horizontal move of two squares
        if from_y != to_y or abs(to_x - from_x) != 2:
            return False
        
        # Determine rook position
        rook_x = 0 if to_x < from_x else 7  # Queenside or kingside
        rook = self.board[from_y][rook_x]
        
        # Check if there's a rook in the correct position
        if rook.type != PieceType.ROOK or rook.color != piece.color:
            return False
        
        # Rook must not have moved
        if rook.has_moved:
            return False
        
        # Path between king and rook must be clear
        min_x = min(from_x, rook_x) + 1
        max_x = max(from_x, rook_x)
        for x in range(min_x, max_x):
            if self.board[from_y][x].type != PieceType.EMPTY:
                return False
        
        # King must not be in check
        if self._is_in_check(piece.color):
            return False
        
        # King must not pass through or land on a square under attack
        step = -1 if to_x < from_x else 1
        for x in range(from_x, to_x + step, step):
            if self._is_square_attacked(x, from_y, piece.color):
                return False
        
        return True
    
    def _is_en_passant_valid(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if an en passant capture is valid"""
        # En passant is only valid for pawns
        if piece.type != PieceType.PAWN:
            return False
        
        # Must be a diagonal move
        if abs(to_x - from_x) != 1:
            return False
        
        # Ensure the destination is the en passant target
        return self.en_passant_target == (to_x, to_y)
    
    def _would_be_in_check(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if a move would leave the player in check"""
        # Create a copy of the board
        board_copy = copy.deepcopy(self.board)
        
        # Temporarily make the move
        board_copy[to_y][to_x] = board_copy[from_y][from_x]
        board_copy[from_y][from_x] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Find the king's position
        king_pos = None
        for y in range(8):
            for x in range(8):
                piece = board_copy[y][x]
                if (piece.type == PieceType.KING and piece.color == self.current_player.color):
                    king_pos = (x, y)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return True  # If king not found, assume invalid
        
        # Check if any opponent's piece can capture the king
        for y in range(8):
            for x in range(8):
                piece = board_copy[y][x]
                if piece.type != PieceType.EMPTY and piece.color != self.current_player.color:
                    if self._can_piece_move_to(piece, x, y, king_pos[0], king_pos[1], board_copy):
                        return True
        
        return False
    
    def _can_piece_move_to(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int, board) -> bool:
        """Check if a piece can move to a specific position on a given board"""
        # Calculate deltas
        dx = to_x - from_x
        dy = to_y - from_y
        
        if piece.type == PieceType.PAWN:
            # Pawns can only capture diagonally
            direction = 1 if piece.color == PieceColor.WHITE else -1
            return abs(dx) == 1 and dy == direction
        
        elif piece.type == PieceType.KNIGHT:
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        
        elif piece.type == PieceType.BISHOP:
            if abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear_on_board(from_x, from_y, to_x, to_y, board)
        
        elif piece.type == PieceType.ROOK:
            if dx != 0 and dy != 0:
                return False
            
            # Check if the path is clear
            return self._is_path_clear_on_board(from_x, from_y, to_x, to_y, board)
        
        elif piece.type == PieceType.QUEEN:
            if dx != 0 and dy != 0 and abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear_on_board(from_x, from_y, to_x, to_y, board)
        
        elif piece.type == PieceType.KING:
            # Kings move one square in any direction
            return abs(dx) <= 1 and abs(dy) <= 1
        
        return False
    
    def _is_path_clear_on_board(self, from_x: int, from_y: int, to_x: int, to_y: int, board) -> bool:
        """Check if the path between two positions is clear on a given board"""
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Determine step direction
        x_step = 0 if dx == 0 else (1 if dx > 0 else -1)
        y_step = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Start from the square after the origin
        x, y = from_x + x_step, from_y + y_step
        
        # Check each square along the path
        while (x, y) != (to_x, to_y):
            if board[y][x].type != PieceType.EMPTY:
                return False
            x += x_step
            y += y_step
        
        return True
    
    def _is_in_check(self, color: PieceColor) -> bool:
        """Check if a player is in check"""
        # Find the king's position
        king_pos = None
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type == PieceType.KING and piece.color == color:
                    king_pos = (x, y)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # Should not happen in a valid game
        
        # Check if the king is under attack
        return self._is_square_attacked(king_pos[0], king_pos[1], color)
    
    def _is_square_attacked(self, x: int, y: int, color: PieceColor) -> bool:
        """Check if a square is under attack by the opponent"""
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece.type != PieceType.EMPTY and piece.color != color:
                    if self._can_piece_move_to(piece, j, i, x, y, self.board):
                        return True
        return False
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a player's move"""
        if not self.is_valid_move(move):
            return False
        
        # Record the move
        self.record_move(move, self.current_player_idx)
        
        from_x, from_y = move.get("from_x"), move.get("from_y")
        to_x, to_y = move.get("to_x"), move.get("to_y")
        
        # Save the current board state for takeback
        self.board_history.append(copy.deepcopy(self.board))
        self.state_history.append(self._get_game_state_minimal())
        
        # Get the piece being moved
        piece = self.board[from_y][from_x]
        
        # Update halfmove clock (for 50-move rule)
        # Reset on pawn move or capture
        if piece.type == PieceType.PAWN or self.board[to_y][to_x].type != PieceType.EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        # Update fullmove number after Black's move
        if self.current_player.color == PieceColor.BLACK:
            self.fullmove_number += 1
        
        # Update stats
        self.stats["piece_moves"][piece.type.name] += 1
        self.stats["squares_visited"][to_y][to_x] += 1
        
        # Handle en passant
        en_passant_capture = False
        en_passant_captured_pos = None
        if piece.type == PieceType.PAWN and self.en_passant_target == (to_x, to_y):
            en_passant_capture = True
            en_passant_captured_pos = (to_x, from_y)  # The actual position of the captured pawn
        
        # Update en passant target for next move
        self.en_passant_target = None
        if piece.type == PieceType.PAWN and abs(to_y - from_y) == 2:
            # Pawn moved two squares, set en passant target
            y_middle = (from_y + to_y) // 2
            self.en_passant_target = (from_x, y_middle)
        
        # Handle castling
        rook_move = None
        if piece.type == PieceType.KING and abs(to_x - from_x) == 2:
            # Record castling in stats
            self.stats["castling"][self.current_player.id] = True
            
            # Determine rook position and new position
            rook_x = 0 if to_x < from_x else 7
            new_rook_x = 3 if to_x < from_x else 5
            rook_move = (rook_x, from_y, new_rook_x, from_y)
        
        # Check for capture
        captured_piece = None
        if self.board[to_y][to_x].type != PieceType.EMPTY:
            captured_piece = self.board[to_y][to_x]
            self.current_player.captured_pieces.append(captured_piece)
            # Record capture in stats
            self.stats["captures"][self.current_player.id].append(str(captured_piece))
        
        # Move the piece
        self.board[to_y][to_x] = piece
        self.board[from_y][from_x] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Handle en passant capture
        if en_passant_capture:
            captured_pawn = self.board[en_passant_captured_pos[1]][en_passant_captured_pos[0]]
            self.current_player.captured_pieces.append(captured_pawn)
            self.stats["captures"][self.current_player.id].append(str(captured_pawn))
            self.board[en_passant_captured_pos[1]][en_passant_captured_pos[0]] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Execute castling rook move
        if rook_move:
            rx, ry, new_rx, new_ry = rook_move
            self.board[new_ry][new_rx] = self.board[ry][rx]
            self.board[ry][rx] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Handle promotion
        promotion = move.get("promotion")
        if promotion and piece.type == PieceType.PAWN and (to_y == 0 or to_y == 7):
            promotion_type_map = {
                "Q": PieceType.QUEEN,
                "N": PieceType.KNIGHT,
                "R": PieceType.ROOK,
                "B": PieceType.BISHOP
            }
            promotion_type = promotion_type_map.get(promotion.upper(), PieceType.QUEEN)
            self.board[to_y][to_x] = ChessPiece(promotion_type, piece.color)
            self.board[to_y][to_x].has_moved = True
        
        # Mark the piece as moved
        piece.has_moved = True
        
        # Add position to history for threefold repetition check
        position_hash = self._get_position_hash()
        self.position_history.append(position_hash)
        
        # Update player time if in timed mode
        if self.timed_mode and self.last_move_time:
            elapsed = (datetime.now() - self.last_move_time).total_seconds()
            current_player_id = self.current_player.id
            self.player_time[current_player_id] -= elapsed
            
            # Apply increment
            increment = self.time_control.get("increment", 0)
            if increment > 0:
                self.player_time[current_player_id] += increment
                
            # Check for time forfeit
            if self.player_time[current_player_id] <= 0:
                self.is_game_over = True
                self.winner = self.players[(self.current_player_idx + 1) % 2]
                return True
            
            self.last_move_time = datetime.now()
        
        # Move to next player
        self.next_player()
        
        # Calculate material advantage for stats
        self._update_material_advantage()
        
        # Check if the next player is in check
        next_color = self.current_player.color
        check_status = self._is_in_check(next_color)
        self.check_status = check_status
        self.current_player.in_check = check_status
        
        if check_status:
            self.stats["checks"][self.players[(self.current_player_idx + 1) % 2].id] += 1
        
        # Check if the game is over
        if self.check_game_over():
            self.is_game_over = True
            
            # Set winner if it's checkmate
            if self.checkmate_status:
                self.winner = self.players[(self.current_player_idx + 1) % 2]
        
        return True
    
    def get_ai_move(self, player_idx: int) -> Dict[str, Any]:
        """Get a move for an AI player"""
        if player_idx != self.current_player_idx:
            return {"error": "Not this player's turn"}
            
        player = self.players[player_idx]
        
        if not isinstance(player, ChessAIPlayer):
            return {"error": "Not an AI player"}
        
        # Get valid moves
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            return {"error": "No valid moves available"}
        
        # Let the AI choose a move
        return player.choose_move(self.board, valid_moves)
    
    def _update_material_advantage(self):
        """Calculate and record the current material advantage"""
        # Use piece values to calculate advantage
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type != PieceType.EMPTY:
                    value = piece_values.get(piece.type, 0)
                    if piece.color == PieceColor.WHITE:
                        white_material += value
                    else:
                        black_material += value
        
        # Record the advantage (positive = white advantage)
        self.stats["material_advantage"].append({
            "move": self.fullmove_number,
            "advantage": white_material - black_material
        })
    
    def _get_position_hash(self) -> str:
        """Get a hash of the current position for repetition detection"""
        # Simple string representation for now
        # A full implementation would include castling rights, en passant, etc.
        position = ""
        for row in self.board:
            for piece in row:
                position += str(piece)
        return position
    
    def _get_game_state_minimal(self) -> Dict[str, Any]:
        """Get minimal game state for history"""
        return {
            "board": copy.deepcopy(self.board),
            "current_player_idx": self.current_player_idx,
            "check_status": self.check_status,
            "en_passant_target": self.en_passant_target,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number
        }
    
    def takeback_move(self) -> bool:
        """Take back the last move"""
        if not self.allow_takeback or len(self.board_history) <= 1:
            return False
        
        # Remove the last position from repetition history
        if self.position_history:
            self.position_history.pop()
        
        # Restore the previous board state
        self.board = self.board_history.pop()
        
        # Restore the previous game state
        previous_state = self.state_history.pop()
        self.current_player_idx = previous_state["current_player_idx"]
        self.check_status = previous_state["check_status"]
        self.

# Register AI profiles
    engine.register_ai_profile("beginner", ChessAIPlayer, {
        "difficulty": GameDifficulty.BEGINNER,
        "adaptive": True
    })
    
    engine.register_ai_profile("expert", ChessAIPlayer, {
        "difficulty": GameDifficulty.EXPERT,
        "adaptive": False
    })
    
    # List available games
    print("Available games:")
    available_games = engine.get_available_games()
    for game_name, game_info in available_games.items():
        variants = ', '.join([v["name"] for v in game_info.get("variants", [])])
        variant_text = f" (Variants: {variants})" if variants else ""
        print(f"- {game_name}{variant_text}")
    
    # Example: Create and initialize a chess game
    print("\nCreating Chess game...")
    player1 = Player("Human Player")
    player2 = engine.create_ai_player("expert", "AI Challenger")
    chess_game = engine.create_game("ChessGame", [player1, player2], settings={"timed_mode": True})
    chess_game.display()
    
    # Make a sample move (e2 to e4)
    print("\nMaking a move (e2 to e4)...")
    move = {"from_x": 4, "from_y": 1, "to_x": 4, "to_y": 3}
    if chess_game.is_valid_move(move):
        chess_game.make_move(move)
        chess_game.display()
        
        # Get AI response
        ai_move = chess_game.get_ai_move(chess_game.current_player_idx)
        print(f"\nAI's move: {ai_move}")
        chess_game.make_move(ai_move)
        chess_game.display()
    else:
        print("Invalid move!")
def __init__(self, x: int, y: int, width: int, height: int, parent=None, 
                 color: Optional[Tuple[int, int, int]] = None,
                 border_color: Optional[Tuple[int, int, int]] = None,
                 border_width: int = 0,
                 border_radius: int = 0,
                 scrollable: bool = False,
                 padding: int = 10):
        """
        Initialize a panel.
        
        Args:
            x: X position
            y: Y position
            width: Panel width
            height: Panel height
            parent: Parent UI element
            color: Panel background color (or None for theme default)
            border_color: Border color (or None for theme default)
            border_width: Border width
            border_radius: Radius for rounded corners
            scrollable: Whether the panel is scrollable
            padding: Internal padding
        """
        super().__init__(x, y, width, height, parent)
        self.color = color
        self.border_color = border_color
        self.border_width = border_width
        self.border_radius = border_radius
        self.scrollable = scrollable
        self.padding = padding
        self.scroll_y = 0
        self.max_scroll_y = 0
        self.dragging = False
        self.drag_start_y = 0
        self.drag_start_scroll = 0
    
    def draw(self, surface: pygame.Surface):
        """Draw the panel"""
        if not self.visible:
            return

 for color, image in self.stone_images.items():
            self.stone_images[color] = pygame.transform.scale(image, (stone_size, stone_size))
    
    def screen_to_board(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to board coordinates (0-(size-1), 0-(size-1))"""
        # Calculate grid coordinates (floating point)
        grid_x = (screen_x - self.board_offset_x) / self.cell_size
        grid_y = (screen_y - self.board_offset_y) / self.cell_size
        
        # Round to nearest intersection
        x = round(grid_x)
        y = round(grid_y)
        
        # Check if the coordinates are valid
        if not (0 <= x < self.game.board_size and 0 <= y < self.game.board_size):
            return (-1, -1)
            
        return (x, y)
    
    def board_to_screen(self, board_x: int, board_y: int) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates"""
        x = self.board_offset_x + board_x * self.cell_size
        y = self.board_offset_y + board_y * self.cell_size
        
        return (x, y)
    
    def _draw_board(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the Go board background"""
        # Draw the board background
        board_color = Theme.get(self.theme, "board_light") if self.board_style == "light" else (224, 165, 100)
        
        board_rect = pygame.Rect(
            abs_rect.left + self.board_offset_x - self.cell_size * 0.6,
            abs_rect.top + self.board_offset_y - self.cell_size * 0.6,
            self.cell_size * (self.game.board_size - 1 + 1.2),
            self.cell_size * (self.game.board_size - 1 + 1.2)
        )
        
        pygame.draw.rect(surface, board_color

, surface, border_radius=8)
        pygame.draw.rect(surface, Theme.get(self.theme, "border"), board_rect, 2, border_radius=8)

        # Draw grid lines
        grid_color = (0, 0, 0)
        for i in range(self.game.board_size):
            # Draw horizontal line
            start_x = abs_rect.left + self.board_offset_x
            end_x = abs_rect.left + self.board_offset_x + self.cell_size * (self.game.board_size - 1)
            y = abs_rect.top + self.board_offset_y + i * self.cell_size
            pygame.draw.line(surface, grid_color, (start_x, y), (end_x, y), 2 if i == 0 or i == self.game.board_size - 1 else 1)
            
            # Draw vertical line
            start_y = abs_rect.top + self.board_offset_y
            end_y = abs_rect.top + self.board_offset_y + self.cell_size * (self.game.board_size - 1)
            x = abs_rect.left + self.board_offset_x + i * self.cell_size
            pygame.draw.line(surface, grid_color, (x, start_y), (x, end_y), 2 if i == 0 or i == self.game.board_size - 1 else 1)
        
        # Draw star points (hoshi)
        star_points = self._get_star_points()
        for x, y in star_points:
            screen_x, screen_y = self.board_to_screen(x, y)
            pygame.draw.circle(
                surface, 
                grid_color, 
                (abs_rect.left + screen_x, abs_rect.top + screen_y), 
                self.cell_size * 0.1
            )
        
        # Draw coordinates if in debug mode
        if config.debug:
            font = pygame.font.SysFont("Arial", 10)
            
            # Draw column labels (A, B, C, ...)
            for i in range(self.game.board_size):
                coord_text = chr(65 + i) if i < 8 else chr(66 + i)  # Skip 'I'
                text_surface = font.render(coord_text, True, Theme.get(self.theme, "text_secondary"))
                text_x = abs_rect.left + self.board_offset_x + i * self.cell_size - text_surface.get_width() // 2
                text_y = abs_rect.top + self.board_offset_y + self.cell_size * (self.game.board_size - 1) + 10
                surface.blit(text_surface, (text_x, text_y))
                
                # Row labels (1, 2, 3, ...)
                row_text = str(i + 1)
                text_surface = font.render(row_text, True, Theme.get(self.theme, "text_secondary"))
                text_x = abs_rect.left + self.board_offset_x - 15
                text_y = abs_rect.top + self.board_offset_y + i * self.cell_size - text_surface.get_height() // 2
                surface.blit(text_surface, (text_x, text_y))
    
    def _get_star_points(self) -> List[Tuple[int, int]]:
        """Get the positions of star points based on board size"""
        star_points = []
        
        board_size = self.game.board_size
        
        if board_size == 19:
            # Traditional 19x19 board has 9 star points
            points = [3, 9, 15]
            for y in points:
                for x in points:
                    star_points.append((x, y))
        elif board_size == 13:
            # 13x13 board has 5 star points
            points = [3, 6, 9]
            middle = 6
            star_points.append((middle, middle))
            for point in points:
                if point != middle:
                    star_points.append((point, middle))
                    star_points.append((middle, point))
        elif board_size == 9:
            # 9x9 board has 5 star points
            points = [2, 4, 6]
            middle = 4
            star_points.append((middle, middle))
            for point in points:
                if point != middle:
                    star_points.append((point, middle))
                    star_points.append((middle, point))
        
        return star_points
    
    def _draw_pieces(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the Go stones"""
        # Draw stones on the board
        for y in range(self.game.board_size):
            for x in range(self.game.board_size):
                stone = self.game.board[y][x]
                if stone != Stone.EMPTY and (x, y) not in self.animated_pieces:
                    self._draw_stone(surface, abs_rect, stone, x, y)
        
        # Draw hover stone preview
        if self.hover_pos and not self.game.is_game_over:
            x, y = self.hover_pos
            # Check if the move would be valid
            move = {"type": "place", "x": x, "y": y}
            if self.game.is_valid_move(move):
                stone = self.game.current_player.stone
                self._draw_stone_preview(surface, abs_rect, stone, x, y)
        
        # Draw animated stones
        for (x, y), stone_info in list(self.animated_pieces.items()):
            stone, start_pos, end_pos, progress = stone_info
            
            if progress >= 1.0:
                # Animation complete, remove from animated pieces
                del self.animated_pieces[(x, y)]
                
                # Draw the stone at its final position
                if stone != Stone.EMPTY:
                    self._draw_stone(surface, abs_rect, stone, x, y)
            else:
                # Interpolate position
                start_x, start_y = self.board_to_screen(start_pos[0], start_pos[1])
                end_x, end_y = self.board_to_screen(end_pos[0], end_pos[1])
                
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress
                
                self._draw_stone_at_pos(surface, abs_rect, stone, current_x, current_y)
                
                # Update progress
                self.animated_pieces[(x, y)] = (stone, start_pos, end_pos, min(1.0, progress + 0.05))
    
    def _draw_stone(self, surface: pygame.Surface, abs_rect: pygame.Rect, stone: Stone, x: int, y: int):
        """Draw a Go stone at board coordinates"""
        screen_x, screen_y = self.board_to_screen(x, y)
        self._draw_stone_at_pos(surface, abs_rect, stone, screen_x, screen_y)
    
    def _draw_stone_at_pos(self, surface: pygame.Surface, abs_rect: pygame.Rect, stone: Stone, x: float, y: float):
        """Draw a Go stone at specific screen coordinates"""
        color = "black" if stone == Stone.BLACK else "white"
        
        if color in self.stone_images:
            image = self.stone_images[color]
            image_rect = image.get_rect(center=(abs_rect.left + x, abs_rect.top + y))
            surface.blit(image, image_rect)
        else:
            # Fallback if image not available
            radius = int(self.cell_size * 0.45)
            center = (abs_rect.left + x, abs_rect.top + y)
            
            # Draw shadow
            shadow_radius = radius + 2
            shadow_center = (center[0] + 2, center[1] + 2)
            pygame.draw.circle(surface, (0, 0, 0, 100), shadow_center, shadow_radius)
            
            # Draw stone
            stone_color = (0, 0, 0) if stone == Stone.BLACK else (255, 255, 255)
            pygame.draw.circle(surface, stone_color, center, radius)
            
            # Draw border
            border_color = (64, 64, 64) if stone == Stone.BLACK else (192, 192, 192)
            pygame.draw.circle(surface, border_color, center, radius, 1)
    
    def _draw_stone_preview(self, surface: pygame.Surface, abs_rect: pygame.Rect, stone: Stone, x: int, y: int):
        """Draw a semi-transparent stone preview at the hover position"""
        screen_x, screen_y = self.board_to_screen(x, y)
        center = (abs_rect.left + screen_x, abs_rect.top + screen_y)
        radius = int(self.cell_size * 0.45)
        
        # Draw semi-transparent stone
        stone_color = (0, 0, 0, 160) if stone == Stone.BLACK else (255, 255, 255, 160)
        
        # Create a surface with alpha channel
        preview_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(preview_surface, stone_color, (radius, radius), radius)
        
        # Draw border
        border_color = (64, 64, 64, 160) if stone == Stone.BLACK else (192, 192, 192, 160)
        pygame.draw.circle(preview_surface, border_color, (radius, radius), radius, 1)
        
        # Blit the preview
        preview_rect = preview_surface.get_rect(center=center)
        surface.blit(preview_surface, preview_rect)
    
    def _draw_highlights(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw highlights for last move, etc."""
        # Draw last move marker
        if self.show_last_move and self.last_move:
            if self.last_move["type"] == "place":
                x, y = self.last_move["x"], self.last_move["y"]
                screen_x, screen_y = self.board_to_screen(x, y)
                center = (abs_rect.left + screen_x, abs_rect.top + screen_y)
                
                # Draw a small marker
                marker_color = (255, 255, 255) if self.game.board[y][x] == Stone.BLACK else (0, 0, 0)
                pygame.draw.circle(surface, marker_color, center, self.cell_size * 0.1)
    
    def _handle_board_click(self, board_x: int, board_y: int, event: pygame.event.Event) -> bool:
        """Handle a click on the board"""
        if not (0 <= board_x < self.game.board_size and 0 <= board_y < self.game.board_size):
            return False
            
        # Try to make a move
        move = {"type": "place", "x": board_x, "y": board_y}
        
        if self.game.is_valid_move(move):
            # Store the last move
            self.last_move = move
            
            # Make the move
            self.game.make_move(move)
            
            # Update the board
            self.update_from_game()
            
            return True
            
        return False
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        result = super().handle_event(event)
        if result:
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        # Track mouse position for hover effect
        if event.type == pygame.MOUSEMOTION:
            pos_x, pos_y = event.pos[0] - abs_rect.left, event.pos[1] - abs_rect.top
            board_x, board_y = self.screen_to_board(pos_x, pos_y)
            
            if 0 <= board_x < self.game.board_size and 0 <= board_y < self.game.board_size:
                self.hover_pos = (board_x, board_y)
            else:
                self.hover_pos = None
                
        return False
    
    def update_from_game(self):
        """Update the board state from the game"""
        # Nothing special to update for Go
        pass

class MahjongBoard(GameBoard):
    """Mahjong board UI for the Mahjong game"""
    
    def __init__(self, x: int, y: int, width: int, height: int, game: MahjongGame, parent=None):
        """
        Initialize a Mahjong board.
        
        Args:
            x: X position
            y: Y position
            width: Board width
            height: Board height
            game: MahjongGame instance
            parent: Parent UI element
        """
        super().__init__(x, y, width, height, game, parent)
        self.tile_width = 40
        self.tile_height = 60
        self.tile_images = {}
        self.selected_tile_idx = None
        self.player_view = 0  # Which player's perspective to show
        
        # Load tile images
        self._load_tile_images()
        
        # Calculate layout dimensions
        self._update_dimensions()
    
    def _update_dimensions(self):
        """Update board dimensions based on available space"""
        super()._update_dimensions()
        
        # Calculate tile dimensions based on available space
        self.tile_width = min(50, self.rect.width // 15)
        self.tile_height = int(self.tile_width * 1.5)
        
        # Resize tile images
        self._resize_tile_images()
    
    def _load_tile_images(self):
        """Load Mahjong tile images"""
        tile_set = config.piece_set
        tile_dir = IMAGE_DIR / "mahjong" / tile_set
        
        if not tile_dir.exists():
            # Fall back to default tile set
            tile_set = "default"
            tile_dir = IMAGE_DIR / "mahjong" / tile_set
            
            if not tile_dir.exists():
                # Create directory if it doesn't exist
                tile_dir.mkdir(parents=True, exist_ok=True)
        
        self.tile_images = {}
        
        # Try to load tile images for each type
        for suit in ["dots", "bamboo", "characters"]:
            for value in range(1, 10):
                image_path = tile_dir / f"{suit}_{value}.png"
                self._load_tile_image(TileType[suit.upper()], value, image_path)
        
        # Load wind tiles
        for wind in ["east", "south", "west", "north"]:
            image_path = tile_dir / f"wind_{wind}.png"
            self._load_tile_image(TileType.WIND, Wind[wind.upper()], image_path)
        
        # Load dragon tiles
        for dragon in ["red", "green", "white"]:
            image_path = tile_dir / f"dragon_{dragon}.png"
            self._load_tile_image(TileType.DRAGON, Dragon[dragon.upper()], image_path)
        
        # Load flower tiles
        for i in range(1, 5):
            image_path = tile_dir / f"flower_{i}.png"
            self._load_tile_image(TileType.FLOWER, i, image_path)
        
        # Load season tiles
        for i in range(1, 5):
            image_path = tile_dir / f"season_{i}.png"
            self._load_tile_image(TileType.SEASON, i, image_path)
    
    def _load_tile_image(self, tile_type: TileType, value: Any, image_path: Path):
        """Load a specific tile image"""
        try:
            if image_path.exists():
                image = pygame.image.load(str(image_path))
                self.tile_images[(tile_type, value)] = image
            else:
                # Create placeholder images
                self._create_placeholder_tile_image(tile_type, value)
        except Exception as e:
            logger.error(f"Failed to load tile image {image_path}: {e}")
            # Create placeholder images
            self._create_placeholder_tile_image(tile_type, value)
    
    def _create_placeholder_tile_image(self, tile_type: TileType, value: Any):
        """Create a placeholder tile image"""
        width, height = 40, 60
        image = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw tile background and border
        pygame.draw.rect(image, (245, 245, 220), (0, 0, width, height), border_radius=4)
        pygame.draw.rect(image, (100, 100, 100), (0, 0, width, height), 1, border_radius=4)
        
        # Draw symbol
        font = pygame.font.SysFont("Arial", 14)
        
        if tile_type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            # Draw suit symbol
            value_text = str(value)
            if tile_type == TileType.DOTS:
                suit_text = "•"
            elif tile_type == TileType.BAMBOO:
                suit_text = "/"
            else:  # Characters
                suit_text = "字"
                
            value_surf = font.render(value_text, True, (0, 0, 0))
            suit_surf = font.render(suit_text, True, (0, 0, 0))
            
            image.blit(value_surf, (width // 2 - value_surf.get_width() // 2, 10))
            image.blit(suit_surf, (width // 2 - suit_surf.get_width() // 2, 30))
            
        elif tile_type == TileType.WIND:
            # Draw wind symbol
            wind_text = value.name[0]
            text_surf = font.render(wind_text, True, (0, 0, 200))
            image.blit(text_surf, (width // 2 - text_surf.get_width() // 2, height // 2 - text_surf.get_height() // 2))
            
        elif tile_type == TileType.DRAGON:
            # Draw dragon symbol
            if value == Dragon.RED:
                text = "R"
                color = (200, 0, 0)
            elif value == Dragon.GREEN:
                text = "G"
                color = (0, 150, 0)
            else:  # White
                text = "W"
                color = (100, 100, 100)
                
            text_surf = font.render(text, True, color)
            image.blit(text_surf, (width // 2 - text_surf.get_width() // 2, height // 2 - text_surf.get_height() // 2))
            
        elif tile_type in [TileType.FLOWER, TileType.SEASON]:
            # Draw flower/season symbol
            prefix = "F" if tile_type == TileType.FLOWER else "S"
            text = f"{prefix}{value}"
            text_surf = font.render(text, True, (0, 150, 0) if tile_type == TileType.FLOWER else (200, 100, 0))
            image.blit(text_surf, (width // 2 - text_surf.get_width() // 2, height // 2 - text_surf.get_height() // 2))
        
        # Store the image
        self.tile_images[(tile_type, value)] = image
    
    def _resize_tile_images(self):
        """Resize tile images to fit the current dimensions"""
        for key, image in self.tile_images.items():
            self.tile_images[key] = pygame.transform.scale(image, (self.tile_width, self.tile_height))
    
    def _draw_board(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the Mahjong board background"""
        # Draw a green felt background
        background_color = (0, 100, 0)
        pygame.draw.rect(surface, background_color, abs_rect)
        
        # Draw table border
        border_rect = pygame.Rect(
            abs_rect.left + 10,
            abs_rect.top + 10,
            abs_rect.width - 20,
            abs_rect.height - 20
        )
        pygame.draw.rect(surface, (0, 80, 0), border_rect, border_radius=10)
        pygame.draw.rect(surface, (0, 60, 0), border_rect, 2, border_radius=10)
        
        # Draw game info
        font = pygame.font.SysFont("Arial", 14)
        
        # Draw wall size
        wall_text = f"Wall: {len(self.game.wall)} tiles"
        text_surf = font.render(wall_text, True, WHITE)
        surface.blit(text_surf, (abs_rect.left + 20, abs_rect.top + 20))
        
        # Draw round wind
        wind_text = f"Round Wind: {self.game.round_wind.value}"
        text_surf = font.render(wind_text, True, WHITE)
        surface.blit(text_surf, (abs_rect.left + 20, abs_rect.top + 40))
        
        # Draw dora indicators
        dora_text = "Dora Indicators: " + ", ".join(str(tile) for tile in self.game.dora_indicators)
        text_surf = font.render(dora_text, True, WHITE)
        surface.blit(text_surf, (abs_rect.left + 20, abs_rect.top + 60))
    
    def _draw_pieces(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw the Mahjong tiles"""
        # Get player view data
        player_view = self.game.get_player_hand_view(self.player_view)
        player_data = player_view["players"][self.player_view]
        
        # Draw player's hand
        self._draw_player_hand(surface, abs_rect, player_data)
        
        # Draw discards
        self._draw_discards(surface, abs_rect, player_view)
        
        # Draw current discard
        if player_view["current_discard"]:
            self._draw_current_discard(surface, abs_rect, player_view["current_discard"])
    
    def _draw_player_hand(self, surface: pygame.Surface, abs_rect: pygame.Rect, player_data: Dict[str, Any]):
        """Draw the current player's hand"""
        # Calculate starting position for hand
        hand_center_x = abs_rect.centerx
        hand_y = abs_rect.bottom - self.tile_height - 20
        
        # Draw player name
        font = pygame.font.SysFont("Arial", 16, bold=True)
        name_text = f"{player_data['name']} ({player_data['wind']})"
        name_surf = font.render(name_text, True, WHITE)
        name_rect = name_surf.get_rect(center=(hand_center_x, hand_y - 20))
        surface.blit(name_surf, name_rect)
        
        # Draw hand tiles
        hand = player_data.get("hand", [])
        total_width = len(hand) * self.tile_width
        start_x = hand_center_x - total_width // 2
        
        for i, tile_str in enumerate(hand):
            # Parse tile string
            tile = self._parse_tile_str(tile_str)
            if not tile:
                continue
                
            # Draw tile
            tile_x = start_x + i * self.tile_width
            tile_y = hand_y
            
            # Highlight selected tile
            if i == self.selected_tile_idx:
                highlight_rect = pygame.Rect(
                    tile_x - 2, 
                    tile_y - 2, 
                    self.tile_width + 4, 
                    self.tile_height + 4
                )
                pygame.draw.rect(surface, (0, 255, 0), highlight_rect, 2, border_radius=5)
            
            # Draw the tile
            self._draw_tile(surface, tile, tile_x, tile_y)
        
        # Draw revealed sets
        revealed_sets = player_data.get("revealed_sets", [])
        if revealed_sets:
            # Calculate position for revealed sets
            set_y = hand_y - self.tile_height - 30
            set_x = hand_center_x - (len(revealed_sets) * 3 * self.tile_width) // 2
            
            for i, revealed_set in enumerate(revealed_sets):
                set_type, tiles = revealed_set
                
                # Draw each tile in the set
                for j, tile_str in enumerate(tiles):
                    tile = self._parse_tile_str(tile_str)
                    if not tile:
                        continue
                        
                    tile_x = set_x + i * 3 * self.tile_width + j * self.tile_width * 0.7
                    tile_y = set_y
                    
                    self._draw_tile(surface, tile, tile_x, tile_y)
    
    def _draw_discards(self, surface: pygame.Surface, abs_rect: pygame.Rect, player_view: Dict[str, Any]):
        """Draw the discards for all players"""
        # Calculate positions for each player's discards
        positions = [
            (abs_rect.centerx, abs_rect.bottom - self.tile_height * 3),  # Bottom (current player)
            (abs_rect.left + self.tile_width * 2, abs_rect.centery),     # Left
            (abs_rect.centerx, abs_rect.top + self.tile_height * 3),     # Top
            (abs_rect.right - self.tile_width * 2, abs_rect.centery)     # Right
        ]
        
        for i, player_data in enumerate(player_view["players"]):
            x, y = positions[i]
            
            # Draw player name
            font = pygame.font.SysFont("Arial", 14)
            name_text = f"{player_data['name']} ({player_data['wind']})"
            name_surf = font.render(name_text, True, WHITE)
            
            # Position name text based on player position
            if i == 0:  # Bottom
                name_rect = name_surf.get_rect(center=(x, y - 15))
            elif i == 1:  # Left
                name_rect = name_surf.get_rect(midright=(x - 10, y - 60))
            elif i == 2:  # Top
                name_rect = name_surf.get_rect(center=(x, y + 15))
            else:  # Right
                name_rect = name_surf.get_rect(midleft=(x + 10, y - 60))
                
            surface.blit(name_surf, name_rect)
            
            # Draw discards
            discards = player_data.get("discards", [])
            if not discards:
                continue
                
            # Calculate discard layout
            cols = 6
            rows = (len(discards) + cols - 1) // cols
            
            # Adjust position based on player
            if i == 0:  # Bottom
                start_x = x - (cols * self.tile_width * 0.6) // 2
                start_y = y
                dx, dy = self.tile_width * 0.6, self.tile_height * 0.6
            elif i == 1:  # Left
                start_x = x
                start_y = y - (rows * self.tile_height * 0.6) // 2
                dx, dy = self.tile_width * 0.6, self.tile_height * 0.6
            elif i == 2:  # Top
                start_x = x - (cols * self.tile_width * 0.6) // 2
                start_y = y - self.tile_height * 0.6 * rows
                dx, dy = self.tile_width * 0.6, self.tile_height * 0.6
            else:  # Right
                start_x = x - self.tile_width * 0.6
                start_y = y - (rows * self.tile_height * 0.6) // 2
                dx, dy = self.tile_width * 0.6, self.tile_height * 0.6
            
            # Draw each discard
            for j, tile_str in enumerate(discards):
                col = j % cols
                row = j // cols
                
                tile = self._parse_tile_str(tile_str)
                if not tile:
                    continue
                    
                tile_x = start_x + col * dx
                tile_y = start_y + row * dy
                
                # Draw at smaller scale
                self._draw_tile(surface, tile, tile_x, tile_y, scale=0.6)
    
    def _draw_current_discard(self, surface: pygame.Surface, abs_rect: pygame.Rect, tile_str: str):
        """Draw the current discard in the center"""
        tile = self._parse_tile_str(tile_str)
        if not tile:
            return
            
        # Draw in center of board
        tile_x = abs_rect.centerx - self.tile_width // 2
        tile_y = abs_rect.centery - self.tile_height // 2
        
        # Draw with highlight
        highlight_rect = pygame.Rect(
            tile_x - 5, 
            tile_y - 5, 
            self.tile_width + 10, 
            self.tile_height + 10
        )
        pygame.draw.rect(surface, (255, 255, 0, 150), highlight_rect, border_radius=5)
        
        self._draw_tile(surface, tile, tile_x, tile_y)
    
    def _draw_tile(self, surface: pygame.Surface, tile: Tuple[TileType, Any], x: int, y: int, scale: float = 1.0):
        """Draw a Mahjong tile"""
        if tile in self.tile_images:
            image = self.tile_images[tile]
            
            # Scale if needed
            if scale != 1.0:
                width = int(self.tile_width * scale)
                height = int(self.tile_height * scale)
                image = pygame.transform.scale(image, (width, height))
                
            surface.blit(image, (x, y))
        else:
            # Fallback if image not available
            width = int(self.tile_width * scale)
            height = int(self.tile_height * scale)
            
            tile_rect = pygame.Rect(x, y, width, height)
            pygame.draw.rect(surface, (245, 245, 220), tile_rect, border_radius=4)
            pygame.draw.rect(surface, (100, 100, 100), tile_rect, 1, border_radius=4)
            
            # Draw simple text
            font = pygame.font.SysFont("Arial", int(14 * scale))
            text = f"{tile[0].name[0]}{tile[1]}"
            text_surf = font.render(text, True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
            surface.blit(text_surf, text_rect)
    
    def _parse_tile_str(self, tile_str: str) -> Optional[Tuple[TileType, Any]]:
        """Parse a tile string into a tile tuple"""
        try:
            if "Dots" in tile_str:
                value = int(tile_str.split()[0])
                return (TileType.DOTS, value)
            elif "Bamboo" in tile_str:
                value = int(tile_str.split()[0])
                return (TileType.BAMBOO, value)
            elif "Characters" in tile_str:
                value = int(tile_str.split()[0])
                return (TileType.CHARACTERS, value)
            elif "Wind" in tile_str:
                wind_name = tile_str.split()[0].upper()
                return (TileType.WIND, Wind[wind_name])
            elif "Dragon" in tile_str:
                dragon_name = tile_str.split()[0].upper()
                return (TileType.DRAGON, Dragon[dragon_name])
            elif "Flower" in tile_str:
                value = int(tile_str.split()[1])
                return (TileType.FLOWER, value)
            elif "Season" in tile_str:
                value = int(tile_str.split()[1])
                return (TileType.SEASON, value)
            else:
                return None
        except Exception as e:
            logger.error(f"Error parsing tile string '{tile_str}': {e}")
            return None
    
    def _draw_highlights(self, surface: pygame.Surface, abs_rect: pygame.Rect):
        """Draw highlights for valid moves, etc."""
        # Nothing to highlight in Mahjong
        pass
    
    def _handle_board_click(self, board_x: int, board_y: int, event: pygame.event.Event) -> bool:
        """Handle a click on the board - not used in Mahjong"""
        # Mahjong doesn't use board coordinates for clicks
        return False
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events"""
        if super().handle_event(event):
            return True
            
        if not self.visible or not self.enabled:
            return False
            
        abs_rect = self.get_absolute_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if clicking on player's hand
            player_view = self.game.get_player_hand_view(self.player_view)
            player_data = player_view["players"][self.player_view]
            hand = player_data.get("hand", [])
            
            # Calculate hand position
            hand_center_x = abs_rect.centerx
            hand_y = abs_rect.bottom - self.tile_height - 20
            total_width = len(hand) * self.tile_width
            start_x = hand_center_x - total_width // 2
            
            for i in range(len(hand)):
                tile_x = start_x + i * self.tile_width
                tile_rect = pygame.Rect(
                    abs_rect.left + tile_x,
                    abs_rect.top + hand_y,
                    self.tile_width,
                    self.tile_height
                )
                
                if tile_rect.collidepoint(event.pos):
                    self.selected_tile_idx = i
                    return True
            
            # Check if clicking on current discard for potential calls
            if player_view["current_discard"]:
                discard_x = abs_rect.centerx - self.tile_width // 2
                discard_y = abs_rect.centery - self.tile_height // 2
                discard_rect = pygame.Rect(
                    abs_rect.left + discard_x,
                    abs_rect.top + discard_y,
                    self.tile_width,
                    self.tile_height
                )
                
                if discard_rect.collidepoint(event.pos):
                    # Check for valid calls
                    valid_calls = player_view.get("valid_calls", [])
                    if valid_calls:
                        # Handle calls (would need UI for selecting which call to make)
                        pass
                    return True
        
        return False
    
    def update_from_game(self):
        """Update the board state from the game"""
        # Nothing special to update for Mahjong
        pass

# Main game UI integration with the engine
class GameApp:
    """Main application class for the game visualizer"""
    
    def __init__(self):
        """Initialize the game application"""
        global SCREEN_WIDTH, SCREEN_HEIGHT
        
        # Initialize pygame if available
        if not PYGAME_AVAILABLE:
            print("Error: pygame not available. Cannot run the game visualizer.")
            return
            
        # Initialize game engine if available
        if GAMES_AVAILABLE:
            self.engine = GameEngine()
            self.engine.register_game(MahjongGame)
            self.engine.register_game(ChessGame)
            self.engine.register_game(GoGame)
            
            # Register AI profiles
            self._register_ai_profiles()
        else:
            self.engine = None
            print("Warning: Game engine not available. Running in demo mode.")
        
        # Initialize Sully integration if available
        if SULLY_AVAILABLE and config.sully_integration:
            try:
                self.sully = Sully()
                self.sully_available = True
                print("Sully integration initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Sully: {e}")
                self.sully = None
                self.sully_available = False
        else:
            self.sully = None
            self.sully_available = False
        
        # Set up the display
        flags = pygame.RESIZABLE
        if config.fullscreen:
            flags |= pygame.FULLSCREEN
            
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        pygame.display.set_caption("Sully Game System")
        
        # Set up the clock
        self.clock = pygame.time.Clock()
        
        # Set up the sound manager
        self.sound_manager = SoundManager()
        
        # Current game and state
        self.current_game = None
        self.current_game_type = None
        self.current_board = None
        self.is_running = True
        self.current_screen = "main_menu"  # main_menu, game, settings, etc.
        
        # Auto-save timer
        self.last_save_time = time.time()
        
        # Initialize UI elements
        self._init_ui()
        
        # Start background music
        if config.music_enabled:
            self.sound_manager.play_music("menu")
    
    def _register_ai_profiles(self):
        """Register AI profiles with the game engine"""
        if not self.engine:
            return
            
        # Chess AI profiles
        self.engine.register_ai_profile("chess_beginner", ChessAIPlayer, {
            "difficulty": GameDifficulty.BEGINNER,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("chess_easy", ChessAIPlayer, {
            "difficulty": GameDifficulty.EASY,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("chess_medium", ChessAIPlayer, {
            "difficulty": GameDifficulty.MEDIUM,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("chess_hard", ChessAIPlayer, {
            "difficulty": GameDifficulty.HARD,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("chess_expert", ChessAIPlayer, {
            "difficulty": GameDifficulty.EXPERT,
            "adaptive": False
        })
        
        # Go AI profiles
        self.engine.register_ai_profile("go_beginner", GoAIPlayer, {
            "difficulty": GameDifficulty.BEGINNER,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("go_easy", GoAIPlayer, {
            "difficulty": GameDifficulty.EASY,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("go_medium", GoAIPlayer, {
            "difficulty": GameDifficulty.MEDIUM,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("go_hard", GoAIPlayer, {
            "difficulty": GameDifficulty.HARD,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("go_expert", GoAIPlayer, {
            "difficulty": GameDifficulty.EXPERT,
            "adaptive": False
        })
        
        # Mahjong AI profiles
        self.engine.register_ai_profile("mahjong_easy", MahjongAIPlayer, {
            "difficulty": GameDifficulty.EASY,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("mahjong_medium", MahjongAIPlayer, {
            "difficulty": GameDifficulty.MEDIUM,
            "adaptive": True
        })
        
        self.engine.register_ai_profile("mahjong_hard", MahjongAIPlayer, {
            "difficulty": GameDifficulty.HARD,
            "adaptive": True
        })
    
    def _init_ui(self):
        """Initialize UI elements"""
        self.ui_elements = {}
        
        # Main menu screen
        self.ui_elements["main_menu"] = Panel(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, None)
        
        # Main menu title
        title_label = Label(
            SCREEN_WIDTH // 2 - 200, 
            50, 
            400, 
            50, 
            "Sully Game System", 
            self.ui_elements["main_menu"],
            font_size=32,
            align="center"
        )
        
        # Game buttons
        button_width = 200
        button_height = 50
        button_spacing = 20
        start_y = 150
        
        chess_button = Button(
            SCREEN_WIDTH // 2 - button_width // 2,
            start_y,
            button_width,
            button_height,
            "Chess",
            lambda: self._show_game_setup("chess"),
            self.ui_elements["main_menu"],
            icon="chess"
        )
        
        go_button = Button(
            SCREEN_WIDTH // 2 - button_width // 2,
            start_y + button_height + button_spacing,
            button_width,
            button_height,
            "Go",
            lambda: self._show_game_setup("go"),
            self.ui_elements["main_menu"],
            icon="go"
        )
        
        mahjong_button = Button(
            SCREEN_WIDTH // 2 - button_width // 2,
            start_y + 2 * (button_height + button_spacing),
            button_width,
            button_height,
            "Mahjong",
            lambda: self._show_game_setup("mahjong"),
            self.ui_elements["main_menu"],
            icon="mahjong"
        )
        
        # Options button
        options_button = Button(
            SCREEN_WIDTH // 2 - button_width // 2,
            start_y + 3 * (button_height + button_spacing),
            button_width,
            button_height,
            "Options",
            self._show_options,
            self.ui_elements["main_menu"],
            icon="settings"
        )
        
        # Exit button
        exit_button = Button(
            SCREEN_WIDTH // 2 - button_width // 2,
            start_y + 4 * (button_height + button_spacing),
            button_width,
            button_height,
            "Exit",
            self.quit,
            self.ui_elements["main_menu"],
            icon="exit"
        )
        
        # Version info
        version_label = Label(
            10, 
            SCREEN_HEIGHT - 30, 
            200, 
            20, 
            "Version 1.0", 
            self.ui_elements["main_menu"],
            color=Theme.get(config.theme, "text_secondary")
        )
        
        # Game setup screen (will be populated dynamically)
        self.ui_elements["game_setup"] = Panel(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, None)
        
        # Game screen
        self.ui_elements["game"] = Panel(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, None)
        
        # Game sidebar
        sidebar_width = 250
        self.game_sidebar = Panel(
            SCREEN_WIDTH - sidebar_width, 
            0, 
            sidebar_width, 
            SCREEN_HEIGHT, 
            self.ui_elements["game"],
            color=Theme.get(config.theme, "panel"),
            border_width=1,
            border_color=Theme.get(config.theme, "border")
        )
        
        # Game board container (will be replaced with actual game board)
        self.game_board_container = Panel(
            0,
            0,
            SCREEN_WIDTH - sidebar_width,
            SCREEN_HEIGHT,
            self.ui_elements["game"]
        )
        
        # Add sidebar elements
        sidebar_title = Label(
            10,
            20,
            sidebar_width - 20,
            30,
            "Game Controls",
            self.game_sidebar,
            font_size=20,
            align="center",
            bold=True
        )
        
        # Back to menu button
        back_button = Button(
            10,
            SCREEN_HEIGHT - 60,
            sidebar_width - 20,
            40,
            "Back to Menu",
            self._back_to_menu,
            self.game_sidebar,
            icon="home"
        )
    
    def _show_game_setup(self, game_type: str):
        """Show the game setup screen for a specific game type"""
        self.current_screen = "game_setup"
        self.current_game_type = game_type
        
        # Clear existing elements
        self.ui_elements["game_setup"].children = []
        
        # Add title
        title_text = f"{game_type.capitalize()} Game Setup"
        title_label = Label(
            SCREEN_WIDTH // 2 - 200, 
            50, 
            400, 
            50, 
            title_text, 
            self.ui_elements["game_setup"],
            font_size=24,
            align="center",
            bold=True
        )
        
        # Add back button
        back_button = Button(
            50,
            50,
            100,
            40,
            "Back",
            lambda: self._set_screen("main_menu"),
            self.ui_elements["game_setup"],
            icon="back"
        )
        
        # Add game-specific setup options
        if game_type == "chess":
            self._setup_chess_game()
        elif game_type == "go":
            self._setup_go_game()
        elif game_type == "mahjong":
            self._setup_mahjong_game()
    
    def _setup_chess_game(self):
        """Set up the chess game options"""
        panel_width = 400
        panel_height = 400
        panel_x = SCREEN_WIDTH // 2 - panel_width // 2
        panel_y = 120
        
        setup_panel = Panel(
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            self.ui_elements["game_setup"],
            color=Theme.get(config.theme, "panel"),
            border_width=1,
            border_color=Theme.get(config.theme, "border"),
            border_radius=10
        )
        
        # Player 1 settings
        Label(20, 20, 200, 30, "White Player:", setup_panel, font_size=18, bold=True)
        
        player1_type = Dropdown(
            20, 60, 160, 30,
            ["Human", "AI (Easy)", "AI (Medium)", "AI (Hard)", "AI (Expert)"],
            None,
            setup_panel
        )
        
        player1_name = Label(20, 100, 80, 30, "Name:", setup_panel)
        player1_name_input = Label(100, 100, 150, 30, "Player 1", setup_panel)  # Would be an input field
        
        # Player 2 settings
        Label(20, 150, 200, 30, "Black Player:", setup_panel, font_size=18, bold=True)
        
        player2_type = Dropdown(
            20, 190, 160, 30,
            ["Human", "AI (Easy)", "AI (Medium)", "AI (Hard)", "AI (Expert)"],
            None,
            setup_panel
        )
        player2_type.set_selected_index(2)  # Default to Medium AI
        
        player2_name = Label(20, 230, 80, 30, "Name:", setup_panel)
        player2_name_input = Label(100, 230, 150, 30, "AI Player", setup_panel)  # Would be an input field
        
        # Game settings
        Label(20, 280, 200, 30, "Game Settings:", setup_panel, font_size=18, bold=True)
        
        timed_game = Checkbox(20, 320, 200, 30, "Timed Game", False, None, setup_panel)
        
        # Start game button
        start_button = Button(
            panel_width // 2 - 75,
            panel_height - 60,
            150,
            40,
            "Start Game",
            lambda: self._start_chess_game(player1_type.get_selected(), player2_type.get_selected()),
            setup_panel,
            color=Theme.get(config.theme, "accent"),
            text_color=WHITE
        )
    
    def _setup_go_game(self):
        """Set up the Go game options"""
        panel_width = 400
        panel_height = 450
        panel_x = SCREEN_WIDTH // 2 - panel_width // 2
        panel_y = 120
        
        setup_panel = Panel(
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            self.ui_elements["game_setup"],
            color=Theme.get(config.theme, "panel"),
            border_width=1,
            border_color=Theme.get(config.theme, "border"),
            border_radius=10
        )
        
        # Player 1 settings
        Label(20, 20, 200, 30, "Black Player:", setup_panel, font_size=18, bold=True)
        
        player1_type = Dropdown(
            20, 60, 160, 30,
            ["Human", "AI (Easy)", "AI (Medium)", "AI (Hard)", "AI (Expert)"],
            None,
            setup_panel
        )
        
        player1_name = Label(20, 100, 80, 30, "Name:", setup_panel)
        player1_name_input = Label(100, 100, 150, 30, "Player 1", setup_panel)  # Would be an input field
        
        # Player 2 settings
        Label(20, 150, 200, 30, "White Player:", setup_panel, font_size=18, bold=True)
        
        player2_type = Dropdown(
            20, 190, 160, 30,
            ["Human", "AI (Easy)", "AI (Medium)", "AI (Hard)", "AI (Expert)"],
            None,
            setup_panel
        )
        player2_type.set_selected_index(2)  # Default to Medium AI
        
        player2_name = Label(20, 230, 80, 30, "Name:", setup_panel)
        player2_name_input = Label(100, 230, 150, 30, "AI Player", setup_panel)  # Would be an input field
        
        # Game settings
        Label(20, 280, 200, 30, "Game Settings:", setup_panel, font_size=18, bold=True)
        
        # Board size
        board_size_label = Label(20, 320, 100, 30, "Board Size:", setup_panel)
        board_size = Dropdown(
            130, 320, 100, 30,
            ["9×9", "13×13", "19×19"],
            None,
            setup_panel
        )
        board_size.set_selected_index(2)  # Default to 19×19
        
        # Komi
        komi_label = Label(20, 360, 100, 30, "Komi:", setup_panel)
        komi = Dropdown(
            130, 360, 100, 30,
            ["5.5", "6.5", "7.5"],
            None,
            setup_panel
        )
        komi.set_selected_index(1)  # Default to 6.5
        
        # Start game button
        start_button = Button(
            panel_width // 2 - 75,
            panel_height - 60,
            150,
            40,
            "Start Game",
            lambda: self._start_go_game(player1_type.get_selected(), player2_type.get_selected(), board_size.get_selected()),
            setup_panel,
            color=Theme.get(config.theme, "accent"),
            text_color=WHITE
        )
    
    def _setup_mahjong_game(self):
        """Set up the Mahjong game options"""
        panel_width = 400
        panel_height = 500
        panel_x = SCREEN_WIDTH // 2 - panel_width // 2
        panel_y = 120
        
        setup_panel = Panel(
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            self.ui_elements["game_setup"],
            color=Theme.get(config.theme, "panel"),
            border_width=1,
            border_color=Theme.get(config.theme, "border"),
            border_radius=10
        )
        
        # Player settings
        Label(20, 20, 200, 30, "Players:", setup_panel, font_size=18, bold=True)
        
        player_types = []
        player_names = []
        
        for i in range(4):
            wind = ["East", "South", "West", "North"][i]
            Label(20, 60 + i * 70, 200, 30, f"{wind} Player:", setup_panel, font_size=16)
            
            player_type = Dropdown(
                20, 90 + i * 70, 160, 30,
                ["Human", "AI (Easy)", "AI (Medium)", "AI (Hard)"],
                None,
                setup_panel
            )
            
            if i > 0:
                player_type.set_selected_index(2)  # Default to Medium AI for non-human players
                
            player_types.append(player_type)
            
            name_label = Label(200, 90 + i * 70, 60, 30, "Name:", setup_panel)
            name_input = Label(260, 90 + i * 70, 120, 30, f"Player {i+1}" if i == 0 else f"AI {i}", setup_panel)
            player_names.append(name_input)
        
        # Game settings
        Label(20, 340, 200, 30, "Game Settings:", setup_panel, font_size=18, bold=True)
        
        # Ruleset
        ruleset_label = Label(20, 380, 100, 30, "Ruleset:", setup_panel)
        ruleset = Dropdown(
            130, 380, 140, 30,
            ["Japanese", "Chinese", "American"],
            None,
            setup_panel
        )
        
        # Red fives option
        red_fives = Checkbox(20, 420, 200, 30, "Use Red Fives", True, None, setup_panel)
        
        # Start game button
        start_button = Button(
            panel_width // 2 - 75,
            panel_height - 60,
            150,
            40,
            "Start Game",
            lambda: self._start_mahjong_game([pt.get_selected() for pt in player_types], ruleset.get_selected()),
            setup_panel,
            color=Theme.get(config.theme, "accent"),
            text_color=WHITE
        )
    
    def _start_chess_game(self, player1_type: str, player2_type: str):
        """Start a chess game with the selected options"""
        if not self.engine:
            self._show_error("Game engine not available")
            return
            
        try:
            # Create players
            players = []
            
            # Player 1 (White)
            if "AI" in player1_type:
                difficulty = player1_type.split("(")[1].split(")")[0].lower()
                ai_profile = f"chess_{difficulty}"
                player1 = self.engine.create_ai_player(ai_profile, f"AI {difficulty.capitalize()}")
            else:
                player1 = ChessPlayer("Player 1", PieceColor.WHITE)
                
            players.append(player1)
            
            # Player 2 (Black)
            if "AI" in player2_type:
                difficulty = player2_type.split("(")[1].split(")")[0].lower()
                ai_profile = f"chess_{difficulty}"
                player2 = self.engine.create_ai_player(ai_profile, f"AI {difficulty.capitalize()}")
            else:
                player2 = ChessPlayer("Player 2", PieceColor.BLACK)
                
            players.append(player2)
            
            # Create game
            self.current_game = self.engine.create_game("ChessGame", players)
            self.current_game_type = "chess"
            
            # Create game board
            self.current_board = ChessBoard(
                0, 0, 
                self.game_board_container.rect.width,
                self.game_board_container.rect.height,
                self.current_game,
                self.game_board_container
            )
            
            # Switch to game screen
            self._set_screen("game")
            
            # Start gameplay music
            if config.music_enabled:
                self.sound_manager.play_music("gameplay")
                
        except Exception as e:
            logger.error(f"Error starting chess game: {e}", exc_info=True)
            self._show_error(f"Failed to start game: {str(e)}")
    
    def _start_go_game(self, player1_type: str, player2_type: str, board_size: str):
        """Start a Go game with the selected options"""
        if not self.engine:
            self._show_error("Game engine not available")
            return
            
        try:
            # Parse board size
            size = int(board_size.split("×")[0])
            
            # Create players
            players = []
            
            # Player 1 (Black)
            if "AI" in player1_type:
                difficulty = player1_type.split("(")[1].split(")")[0].lower()
                ai_profile = f"go_{difficulty}"
                player1 = self.engine.create_ai_player(ai_profile, f"AI {difficulty.capitalize()}")
            else:
                player1 = GoPlayer("Player 1", Stone.BLACK)
                
            players.append(player1)
            
            # Player 2 (White)
            if "AI" in player2_type:
                difficulty = player2_type.split("(")[1].split(")")[0].lower()
                ai_profile = f"go_{difficulty}"
                player2 = self.engine.create_ai_player(ai_profile, f"AI {difficulty.capitalize()}")
            else:
                player2 = GoPlayer("Player 2", Stone.WHITE)
                
            players.append(player2)
            
            # Create game
            self.current_game = self.engine.create_game("GoGame", players, board_size=size)
            self.current_game_type = "go"
            
            # Create game board
            self.current_board = GoBoard(
                0, 0, 
                self.game_board_container.rect.width,
                self.game_board_container.rect.height,
                self.current_game,
                self.game_board_container
            )
            
            # Switch to game screen
            self._set_screen("game")
            
            # Start gameplay music
            if config.music_enabled:
                self.sound_manager.play_music("gameplay")
                
        except Exception as e:
            logger.error(f"Error starting Go game: {e}", exc_info=True)
            self._show_error(f"Failed to start game: {str(e)}")
    
    def _start_mahjong_game(self, player_types: List[str], ruleset: str):
        """Start a Mahjong game with the selected options"""
        if not self.engine:
            self._show_error("Game engine not available")
            return
            
        try:
            # Create players
            players = []
            winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
            
            for i, player_type in enumerate(player_types):
                if "AI" in player_type:
                    difficulty = player_type.split("(")[1].split(")")[0].lower()
                    ai_profile = f"mahjong_{difficulty}"
                    player = self.engine.create_ai_player(ai_profile, f"AI {difficulty.capitalize()}")
                    
                    # Set wind for AI player
                    if isinstance(player, MahjongPlayer):
                        player.wind = winds[i]
                    else:
                        player = MahjongPlayer(player.name, winds[i], player.id)
                else:
                    player = MahjongPlayer(f"Player {i+1}", winds[i])
                    
                players.append(player)
            
            # Create game with settings
            settings = {
                "ruleset": ruleset.lower(),
                "use_red_fives": True
            }
            
            self.current_game = self.engine.create_game("MahjongGame", players, settings=settings)
            self.current_game_type = "mahjong"
            
            # Create game board
            self.current_board = MahjongBoard(
                0, 0, 
                self.game_board_container.rect.width,
                self.game_board_container.rect.height,
                self.current_game,
                self.game_board_container
            )
            
            # Switch to game screen
            self._set_screen("game")
            
            # Start gameplay music
            if config.music_enabled:
                self.sound_manager.play_music("gameplay")
                
        except Exception as e:
            logger.error(f"Error starting Mahjong game