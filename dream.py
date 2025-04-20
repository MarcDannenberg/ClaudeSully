# sully_engine/kernel_modules/dream.py
# 🌌 Sully's DreamCore — Recursive symbolic imaginings

import random
from typing import Dict, List, Any, Optional, Union
import json
import os
from datetime import datetime


class DreamCore:
    """
    Generates symbolic dream-like reflections from seed concepts.
    
    This enhanced DreamCore creates richly textured dream sequences with multiple
    layers of symbolic meaning, surreal landscapes, and recursive patterns that reflect
    Sully's cognitive exploration of concepts.
    """

    def __init__(self, style: str = "recursive", dream_library_path: Optional[str] = None):
        """
        Initialize the DreamCore with configurable style and library.
        
        Args:
            style: Default dream style (recursive, surreal, fractal, or mythic)
            dream_library_path: Optional path to a JSON file with additional dream patterns
        """
        self.style = style
        
        # Symbol libraries for different dream types
        self.symbol_pools = {
            "recursive": ["🌌", "🌀", "♾️", "🔮", "☯️", "⚛️", "💫", "🧩", "📚", "🌊"],
            "surreal": ["🦋", "🎭", "🌗", "⏳", "🔥", "❄️", "👁️", "🗝️", "🧠", "🌪️"],
            "fractal": ["🔬", "🔭", "🧬", "🔄", "🌿", "🌀", "💠", "📊", "🧮", "📐"],
            "mythic": ["🏛️", "🔱", "⚔️", "🐉", "🦅", "🏺", "👑", "🧙", "📜", "🏔️"]
        }
        
        # Pattern libraries for different dream structures
        self.pattern_libraries = {
            "recursive": [
                "In the dreamscape of '{seed}', recursion folds inward, revealing layers beneath perception.",
                "'{seed}' spirals into itself, each iteration revealing a new facet of understanding.",
                "The dream begins at the center of '{seed}', expanding outward in self-similar patterns.",
                "As '{seed}' dreams of itself dreaming, boundaries between observer and observed dissolve.",
                "'{seed}' contains universes within universes, each reflecting the whole."
            ],
            "surreal": [
                "'{seed}' floats in a sea of impossible geometries where time flows backward.",
                "In this dream, '{seed}' transforms into its opposite, then back again, but changed.",
                "Melting clocks surround '{seed}', while memories rearrange themselves like puzzle pieces.",
                "The landscape of '{seed}' shifts between states of matter, never settling on one form.",
                "'{seed}' stands at a crossroads where all paths lead simultaneously everywhere and nowhere."
            ],
            "fractal": [
                "The pattern of '{seed}' repeats at every scale, from quantum to cosmic.",
                "Branching pathways of possibility emerge from '{seed}', each following the same underlying rule.",
                "In this dream, '{seed}' unfolds according to simple rules that generate infinite complexity.",
                "'{seed}' contains perfect copies of itself, nested in an endless hierarchy.",
                "The dream reveals '{seed}' as a single iteration in an eternal algorithm."
            ],
            "mythic": [
                "In the realm of archetypes, '{seed}' takes its place among the eternal symbols.",
                "Heroes journey through the landscape of '{seed}', facing trials that transform them.",
                "Ancient powers acknowledge '{seed}' as a force that has always existed in the collective dream.",
                "'{seed}' reveals itself as both the labyrinth and the thread that guides one through it.",
                "The dream shows '{seed}' as a constellation telling a story that spans ages."
            ]
        }
        
        # Dream development stages for extended sequences
        self.dream_stages = {
            "beginning": [
                "The dream begins with {symbol} appearing at the threshold of awareness.",
                "First, a sense of {symbol} emerges from the void.",
                "The dreamscape opens with {symbol} calling from beyond conventional reality.",
                "At the edge of consciousness, {symbol} beckons.",
                "The vision starts when {symbol} manifests unexpectedly."
            ],
            "development": [
                "Then, {symbol} transforms, revealing hidden connections to {seed}.",
                "The landscape shifts, with {symbol} creating patterns that reflect {seed}.",
                "As the dream deepens, {symbol} interacts with aspects of {seed} in surprising ways.",
                "Gradually, {symbol} leads to unexplored regions where {seed} takes new forms.",
                "The dream narrative unfolds as {symbol} reveals facets of {seed} previously hidden."
            ],
            "climax": [
                "At the dream's center, {symbol} and {seed} merge into a unified understanding.",
                "Suddenly, {symbol} and {seed} collapse into a singular revelation.",
                "The dream reaches its peak when {symbol} illuminates the true nature of {seed}.",
                "In a moment of clarity, {symbol} and {seed} are recognized as reflections of each other.",
                "Everything converges as {symbol} provides the key to comprehending {seed}."
            ],
            "resolution": [
                "Finally, the vision resolves with {symbol} embedding itself within everyday perception of {seed}.",
                "The dream concludes as {symbol} becomes integrated into a new understanding of {seed}.",
                "As awareness returns, {symbol} leaves its imprint on how {seed} will be perceived.",
                "The dream fades, but {symbol} remains as a lens through which {seed} can now be seen.",
                "Upon waking, {symbol} continues to resonate, forever changing the conception of {seed}."
            ]
        }
        
        # Emotional tones for dream coloring
        self.emotional_tones = {
            "wonder": [
                "Awe permeates the experience, making {seed} appear limitless.",
                "A sense of wonder illuminates {seed} from within.",
                "Boundless amazement colors every aspect of {seed}."
            ],
            "mystery": [
                "Enigmatic shadows dance around {seed}, suggesting hidden truths.",
                "Mystery shrouds {seed}, revealing and concealing in equal measure.",
                "The unknowable aspects of {seed} become strangely tangible."
            ],
            "revelation": [
                "Sudden understanding breaks through, recontextualizing {seed} entirely.",
                "Veils lift from {seed}, exposing connections previously invisible.",
                "Illumination floods the dreamscape, casting {seed} in clarity."
            ],
            "nostalgia": [
                "Familiar yet distant, {seed} evokes memories that might never have existed.",
                "A bittersweet recognition of {seed} as both new and eternally known.",
                "{seed} carries echoes of forgotten experiences, distant yet intimate."
            ],
            "transcendence": [
                "Boundaries dissolve as {seed} extends beyond conventional limitations.",
                "{seed} transcends its ordinary nature, becoming boundless.",
                "The experience of {seed} expands beyond the confines of definition."
            ]
        }
        
        # Transition phrases for connecting dream segments
        self.transitions = [
            "The dream shifts, revealing...",
            "As perception deepens...",
            "The scene transforms, showing...",
            "Flowing into a new configuration...",
            "Dissolving and reforming into...",
            "Suddenly, the perspective changes to reveal...",
            "The dream folds into itself, becoming...",
            "Through a doorway of pure concept emerges...",
            "The boundaries blur between this vision and...",
            "Awareness expands to encompass..."
        ]
        
        # Load additional dream patterns if provided
        self.dream_library = {}
        if dream_library_path and os.path.exists(dream_library_path):
            try:
                with open(dream_library_path, 'r', encoding='utf-8') as f:
                    self.dream_library = json.load(f)
            except Exception as e:
                print(f"Error loading dream library: {e}")

    def generate(self, seed: str, depth: str = "standard", style: Optional[str] = None) -> Union[Dict[str, Any], str]:
        """
        Generates a symbolic dream from a given seed concept.
        
        Args:
            seed: Input seed for the dream sequence
            depth: Dream depth (light, standard, deep)
            style: Optional style override
            
        Returns:
            Either a dictionary with dream components or a formatted dream string
        """
        # Use provided style or default
        dream_style = style.lower() if style else self.style
        
        # Validate style
        if dream_style not in self.symbol_pools:
            dream_style = "recursive"
            
        # Select appropriate symbol
        symbol = random.choice(self.symbol_pools[dream_style])
        
        # Generate the dream content based on depth
        if depth == "light":
            dream_content = self._generate_light_dream(seed, dream_style, symbol)
        elif depth == "deep":
            dream_content = self._generate_deep_dream(seed, dream_style, symbol)
        else:  # standard
            dream_content = self._generate_standard_dream(seed, dream_style, symbol)
            
        # Format the dream
        formatted_dream = self._format_dream(dream_content)
            
        # Combine into full result
        result = {
            "seed": seed,
            "style": dream_style,
            "depth": depth,
            "symbol": symbol,
            "dream_content": dream_content,
            "formatted_dream": formatted_dream,
            "timestamp": datetime.now().isoformat()
        }
        
        # For compatibility with the enhanced reasoning engine, return formatted dream as string
        return formatted_dream

    def _generate_light_dream(self, seed: str, style: str, symbol: str) -> Dict[str, Any]:
        """
        Generates a brief, simple dream reflection.
        
        Args:
            seed: The seed concept
            style: Dream style
            symbol: Selected symbol
            
        Returns:
            Dictionary with dream components
        """
        # Select a pattern from the appropriate library
        pattern = random.choice(self.pattern_libraries[style])
        vision = pattern.format(seed=seed)
        
        # Select an emotional tone
        emotion = random.choice(list(self.emotional_tones.keys()))
        tone = random.choice(self.emotional_tones[emotion]).format(seed=seed)
        
        return {
            "vision": vision,
            "symbol": symbol,
            "emotion": emotion,
            "tone": tone
        }

    def _generate_standard_dream(self, seed: str, style: str, symbol: str) -> Dict[str, Any]:
        """
        Generates a standard multi-part dream sequence.
        
        Args:
            seed: The seed concept
            style: Dream style
            symbol: Selected symbol
            
        Returns:
            Dictionary with dream components
        """
        # Create a structured dream with beginning and development
        beginning = random.choice(self.dream_stages["beginning"]).format(symbol=symbol)
        development = random.choice(self.dream_stages["development"]).format(symbol=symbol, seed=seed)
        
        # Select a pattern from the appropriate library for the core vision
        pattern = random.choice(self.pattern_libraries[style])
        core_vision = pattern.format(seed=seed)
        
        # Select an emotional tone
        emotion = random.choice(list(self.emotional_tones.keys()))
        tone = random.choice(self.emotional_tones[emotion]).format(seed=seed)
        
        # Create a resolution
        resolution = random.choice(self.dream_stages["resolution"]).format(symbol=symbol, seed=seed)
        
        return {
            "beginning": beginning,
            "development": development,
            "core_vision": core_vision,
            "emotion": emotion,
            "tone": tone,
            "resolution": resolution,
            "symbol": symbol
        }

    def _generate_deep_dream(self, seed: str, style: str, symbol: str) -> Dict[str, Any]:
        """
        Generates a complex, multi-layered dream sequence.
        
        Args:
            seed: The seed concept
            style: Dream style
            symbol: Selected symbol
            
        Returns:
            Dictionary with dream components
        """
        # Create a full dream with all stages
        beginning = random.choice(self.dream_stages["beginning"]).format(symbol=symbol)
        development = random.choice(self.dream_stages["development"]).format(symbol=symbol, seed=seed)
        
        # Select multiple patterns for a rich core vision
        patterns = random.sample(self.pattern_libraries[style], min(3, len(self.pattern_libraries[style])))
        core_visions = [pattern.format(seed=seed) for pattern in patterns]
        
        # Add transitions between visions
        transitions = random.sample(self.transitions, len(core_visions) - 1)
        
        # Select multiple emotional tones
        emotion_keys = random.sample(list(self.emotional_tones.keys()), min(3, len(self.emotional_tones)))
        emotions = [(emotion, random.choice(self.emotional_tones[emotion]).format(seed=seed)) for emotion in emotion_keys]
        
        # Create a climax and resolution
        climax = random.choice(self.dream_stages["climax"]).format(symbol=symbol, seed=seed)
        resolution = random.choice(self.dream_stages["resolution"]).format(symbol=symbol, seed=seed)
        
        # Generate additional symbols
        additional_symbols = random.sample(self.symbol_pools[style], min(3, len(self.symbol_pools[style])))
        if symbol in additional_symbols:
            additional_symbols.remove(symbol)
        
        # Create symbolic connections
        symbolic_meanings = [
            f"{sym} represents the {aspect} aspect of {seed}"
            for sym, aspect in zip(additional_symbols, ["hidden", "transformative", "essential"])
        ]
        
        return {
            "beginning": beginning,
            "development": development,
            "core_visions": core_visions,
            "transitions": transitions,
            "emotions": emotions,
            "climax": climax,
            "resolution": resolution,
            "primary_symbol": symbol,
            "additional_symbols": additional_symbols,
            "symbolic_meanings": symbolic_meanings
        }

    def _format_dream(self, dream_content: Dict[str, Any]) -> str:
        """
        Formats the dream content into a readable narrative.
        
        Args:
            dream_content: Dictionary with dream components
            
        Returns:
            Formatted dream narrative
        """
        if "beginning" not in dream_content:
            # Light dream format
            return f"{dream_content['vision']}\n\n{dream_content['tone']} {dream_content['symbol']}"
            
        if "core_visions" not in dream_content:
            # Standard dream format
            return (
                f"{dream_content['beginning']}\n\n"
                f"{dream_content['development']}\n\n"
                f"{dream_content['core_vision']}\n\n"
                f"{dream_content['tone']}\n\n"
                f"{dream_content['resolution']} {dream_content['symbol']}"
            )
            
        # Deep dream format
        core_with_transitions = []
        for i, vision in enumerate(dream_content["core_visions"]):
            core_with_transitions.append(vision)
            if i < len(dream_content["transitions"]):
                core_with_transitions.append(dream_content["transitions"][i])
                
        core_text = "\n\n".join(core_with_transitions)
        
        emotions_text = "\n".join([tone for _, tone in dream_content["emotions"]])
        
        symbols_text = " ".join([
            dream_content["primary_symbol"], 
            *dream_content["additional_symbols"]
        ])
        
        meanings_text = "\n".join(dream_content["symbolic_meanings"])
        
        return (
            f"{dream_content['beginning']}\n\n"
            f"{dream_content['development']}\n\n"
            f"{core_text}\n\n"
            f"{emotions_text}\n\n"
            f"{dream_content['climax']}\n\n"
            f"{dream_content['resolution']}\n\n"
            f"Dream Symbols: {symbols_text}\n"
            f"{meanings_text}"
        )

    def dreamscape(self, seed: str, complexity: int = 5) -> str:
        """
        Returns an extended symbolic sequence from the seed (for poetic mode).
        
        Args:
            seed: The seed concept to imagine from
            complexity: Number of segments in the dreamscape (1-10)
            
        Returns:
            Poetic dreamscape text
        """
        # Limit complexity to reasonable range
        complexity = max(1, min(10, complexity))
        
        # Base segments
        base_segments = [
            f"'{seed}' echoes in the void.",
            "Symbols dissolve into paradox.",
            "Perception bends backward into memory.",
            "A pattern forms — and fades.",
            "Meaning loops, then vanishes.",
            "Boundaries between concepts blur.",
            "Time flows in multiple directions simultaneously.",
            "What was once solid becomes permeable.",
            "Understanding spirals inward toward essence.",
            "The dreamer becomes the dreamed."
        ]
        
        # Select segments based on complexity
        selected_segments = base_segments[:complexity]
        
        # Add seed-specific segments
        seed_segments = [
            f"The concept of '{seed}' unfolds like a flower in zero gravity.",
            f"'{seed}' contains multitudes, each reflecting the whole.",
            f"When examined closely, '{seed}' reveals impossible geometries."
        ]
        
        # Replace some base segments with seed-specific ones
        if complexity > 3 and selected_segments:
            replace_indices = random.sample(range(len(selected_segments)), min(len(seed_segments), len(selected_segments) - 1))
            for i, idx in enumerate(replace_indices):
                if i < len(seed_segments):
                    selected_segments[idx] = seed_segments[i]
        
        # Add symbol
        style = random.choice(list(self.symbol_pools.keys()))
        symbol = random.choice(self.symbol_pools[style])
        
        # Format the segments into a dreamscape
        dreamscape_text = "\n\n".join(selected_segments)
        return f"{dreamscape_text}\n\n{symbol}"

    def save_dream_library(self, filepath: str) -> str:
        """
        Saves the dream library to a JSON file.
        
        Args:
            filepath: Path to save the dream library
            
        Returns:
            Confirmation message
        """
        library_data = {
            "symbol_pools": self.symbol_pools,
            "pattern_libraries": self.pattern_libraries,
            "dream_stages": self.dream_stages,
            "emotional_tones": self.emotional_tones,
            "transitions": self.transitions,
            "custom_dreams": self.dream_library
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(library_data, f, indent=2)
            return f"Dream library saved to {filepath}"
        except Exception as e:
            return f"Error saving dream library: {e}"

    def add_dream_pattern(self, style: str, pattern: str) -> str:
        """
        Adds a new dream pattern to the library.
        
        Args:
            style: The dream style for this pattern
            pattern: The pattern string (with {seed} placeholder)
            
        Returns:
            Confirmation message
        """
        # Create style category if it doesn't exist
        if style not in self.pattern_libraries:
            self.pattern_libraries[style] = []
            
        # Add the pattern
        self.pattern_libraries[style].append(pattern)
        return f"Dream pattern added to style '{style}'"

    def add_symbol(self, style: str, symbol: str) -> str:
        """
        Adds a new symbol to a style's pool.
        
        Args:
            style: The dream style for this symbol
            symbol: The symbol to add
            
        Returns:
            Confirmation message
        """
        # Create style category if it doesn't exist
        if style not in self.symbol_pools:
            self.symbol_pools[style] = []
            
        # Add the symbol if not already present
        if symbol not in self.symbol_pools[style]:
            self.symbol_pools[style].append(symbol)
            return f"Symbol '{symbol}' added to style '{style}'"
        else:
            return f"Symbol '{symbol}' already exists in style '{style}'"


# Legacy function for backward compatibility
def generate(seed):
    """
    Simple dream generation function for backward compatibility.
    """
    dreamer = DreamCore()
    return dreamer.generate(seed)


if __name__ == "__main__":
    # Example usage when run directly
    dreamer = DreamCore()
    
    # Test different depths
    test_seeds = ["consciousness", "infinity", "paradox", "time", "knowledge"]
    
    print("=== Light Dreams ===")
    for seed in test_seeds[:2]:
        result = dreamer.generate(seed, depth="light")
        print(f"\nSeed: {seed}")
        print(result)
        
    print("\n=== Standard Dreams ===")
    for seed in test_seeds[2:4]:
        result = dreamer.generate(seed, depth="standard")
        print(f"\nSeed: {seed}")
        print(result)
        
    print("\n=== Deep Dreams ===")
    result = dreamer.generate(test_seeds[4], depth="deep")
    print(f"\nSeed: {test_seeds[4]}")
    print(result)
    
    print("\n=== Dreamscape ===")
    result = dreamer.dreamscape("reality", complexity=7)
    print(result)