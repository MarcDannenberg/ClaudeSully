# sully_engine/kernel_modules/math_translator.py
# 🔢 Symbolic-to-Mathematical Expression Translator - Enhanced Version

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import re
import json
import os
import random
import numpy as np
from sympy import symbols, sympify, solve, diff, integrate, simplify, expand, factor
from sympy.parsing.sympy_parser import parse_expr
import sympy.physics.units as units
from sympy.abc import x, y, z, t

class SymbolicMathTranslator:
    """
    Translates between symbolic/linguistic expressions and mathematical representations.
    
    This enhanced translator supports bidirectional translation (language to math, math to language),
    contextual interpretations, multiple translation styles, and mathematical computation capabilities.
    """

    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize the translator with standard mappings or from a custom file.
        
        Args:
            mapping_file: Optional path to a JSON file with additional mappings
        """
        # Core symbolic mappings (concept to math notation)
        self.math_mappings = {
            # Abstract concepts
            "infinity": "∞",
            "infinite": "∞",
            "endless": "∞",
            "boundless": "∞",
            "forever": "∞",
            "eternity": "∞",
            
            # Change and movement
            "change": "d/dx",
            "derivative": "d/dx",
            "rate of change": "d/dx",
            "flow": "∇",
            "gradient": "∇",
            "direction": "→",
            "vector": "→",
            "movement": "→",
            
            # Integration and accumulation
            "sum": "∑",
            "summation": "∑",
            "series": "∑",
            "accumulation": "∫",
            "total": "∫",
            "area under curve": "∫",
            "integral": "∫",
            "area": "∫",
            
            # Growth and comparison
            "growth": "f'(x) > 0",
            "increase": "f'(x) > 0",
            "expand": "f'(x) > 0",
            "decrease": "f'(x) < 0",
            "shrink": "f'(x) < 0",
            "greater than": ">",
            "less than": "<",
            "equal to": "=",
            "equivalence": "≡",
            "approximately": "≈",
            
            # Balance and systems
            "equilibrium": "∇ · F = 0",
            "balance": "∇ · F = 0",
            "harmony": "∇ · F = 0",
            "system": "S = {x : P(x)}",
            "set": "S = {x : P(x)}",
            "collection": "{x₁, x₂, ..., xₙ}",
            
            # Logic and truth
            "therefore": "∴",
            "thus": "∴",
            "because": "∵",
            "since": "∵",
            "for all": "∀",
            "every": "∀",
            "each": "∀",
            "there exists": "∃",
            "exists": "∃",
            "some": "∃",
            "not": "¬",
            "contradiction": "⊥",
            "impossible": "⊥",
            "and": "∧",
            "or": "∨",
            "implies": "→",
            "if then": "→",
            
            # Relationships and structure
            "belongs to": "∈",
            "element of": "∈",
            "contains": "⊃",
            "subset": "⊂",
            "intersection": "∩",
            "overlap": "∩",
            "union": "∪",
            "combine": "∪",
            "join": "∪",
            "empty": "∅",
            "nothing": "∅",
            "void": "∅",
            
            # Quantum and uncertainty
            "uncertainty": "Δx · Δp ≥ ℏ/2",
            "wave function": "Ψ",
            "quantum": "ℏ",
            "planck": "ℏ",
            "probability": "P(A)",
            "chance": "P(A)",
            "likelihood": "P(A)",
            
            # Time and space
            "time": "t",
            "space": "s",
            "distance": "d",
            "position": "(x,y,z)",
            "location": "(x,y,z)",
            "spacetime": "(x,y,z,t)",
            
            # Constants and notable values
            "transcendental": "π, e, φ",
            "pi": "π",
            "golden ratio": "φ",
            "euler number": "e",
            "exponential": "e^x",
            "natural log": "ln(x)",
            "logarithm": "log₂(x)",
            
            # Physics
            "energy": "E = mc²",
            "mass": "m",
            "light": "c",
            "gravity": "G",
            "force": "F = ma",
            "acceleration": "a",
            "relativity": "E = mc²",
            
            # Complex and abstract
            "imaginary": "i",
            "complex": "a + bi",
            "fractal": "z ← z² + c",
            "recursion": "f(f(x))",
            "self-reference": "f(f)",
            "paradox": "P ⟺ ¬P",
            "contradiction": "A ∧ ¬A",
            
            # Additional domains
            "entropy": "S = k · ln(W)",
            "information": "I = -log₂(p)",
            "chaos": "xₙ₊₁ = r·xₙ·(1-xₙ)",
            "network": "G = (V, E)",
            "graph": "G = (V, E)",
            "cycle": "f(x+T) = f(x)",
            
            # New mathematical concepts
            "tensor": "T^{μν}",
            "group": "(G, ∘)",
            "category": "C = (Ob(C), hom(C), ∘)",
            "topology": "(X, τ)",
            "manifold": "M",
            "differential form": "ω",
            "eigenvalue": "Av = λv",
            "laplacian": "∇²",
            "fourier transform": "F[f](ω) = ∫ f(t)e^{-iωt} dt",
            "divergence": "∇·",
            "curl": "∇×",
            "matrix": "[aᵢⱼ]",
            "determinant": "det(A)",
            "trace": "tr(A)",
            "modular": "a ≡ b (mod n)",
            "stochastic": "P(X₁,...,Xₙ)",
            "martingale": "E[Xₙ₊₁|X₁,...,Xₙ] = Xₙ"
        }
        
        # Enhanced mathematical notations (symbol to expanded form)
        self.expanded_math = {
            "∞": "infinity",
            "d/dx": "the derivative with respect to x",
            "∇": "the gradient operator",
            "→": "a vector or direction",
            "∑": "the sum of a series",
            "∫": "the integral of",
            "∇ · F = 0": "a system in equilibrium",
            "f'(x) > 0": "a function with positive derivative (increasing)",
            "f'(x) < 0": "a function with negative derivative (decreasing)",
            ">": "greater than",
            "<": "less than",
            "=": "equals",
            "≡": "is identical to",
            "≈": "is approximately equal to",
            "∴": "therefore",
            "∵": "because",
            "∀": "for all",
            "∃": "there exists",
            "¬": "not",
            "⊥": "contradiction",
            "∧": "and",
            "∨": "or",
            "∈": "belongs to the set",
            "⊂": "is a subset of",
            "⊃": "contains",
            "∩": "intersection",
            "∪": "union",
            "∅": "the empty set",
            "Ψ": "the wave function",
            "ℏ": "Planck's constant",
            "P(A)": "probability of event A",
            "π": "pi (approximately 3.14159)",
            "e": "Euler's number (approximately 2.71828)",
            "φ": "the golden ratio (approximately 1.61803)",
            "i": "the imaginary unit, sqrt(-1)",
            "E = mc²": "energy equals mass times the speed of light squared",
            "G = (V, E)": "a graph with vertices V and edges E",
            "T^{μν}": "a tensor of type (μ,ν)",
            "(G, ∘)": "a group with operation ∘",
            "∇²": "the Laplacian operator",
            "F[f](ω)": "the Fourier transform of f evaluated at ω",
            "∇·": "the divergence operator",
            "∇×": "the curl operator",
            "[aᵢⱼ]": "a matrix with elements aᵢⱼ",
            "det(A)": "the determinant of matrix A",
            "tr(A)": "the trace of matrix A",
            "a ≡ b (mod n)": "a is congruent to b modulo n"
        }
        
        # Translation styles
        self.translation_styles = {
            "formal": {
                "template": "{concept} can be formally represented as {symbol}.",
                "connectors": [
                    "which is expressed as",
                    "formally denoted as",
                    "symbolically represented by",
                    "mathematically equivalent to",
                    "denoted in formal notation as"
                ]
            },
            "intuitive": {
                "template": "Think of {symbol} as representing {concept}.",
                "connectors": [
                    "which intuitively captures",
                    "giving us a way to visualize",
                    "offering an intuitive representation of",
                    "providing a mental model for",
                    "helping us grasp the idea of"
                ]
            },
            "poetic": {
                "template": "The concept of {concept} unfolds into the symbolic rhythm of {symbol}.",
                "connectors": [
                    "dancing with the essence of",
                    "resonating with the meaning of",
                    "flowing into the symbolic realm of",
                    "transcending into the notation",
                    "echoing the pattern of"
                ]
            },
            "philosophical": {
                "template": "{symbol} emerges as the embodiment of {concept}, a bridge between thought and form.",
                "connectors": [
                    "revealing the deeper truth of",
                    "transcending the boundaries between",
                    "illuminating the essence of",
                    "dissolving the distinction between",
                    "unfolding the meaning within"
                ]
            },
            "pedagogical": {
                "template": "We can understand {concept} through the mathematical lens of {symbol}.",
                "connectors": [
                    "which helps students grasp",
                    "clarifying our understanding of",
                    "providing a structured way to approach",
                    "offering a framework for comprehending",
                    "building a foundation for exploring"
                ]
            },
            "computational": {
                "template": "The operation {concept} translates to the computable expression {symbol}.",
                "connectors": [
                    "which algorithmically represents",
                    "enabling computational approaches to",
                    "providing a calculable form of",
                    "translating into the language of computation as",
                    "allowing us to compute aspects of"
                ]
            },
            "historical": {
                "template": "The concept of {concept}, formalized as {symbol}, has evolved through centuries of mathematical thought.",
                "connectors": [
                    "connecting to the historical development of",
                    "emerging from the intellectual tradition behind",
                    "revealing the historical progression toward",
                    "tracing its lineage to early explorations of",
                    "showing the culmination of thinking about"
                ]
            }
        }
        
        # Domain-specific notation sets
        self.domain_notations = {
            "physics": {
                "force": "F = ma",
                "energy": "E = mc²",
                "work": "W = F·d",
                "power": "P = dW/dt",
                "momentum": "p = mv",
                "relativity": "E² = (mc²)² + (pc)²",
                "wave": "Ψ(x,t) = A·sin(kx - ωt)",
                "gravity": "F = G·(m₁m₂)/r²",
                "electric field": "E = F/q",
                "magnetic field": "B = μ₀I/(2πr)",
                "electromagnetic": "∇ × E = -∂B/∂t",
                "thermodynamics": "dS ≥ 0",
                "entropy": "S = k·ln(W)",
                "heat": "Q = m·c·ΔT"
            },
            "calculus": {
                "derivative": "f'(x) = lim_{h→0} (f(x+h) - f(x))/h",
                "integral": "∫ab f(x)dx = F(b) - F(a)",
                "series": "∑n=0∞ aₙ",
                "taylor": "f(x) = ∑n=0∞ (f⁽ⁿ⁾(a)/n!)·(x-a)ⁿ",
                "partial derivative": "∂f/∂x",
                "gradient": "∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)",
                "vector calculus": "∇·(∇×F) = 0",
                "divergence theorem": "∫∫∫ₚ(∇·F)dV = ∫∫ₛF·dS",
                "stokes theorem": "∫ₛ(∇×F)·dS = ∮ₗF·dr"
            },
            "set_theory": {
                "union": "A ∪ B = {x : x ∈ A or x ∈ B}",
                "intersection": "A ∩ B = {x : x ∈ A and x ∈ B}",
                "complement": "Aᶜ = {x ∈ U : x ∉ A}",
                "power_set": "P(A) = {S : S ⊆ A}",
                "cardinality": "|A| = n",
                "empty set": "∅",
                "subset": "A ⊆ B",
                "proper subset": "A ⊂ B",
                "set difference": "A \\ B = {x ∈ A : x ∉ B}",
                "symmetric difference": "A Δ B = (A \\ B) ∪ (B \\ A)",
                "cartesian product": "A × B = {(a,b) : a ∈ A, b ∈ B}"
            },
            "logic": {
                "conjunction": "A ∧ B",
                "disjunction": "A ∨ B",
                "implication": "A → B",
                "biconditional": "A ↔ B",
                "negation": "¬A",
                "universal": "∀x P(x)",
                "existential": "∃x P(x)",
                "exclusive or": "A ⊕ B",
                "tautology": "⊤",
                "contradiction": "⊥",
                "de morgan": "¬(A ∧ B) ≡ ¬A ∨ ¬B",
                "modus ponens": "A, A → B ⊢ B",
                "modus tollens": "¬B, A → B ⊢ ¬A"
            },
            "probability": {
                "probability": "P(A)",
                "conditional": "P(A|B) = P(A ∩ B)/P(B)",
                "bayes": "P(A|B) = P(B|A)·P(A)/P(B)",
                "independence": "P(A ∩ B) = P(A)·P(B)",
                "expectation": "E[X] = ∑x·P(X=x)",
                "variance": "Var(X) = E[(X - E[X])²]",
                "normal": "f(x) = (1/(σ√2π))·e^(-(x-μ)²/(2σ²))",
                "binomial": "P(X=k) = (n choose k)·p^k·(1-p)^(n-k)",
                "poisson": "P(X=k) = (λ^k·e^(-λ))/k!",
                "markov chain": "P(Xₙ₊₁=j|Xₙ=i) = pᵢⱼ"
            },
            "quantum": {
                "uncertainty": "Δx·Δp ≥ ℏ/2",
                "schrodinger": "iℏ·∂Ψ/∂t = ĤΨ",
                "wavefunction": "Ψ(x,t)",
                "superposition": "|Ψ⟩ = α|0⟩ + β|1⟩",
                "bra-ket": "⟨φ|ψ⟩",
                "observable": "Â|ψ⟩ = a|ψ⟩",
                "measurement": "P(a) = |⟨a|ψ⟩|²",
                "spin": "|↑⟩, |↓⟩",
                "entanglement": "|Ψ⟩ = (|00⟩ + |11⟩)/√2",
                "density matrix": "ρ = ∑ᵢpᵢ|ψᵢ⟩⟨ψᵢ|",
                "unitary": "U†U = I"
            },
            "computation": {
                "algorithm": "O(n log n)",
                "recursion": "f(n) = f(n-1) + f(n-2)",
                "turing": "M = (Q, Σ, Γ, δ, q₀, F)",
                "boolean": "A ∧ (B ∨ C)",
                "complexity": "P vs NP",
                "lambda calculus": "λx.x",
                "computation": "f : X → Y",
                "automaton": "δ : Q × Σ → Q",
                "register": "r := v",
                "loop": "while (condition) { action }",
                "sorting": "sort(A) = A'",
                "graph theory": "path(u, v)",
                "recursion theory": "R = {e | φₑ(e)↓}"
            },
            "algebra": {
                "group": "(G, ∘)",
                "ring": "(R, +, ·)",
                "field": "(F, +, ·)",
                "polynomial": "p(x) = a₀ + a₁x + ... + aₙxⁿ",
                "homomorphism": "φ(a∘b) = φ(a)∗φ(b)",
                "isomorphism": "G ≅ H",
                "algebraic structure": "(S, ∘₁, ∘₂, ...)",
                "matrix": "A = [aᵢⱼ]",
                "eigenvalue": "Av = λv",
                "eigenspace": "E_λ = {v ∈ V | Av = λv}",
                "determinant": "det(A)",
                "trace": "tr(A)",
                "nilpotent": "A^n = 0",
                "idempotent": "A² = A"
            },
            "topology": {
                "open set": "(X, τ)",
                "continuity": "f : X → Y is continuous",
                "homeomorphism": "X ≅ Y",
                "compactness": "K is compact",
                "connectedness": "X is connected",
                "neighborhood": "N_ε(x₀)",
                "hausdorff": "X is Hausdorff",
                "manifold": "M is a manifold",
                "differential form": "ω = f dx + g dy",
                "homotopy": "F : X × [0,1] → Y",
                "homology": "H_n(X)",
                "covering space": "p : E → X",
                "fundamental group": "π₁(X, x₀)"
            },
            "category_theory": {
                "category": "C = (Ob(C), hom(C), ∘)",
                "functor": "F : C → D",
                "natural transformation": "η : F ⇒ G",
                "adjunction": "F ⊣ G",
                "limit": "lim F",
                "colimit": "colim F",
                "monoid": "(M, ∘, e)",
                "initial object": "0",
                "terminal object": "1",
                "product": "A × B",
                "coproduct": "A ⊔ B",
                "pullback": "P = A ×_C B",
                "pushout": "Q = A ⊔_C B",
                "monad": "(T, η, μ)"
            },
            "number_theory": {
                "divisibility": "a | b",
                "prime": "p is prime",
                "congruence": "a ≡ b (mod n)",
                "fermat's little": "a^(p-1) ≡ 1 (mod p)",
                "euler's totient": "φ(n)",
                "euler's formula": "e^(iπ) + 1 = 0",
                "gcd": "gcd(a,b)",
                "lcm": "lcm(a,b)",
                "diophantine": "ax + by = c",
                "quadratic reciprocity": "(p/q)(q/p) = (-1)^((p-1)(q-1)/4)",
                "continued fraction": "[a₀; a₁, a₂, ...]",
                "bernoulli numbers": "B_n",
                "zeta function": "ζ(s) = ∑n^(-s)"
            }
        }
        
        # Computational operators for symbolic math
        self.computational_operators = {
            "derivative": lambda expr, var='x': str(diff(parse_expr(expr), symbols(var))),
            "integrate": lambda expr, var='x': str(integrate(parse_expr(expr), symbols(var))),
            "solve": lambda expr, var='x': str(solve(parse_expr(expr), symbols(var))),
            "simplify": lambda expr: str(simplify(parse_expr(expr))),
            "expand": lambda expr: str(expand(parse_expr(expr))),
            "factor": lambda expr: str(factor(parse_expr(expr))),
            "evaluate": lambda expr, val_dict=None: str(parse_expr(expr).subs(val_dict or {})),
            "series": lambda expr, var='x', point=0, n=5: str(parse_expr(expr).series(symbols(var), point, n)),
            "limit": lambda expr, var='x', point=0: str(parse_expr(expr).limit(symbols(var), point))
        }
        
        # Common mathematical patterns for natural language parsing
        self.math_patterns = {
            r"derivative of (.+?) with respect to (.+?)": 
                lambda match: self.computational_operators["derivative"](match.group(1), match.group(2)),
            r"derivative of (.+)": 
                lambda match: self.computational_operators["derivative"](match.group(1)),
            r"integrate (.+?) with respect to (.+?)": 
                lambda match: self.computational_operators["integrate"](match.group(1), match.group(2)),
            r"integrate (.+)": 
                lambda match: self.computational_operators["integrate"](match.group(1)),
            r"solve (.+?) for (.+?)": 
                lambda match: self.computational_operators["solve"](match.group(1), match.group(2)),
            r"solve (.+)": 
                lambda match: self.computational_operators["solve"](match.group(1)),
            r"simplify (.+)": 
                lambda match: self.computational_operators["simplify"](match.group(1)),
            r"expand (.+)": 
                lambda match: self.computational_operators["expand"](match.group(1)),
            r"factor (.+)": 
                lambda match: self.computational_operators["factor"](match.group(1)),
            r"compute series of (.+?) around (.+?) to order (.+?)": 
                lambda match: self.computational_operators["series"](match.group(1), 'x', float(match.group(2)), int(match.group(3))),
            r"compute limit of (.+?) as (.+?) approaches (.+?)": 
                lambda match: self.computational_operators["limit"](match.group(1), match.group(2), match.group(3))
        }
        
        # Physical constants with units
        self.physical_constants = {
            "speed of light": {"symbol": "c", "value": 299792458, "unit": "m/s"},
            "gravitational constant": {"symbol": "G", "value": 6.67430e-11, "unit": "m³/(kg·s²)"},
            "planck constant": {"symbol": "h", "value": 6.62607015e-34, "unit": "J·s"},
            "reduced planck constant": {"symbol": "ℏ", "value": 1.054571817e-34, "unit": "J·s"},
            "boltzmann constant": {"symbol": "k", "value": 1.380649e-23, "unit": "J/K"},
            "avogadro number": {"symbol": "N_A", "value": 6.02214076e23, "unit": "1/mol"},
            "gas constant": {"symbol": "R", "value": 8.31446261815324, "unit": "J/(mol·K)"},
            "electron charge": {"symbol": "e", "value": 1.602176634e-19, "unit": "C"},
            "vacuum permittivity": {"symbol": "ε₀", "value": 8.8541878128e-12, "unit": "F/m"},
            "vacuum permeability": {"symbol": "μ₀", "value": 1.25663706212e-6, "unit": "H/m"},
            "electron mass": {"symbol": "m_e", "value": 9.1093837015e-31, "unit": "kg"},
            "proton mass": {"symbol": "m_p", "value": 1.67262192369e-27, "unit": "kg"},
            "neutron mass": {"symbol": "m_n", "value": 1.67492749804e-27, "unit": "kg"},
            "fine structure constant": {"symbol": "α", "value": 7.2973525693e-3, "unit": ""},
            "rydberg constant": {"symbol": "R_∞", "value": 10973731.568160, "unit": "1/m"}
        }
        
        # Unit conversion system
        self.unit_conversions = {
            "length": {
                "m": 1.0,  # base: meter
                "km": 1000.0,
                "cm": 0.01,
                "mm": 0.001,
                "in": 0.0254,
                "ft": 0.3048,
                "yd": 0.9144,
                "mi": 1609.344
            },
            "mass": {
                "kg": 1.0,  # base: kilogram
                "g": 0.001,
                "mg": 0.000001,
                "lb": 0.45359237,
                "oz": 0.028349523125
            },
            "time": {
                "s": 1.0,  # base: second
                "min": 60.0,
                "hr": 3600.0,
                "day": 86400.0,
                "week": 604800.0,
                "month": 2592000.0,  # 30 days
                "year": 31536000.0  # 365 days
            },
            "temperature": {
                "K": lambda t: t,  # base: kelvin
                "C": lambda t: t + 273.15,  # to kelvin
                "F": lambda t: (t + 459.67) * 5/9  # to kelvin
            },
            "energy": {
                "J": 1.0,  # base: joule
                "kJ": 1000.0,
                "cal": 4.184,
                "kcal": 4184.0,
                "eV": 1.602176634e-19,
                "kWh": 3600000.0
            }
        }
        
        # Load additional mappings from file if provided
        if mapping_file and os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    additional_mappings = json.load(f)
                    self.math_mappings.update(additional_mappings)
            except Exception as e:
                print(f"Error loading additional mappings: {e}")
    
    def translate(self, phrase: str, style: str = "formal", domain: Optional[str] = None) -> Union[Dict[str, Any], str]:
        """
        Translates natural language into mathematical notation.
        
        Args:
            phrase: The text to translate
            style: Translation style (formal, intuitive, poetic, philosophical, pedagogical)
            domain: Optional domain focus (physics, calculus, set_theory, etc.)
            
        Returns:
            Either a dictionary with translation details or a formatted string
        """
        # Check if the phrase contains a computational request
        computation_result = self._check_for_computation(phrase)
        if computation_result:
            return computation_result
        
        # Normalize inputs
        phrase_lower = phrase.lower()
        style = style.lower() if style else "formal"
        domain = domain.lower() if domain else None
        
        # Use domain-specific notation if requested
        active_mappings = self.math_mappings.copy()
        if domain and domain in self.domain_notations:
            active_mappings.update(self.domain_notations[domain])
        
        # Find matches in phrase
        matches = {}
        for word, symbol in active_mappings.items():
            if word in phrase_lower:
                matches[word] = symbol
        
        # If no direct matches, try to find related concepts
        if not matches:
            # Look for partial matches
            for word, symbol in active_mappings.items():
                words = word.split()
                if len(words) > 1:  # For multi-word concepts
                    # Check if at least half the words match
                    matching_words = sum(1 for w in words if w in phrase_lower)
                    if matching_words >= len(words) / 2:
                        matches[word] = symbol
                elif len(word) >= 5:  # For single longer words, check partial matches
                    # If the word is at least 5 chars, check if a substantial part appears
                    if word[:4] in phrase_lower:
                        matches[word] = symbol
        
        # Check for physical constants references
        for constant_name, constant_info in self.physical_constants.items():
            if constant_name in phrase_lower:
                matches[constant_name] = f"{constant_info['symbol']} = {constant_info['value']} {constant_info['unit']}"
        
        # Prepare the response
        if not matches:
            # Create a symbolic response even when no direct match
            explanation = self._generate_symbolic_reflection(phrase)
            return {
                "matches": {},
                "explanation": explanation
            }
        
        # Format the explanation based on the requested style
        explanation = self._format_translation(matches, style)
        
        # Return detailed or simple response
        return {
            "matches": matches,
            "explanation": explanation
        }

    def translate_to_text(self, math_expression: str, style: str = "formal") -> str:
        """
        Translates mathematical notation into natural language explanation.
        
        Args:
            math_expression: The mathematical expression to translate
            style: Translation style
            
        Returns:
            Natural language explanation
        """
        # Clean the expression
        expression = math_expression.strip()
        
        # Check for direct matches in expanded forms
        if expression in self.expanded_math:
            base_explanation = self.expanded_math[expression]
        else:
            # Look for symbols within the expression
            found_symbols = []
            explained_parts = []
            
            for symbol, explanation in self.expanded_math.items():
                if symbol in expression and len(symbol) > 1:  # Avoid single character false positives
                    found_symbols.append(symbol)
                    explained_parts.append(f"{symbol} represents {explanation}")
            
            if not found_symbols:
                # Check for individual symbols
                for symbol, explanation in self.expanded_math.items():
                    if len(symbol) == 1 and symbol in expression:
                        found_symbols.append(symbol)
                        explained_parts.append(f"{symbol} represents {explanation}")
            
            if explained_parts:
                base_explanation = "This expression contains " + ", ".join(explained_parts)
            else:
                # Try to parse and explain the mathematical structure
                try:
                    parsed_expr = parse_expr(expression.replace('∞', 'oo'))
                    base_explanation = f"a mathematical expression{self._explain_math_structure(parsed_expr)}"
                except Exception:
                    base_explanation = "a mathematical expression that combines multiple symbolic elements"
        
        # Apply the requested style
        if style == "formal":
            return f"The expression {math_expression} represents {base_explanation}."
        elif style == "intuitive":
            return f"Think of {math_expression} as {base_explanation} in a more intuitive sense."
        elif style == "poetic":
            return f"The symbolic dance of {math_expression} reveals {base_explanation}, a pattern unfolding in the language of mathematics."
        elif style == "philosophical":
            return f"In the realm of symbolic thought, {math_expression} emerges as {base_explanation}, bridging the concrete and abstract."
        elif style == "pedagogical":
            return f"When teaching {math_expression}, we explain it as {base_explanation}, which helps build conceptual understanding."
        elif style == "computational":
            try:
                # Try to compute or evaluate
                parsed = parse_expr(expression.replace('∞', 'oo'))
                simplified = simplify(parsed)
                expanded = expand(parsed)
                
                return f"Computationally, {math_expression} can be expressed as {simplified} in simplified form or as {expanded} when expanded."
            except Exception:
                return f"The expression {math_expression} represents {base_explanation} and can be analyzed computationally through various transformations."
        elif style == "historical":
            return f"The notation {math_expression}, representing {base_explanation}, reflects the evolving language of mathematics developed over centuries by mathematicians seeking precise symbolic representation."
        else:
            return f"{math_expression}: {base_explanation}"

    def _explain_math_structure(self, expr) -> str:
        """
        Generates a structured explanation of a mathematical expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            Explanation of mathematical structure
        """
        import sympy
        
        try:
            if isinstance(expr, sympy.Add):
                return " involving addition of terms"
            elif isinstance(expr, sympy.Mul):
                return " involving multiplication of factors"
            elif isinstance(expr, sympy.Pow):
                return " involving exponentiation"
            elif isinstance(expr, sympy.sin):
                return " involving the sine function"
            elif isinstance(expr, sympy.cos):
                return " involving the cosine function"
            elif isinstance(expr, sympy.log):
                return " involving logarithms"
            elif isinstance(expr, sympy.exp):
                return " involving exponentials"
            elif isinstance(expr, sympy.Integral):
                return " representing an integral"
            elif isinstance(expr, sympy.Derivative):
                return " representing a derivative"
            elif isinstance(expr, sympy.Sum):
                return " representing a summation"
            elif isinstance(expr, sympy.Matrix):
                return " representing a matrix"
            elif isinstance(expr, sympy.Eq):
                return " representing an equation"
            else:
                return ""
        except Exception:
            return ""

    def _check_for_computation(self, phrase: str) -> Optional[Dict[str, Any]]:
        """
        Checks if the phrase contains a computational request and processes it.
        
        Args:
            phrase: The input phrase
            
        Returns:
            Computation result if applicable, None otherwise
        """
        # Check for mathematical operation patterns
        for pattern, operation in self.math_patterns.items():
            match = re.search(pattern, phrase, re.IGNORECASE)
            if match:
                try:
                    result = operation(match)
                    return {
                        "computation": True,
                        "input": phrase,
                        "pattern_matched": pattern,
                        "result": result,
                        "explanation": f"Computed: {result}"
                    }
                except Exception as e:
                    return {
                        "computation": True,
                        "input": phrase,
                        "error": str(e),
                        "explanation": f"Could not compute the expression due to: {str(e)}"
                    }
        
        # Check for unit conversion requests
        unit_match = re.search(r"convert\s+(\d+(?:\.\d+)?)\s+(\w+)\s+to\s+(\w+)", phrase, re.IGNORECASE)
        if unit_match:
            try:
                value = float(unit_match.group(1))
                from_unit = unit_match.group(2).lower()
                to_unit = unit_match.group(3).lower()
                
                result = self._convert_units(value, from_unit, to_unit)
                
                if result:
                    return {
                        "computation": True,
                        "input": phrase,
                        "value": value,
                        "from_unit": from_unit,
                        "to_unit": to_unit,
                        "result": result,
                        "explanation": f"{value} {from_unit} = {result} {to_unit}"
                    }
            except Exception as e:
                return {
                    "computation": True,
                    "input": phrase,
                    "error": str(e),
                    "explanation": f"Could not convert units due to: {str(e)}"
                }
        
        # Check for physical constant requests
        constant_match = re.search(r"value of\s+(.+?)(?:\s+in\s+(\w+))?$", phrase, re.IGNORECASE)
        if constant_match:
            constant_name = constant_match.group(1).lower()
            target_unit = constant_match.group(2).lower() if constant_match.group(2) else None
            
            for name, info in self.physical_constants.items():
                if name in constant_name:
                    if target_unit and info["unit"] != target_unit:
                        try:
                            converted_value = self._convert_units(info["value"], info["unit"], target_unit)
                            if converted_value:
                                return {
                                    "computation": True,
                                    "input": phrase,
                                    "constant": name,
                                    "value": converted_value,
                                    "unit": target_unit,
                                    "explanation": f"The {name} ({info['symbol']}) = {converted_value} {target_unit}"
                                }
                        except Exception:
                            pass
                    
                    return {
                        "computation": True,
                        "input": phrase,
                        "constant": name,
                        "value": info["value"],
                        "unit": info["unit"],
                        "explanation": f"The {name} ({info['symbol']}) = {info['value']} {info['unit']}"
                    }
        
        # No computational request found
        return None

    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Converts a value from one unit to another.
        
        Args:
            value: The numeric value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value or None if conversion not possible
        """
        # Find the unit category
        category = None
        for cat, units in self.unit_conversions.items():
            if from_unit in units and to_unit in units:
                category = cat
                break
        
        if not category:
            return None
        
        # Temperature requires special handling due to offsets
        if category == "temperature":
            kelvin_value = self.unit_conversions[category][from_unit](value)
            
            # Convert from kelvin to target
            if to_unit == "K":
                return kelvin_value
            elif to_unit == "C":
                return kelvin_value - 273.15
            elif to_unit == "F":
                return kelvin_value * 9/5 - 459.67
        
        # Standard linear conversion
        from_value = self.unit_conversions[category][from_unit]
        to_value = self.unit_conversions[category][to_unit]
        
        return value * (from_value / to_value)

    def _format_translation(self, matches: Dict[str, str], style: str) -> str:
        """
        Formats the translation results according to the requested style.
        
        Args:
            matches: Dictionary of concept to symbol matches
            style: Translation style
            
        Returns:
            Formatted explanation string
        """
        # Get style configuration
        style_config = self.translation_styles.get(style, self.translation_styles["formal"])
        template = style_config["template"]
        connectors = style_config["connectors"]
        
        # Format individual matches
        formatted_matches = []
        for concept, symbol in matches.items():
            formatted = template.format(concept=concept, symbol=symbol)
            formatted_matches.append(formatted)
        
        # Combine with appropriate connectors
        if len(formatted_matches) == 1:
            return formatted_matches[0]
        
        # Use connectors for multiple matches
        result = [formatted_matches[0]]
        for i in range(1, len(formatted_matches)):
            connector = random.choice(connectors)
            result.append(f"This {connector} {formatted_matches[i].lower()}")
        
        return " ".join(result)

    def _generate_symbolic_reflection(self, phrase: str) -> str:
        """
        Generates a symbolic reflection for phrases without direct matches.
        
        Args:
            phrase: The input phrase
            
        Returns:
            A symbolic reflection
        """
        reflections = [
            f"While there's no direct mathematical mapping for '{phrase}', it suggests a conceptual space where symbolic representation could emerge.",
            f"The phrase '{phrase}' transcends current symbolic notation, existing in the liminal space between language and mathematics.",
            f"'{phrase}' represents a concept that might be expressible through a combination of existing notations or a new symbolic framework.",
            f"In considering '{phrase}', we approach the boundary where language meets formalism, suggesting potential for new notation.",
            f"Though lacking a standardized notation, '{phrase}' invites us to consider how mathematical language might evolve to capture such concepts."
        ]
        return random.choice(reflections)

    def add_mapping(self, symbol_phrase: str, math_form: str, domain: Optional[str] = None) -> str:
        """
        Adds a new symbolic → math mapping at runtime.
        
        Args:
            symbol_phrase: The symbolic phrase to map
            math_form: The corresponding mathematical representation
            domain: Optional domain category
            
        Returns:
            Confirmation message
        """
        symbol_phrase = symbol_phrase.lower()
        
        # Add to appropriate dictionary
        if domain and domain in self.domain_notations:
            self.domain_notations[domain][symbol_phrase] = math_form
            return f"Domain-specific mapping added: '{symbol_phrase}' → '{math_form}' in {domain}"
        else:
            self.math_mappings[symbol_phrase] = math_form
            return f"Mapping added: '{symbol_phrase}' → '{math_form}'"

    def add_expanded_form(self, symbol: str, explanation: str) -> str:
        """
        Adds an expanded text explanation for a mathematical symbol.
        
        Args:
            symbol: The mathematical symbol or expression
            explanation: The natural language explanation
            
        Returns:
            Confirmation message
        """
        self.expanded_math[symbol] = explanation
        return f"Expanded form added: '{symbol}' → '{explanation}'"

    def add_computational_pattern(self, pattern: str, operation_function: Callable) -> str:
        """
        Adds a new computational pattern for processing natural language math requests.
        
        Args:
            pattern: Regular expression pattern to match
            operation_function: Function to perform the operation
            
        Returns:
            Confirmation message
        """
        self.math_patterns[pattern] = operation_function
        return f"Computational pattern added: '{pattern}'"

    def add_physical_constant(self, name: str, symbol: str, value: float, unit: str) -> str:
        """
        Adds a new physical constant to the knowledge base.
        
        Args:
            name: Name of the constant
            symbol: Mathematical symbol
            value: Numeric value
            unit: Unit of measurement
            
        Returns:
            Confirmation message
        """
        self.physical_constants[name.lower()] = {
            "symbol": symbol,
            "value": value,
            "unit": unit
        }
        return f"Physical constant added: {name} ({symbol}) = {value} {unit}"

    def symbolic_compute(self, expression: str, operation: str, *args) -> Dict[str, Any]:
        """
        Performs symbolic computation on a mathematical expression.
        
        Args:
            expression: The mathematical expression to compute with
            operation: Operation to perform (derivative, integrate, solve, etc.)
            *args: Additional arguments for the operation
            
        Returns:
            Computation result details
        """
        try:
            if operation in self.computational_operators:
                operator_func = self.computational_operators[operation]
                result = operator_func(expression, *args)
                
                return {
                    "computation": True,
                    "input": expression,
                    "operation": operation,
                    "args": args,
                    "result": result,
                    "explanation": f"Applied {operation} to {expression}: {result}"
                }
            else:
                return {
                    "computation": False,
                    "error": f"Unknown operation: {operation}",
                    "explanation": f"The operation '{operation}' is not supported."
                }
        except Exception as e:
            return {
                "computation": False,
                "input": expression,
                "operation": operation,
                "error": str(e),
                "explanation": f"Error computing {operation} of {expression}: {str(e)}"
            }

    def save_mappings(self, filepath: str) -> str:
        """
        Saves all mappings to a JSON file.
        
        Args:
            filepath: Path to save the mappings
            
        Returns:
            Confirmation message
        """
        data = {
            "math_mappings": self.math_mappings,
            "expanded_math": self.expanded_math,
            "domain_notations": self.domain_notations
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return f"Mappings saved to {filepath}"
        except Exception as e:
            return f"Error saving mappings: {e}"

    def load_mappings(self, filepath: str) -> str:
        """
        Loads mappings from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Confirmation message
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "math_mappings" in data:
                self.math_mappings.update(data["math_mappings"])
            if "expanded_math" in data:
                self.expanded_math.update(data["expanded_math"])
            if "domain_notations" in data:
                for domain, mappings in data["domain_notations"].items():
                    if domain in self.domain_notations:
                        self.domain_notations[domain].update(mappings)
                    else:
                        self.domain_notations[domain] = mappings
                        
            return f"Mappings loaded from {filepath}"
        except Exception as e:
            return f"Error loading mappings: {e}"

    def analyze_expression(self, math_expression: str) -> Dict[str, Any]:
        """
        Analyzes a mathematical expression to identify its components and structure.
        
        Args:
            math_expression: The mathematical expression to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Clean the expression
        expression = math_expression.strip()
        
        # Identify symbols present
        symbols = []
        for symbol in self.expanded_math:
            if len(symbol) > 1 and symbol in expression:  # Avoid single character false positives
                symbols.append({
                    "symbol": symbol,
                    "meaning": self.expanded_math[symbol]
                })
        
        # Look for individual symbols if none found
        if not symbols:
            for symbol in self.expanded_math:
                if len(symbol) == 1 and symbol in expression:
                    symbols.append({
                        "symbol": symbol,
                        "meaning": self.expanded_math[symbol]
                    })
        
        # Attempt to identify the domain
        possible_domains = []
        for domain, notations in self.domain_notations.items():
            domain_matches = 0
            domain_symbols = []
            
            for notation_key, notation_value in notations.items():
                if notation_value in expression:
                    domain_matches += 1
                    domain_symbols.append(notation_value)
            
            if domain_matches > 0:
                possible_domains.append({
                    "domain": domain,
                    "confidence": min(domain_matches / len(notations) * 10, 1.0),
                    "matching_symbols": domain_symbols
                })
        
        # Sort domains by confidence
        possible_domains.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Try to perform symbolic analysis if possible
        symbolic_analysis = {}
        try:
            parsed_expr = parse_expr(expression.replace('∞', 'oo'))
            
            # Check for special properties
            symbolic_analysis["simplified"] = str(simplify(parsed_expr))
            symbolic_analysis["expanded"] = str(expand(parsed_expr))
            symbolic_analysis["factors"] = str(factor(parsed_expr))
            
            # Check expression type
            expr_type = self._get_expression_type(parsed_expr)
            if expr_type:
                symbolic_analysis["type"] = expr_type
                
        except Exception as e:
            symbolic_analysis["error"] = str(e)
        
        # Return the analysis
        return {
            "expression": expression,
            "symbols_identified": symbols,
            "possible_domains": possible_domains,
            "complexity": len(expression) / 10,  # Simple heuristic for complexity
            "symbolic_analysis": symbolic_analysis
        }
    
    def _get_expression_type(self, expr) -> str:
        """
        Identifies the type of mathematical expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            Type of expression
        """
        import sympy
        
        try:
            if isinstance(expr, sympy.Eq):
                return "equation"
            elif isinstance(expr, sympy.Inequality):
                return "inequality"
            elif isinstance(expr, sympy.Integral):
                return "integral"
            elif isinstance(expr, sympy.Derivative):
                return "derivative"
            elif isinstance(expr, sympy.Sum):
                return "summation"
            elif isinstance(expr, sympy.Product):
                return "product"
            elif isinstance(expr, sympy.Matrix):
                return "matrix"
            elif isinstance(expr, sympy.Poly):
                return "polynomial"
            elif isinstance(expr, sympy.Rational):
                return "rational"
            elif isinstance(expr, sympy.sin) or isinstance(expr, sympy.cos) or isinstance(expr, sympy.tan):
                return "trigonometric"
            elif isinstance(expr, sympy.log) or isinstance(expr, sympy.exp):
                return "transcendental"
            else:
                return "algebraic"
        except Exception:
            return "unknown"

    def visualize_expression(self, expression: str) -> Dict[str, Any]:
        """
        Generates visualization data for a mathematical expression.
        
        Args:
            expression: Mathematical expression to visualize
            
        Returns:
            Visualization data
        """
        # This would provide data for visualization rather than the actual visual
        try:
            parsed_expr = parse_expr(expression.replace('∞', 'oo'))
            
            # Generate data for visualization
            vis_data = {
                "expression": expression,
                "visualization_type": "function_plot" if 'x' in str(parsed_expr) else "static",
                "data_points": []
            }
            
            # If expression contains x, generate function plot data
            if 'x' in str(parsed_expr):
                try:
                    x_values = np.linspace(-10, 10, 100)
                    y_values = []
                    
                    x_sym = symbols('x')
                    lambda_func = sympy.lambdify(x_sym, parsed_expr, "numpy")
                    
                    for x_val in x_values:
                        try:
                            y_val = lambda_func(x_val)
                            if isinstance(y_val, (int, float, complex)) and not np.isnan(y_val) and not np.isinf(y_val):
                                # Handle complex numbers
                                if isinstance(y_val, complex):
                                    y_values.append({"x": float(x_val), "y": abs(y_val), "complex": True})
                                else:
                                    y_values.append({"x": float(x_val), "y": float(y_val)})
                        except Exception:
                            # Skip points where function is undefined
                            pass
                    
                    vis_data["data_points"] = y_values
                    vis_data["x_range"] = [-10, 10]
                    vis_data["visualization_type"] = "function_plot"
                    
                except Exception as e:
                    vis_data["error"] = f"Function plot generation error: {str(e)}"
            
            return vis_data
            
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "visualization_type": "error"
            }

    def generate_mathematical_concept_map(self, concept: str) -> Dict[str, Any]:
        """
        Generates a conceptual map of mathematical relationships for a given concept.
        
        Args:
            concept: Mathematical concept to map
            
        Returns:
            Conceptual map data
        """
        concept_lower = concept.lower()
        
        # Find the concept in our knowledge base
        concept_data = {
            "concept": concept,
            "related_concepts": [],
            "domains": [],
            "symbols": []
        }
        
        # Find in mappings
        for word, symbol in self.math_mappings.items():
            if concept_lower in word or word in concept_lower:
                concept_data["symbols"].append({
                    "term": word,
                    "notation": symbol
                })
        
        # Find in domains
        for domain, notations in self.domain_notations.items():
            domain_concepts = []
            
            for notation_key, notation_value in notations.items():
                if concept_lower in notation_key or concept_lower in notation_value:
                    domain_concepts.append({
                        "term": notation_key,
                        "notation": notation_value
                    })
            
            if domain_concepts:
                concept_data["domains"].append({
                    "domain": domain,
                    "concepts": domain_concepts
                })
        
        # Generate related concepts based on mathematical relationships
        relations = {
            "calculus": ["derivative", "integral", "limit", "series", "differential", "gradient"],
            "algebra": ["equation", "matrix", "vector", "polynomial", "group", "ring", "field"],
            "geometry": ["space", "curve", "surface", "manifold", "topology", "metric"],
            "analysis": ["function", "continuity", "convergence", "metric", "space", "norm"],
            "probability": ["distribution", "random", "statistical", "bayesian", "markov"],
            "number_theory": ["prime", "divisor", "congruence", "diophantine", "integer"],
            "logic": ["proof", "theorem", "proposition", "axiom", "model"]
        }
        
        # Check which areas the concept might relate to
        for area, keywords in relations.items():
            for keyword in keywords:
                if keyword in concept_lower:
                    concept_data["related_concepts"].extend([
                        {"concept": related, "relationship": f"related via {area}"} 
                        for related in relations[area] if related != keyword
                    ])
        
        return concept_data

    def evaluate_mathematical_statement(self, statement: str) -> Dict[str, Any]:
        """
        Evaluates the truth value of a mathematical statement.
        
        Args:
            statement: Mathematical statement to evaluate
            
        Returns:
            Evaluation result
        """
        try:
            # Check if this is an equation to solve
            if '=' in statement and '==' not in statement:
                # Replace = with == for sympy equation parsing
                eq_statement = statement.replace('=', '==', 1)
                
                try:
                    # Try to evaluate as a boolean expression
                    result = eval(eq_statement, {"__builtins__": {}}, {
                        "sin": np.sin, "cos": np.cos, "tan": np.tan, 
                        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                        "pi": np.pi, "e": np.e
                    })
                    
                    return {
                        "statement": statement,
                        "truth_value": bool(result),
                        "evaluation_type": "direct",
                        "explanation": f"The statement is {'true' if result else 'false'}"
                    }
                except Exception:
                    # Try to solve the equation
                    try:
                        lhs, rhs = statement.split('=', 1)
                        equation = f"{lhs}-({rhs})"
                        solutions = self.computational_operators["solve"](equation)
                        
                        return {
                            "statement": statement,
                            "truth_value": None,  # Truth depends on variable values
                            "evaluation_type": "equation",
                            "solutions": solutions,
                            "explanation": f"The equation is satisfied when: {solutions}"
                        }
                    except Exception as e:
                        return {
                            "statement": statement,
                            "error": str(e),
                            "explanation": f"Could not evaluate equation: {str(e)}"
                        }
            
            # Check if it's a comparison
            for op in ['>', '<', '>=', '<=', '==', '!=']:
                if op in statement:
                    try:
                        # Try to evaluate as a boolean expression
                        result = eval(statement, {"__builtins__": {}}, {
                            "sin": np.sin, "cos": np.cos, "tan": np.tan, 
                            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                            "pi": np.pi, "e": np.e
                        })
                        
                        return {
                            "statement": statement,
                            "truth_value": bool(result),
                            "evaluation_type": "comparison",
                            "explanation": f"The comparison is {'true' if result else 'false'}"
                        }
                    except Exception as e:
                        return {
                            "statement": statement,
                            "error": str(e),
                            "explanation": f"Could not evaluate comparison: {str(e)}"
                        }
            
            # Try to evaluate as an expression
            try:
                result = eval(statement, {"__builtins__": {}}, {
                    "sin": np.sin, "cos": np.cos, "tan": np.tan, 
                    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                    "pi": np.pi, "e": np.e
                })
                
                return {
                    "statement": statement,
                    "value": result,
                    "evaluation_type": "expression",
                    "explanation": f"The expression evaluates to {result}"
                }
            except Exception as e:
                # If all else fails, try symbolic analysis
                try:
                    analysis = self.analyze_expression(statement)
                    
                    return {
                        "statement": statement,
                        "evaluation_type": "symbolic",
                        "symbolic_analysis": analysis["symbolic_analysis"],
                        "explanation": "Performed symbolic analysis of the statement"
                    }
                except Exception as inner_e:
                    return {
                        "statement": statement,
                        "error": f"{str(e)}; {str(inner_e)}",
                        "explanation": "Could not evaluate the mathematical statement"
                    }
        
        except Exception as e:
            return {
                "statement": statement,
                "error": str(e),
                "explanation": f"Evaluation error: {str(e)}"
            }

    def extend_translation_style(self, style_name: str, template: str, connectors: List[str]) -> str:
        """
        Adds a new translation style.
        
        Args:
            style_name: Name of the style
            template: Template string with {concept} and {symbol} placeholders
            connectors: List of connector phrases
            
        Returns:
            Confirmation message
        """
        style_name = style_name.lower()
        
        self.translation_styles[style_name] = {
            "template": template,
            "connectors": connectors
        }
        
        return f"Translation style '{style_name}' added successfully"
    
    def create_mathematical_narrative(self, concepts: List[str], narrative_style: str = "educational") -> str:
        """
        Creates a narrative that weaves together multiple mathematical concepts.
        
        Args:
            concepts: List of mathematical concepts to include
            narrative_style: Style of narrative (educational, historical, poetic, etc.)
            
        Returns:
            Mathematical narrative
        """
        matched_concepts = {}
        
        # Find symbols for each concept
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Direct matches
            for word, symbol in self.math_mappings.items():
                if concept_lower in word or word in concept_lower:
                    matched_concepts[concept] = {
                        "symbol": symbol,
                        "expanded": self.expanded_math.get(symbol, "")
                    }
                    break
            
            # Check domain-specific notations if not found
            if concept not in matched_concepts:
                for domain, notations in self.domain_notations.items():
                    for notation_key, notation_value in notations.items():
                        if concept_lower in notation_key:
                            matched_concepts[concept] = {
                                "symbol": notation_value,
                                "expanded": self.expanded_math.get(notation_value, ""),
                                "domain": domain
                            }
                            break
                    
                    if concept in matched_concepts:
                        break
            
            # If still not found, create a placeholder
            if concept not in matched_concepts:
                matched_concepts[concept] = {
                    "symbol": f"[{concept}]",
                    "expanded": f"representation of {concept}"
                }
        
        # Create narrative based on style
        if narrative_style == "educational":
            return self._create_educational_narrative(matched_concepts)
        elif narrative_style == "historical":
            return self._create_historical_narrative(matched_concepts)
        elif narrative_style == "poetic":
            return self._create_poetic_narrative(matched_concepts)
        elif narrative_style == "philosophical":
            return self._create_philosophical_narrative(matched_concepts)
        else:
            return self._create_educational_narrative(matched_concepts)
    
    def _create_educational_narrative(self, concepts: Dict[str, Dict[str, str]]) -> str:
        """Creates an educational narrative connecting mathematical concepts."""
        narrative = "Let's explore how these mathematical concepts interconnect:\n\n"
        
        for i, (concept, info) in enumerate(concepts.items()):
            if i == 0:
                narrative += f"We begin with {concept}, represented symbolically as {info['symbol']}. "
                if info['expanded']:
                    narrative += f"This represents {info['expanded']}. "
            else:
                narrative += f"\n\nBuilding on this, we can consider {concept}, denoted as {info['symbol']}. "
                if info['expanded']:
                    narrative += f"This notation represents {info['expanded']}. "
            
            # Add domain context if available
            if 'domain' in info:
                narrative += f"This concept is particularly important in the field of {info['domain']}. "
            
            # Add connections to previous concepts
            if i > 0:
                prev_concept = list(concepts.keys())[i-1]
                narrative += f"There's an interesting relationship between {concept} and {prev_concept}, as both involve structured mathematical thinking. "
        
        narrative += f"\n\nBy understanding these {len(concepts)} concepts together, we gain insight into the interconnected nature of mathematical ideas and their powerful applications."
        
        return narrative
    
    def _create_historical_narrative(self, concepts: Dict[str, Dict[str, str]]) -> str:
        """Creates a historical narrative of mathematical concepts."""
        narrative = "Throughout the history of mathematics, ideas have built upon each other in fascinating ways:\n\n"
        
        for concept, info in concepts.items():
            narrative += f"The concept of {concept}, now symbolized as {info['symbol']}, emerged through centuries of mathematical evolution. "
            if info['expanded']:
                narrative += f"Mathematicians developed this notation to represent {info['expanded']}. "
            
            # Add domain context if available
            if 'domain' in info:
                narrative += f"This revolutionized thinking in {info['domain']}. "
            
            narrative += "\n\n"
        
        narrative += "These mathematical ideas didn't develop in isolation but influenced each other across cultures and time periods, gradually forming the cohesive language of mathematics we use today."
        
        return narrative
    
    def _create_poetic_narrative(self, concepts: Dict[str, Dict[str, str]]) -> str:
        """Creates a poetic narrative of mathematical concepts."""
        narrative = "A symphony of mathematical symbols unfolds:\n\n"
        
        for concept, info in concepts.items():
            narrative += f"The {concept} dances into view as {info['symbol']},\n"
            if info['expanded']:
                narrative += f"Whispering secrets of {info['expanded']},\n"
            
            # Add domain context if available
            if 'domain' in info:
                narrative += f"Resonating through the chambers of {info['domain']},\n"
            
            narrative += "\n"
        
        narrative += "Together they form a tapestry of relationships,\nPatterns interwoven in the fabric of mathematical truth,\nEach symbol a note in the eternal song of logic and form."
        
        return narrative
    
    def _create_philosophical_narrative(self, concepts: Dict[str, Dict[str, str]]) -> str:
        """Creates a philosophical narrative of mathematical concepts."""
        narrative = "Mathematics reveals profound philosophical truths through its symbols:\n\n"
        
        for concept, info in concepts.items():
            narrative += f"When we contemplate {concept}, represented as {info['symbol']}, we encounter a form of truth that transcends mere calculation. "
            if info['expanded']:
                narrative += f"This symbol manifests {info['expanded']} as a bridge between thought and reality. "
            
            # Add domain context if available
            if 'domain' in info:
                narrative += f"In the realm of {info['domain']}, this concept illuminates deeper structures of existence. "
            
            narrative += "\n\n"
        
        narrative += "These mathematical concepts together suggest that reality itself may have mathematical underpinnings, raising profound questions about the nature of knowledge, existence, and the human mind's capacity to grasp abstract truth."
        
        return narrative

    def process_chain_of_operations(self, initial_expression: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes a chain of mathematical operations sequentially.
        
        Args:
            initial_expression: Starting expression
            operations: List of operations to apply, each containing type and parameters
            
        Returns:
            Results of the operation chain
        """
        results = []
        current_expr = initial_expression
        
        try:
            # Process each operation in sequence
            for i, operation in enumerate(operations):
                op_type = operation.get("type", "")
                
                if op_type == "derivative":
                    var = operation.get("variable", "x")
                    result = self.computational_operators["derivative"](current_expr, var)
                    results.append({
                        "step": i+1,
                        "operation": f"derivative with respect to {var}",
                        "input": current_expr,
                        "output": result
                    })
                    current_expr = result
                
                elif op_type == "integrate":
                    var = operation.get("variable", "x")
                    result = self.computational_operators["integrate"](current_expr, var)
                    results.append({
                        "step": i+1,
                        "operation": f"integrate with respect to {var}",
                        "input": current_expr,
                        "output": result
                    })
                    current_expr = result
                
                elif op_type == "simplify":
                    result = self.computational_operators["simplify"](current_expr)
                    results.append({
                        "step": i+1,
                        "operation": "simplify",
                        "input": current_expr,
                        "output": result
                    })
                    current_expr = result
                
                elif op_type == "expand":
                    result = self.computational_operators["expand"](current_expr)
                    results.append({
                        "step": i+1,
                        "operation": "expand",
                        "input": current_expr,
                        "output": result
                    })
                    current_expr = result
                
                elif op_type == "factor":
                    result = self.computational_operators["factor"](current_expr)
                    results.append({
                        "step": i+1,
                        "operation": "factor",
                        "input": current_expr,
                        "output": result
                    })
                    current_expr = result
                
                elif op_type == "substitute":
                    var = operation.get("variable", "x")
                    value = operation.get("value", "0")
                    
                    # Create substitution dictionary
                    subs_dict = {symbols(var): parse_expr(value)}
                    
                    # Evaluate with substitution
                    result = self.computational_operators["evaluate"](current_expr, subs_dict)
                    results.append({
                        "step": i+1,
                        "operation": f"substitute {var}={value}",
                        "input": current_expr,
                        "output": result
                    })
                    current_expr = result
                
                else:
                    results.append({
                        "step": i+1,
                        "operation": op_type,
                        "input": current_expr,
                        "error": f"Unknown operation type: {op_type}"
                    })
        
        except Exception as e:
            results.append({
                "step": len(results) + 1,
                "error": str(e),
                "input": current_expr
            })
        
        return {
            "initial_expression": initial_expression,
            "final_result": current_expr,
            "steps": results
        }


# Legacy method for backward compatibility
def translate(phrase):
    """
    Simple translation function for backward compatibility.
    """
    translator = SymbolicMathTranslator()
    result = translator.translate(phrase)
    return result["explanation"]


if __name__ == "__main__":
    # Example usage when run directly
    translator = SymbolicMathTranslator()
    
    # Test translations
    tests = [
        "The infinite sum approaches equilibrium",
        "The rate of change increases over time",
        "For all systems, there exists a state of balance",
        "The wave function collapses upon observation",
        "Knowledge increases entropy while decreasing uncertainty"
    ]
    
    print("=== Mathematical Translations ===")
    for test in tests:
        result = translator.translate(test)
        print(f"\nInput: {test}")
        print(f"Output: {result['explanation']}")
        print(f"Matches: {result['matches']}")
    
    # Test reverse translations
    math_tests = [
        "∇ · F = 0",
        "E = mc²",
        "P(A|B) = P(A ∩ B)/P(B)",
        "Δx·Δp ≥ ℏ/2"
    ]
    
    print("\n=== Natural Language Translations ===")
    for test in math_tests:
        result = translator.translate_to_text(test)
        print(f"\nInput: {test}")
        print(f"Output: {result}")
    
    # Test computational capabilities
    print("\n=== Computational Capabilities ===")
    
    comp_tests = [
        "derivative of x^2 + 3*x with respect to x",
        "integrate x^2 from 0 to 1",
        "solve x^2 - 4 = 0 for x"
    ]
    
    for test in comp_tests:
        result = translator._check_for_computation(test)
        if result:
            print(f"\nInput: {test}")
            print(f"Result: {result.get('result', 'Error: ' + result.get('error', 'unknown error'))}")
            print(f"Explanation: {result['explanation']}")