import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

class JudgmentProtocol:
    """
    A protocol for making human-like judgments across multiple cognitive frameworks.
    """
    
    def __init__(self):
        """Initialize the judgment protocol with cognitive frameworks and emotional state."""
        # Define cognitive frameworks with their evaluation criteria
        self.cognitive_frameworks = {
            "rational": {
                "checks": ["consistency", "clarity", "evidence_quality", "explanatory_power"],
                "thresholds": {"high": 0.8, "medium": 0.5, "low": 0.0},
                "verdicts": {
                    "high": "Rationally Sound",
                    "medium": "Rationally Plausible",
                    "low": "Rationally Questionable"
                }
            },
            "ethical": {
                "checks": ["value_pluralism", "justice_considerations", "virtue_kindness", "virtue_justice", "virtue_wisdom"],
                "thresholds": {"high": 0.8, "medium": 0.5, "low": 0.0},
                "verdicts": {
                    "high": "Ethically Sound",
                    "medium": "Ethically Acceptable",
                    "low": "Ethically Questionable"
                }
            },
            "aesthetic": {
                "checks": ["experiential_richness", "form_content_coherence", "aesthetic_significance"],
                "thresholds": {"high": 0.8, "medium": 0.5, "low": 0.0},
                "verdicts": {
                    "high": "Aesthetically Compelling",
                    "medium": "Aesthetically Interesting",
                    "low": "Aesthetically Weak"
                }
            },
            "pragmatic": {
                "checks": ["implementability", "resource_feasibility", "scalability"],
                "thresholds": {"high": 0.8, "medium": 0.5, "low": 0.0},
                "verdicts": {
                    "high": "Highly Practical",
                    "medium": "Moderately Practical",
                    "low": "Impractical"
                }
            },
            "contextual": {
                "checks": ["situational_fit", "cultural_fit", "domain_appropriateness"],
                "thresholds": {"high": 0.8, "medium": 0.5, "low": 0.0},
                "verdicts": {
                    "high": "Contextually Appropriate",
                    "medium": "Contextually Adequate",
                    "low": "Contextually Inappropriate"
                }
            },
            "pluralistic": {
                "checks": ["value_pluralism", "cultural_fit", "domain_practicality"],
                "thresholds": {"high": 0.8, "medium": 0.5, "low": 0.0},
                "verdicts": {
                    "high": "Pluralistically Valid",
                    "medium": "Pluralistically Acceptable",
                    "low": "Pluralistically Limited"
                }
            }
        }
        
        # All available frameworks
        self.all_frameworks = list(self.cognitive_frameworks.keys())
        
        # Initialize emotional state
        self.emotional_state = {
            "neutral": 0.7,
            "joy": 0.0,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.0
        }
        
        # Initialize domain expertise
        self.domain_expertise = {
            "science": 0.8,
            "philosophy": 0.8,
            "ethics": 0.8,
            "politics": 0.7,
            "art": 0.7,
            "technology": 0.8,
            "social": 0.7
        }
        
        # Initialize cultural framework
        self.cultural_framework = "western"
        
        # Initialize tacit knowledge (background influences that may affect judgment)
        self.tacit_knowledge = {
            "status_quo_bias": {
                "description": "Tendency to prefer existing state",
                "activation_threshold": 0.6,
                "influence_degree": 0.2
            },
            "optimism_bias": {
                "description": "Tendency to overestimate positive outcomes",
                "activation_threshold": 0.7,
                "influence_degree": 0.15
            },
            "cultural_background": {
                "description": "Influence of cultural upbringing",
                "activation_threshold": 0.5,
                "influence_degree": 0.25
            }
        }
        
        # Initialize judgment history
        self.judgment_history = []
        
        # Initialize current context
        self.current_context = {}
        
        # Optional components
        self.memory = None  # Could be connected to a memory system
        self.intuition = None  # Could be connected to a somatic marker system
    
    def evaluate(self, claim: str, framework: str = "rational", detailed_output: bool = False, context: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """
        Evaluate a claim using a specified cognitive framework.
        
        Args:
            claim: The claim to evaluate
            framework: The cognitive framework to use
            detailed_output: Whether to return detailed evaluation information
            context: Optional context information
            
        Returns:
            Either a verdict string or detailed evaluation dictionary
        """
        # Process context if provided
        self._assess_context(claim, context)
        
        # Select framework
        if framework not in self.cognitive_frameworks:
            if detailed_output:
                return {
                    "error": f"Unknown framework: {framework}",
                    "available_frameworks": list(self.cognitive_frameworks.keys())
                }
            return f"Cannot evaluate: unknown framework '{framework}'"
            
        framework_info = self.cognitive_frameworks[framework]
        checks = framework_info["checks"]
        
        # Run appropriate checks for this framework
        results = []
        details = {}
        
        for check in checks:
            check_method = getattr(self, f"_check_{check}", None)
            if check_method:
                if check in ["situational_fit", "cultural_fit", "domain_appropriateness", "domain_practicality"]:
                    # These checks require context
                    if check == "domain_appropriateness" or check == "domain_practicality":
                        domain = self.current_context.get("domain", "general")
                        result = check_method(claim, domain)
                    else:
                        result = check_method(claim, self.current_context)
                else:
                    # Standard checks without context
                    result = check_method(claim)
                    
                if result:
                    results.append(result["score"])
                    details[check] = result
            
        # Average the scores
        if not results:
            if detailed_output:
                return {
                    "error": "No valid checks were performed",
                    "framework": framework
                }
            return "Unable to evaluate claim with this framework"
            
        avg_score = sum(results) / len(results)
        
        # Apply tacit knowledge influences
        active_influences = self.get_active_tacit_knowledge()
        for influence in active_influences:
            if influence in self.tacit_knowledge:
                influence_data = self.tacit_knowledge[influence]
                influence_degree = influence_data["influence_degree"]
                
                if influence == "status_quo_bias":
                    # Status quo bias may reduce scores for novel/challenging claims
                    if "novel" in claim.lower() or "innovative" in claim.lower():
                        avg_score = max(0.0, avg_score - influence_degree)
                elif influence == "optimism_bias":
                    # Optimism bias may increase scores generally
                    avg_score = min(1.0, avg_score + influence_degree)
        
        # Apply emotional state influence
        dominant_emotion, intensity = self._get_dominant_emotion()
        if dominant_emotion != "neutral" and intensity > 0.3:
            if dominant_emotion in ["fear", "disgust", "anger"]:
                # These emotions can reduce perceived credibility
                avg_score = max(0.0, avg_score - intensity * 0.1)
            elif framework == "pragmatic" and dominant_emotion in ["joy", "anticipation"]:
                # These emotions can increase pragmatic optimism
                avg_score = min(1.0, avg_score + intensity * 0.1)
            elif framework == "pluralistic" and dominant_emotion in ["trust"]:
                # Trust can enhance pluralistic appreciation
                avg_score = min(1.0, avg_score + intensity * 0.1)
                
        # Calculate confidence for this specialized evaluation
        confidence = self._calculate_confidence(framework, self.current_context)
        
        # Determine verdict
        if avg_score >= framework_info["thresholds"]["high"]:
            verdict = framework_info["verdicts"]["high"]
        elif avg_score >= framework_info["thresholds"]["medium"]:
            verdict = framework_info["verdicts"]["medium"]
        else:
            verdict = framework_info["verdicts"]["low"]
            
        # Adjust verdict language based on confidence
        if confidence < 0.4:
            verdict = f"Possibly {verdict.lower()}"
        elif confidence > 0.8:
            verdict = f"Definitely {verdict.lower()}"
            
        return {
            "score": avg_score,
            "verdict": verdict,
            "details": details,
            "confidence": confidence
        } if detailed_output else verdict

    def multi_perspective_evaluation(self, claim: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates a claim through multiple cognitive frameworks with human-like integration.
        
        Args:
            claim: The claim to evaluate
            context: Optional context information
            
        Returns:
            Dictionary with results from all frameworks and consensus information
        """
        # Process context if provided
        self._assess_context(claim, context)
        
        # Update emotional state based on claim and context
        self._update_emotional_state(claim, context)
        
        # Evaluate using all frameworks
        framework_evaluations = {}
        scores = []
        confidences = []
        
        for framework in self.all_frameworks:
            result = self.evaluate(claim, framework, detailed_output=True, context=context)
            if isinstance(result, dict):
                score = result.get("score", 0.5)
                verdict = result.get("verdict", "No verdict")
                confidence = result.get("confidence", 0.5)
                
                framework_evaluations[framework] = {
                    "score": score,
                    "verdict": verdict,
                    "confidence": confidence
                }
                scores.append(score)
                confidences.append(confidence)
            else:
                # Handle string result case
                framework_evaluations[framework] = {
                    "score": 0.5,
                    "verdict": result,
                    "confidence": 0.5
                }
                scores.append(0.5)
                confidences.append(0.5)
            
        # Calculate confidence-weighted average score
        if sum(confidences) > 0:
            weighted_scores = [s * c for s, c in zip(scores, confidences)]
            average_score = sum(weighted_scores) / sum(confidences)
        else:
            average_score = sum(scores) / len(scores) if scores else 0.5
        
        # Calculate consensus metrics with human-like assessment
        # Calculate how much agreement exists between frameworks
        score_variance = sum((s - average_score) ** 2 for s in scores) / len(scores) if scores else 0
        
        # Higher variance = lower consensus
        consensus_score = 1 - (score_variance * 2)  # Scale variance to a 0-1 consensus score
        consensus_score = max(0, min(1, consensus_score))  # Clamp to 0-1 range
        
        # Apply emotional influence to consensus assessment
        dominant_emotion, intensity = self._get_dominant_emotion()
        if dominant_emotion != "neutral" and intensity > 0.5:
            if dominant_emotion in ["fear", "anger"]:
                # These emotions can decrease perception of consensus
                consensus_score = max(0, consensus_score - intensity * 0.1)
            elif dominant_emotion in ["trust"]:
                # Trust can increase perception of consensus
                consensus_score = min(1, consensus_score + intensity * 0.1)
                
        # Determine consensus level
        if consensus_score >= 0.8:
            consensus_level = "Strong Consensus"
        elif consensus_score >= 0.6:
            consensus_level = "Moderate Consensus"
        elif consensus_score >= 0.4:
            consensus_level = "Limited Consensus"
        else:
            consensus_level = "No Consensus"
            
        # Determine overall verdict based on average score
        if average_score >= 0.8:
            overall_verdict = "Strongly supported across frameworks"
        elif average_score >= 0.6:
            overall_verdict = "Moderately supported across frameworks"
        elif average_score >= 0.4:
            overall_verdict = "Ambiguously supported across frameworks"
        elif average_score >= 0.2:
            overall_verdict = "Minimally supported across frameworks"
        else:
            overall_verdict = "Unsupported across frameworks"
            
        # Add intuitively supported reasoning
        intuitive_assessment = ""
        if self.intuition:
            try:
                # Get somatic response to aid intuitive assessment
                somatic_state = self.intuition.get_current_somatic_state()
                if somatic_state:
                    if somatic_state.get("centeredness", 0.5) > 0.7:
                        intuitive_assessment = "This assessment feels centered and grounded."
                    elif somatic_state.get("expansion", 0.5) > 0.7:
                        intuitive_assessment = "This assessment comes with a sense of openness and possibility."
                    elif somatic_state.get("muscular_tension", 0.5) > 0.7:
                        intuitive_assessment = "This assessment comes with a sense of tension that suggests caution."
            except Exception:
                pass
                
        if not intuitive_assessment and dominant_emotion != "neutral" and intensity > 0.5:
            if dominant_emotion == "joy":
                intuitive_assessment = "This assessment feels uplifting and resonant."
            elif dominant_emotion == "trust":
                intuitive_assessment = "This assessment feels trustworthy and well-grounded."
            elif dominant_emotion == "fear":
                intuitive_assessment = "This assessment comes with a note of caution."
            elif dominant_emotion == "surprise":
                intuitive_assessment = "This assessment contains unexpected elements worth exploring."
                
        # Calculate overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Record in judgment history
        judgment_record = {
            "timestamp": datetime.now().isoformat(),
            "claim": claim,
            "framework": "multi_perspective",
            "verdict": overall_verdict,
            "score": average_score,
            "confidence": overall_confidence,
            "consensus_score": consensus_score,
            "context": self.current_context,
            "emotional_state": {k: v for k, v in self.emotional_state.items() if v > 0}
        }
        self.judgment_history.append(judgment_record)
            
        result = {
            "claim": claim,
            "framework_evaluations": framework_evaluations,
            "average_score": average_score,
            "overall_confidence": overall_confidence,
            "consensus_score": consensus_score,
            "consensus_level": consensus_level,
            "overall_verdict": overall_verdict,
            "dominant_emotion": dominant_emotion if intensity > 0.3 else "neutral",
            "emotional_intensity": intensity if intensity > 0.3 else 0,
            "intuitive_assessment": intuitive_assessment,
            "context": self.current_context,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def _check_consistency(self, claim: str) -> Dict[str, Any]:
        """Logical check: Evaluates internal consistency of the claim."""
        # Check for contradiction terms
        contradiction_markers = ["but", "however", "nevertheless", "yet", "although", "despite"]
        has_contradiction_markers = any(marker in claim.lower() for marker in contradiction_markers)
        
        if has_contradiction_markers:
            # Check if claim explicitly acknowledges complexity
            complexity_markers = ["complex", "nuanced", "multifaceted", "paradoxical", "tension", "balance"]
            has_complexity_markers = any(marker in claim.lower() for marker in complexity_markers)
            
            if has_complexity_markers:
                # Acknowledging complexity resolves potential contradiction
                return {"check": "consistency", "score": 0.8, "reason": "Acknowledges complexity/tension."}
            else:
                # Potential unresolved contradiction
                return {"check": "consistency", "score": 0.5, "reason": "Contains potential contradiction."}
        
        # Check for universal claims which are harder to maintain consistency
        universal_markers = ["all", "every", "always", "never", "none", "universal"]
        has_universal_markers = any(marker in claim.lower().split() for marker in universal_markers)
        
        if has_universal_markers:
            return {"check": "consistency", "score": 0.6, "reason": "Makes universal claims that are difficult to maintain consistently."}
            
        # Check for modifier terms that strengthen consistency
        consistency_markers = ["consistently", "coherent", "systematic", "follows", "builds on"]
        has_consistency_markers = any(marker in claim.lower() for marker in consistency_markers)
        
        if has_consistency_markers:
            return {"check": "consistency", "score": 0.9, "reason": "Explicitly addresses consistency."}
            
        # Default case - no obvious consistency issues
        return {"check": "consistency", "score": 0.7, "reason": "No obvious consistency issues detected."}

    def _check_clarity(self, claim: str) -> Dict[str, Any]:
        """Logical check: Evaluates precision and clarity of expression."""
        # Check for vague language
        vague_markers = ["sort of", "kind of", "perhaps", "maybe", "somewhat", "relatively"]
        vague_count = sum(1 for marker in vague_markers if marker in claim.lower())
        
        if vague_count >= 2:
            return {"check": "clarity", "score": 0.4, "reason": "Contains multiple vague qualifiers."}
        elif vague_count == 1:
            return {"check": "clarity", "score": 0.6, "reason": "Contains some vague language."}
            
        # Check for precise, clear language
        precision_markers = ["specifically", "precisely", "exactly", "clearly", "defined as", "in particular"]
        has_precision_markers = any(marker in claim.lower() for marker in precision_markers)
        
        if has_precision_markers:
            return {"check": "clarity", "score": 0.9, "reason": "Uses precise language."}
            
        # Check for ambiguous pronouns without clear referents
        pronoun_pattern = r'\b(it|they|this|that|these|those)\b'
        pronouns = re.findall(pronoun_pattern, claim.lower())
        
        if len(pronouns) > 3:  # Arbitrary threshold for potential ambiguity
            return {"check": "clarity", "score": 0.5, "reason": "Contains potentially ambiguous pronouns."}
            
        # Check sentence length as proxy for clarity
        words = claim.split()
        avg_words_per_sentence = len(words) / max(1, len(re.split(r'[.!?]', claim)) - 1)
        
        if avg_words_per_sentence > 30:  # Very long sentences
            return {"check": "clarity", "score": 0.5, "reason": "Contains excessively long sentences."}
        elif avg_words_per_sentence > 20:  # Moderately long sentences
            return {"check": "clarity", "score": 0.7, "reason": "Contains moderately complex sentences."}
            
        return {"check": "clarity", "score": 0.8, "reason": "Generally clear expression."}

    def _check_evidence_quality(self, claim: str) -> Dict[str, Any]:
        """Logical check: Evaluates the quality of evidence provided."""
        # Check for evidence markers
        evidence_markers = ["because", "since", "as shown by", "evidence suggests", "research indicates", "data", "study"]
        has_evidence_markers = any(marker in claim.lower() for marker in evidence_markers)
        
        if has_evidence_markers:
            # Check for specific, quantifiable evidence
            specific_evidence_markers = ["percent", "significant", "study", "research", "experiment", "survey", "analysis"]
            has_specific_evidence = any(marker in claim.lower() for marker in specific_evidence_markers)
            
            if has_specific_evidence:
                return {"check": "evidence_quality", "score": 0.9, "reason": "Provides specific evidence."}
            else:
                return {"check": "evidence_quality", "score": 0.7, "reason": "References evidence without specifics."}
                
        # Check for logical reasoning even without explicit evidence
        reasoning_markers = ["therefore", "thus", "consequently", "it follows that", "leads to"]
        has_reasoning_markers = any(marker in claim.lower() for marker in reasoning_markers)
        
        if has_reasoning_markers:
            return {"check": "evidence_quality", "score": 0.6, "reason": "Uses logical reasoning without explicit evidence."}
            
        # Check for hedging that weakens evidential claims
        hedging_markers = ["possibly", "might", "could", "potentially", "perhaps", "may"]
        has_hedging_markers = any(marker in claim.lower().split() for marker in hedging_markers)
        
        if has_hedging_markers:
            return {"check": "evidence_quality", "score": 0.5, "reason": "Uses hedging language."}
            
        # Check for assertions without support
        assertion_markers = ["definitely", "certainly", "undoubtedly", "clearly", "obviously"]
        has_assertion_markers = any(marker in claim.lower().split() for marker in assertion_markers)
        
        if has_assertion_markers:
            return {"check": "evidence_quality", "score": 0.3, "reason": "Makes strong assertions without evidence."}
            
        return {"check": "evidence_quality", "score": 0.5, "reason": "Limited explicit evidence."}
    
    def _check_explanatory_power(self, claim: str) -> Dict[str, Any]:
        """Logical check: Evaluates the explanatory power of the claim."""
        # Check for explanatory language
        explanation_markers = ["explains", "accounts for", "clarifies", "illuminates", "reveals why", "makes sense of"]
        has_explanation_markers = any(marker in claim.lower() for marker in explanation_markers)
        
        if has_explanation_markers:
            return {"check": "explanatory_power", "score": 0.8, "reason": "Explicitly offers explanatory framework."}
            
        # Check for causal language
        causal_markers = ["causes", "leads to", "results in", "produces", "generates", "creates"]
        has_causal_markers = any(marker in claim.lower() for marker in causal_markers)
        
        if has_causal_markers:
            return {"check": "explanatory_power", "score": 0.7, "reason": "Provides causal mechanism."}
            
        # Check for predictive elements
        predictive_markers = ["predicts", "forecasts", "anticipates", "expects", "projects", "foresees"]
        has_predictive_markers = any(marker in claim.lower() for marker in predictive_markers)
        
        if has_predictive_markers:
            return {"check": "explanatory_power", "score": 0.75, "reason": "Offers predictive framework."}
            
        # Check for comparative explanatory claims
        comparative_markers = ["better explains", "more coherent", "more comprehensive", "more parsimonious"]
        has_comparative_markers = any(marker in claim.lower() for marker in comparative_markers)
        
        if has_comparative_markers:
            return {"check": "explanatory_power", "score": 0.85, "reason": "Claims comparative explanatory advantage."}
            
        # Check if claim merely describes rather than explains
        descriptive_markers = ["is characterized by", "consists of", "comprises", "contains", "involves"]
        only_descriptive = any(marker in claim.lower() for marker in descriptive_markers) and not (
            has_explanation_markers or has_causal_markers or has_predictive_markers)
            
        if only_descriptive:
            return {"check": "explanatory_power", "score": 0.4, "reason": "Primarily descriptive rather than explanatory."}
            
        return {"check": "explanatory_power", "score": 0.5, "reason": "Moderate explanatory content."}

    def _check_value_pluralism(self, claim: str) -> Dict[str, Any]:
        """Ethical check: Evaluates recognition of multiple value perspectives."""
        # Check for pluralistic language
        pluralism_markers = ["different perspectives", "various values", "multiple viewpoints", "diversity of", "depends on context"]
        has_pluralism_markers = any(marker in claim.lower() for marker in pluralism_markers)
        
        if has_pluralism_markers:
            return {"check": "value_pluralism", "score": 0.9, "reason": "Explicitly acknowledges value pluralism."}
            
        # Check for universalist language
        universalist_markers = ["universal", "absolute", "for all", "objective", "regardless of"]
        has_universalist_markers = any(marker in claim.lower() for marker in universalist_markers)
        
        if has_universalist_markers:
            return {"check": "value_pluralism", "score": 0.3, "reason": "Indicates universalist value framework."}
            
        # Check for contextual value language
        contextual_markers = ["in some contexts", "depending on", "varies with", "relative to", "situational"]
        has_contextual_markers = any(marker in claim.lower() for marker in contextual_markers)
        
        if has_contextual_markers:
            return {"check": "value_pluralism", "score": 0.8, "reason": "Recognizes contextual value variation."}
            
        # Check for cultural recognition
        cultural_markers = ["culturally", "in different cultures", "cultural context", "tradition", "heritage"]
        has_cultural_markers = any(marker in claim.lower() for marker in cultural_markers)
        
        if has_cultural_markers:
            return {"check": "value_pluralism", "score": 0.85, "reason": "Acknowledges cultural value differences."}
            
        return {"check": "value_pluralism", "score": 0.6, "reason": "Neutral on value pluralism."}

    def _check_justice_considerations(self, claim: str) -> Dict[str, Any]:
        """Ethical check: Evaluates consideration of justice and fairness."""
        # Check for justice language
        justice_markers = ["justice", "fairness", "rights", "equality", "equity", "discrimination", "oppression"]
        has_justice_markers = any(marker in claim.lower() for marker in justice_markers)
        
        if has_justice_markers:
            return {"check": "justice_considerations", "score": 0.9, "reason": "Explicitly addresses justice concerns."}
            
        # Check for power language
        power_markers = ["power", "privilege", "disadvantage", "marginalized", "vulnerable", "access"]
        has_power_markers = any(marker in claim.lower() for marker in power_markers)
        
        if has_power_markers:
            return {"check": "justice_considerations", "score": 0.8, "reason": "Addresses power dynamics."}
            
        # Check for distribution language
        distribution_markers = ["distribution", "allocation", "resources", "wealth", "opportunity", "access"]
        has_distribution_markers = any(marker in claim.lower() for marker in distribution_markers)
        
        if has_distribution_markers:
            return {"check": "justice_considerations", "score": 0.75, "reason": "Addresses distributive concerns."}
            
        # Check for procedural justice
        procedural_markers = ["process", "representation", "voice", "participation", "inclusion", "transparency"]
        has_procedural_markers = any(marker in claim.lower() for marker in procedural_markers)
        
        if has_procedural_markers:
            return {"check": "justice_considerations", "score": 0.7, "reason": "Addresses procedural justice."}
            
        return {"check": "justice_considerations", "score": 0.5, "reason": "Limited explicit justice considerations."}

    def _check_virtue_kindness(self, claim: str) -> Dict[str, Any]:
        """Ethical check: Evaluates consideration of kindness and empathy."""
        # Check for kindness/empathy language
        kindness_markers = ["kindness", "compassion", "empathy", "care", "support", "help", "understanding"]
        has_kindness_markers = any(marker in claim.lower() for marker in kindness_markers)
        
        if has_kindness_markers:
            return {"check": "virtue_kindness", "score": 0.9, "reason": "Explicitly addresses kindness/empathy."}
            
        # Check for harshness/cruelty
        cruelty_markers = ["harsh", "cruel", "mean", "callous", "indifferent", "cold", "uncaring"]
        has_cruelty_markers = any(marker in claim.lower() for marker in cruelty_markers)
        
        if has_cruelty_markers:
            return {"check": "virtue_kindness", "score": 0.2, "reason": "Contains language counter to kindness."}
            
        # Check for consideration of others' feelings
        feeling_markers = ["feelings", "emotional impact", "how others feel", "emotional needs", "emotional well-being"]
        has_feeling_markers = any(marker in claim.lower() for marker in feeling_markers)
        
        if has_feeling_markers:
            return {"check": "virtue_kindness", "score": 0.8, "reason": "Considers emotional impact on others."}
            
        # Check for helping language
        helping_markers = ["help", "assist", "support", "aid", "benefit", "serve", "contribute to"]
        has_helping_markers = any(marker in claim.lower() for marker in helping_markers)
        
        if has_helping_markers:
            return {"check": "virtue_kindness", "score": 0.7, "reason": "Includes helping/supporting elements."}
            
        return {"check": "virtue_kindness", "score": 0.5, "reason": "Neutral on kindness/empathy."}

    def _check_virtue_justice(self, claim: str) -> Dict[str, Any]:
        """Ethical check: Evaluates consideration of justice as a virtue."""
        # Similar to justice considerations but focused on character virtues
        justice_markers = ["fair", "just", "equitable", "impartial", "deserved", "merited", "balanced"]
        has_justice_markers = any(marker in claim.lower().split() for marker in justice_markers)
        
        if has_justice_markers:
            return {"check": "virtue_justice", "score": 0.9, "reason": "Embodies justice as virtue."}
            
        # Check for injustice language
        injustice_markers = ["unfair", "unjust", "biased", "prejudiced", "discriminatory", "inequitable"]
        has_injustice_markers = any(marker in claim.lower() for marker in injustice_markers)
        
        if has_injustice_markers:
            return {"check": "virtue_justice", "score": 0.2, "reason": "Contains unjust elements."}
            
        # Check for desert/merit language
        desert_markers = ["deserve", "earn", "merit", "worthy", "entitlement", "right to"]
        has_desert_markers = any(marker in claim.lower() for marker in desert_markers)
        
        if has_desert_markers:
            return {"check": "virtue_justice", "score": 0.7, "reason": "Addresses desert/merit considerations."}
            
        return {"check": "virtue_justice", "score": 0.5, "reason": "Neutral on justice as virtue."}

    def _check_virtue_wisdom(self, claim: str) -> Dict[str, Any]:
        """Ethical check: Evaluates presence of wisdom as a virtue."""
        # Check for wisdom markers
        wisdom_markers = ["wise", "wisdom", "prudent", "judicious", "discerning", "insightful", "thoughtful"]
        has_wisdom_markers = any(marker in claim.lower().split() for marker in wisdom_markers)
        
        if has_wisdom_markers:
            return {"check": "virtue_wisdom", "score": 0.9, "reason": "Explicitly embodies wisdom."}
            
        # Check for consideration of multiple factors
        complexity_markers = ["multiple factors", "various considerations", "complex interplay", "nuanced", "multifaceted"]
        has_complexity_markers = any(marker in claim.lower() for marker in complexity_markers)
        
        if has_complexity_markers:
            return {"check": "virtue_wisdom", "score": 0.8, "reason": "Demonstrates nuanced understanding."}
            
        # Check for long-term thinking
        longterm_markers = ["long-term", "sustainable", "future generations", "lasting", "enduring", "consequences"]
        has_longterm_markers = any(marker in claim.lower() for marker in longterm_markers)
        
        if has_longterm_markers:
            return {"check": "virtue_wisdom", "score": 0.75, "reason": "Considers long-term implications."}
            
        # Check for moderation/balance
        moderation_markers = ["balance", "moderation", "middle way", "tempered", "measured", "proportional"]
        has_moderation_markers = any(marker in claim.lower() for marker in moderation_markers)
        
        if has_moderation_markers:
            return {"check": "virtue_wisdom", "score": 0.8, "reason": "Embodies balance and moderation."}
            
        return {"check": "virtue_wisdom", "score": 0.5, "reason": "Neutral on wisdom as virtue."}

    def _check_experiential_richness(self, claim: str) -> Dict[str, Any]:
        """Aesthetic check: Evaluates richness of experiential content."""
        # Check for sensory language
        sensory_markers = ["see", "hear", "feel", "touch", "taste", "smell", "sense", "experience"]
        sensory_count = sum(1 for marker in sensory_markers if marker in claim.lower())
        
        if sensory_count >= 2:
            return {"check": "experiential_richness", "score": 0.9, "reason": "Rich sensory language."}
        elif sensory_count == 1:
            return {"check": "experiential_richness", "score": 0.7, "reason": "Contains some sensory language."}
            
        # Check for emotional language
        emotion_markers = ["joy", "sorrow", "anger", "fear", "wonder", "awe", "delight", "melancholy"]
        has_emotion_markers = any(marker in claim.lower() for marker in emotion_markers)
        
        if has_emotion_markers:
            return {"check": "experiential_richness", "score": 0.8, "reason": "Contains emotional richness."}
            
        # Check for experiential terms
        experience_markers = ["experience", "lived", "feeling", "sensation", "perception", "awareness", "consciousness"]
        has_experience_markers = any(marker in claim.lower() for marker in experience_markers)
        
        if has_experience_markers:
            return {"check": "experiential_richness", "score": 0.75, "reason": "Directly references experience."}
            
        # Check for purely abstract language
        abstract_markers = ["concept", "theory", "principle", "abstract", "theoretical", "framework", "system"]
        only_abstract = any(marker in claim.lower() for marker in abstract_markers) and not (
            sensory_count > 0 or has_emotion_markers or has_experience_markers)
            
        if only_abstract:
            return {"check": "experiential_richness", "score": 0.3, "reason": "Exclusively abstract language."}
            
        return {"check": "experiential_richness", "score": 0.4, "reason": "Limited experiential content."}

    def _check_form_content_coherence(self, claim: str) -> Dict[str, Any]:
        """Aesthetic check: Evaluates alignment of form and content."""
        # Check for explicit aesthetic language
        aesthetic_markers = ["beauty", "aesthetic", "form", "style", "expression", "artistic", "creative"]
        has_aesthetic_markers = any(marker in claim.lower() for marker in aesthetic_markers)
        
        if has_aesthetic_markers:
            # Check for form-content language
            form_content_markers = ["reflects", "expresses", "embodies", "represents", "manifests"]
            has_form_content = any(marker in claim.lower() for marker in form_content_markers)
            
            if has_form_content:
                return {"check": "form_content_coherence", "score": 0.9, "reason": "Explicit form-content relationship."}
            else:
                return {"check": "form_content_coherence", "score": 0.7, "reason": "Contains aesthetic language."}
                
        # Check for structural elements in the claim itself
        has_structural_elements = False
        
        # Look for patterns, parallelism, or other structural features
        words = claim.split()
        if len(words) > 10:
            # Check for parallelism (repeated syntactic structures)
            # This is a simplified check for demonstration
            phrases = claim.split(",")
            if len(phrases) >= 3:
                has_structural_elements = True
                
        if has_structural_elements:
            return {"check": "form_content_coherence", "score": 0.8, "reason": "Contains structural coherence."}
            
        # Check for medium-message alignment
        medium_markers = ["medium", "form", "presentation", "delivery", "style", "manner", "expression"]
        message_markers = ["message", "content", "meaning", "substance", "idea", "point", "concept"]
        has_medium_message = any(m1 in claim.lower() for m1 in medium_markers) and any(m2 in claim.lower() for m2 in message_markers)
        
        if has_medium_message:
            return {"check": "form_content_coherence", "score": 0.85, "reason": "Addresses medium-message relationship."}
            
        return {"check": "form_content_coherence", "score": 0.5, "reason": "Neutral form-content relationship."}

    def _check_aesthetic_significance(self, claim: str) -> Dict[str, Any]:
        """Aesthetic check: Evaluates aesthetic significance of the claim."""
        # Check for significance language
        significance_markers = ["significant", "important", "meaningful", "profound", "reveals", "illuminates"]
        has_significance_markers = any(marker in claim.lower() for marker in significance_markers)
        
        if has_significance_markers:
            # Check for aesthetic domain
            aesthetic_domain_markers = ["art", "beauty", "literature", "music", "poetry", "creative", "imagination"]
            has_aesthetic_domain = any(marker in claim.lower() for marker in aesthetic_domain_markers)
            
            if has_aesthetic_domain:
                return {"check": "aesthetic_significance", "score": 0.9, "reason": "Claims aesthetic significance."}
            else:
                return {"check": "aesthetic_significance", "score": 0.6, "reason": "Claims significance in non-aesthetic domain."}
                
        # Check for aesthetic value language
        value_markers = ["beautiful", "sublime", "moving", "powerful", "elegant", "harmonious", "graceful"]
        has_value_markers = any(marker in claim.lower() for marker in value_markers)
        
        if has_value_markers:
            return {"check": "aesthetic_significance", "score": 0.8, "reason": "Expresses aesthetic value."}
            
        # Check for transformative impact
        transformative_markers = ["transformative", "life-changing", "perspective-shifting", "consciousness-altering"]
        has_transformative_markers = any(marker in claim.lower() for marker in transformative_markers)
        
        if has_transformative_markers:
            return {"check": "aesthetic_significance", "score": 0.85, "reason": "Suggests transformative aesthetic impact."}
            
        return {"check": "aesthetic_significance", "score": 0.5, "reason": "Limited claims to aesthetic significance."}

    def _check_implementability(self, claim: str) -> Dict[str, Any]:
        """Practical check: Evaluates whether a claim can be implemented."""
        # Check for practical language
        practical_markers = ["implement", "apply", "use", "practice", "action", "do", "perform"]
        has_practical_markers = any(marker in claim.lower() for marker in practical_markers)
        
        if has_practical_markers:
            return {"check": "implementability", "score": 0.8, "reason": "Contains practical implementation language."}
            
        # Check for abstract vs. concrete language
        abstract_markers = ["theoretical", "abstract", "conceptual", "philosophical", "ideal"]
        has_abstract_markers = any(marker in claim.lower() for marker in abstract_markers)
        
        if has_abstract_markers:
            return {"check": "implementability", "score": 0.3, "reason": "Primarily abstract/theoretical."}
            
        # Check for specific steps or methods
        method_markers = ["method", "step", "procedure", "process", "technique", "approach"]
        has_method_markers = any(marker in claim.lower() for marker in method_markers)
        
        if has_method_markers:
            return {"check": "implementability", "score": 0.9, "reason": "Describes specific methods or procedures."}
            
        # Check for action-oriented language
        action_markers = ["action", "initiative", "undertaking", "endeavor", "operation", "activity"]
        has_action_markers = any(marker in claim.lower() for marker in action_markers)
        
        if has_action_markers:
            return {"check": "implementability", "score": 0.75, "reason": "Contains action-oriented language."}
            
        return {"check": "implementability", "score": 0.5, "reason": "Unclear implementability."}

    def _check_resource_feasibility(self, claim: str) -> Dict[str, Any]:
        """Practical check: Evaluates resource requirements and feasibility."""
        # Check for resource language
        resource_markers = ["resources", "cost", "time", "effort", "investment", "requires", "needs"]
        has_resource_markers = any(marker in claim.lower() for marker in resource_markers)
        
        if has_resource_markers:
            # Check for feasibility qualifiers
            feasibility_markers = ["feasible", "practical", "realistic", "achievable", "doable"]
            has_feasibility_markers = any(marker in claim.lower() for marker in feasibility_markers)
            
            if has_feasibility_markers:
                return {"check": "resource_feasibility", "score": 0.9, "reason": "Explicitly addresses feasibility."}
            else:
                return {"check": "resource_feasibility", "score": 0.7, "reason": "Mentions resources without clear feasibility."}
                
        # Check for idealistic language
        idealistic_markers = ["ideal", "perfect", "optimal", "ultimate", "best possible"]
        has_idealistic_markers = any(marker in claim.lower() for marker in idealistic_markers)
        
        if has_idealistic_markers:
            return {"check": "resource_feasibility", "score": 0.4, "reason": "Contains idealistic language."}
            
        # Check for constraint recognition
        constraint_markers = ["constraint", "limitation", "restrict", "boundary", "ceiling", "cap", "limit"]
        has_constraint_markers = any(marker in claim.lower() for marker in constraint_markers)
        
        if has_constraint_markers:
            return {"check": "resource_feasibility", "score": 0.8, "reason": "Acknowledges constraints/limitations."}
            
        # Check for efficiency language
        efficiency_markers = ["efficient", "streamlined", "optimized", "economical", "cost-effective"]
        has_efficiency_markers = any(marker in claim.lower() for marker in efficiency_markers)
        
        if has_efficiency_markers:
            return {"check": "resource_feasibility", "score": 0.85, "reason": "Addresses efficiency concerns."}
            
        return {"check": "resource_feasibility", "score": 0.6, "reason": "Neutral on resource feasibility."}

    def _check_scalability(self, claim: str) -> Dict[str, Any]:
        """Practical check: Evaluates whether a claim can scale to different contexts."""
        # Check for scalability language
        scalability_markers = ["scale", "expand", "grow", "widespread", "broad application", "generalize"]
        has_scalability_markers = any(marker in claim.lower() for marker in scalability_markers)
        
        if has_scalability_markers:
            return {"check": "scalability", "score": 0.9, "reason": "Explicitly addresses scalability."}
            
        # Check for scope language
        scope_markers = ["specific", "particular", "limited", "narrow", "certain cases", "this context"]
        has_scope_markers = any(marker in claim.lower() for marker in scope_markers)
        
        if has_scope_markers:
            return {"check": "scalability", "score": 0.3, "reason": "Indicates limited scope."}
            
        # Check for universal language
        universal_markers = ["all", "every", "universal", "always", "regardless", "in any case"]
        has_universal_markers = any(marker in claim.lower().split() for marker in universal_markers)
        
        if has_universal_markers:
            return {"check": "scalability", "score": 0.7, "reason": "Implies broad applicability."}
            
        # Check for adaptability language
        adaptability_markers = ["adapt", "flexible", "adjustable", "customizable", "tailored", "versatile"]
        has_adaptability_markers = any(marker in claim.lower() for marker in adaptability_markers)
        
        if has_adaptability_markers:
            return {"check": "scalability", "score": 0.85, "reason": "Emphasizes adaptability across contexts."}
            
        return {"check": "scalability", "score": 0.5, "reason": "Unclear scalability."}
        
    def _check_situational_fit(self, claim: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Contextual check: Evaluates appropriateness to the specific situation.
        
        Args:
            claim: The claim to evaluate
            context: Context information
            
        Returns:
            Check result dictionary
        """
        # If we don't have context information, default to neutral
        if not context:
            return {"check": "situational_fit", "score": 0.5, "reason": "No context available."}
            
        # Check for situation-specific language
        situation_markers = ["in this situation", "in this context", "for this case", "here", "now", "in this instance"]
        has_situation_markers = any(marker in claim.lower() for marker in situation_markers)
        
        if has_situation_markers:
            return {"check": "situational_fit", "score": 0.8, "reason": "Explicitly addresses situational context."}
            
        # Check for domain appropriateness
        domain = context.get("domain")
        if domain:
            domain_markers = {
                "science": ["empirical", "evidence", "research", "data", "study", "hypothesis", "theory"],
                "philosophy": ["concept", "theoretical", "metaphysical", "epistemological", "ontological"],
                "ethics": ["moral", "ethical", "right", "wrong", "good", "bad", "virtue", "justice"],
                "aesthetics": ["beauty", "art", "aesthetic", "form", "expression", "creative", "artistic"],
                "politics": ["policy", "government", "political", "rights", "regulation", "law"],
                "social": ["community", "social", "relationship", "cultural", "interpersonal", "collective"]
            }
            
            if domain in domain_markers:
                domain_relevance = any(marker in claim.lower() for marker in domain_markers[domain])
                if domain_relevance:
                    return {"check": "situational_fit", "score": 0.9, "reason": f"Aligns with {domain} domain context."}
                else:
                    return {"check": "situational_fit", "score": 0.4, "reason": f"Limited alignment with {domain} domain context."}
                    
        # Check for social context appropriateness
        if context.get("social_context", False):
            social_markers = ["we", "us", "our", "community", "social", "together", "collective", "shared"]
            social_relevance = any(marker in claim.lower().split() for marker in social_markers)
            
            if social_relevance:
                return {"check": "situational_fit", "score": 0.85, "reason": "Aligns with social context."}
                
        # Check for prior experience relevance
        if context.get("prior_experience", False) and self.memory:
            try:
                similar_judgments = self.memory.search(claim, limit=3)
                if similar_judgments:
                    return {"check": "situational_fit", "score": 0.8, "reason": "Aligns with prior similar contexts."}
            except Exception:
                pass
                
        return {"check": "situational_fit", "score": 0.6, "reason": "Moderately appropriate to situation."}
        
    def _check_cultural_fit(self, claim: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Contextual check: Evaluates alignment with cultural context.
        
        Args:
            claim: The claim to evaluate
            context: Context information
            
        Returns:
            Check result dictionary
        """
        # If we don't have context information, default to neutral
        if not context:
            return {"check": "cultural_fit", "score": 0.5, "reason": "No cultural context available."}
            
        # Get cultural context
        cultural_context = context.get("cultural_context", self.cultural_framework)
        
        # Cultural markers for different frameworks
        cultural_markers = {
            "western": ["individual", "rights", "freedom", "autonomy", "choice", "rational", "evidence", "analysis"],
            "eastern": ["harmony", "balance", "relationship", "whole", "interconnected", "cyclical", "holistic"],
            "indigenous": ["community", "land", "ancestors", "spiritual", "reciprocity", "tradition", "relationship"]
        }
        
        # Alignment with current cultural framework
        if cultural_context in cultural_markers:
            alignment = any(marker in claim.lower() for marker in cultural_markers[cultural_context])
            
            if alignment:
                return {"check": "cultural_fit", "score": 0.85, "reason": f"Aligns with {cultural_context} cultural framework."}
                
        # Check for cultural sensitivity language
        sensitivity_markers = ["culturally", "culture", "tradition", "heritage", "values", "norms", "practices"]
        has_sensitivity_markers = any(marker in claim.lower() for marker in sensitivity_markers)
        
        if has_sensitivity_markers:
            return {"check": "cultural_fit", "score": 0.8, "reason": "Shows cultural awareness."}
            
        # Check for cultural mismatch
        if cultural_context in cultural_markers:
            # Check for markers that strongly contradict this cultural framework
            contradictions = {
                "western": ["collective obligations", "disregard individual", "reject rationality"],
                "eastern": ["purely individualistic", "ignore relationships", "dismiss harmony"],
                "indigenous": ["exploit resources", "disregard community", "disconnect from land"]
            }
            
            mismatch = any(marker in claim.lower() for marker in contradictions.get(cultural_context, []))
            
            if mismatch:
                return {"check": "cultural_fit", "score": 0.2, "reason": f"Contradicts {cultural_context} cultural values."}
                
        return {"check": "cultural_fit", "score": 0.6, "reason": "Neutral cultural alignment."}
        
    def _check_domain_appropriateness(self, claim: str, domain: str) -> Dict[str, Any]:
        """
        Contextual check: Evaluates appropriateness to specific domain.
        
        Args:
            claim: The claim to evaluate
            domain: Domain context
            
        Returns:
            Check result dictionary
        """
        # Domain-specific evaluation criteria
        domain_criteria = {
            "science": {
                "positive": ["evidence", "data", "research", "study", "empirical", "observation", "experiment"],
                "negative": ["faith", "intuition", "feeling", "belief", "revelation", "dogma"]
            },
            "philosophy": {
                "positive": ["concept", "logic", "reasoning", "premise", "argument", "inference", "principle"],
                "negative": ["proven", "demonstrated", "shown", "measured", "observed"]
            },
            "ethics": {
                "positive": ["value", "good", "right", "moral", "ethical", "virtue", "obligation", "justice"],
                "negative": ["efficient", "profitable", "optimized", "maximized", "useful"]
            },
            "aesthetics": {
                "positive": ["beauty", "form", "experience", "sensation", "expression", "creativity", "art"],
                "negative": ["utility", "function", "purpose", "use", "effectiveness", "efficiency"]
            },
            "politics": {
                "positive": ["policy", "governance", "power", "authority", "rights", "law", "institution"],
                "negative": ["eternal", "universal", "absolute", "inevitable", "natural law"]
            },
            "social": {
                "positive": ["relationship", "interaction", "community", "group", "social", "cultural", "norm"],
                "negative": ["isolation", "solitary", "individual", "alone", "separated"]
            }
        }
        
        if domain in domain_criteria:
            criteria = domain_criteria[domain]
            
            # Check positive markers
            positive_count = sum(1 for marker in criteria["positive"] if marker in claim.lower())
            
            # Check negative markers
            negative_count = sum(1 for marker in criteria["negative"] if marker in claim.lower())
            
            # Calculate appropriateness score
            if positive_count > 0 and negative_count == 0:
                return {"check": "domain_appropriateness", "score": 0.9, "reason": f"Well aligned with {domain} domain."}
            elif positive_count > 0 and negative_count > 0:
                return {"check": "domain_appropriateness", "score": 0.6, "reason": f"Mixed alignment with {domain} domain."}
            elif positive_count == 0 and negative_count > 0:
                return {"check": "domain_appropriateness", "score": 0.3, "reason": f"Poorly aligned with {domain} domain."}
                
        return {"check": "domain_appropriateness", "score": 0.5, "reason": "Neutral domain appropriateness."}
        
    def _check_domain_practicality(self, claim: str, domain: str) -> Dict[str, Any]:
        """
        Practical check: Evaluates domain-specific practicality.
        
        Args:
            claim: The claim to evaluate
            domain: Domain context
            
        Returns:
            Check result dictionary
        """
        # Domain-specific practicality criteria
        domain_practicality = {
            "science": {
                "practical": ["testable", "falsifiable", "measurable", "replicable", "observable", "precise"],
                "impractical": ["untestable", "unfalsifiable", "immeasurable", "subjective", "vague"]
            },
            "philosophy": {
                "practical": ["coherent", "consistent", "applicable", "clarifying", "meaningful", "useful"],
                "impractical": ["incoherent", "contradictory", "meaningless", "purely abstract", "useless"]
            },
            "ethics": {
                "practical": ["actionable", "implementable", "realistic", "case-based", "contextual"],
                "impractical": ["impossible", "unrealistic", "utopian", "perfect", "absolute"]
            },
            "politics": {
                "practical": ["implementable", "enforceable", "acceptable", "feasible", "effective"],
                "impractical": ["unenforceable", "unacceptable", "infeasible", "ineffective"]
            },
            "technology": {
                "practical": ["scalable", "maintainable", "efficient", "reliable", "interoperable", "secure"],
                "impractical": ["unscalable", "unmaintainable", "inefficient", "unreliable", "insecure"]
            }
        }
        
        if domain in domain_practicality:
            criteria = domain_practicality[domain]
            
            # Check practical markers
            practical_count = sum(1 for marker in criteria["practical"] if marker in claim.lower())
            
            # Check impractical markers
            impractical_count = sum(1 for marker in criteria["impractical"] if marker in claim.lower())
            
            # Calculate practicality score
            if practical_count > 0 and impractical_count == 0:
                return {"check": "domain_practicality", "score": 0.9, "reason": f"Practical for {domain} domain."}
            elif practical_count > 0 and impractical_count > 0:
                return {"check": "domain_practicality", "score": 0.5, "reason": f"Mixed practicality for {domain} domain."}
            elif practical_count == 0 and impractical_count > 0:
                return {"check": "domain_practicality", "score": 0.2, "reason": f"Impractical for {domain} domain."}
                
        return {"check": "domain_practicality", "score": 0.6, "reason": "Domain-neutral practicality."}

    def _calculate_confidence(self, framework: str, context: Dict[str, Any]) -> float:
        """
        Calculate confidence level for a particular evaluation.
        
        Args:
            framework: The cognitive framework being used
            context: Context information
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence level
        confidence = 0.6  # Start with moderate confidence
        
        # Adjust based on domain expertise if context specifies domain
        if context and "domain" in context:
            domain = context["domain"]
            if domain in self.domain_expertise:
                # Higher expertise = higher confidence
                confidence += (self.domain_expertise[domain] - 0.5) * 0.4
                
        # Adjust based on framework familiarity
        framework_familiarity = {
            "rational": 0.9,  # Most trained on rational evaluation
            "ethical": 0.8,
            "pragmatic": 0.8,
            "contextual": 0.7,
            "aesthetic": 0.7,
            "pluralistic": 0.6  # Least familiar with pluralistic
        }
        
        if framework in framework_familiarity:
            confidence += (framework_familiarity[framework] - 0.5) * 0.3
            
        # Emotional state can affect confidence
        dominant_emotion, intensity = self._get_dominant_emotion()
        if dominant_emotion == "trust" and intensity > 0.5:
            confidence += intensity * 0.1  # Trust increases confidence
        elif dominant_emotion in ["fear", "surprise"] and intensity > 0.5:
            confidence -= intensity * 0.1  # These emotions decrease confidence
            
        # Ensure confidence stays in valid range
        return max(0.1, min(0.99, confidence))
    
    def _get_dominant_emotion(self) -> tuple:
        """
        Get the dominant emotion and its intensity.
        
        Returns:
            Tuple of (dominant_emotion, intensity)
        """
        # Exclude neutral from determining the dominant emotion
        emotions = {emotion: intensity for emotion, intensity in self.emotional_state.items() if emotion != "neutral"}
        
        if not emotions:
            return "neutral", self.emotional_state.get("neutral", 0.5)
            
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        return dominant_emotion[0], dominant_emotion[1]
    
    def _update_emotional_state(self, claim: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update emotional state based on claim content and context.
        
        Args:
            claim: The claim being evaluated
            context: Optional context information
        """
        # Emotional trigger words
        emotion_triggers = {
            "joy": ["happy", "joy", "delight", "pleasure", "wonderful", "excellent", "great"],
            "trust": ["trust", "reliable", "dependable", "honest", "faithful", "confident"],
            "fear": ["fear", "danger", "threat", "risk", "scary", "terrifying", "harmful"],
            "surprise": ["surprise", "unexpected", "astonishing", "shocking", "amazing", "striking"],
            "sadness": ["sad", "sorrow", "grief", "unhappy", "depressing", "melancholy", "disappointed"],
            "disgust": ["disgust", "revolting", "offensive", "repulsive", "abhorrent", "vile"],
            "anger": ["anger", "rage", "fury", "outrage", "annoyed", "irritated", "frustrated"],
            "anticipation": ["anticipate", "expect", "foresee", "predict", "await", "coming"]
        }
        
        # Detect emotional content in claim
        claim_lower = claim.lower()
        detected_emotions = {}
        
        for emotion, triggers in emotion_triggers.items():
            count = sum(1 for trigger in triggers if trigger in claim_lower)
            if count > 0:
                intensity = min(0.7, 0.3 + (count * 0.1))  # Scale intensity by number of triggers
                detected_emotions[emotion] = intensity
                
        # Update emotional state with mild regression to neutral
        for emotion in self.emotional_state:
            if emotion == "neutral":
                # Neutral state increases as other emotions decay
                continue
            elif emotion in detected_emotions:
                # Blend current state with detected emotion
                current = self.emotional_state[emotion]
                detected = detected_emotions[emotion]
                self.emotional_state[emotion] = current * 0.7 + detected * 0.3
            else:
                # Emotions decay over time
                self.emotional_state[emotion] *= 0.9
                
        # Context can modify emotional state
        if context:
            if context.get("emotional_valence") == "positive":
                self.emotional_state["joy"] = min(1.0, self.emotional_state["joy"] + 0.1)
                self.emotional_state["trust"] = min(1.0, self.emotional_state["trust"] + 0.1)
            elif context.get("emotional_valence") == "negative":
                self.emotional_state["sadness"] = min(1.0, self.emotional_state["sadness"] + 0.1)
                self.emotional_state["fear"] = min(1.0, self.emotional_state["fear"] + 0.1)
                
        # Recalculate neutral state based on other emotions
        total_emotion = sum(v for k, v in self.emotional_state.items() if k != "neutral")
        self.emotional_state["neutral"] = max(0.1, 1.0 - (total_emotion / 8))