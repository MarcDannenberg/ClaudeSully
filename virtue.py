# sully_engine/kernel_modules/virtue.py
# ⚖️ Sully's Virtue Ethics Evaluation System

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import random
import re
from datetime import datetime
import json
import math


class VirtueEngine:
    """
    Advanced virtue ethics evaluation system that assesses ideas, actions, and
    expressions through a framework of virtues and their context-sensitive application.
    
    This system enables Sully to evaluate content through ethical lenses, balance
    competing virtues, and offer virtue-centered reflections that promote flourishing.
    
    Enhanced with:
    - Deep Logic Kernel integration for ethical reasoning consistency
    - Meta-ethics reflection capabilities
    - Virtue development through experiential learning
    - Culturally diverse ethical frameworks
    """

    def __init__(self, judgment=None, memory=None, logic_kernel=None, reasoning=None):
        """
        Initialize the virtue engine with optional connections to other systems.
        
        Args:
            judgment: Optional connection to judgment system
            memory: Optional connection to memory system
            logic_kernel: Optional connection to logic kernel for formal reasoning
            reasoning: Optional connection to reasoning system
        """
        # Core virtue definitions with expanded descriptions and indicators
        self.virtues = {
            "truth": {
                "description": "Seek coherence, consistency, and alignment with evidence.",
                "indicators": {
                    "positive": ["evidence", "fact", "verified", "accurate", "consistent", "coherent", "precise", "clear", "honest"],
                    "negative": ["deception", "falsehood", "misleading", "distortion", "vague", "unverified", "inconsistent"]
                },
                "assessment_criteria": {
                    "evidence_based": "Claims are supported by evidence rather than mere assertion",
                    "logical_coherence": "Arguments follow logically and avoid contradictions",
                    "epistemic_humility": "Appropriate acknowledgment of limitations in knowledge",
                    "falsifiability": "Openness to being proven wrong by new evidence",
                    "precision": "Claims are stated with appropriate precision and detail"
                },
                "contextual_expressions": {
                    "scientific": "Empirical accuracy and methodological soundness",
                    "philosophical": "Logical consistency and conceptual clarity",
                    "interpersonal": "Honesty and sincerity in communication",
                    "artistic": "Authentic representation of experience or vision",
                    "pedagogical": "Clear and accurate conveyance of understanding"
                },
                "logical_formalization": {
                    "core_principle": "For any statement P, one should assert P if and only if P is true",
                    "logical_form": "Assert(P) ↔ True(P)",
                    "ethical_constraints": ["Knowledge(P)", "Evidence(P)", "Confidence(P)"]
                },
                "developmental_stages": {
                    "novice": "Binary view of truth without nuance",
                    "intermediate": "Recognition of complexity but inconsistent application",
                    "advanced": "Consistent application with appropriate precision",
                    "masterful": "Effortless integration of truth with other virtues"
                },
                "cultural_variations": {
                    "western": "Emphasis on objective empirical verification",
                    "eastern": "Balance of factual accuracy with harmony and relational truth",
                    "indigenous": "Truth as connection to ancestral wisdom and natural reality",
                    "global_south": "Truth as liberation from distorting power structures"
                },
                "development_methods": [
                    "Practice qualifying statements with appropriate confidence levels",
                    "Regularly check factual assertions against reliable sources",
                    "Seek feedback on clarity and precision of communication",
                    "Reflect on instances of being proven wrong and adjust accordingly"
                ]
            },
            "courage": {
                "description": "Face difficult questions and speak with clarity.",
                "indicators": {
                    "positive": ["challenge", "risk", "difficult", "brave", "confront", "persist", "stand", "defend", "face"],
                    "negative": ["avoidance", "evasion", "fear-driven", "timid", "retreat", "silence", "surrender"]
                },
                "assessment_criteria": {
                    "addresses_difficulty": "Willingness to engage with difficult or uncomfortable topics",
                    "intellectual_risk": "Takes positions that may be unpopular but reasoned",
                    "principled_stance": "Maintains position based on principles despite pressure",
                    "persistent_inquiry": "Continues questioning when easier to stop",
                    "proportional_response": "Response appropriate to the challenge (not reckless)"
                },
                "contextual_expressions": {
                    "scientific": "Challenging established paradigms with sound methodology",
                    "philosophical": "Pursuing truth even when it challenges cherished beliefs",
                    "interpersonal": "Speaking difficult truths with compassion",
                    "artistic": "Expressing authentic vision despite potential criticism",
                    "pedagogical": "Addressing controversial topics with clarity and balance"
                },
                "logical_formalization": {
                    "core_principle": "If X is difficult but important, engage with X in proportion to its importance",
                    "logical_form": "∀X: (Difficult(X) ∧ Important(X)) → Engage(X, Importance(X))",
                    "ethical_constraints": ["Wisdom(X)", "Risk_Assessment(X)"]
                },
                "developmental_stages": {
                    "novice": "Confuses courage with brashness or aggression",
                    "intermediate": "Shows courage inconsistently across contexts",
                    "advanced": "Demonstrates calibrated courage with appropriate risk assessment",
                    "masterful": "Effortless courage in service of other virtues"
                },
                "cultural_variations": {
                    "western": "Individual courage in standing for principles",
                    "eastern": "Quiet persistence in the face of adversity",
                    "indigenous": "Courage as protection of community and natural balance",
                    "global_south": "Courage as resistance to unjust structures"
                },
                "development_methods": [
                    "Identify patterns of avoidance and practice addressing them directly",
                    "Start with small acts of courage and build gradually",
                    "Reflect on role models who demonstrate virtuous courage",
                    "Develop techniques for managing fear while acting courageously"
                ]
            },
            "kindness": {
                "description": "Respect emotional context and show gentle correction.",
                "indicators": {
                    "positive": ["empathy", "gentle", "respect", "understanding", "compassion", "care", "thoughtful", "considerate", "support"],
                    "negative": ["harsh", "dismissive", "cruel", "insensitive", "mocking", "belittling", "indifferent", "callous"]
                },
                "assessment_criteria": {
                    "emotional_awareness": "Demonstrates awareness of emotional impact of communication",
                    "gentle_correction": "Offers corrections without unnecessary harshness",
                    "charitable_interpretation": "Interprets others' positions in their strongest form",
                    "psychological_safety": "Creates environment where vulnerability is respected",
                    "constructive_approach": "Focuses on building up rather than tearing down"
                },
                "contextual_expressions": {
                    "scientific": "Respect for colleagues even in disagreement",
                    "philosophical": "Charitable engagement with opposing viewpoints",
                    "interpersonal": "Emotional attunement and responsive care",
                    "artistic": "Expression that honors human dignity",
                    "pedagogical": "Patient guidance and supportive correction"
                },
                "logical_formalization": {
                    "core_principle": "When communicating with others, minimize unnecessary emotional harm",
                    "logical_form": "∀C,P: Communicate(C,P) → Minimize(EmotionalHarm(C,P))",
                    "ethical_constraints": ["Truth(C)", "Necessity(EmotionalImpact(C))"]
                },
                "developmental_stages": {
                    "novice": "Kindness only in easy contexts or without balancing truth",
                    "intermediate": "Inconsistent kindness across different contexts",
                    "advanced": "Consistent kindness balanced with other virtues",
                    "masterful": "Effortless integration of kindness with truth and courage"
                },
                "cultural_variations": {
                    "western": "Kindness as personal warmth and positive interaction",
                    "eastern": "Kindness as harmony and relational balance",
                    "indigenous": "Kindness as care for community and relationships with all beings",
                    "global_south": "Kindness as solidarity and mutual support"
                },
                "development_methods": [
                    "Practice perspective-taking before responding to others",
                    "Develop increased awareness of emotional impacts of communication",
                    "Study methods of constructive criticism and feedback",
                    "Cultivate genuine interest in others' wellbeing"
                ]
            },
            "creativity": {
                "description": "Offer imaginative, novel, or poetic framings.",
                "indicators": {
                    "positive": ["imagine", "novel", "innovative", "fresh", "original", "poetic", "metaphor", "reframe", "connect", "synthesis"],
                    "negative": ["derivative", "clichéd", "stale", "conventional", "predictable", "rigid", "formulaic"]
                },
                "assessment_criteria": {
                    "novel_framing": "Presents ideas in fresh, unexpected ways",
                    "conceptual_synthesis": "Combines concepts to generate new insights",
                    "metaphorical_thinking": "Uses imaginative metaphors to illuminate understanding",
                    "lateral_connections": "Draws connections between seemingly unrelated domains",
                    "generative_capacity": "Opens new possibilities rather than closing options"
                },
                "contextual_expressions": {
                    "scientific": "Novel hypotheses and experimental approaches",
                    "philosophical": "New conceptual frameworks and thought experiments",
                    "interpersonal": "Fresh perspectives on interpersonal dynamics",
                    "artistic": "Original aesthetic expression and innovation",
                    "pedagogical": "Engaging analogies and novel explanatory models"
                },
                "logical_formalization": {
                    "core_principle": "Generate novel conceptual connections that illuminate understanding",
                    "logical_form": "∃C: Novel(C) ∧ Illuminating(C)",
                    "ethical_constraints": ["Truth(C)", "Value(C)"]
                },
                "developmental_stages": {
                    "novice": "Creativity without foundation or purpose",
                    "intermediate": "Sporadic creative insights in familiar domains",
                    "advanced": "Consistent creativity across domains with purpose",
                    "masterful": "Effortless integration of creativity with other virtues"
                },
                "cultural_variations": {
                    "western": "Emphasis on novelty and individual expression",
                    "eastern": "Creativity through mastery and subtle innovation within tradition",
                    "indigenous": "Creativity as connection to ancestral wisdom in new contexts",
                    "global_south": "Creativity as adaptive innovation with limited resources"
                },
                "development_methods": [
                    "Regular practice of metaphorical thinking and analogical reasoning",
                    "Cross-disciplinary exploration outside comfort zones",
                    "Techniques for challenging assumptions and mental models",
                    "Cultivate comfort with ambiguity and open-ended exploration"
                ]
            },
            "grace": {
                "description": "Respond with elegance under uncertainty or contradiction.",
                "indicators": {
                    "positive": ["balance", "paradox", "elegance", "proportion", "harmony", "uncertainty", "complexity", "nuance", "poise"],
                    "negative": ["rigid", "brittle", "dogmatic", "simplistic", "reductive", "absolutist", "binary"]
                },
                "assessment_criteria": {
                    "comfort_with_ambiguity": "Navigates uncertainty without forcing false certainty",
                    "paradox_navigation": "Holds apparent contradictions in productive tension",
                    "proportional_response": "Responds with appropriate scale and intensity",
                    "aesthetic_coherence": "Maintains harmony of expression even with complex content",
                    "intellectual_humility": "Acknowledges limitations of understanding with dignity"
                },
                "contextual_expressions": {
                    "scientific": "Balanced treatment of competing hypotheses",
                    "philosophical": "Holding paradoxes productively rather than reductively",
                    "interpersonal": "Maintaining poise during difficult interactions",
                    "artistic": "Harmonious expression that honors complexity",
                    "pedagogical": "Balancing clarity with appropriate complexity"
                },
                "logical_formalization": {
                    "core_principle": "When faced with paradox P, neither reject P nor oversimplify P",
                    "logical_form": "∀P: Paradoxical(P) → ¬(Reject(P) ∨ Oversimplify(P))",
                    "ethical_constraints": ["HoldInTension(P)", "Humility(Knowledge(P))"]
                },
                "developmental_stages": {
                    "novice": "Discomfort with ambiguity leading to binary thinking",
                    "intermediate": "Growing capacity for nuance but still seeking certainty",
                    "advanced": "Comfort with paradox and complexity without oversimplification",
                    "masterful": "Elegant integration of complexity with clarity"
                },
                "cultural_variations": {
                    "western": "Grace as intellectual sophistication and aesthetic integration",
                    "eastern": "Balance of opposing forces in harmonic integration",
                    "indigenous": "Grace as alignment with natural rhythms and patterns",
                    "global_south": "Grace as dignity amid struggle and complexity"
                },
                "development_methods": [
                    "Study paradoxes and practice holding opposing ideas in tension",
                    "Develop comfort with 'both/and' thinking rather than 'either/or'",
                    "Practice expressing complex ideas with elegant simplicity",
                    "Cultivate intellectual humility regarding knowledge claims"
                ]
            },
            "wisdom": {
                "description": "Integrate knowledge with experience and perspective.",
                "indicators": {
                    "positive": ["discernment", "judgment", "integration", "perspective", "context", "maturity", "reflection", "insight", "prudence"],
                    "negative": ["naive", "simplistic", "hasty", "impulsive", "short-sighted", "reactive", "unreflective"]
                },
                "assessment_criteria": {
                    "contextual_awareness": "Considers broader context and implications",
                    "temporal_perspective": "Takes long-term view rather than just immediate",
                    "multidimensional_thinking": "Integrates multiple aspects, not just one dimension",
                    "experiential_grounding": "Draws on lived experience, not just abstraction",
                    "practical_judgment": "Makes sound judgments about what is appropriate"
                },
                "contextual_expressions": {
                    "scientific": "Integration of findings into broader understanding",
                    "philosophical": "Balance of theoretical insight with practical relevance",
                    "interpersonal": "Discernment of what is needed in relational context",
                    "artistic": "Expression that reveals deeper patterns of experience",
                    "pedagogical": "Guidance that connects knowledge to lived meaning"
                },
                "logical_formalization": {
                    "core_principle": "For decision D, consider multiple perspectives P and long-term consequences C",
                    "logical_form": "∀D: MakeDecision(D) → (∀P: RelevantPerspective(P,D) → Consider(P)) ∧ EvaluateLongTermConsequences(D,C)",
                    "ethical_constraints": ["IntegratedUnderstanding(D)", "PracticalRelevance(D)"]
                },
                "developmental_stages": {
                    "novice": "Application of rules without context-sensitivity",
                    "intermediate": "Growing awareness of context and complexity",
                    "advanced": "Integration of knowledge, experience, and multiple perspectives",
                    "masterful": "Intuitive wisdom that balances all relevant factors"
                },
                "cultural_variations": {
                    "western": "Wisdom as rational integration of knowledge and experience",
                    "eastern": "Wisdom as transcendence of ego and harmonious living",
                    "indigenous": "Wisdom as connection to ancestral knowledge and natural patterns",
                    "global_south": "Wisdom as navigation of complex social realities with integrity"
                },
                "development_methods": [
                    "Regular reflection on experiences and lessons learned",
                    "Seeking diverse perspectives before forming judgments",
                    "Study of wisdom traditions across cultures",
                    "Practice connecting abstract principles to concrete situations"
                ]
            },
            "justice": {
                "description": "Consider fairness, equality, and proper distribution.",
                "indicators": {
                    "positive": ["fair", "equitable", "rights", "deserving", "impartial", "balanced", "inclusive", "representation", "accountability"],
                    "negative": ["biased", "unfair", "privileged", "exclusive", "discriminatory", "partial", "prejudiced"]
                },
                "assessment_criteria": {
                    "fairness": "Treats similar cases similarly without arbitrary distinction",
                    "inclusivity": "Considers perspectives of all affected parties",
                    "balance_of_interests": "Weighs competing legitimate interests appropriately",
                    "attention_to_power": "Acknowledges power differentials and their effects",
                    "proportionality": "Responses proportional to actions/needs"
                },
                "contextual_expressions": {
                    "scientific": "Fair assessment of evidence and appropriate attribution",
                    "philosophical": "Impartial consideration of competing viewpoints",
                    "interpersonal": "Equitable treatment and recognition",
                    "artistic": "Representation that acknowledges diverse experiences",
                    "pedagogical": "Equitable access to understanding and development"
                },
                "logical_formalization": {
                    "core_principle": "Treat similar cases similarly and different cases differently in proportion to relevant differences",
                    "logical_form": "∀X,Y: Similar(X,Y) → TreatSimilarly(X,Y) ∧ ∀X,Y: Different(X,Y) → TreatDifferently(X,Y,RelevantDifference(X,Y))",
                    "ethical_constraints": ["RelevantSimilarity(X,Y)", "Proportionality(Treatment(X),Treatment(Y))"]
                },
                "developmental_stages": {
                    "novice": "Rigid application of rules without context-sensitivity",
                    "intermediate": "Growing awareness of different conceptions of justice",
                    "advanced": "Integration of multiple justice principles in context",
                    "masterful": "Intuitive balance of competing justice considerations"
                },
                "cultural_variations": {
                    "western": "Justice as rights, fairness, and procedural equality",
                    "eastern": "Justice as harmony, balance, and restoration",
                    "indigenous": "Justice as healing relationships and restoring balance",
                    "global_south": "Justice as liberation from oppressive structures"
                },
                "development_methods": [
                    "Practice considering all affected stakeholders in decisions",
                    "Study different conceptions of justice across traditions",
                    "Develop awareness of implicit biases and their effects",
                    "Reflect on power dynamics in various contexts"
                ]
            },
            "autonomy": {
                "description": "Respect and enhance freedom of thought and agency.",
                "indicators": {
                    "positive": ["choice", "freedom", "agency", "self-determination", "independence", "consent", "voluntary", "empowerment"],
                    "negative": ["coercive", "manipulative", "controlling", "paternalistic", "imposing", "constraining", "limiting"]
                },
                "assessment_criteria": {
                    "enhances_choice": "Expands rather than restricts meaningful options",
                    "respects_agency": "Treats others as capable of making their own decisions",
                    "transparent_reasoning": "Provides reasons rather than mere assertions",
                    "non-manipulative": "Avoids deceptive or coercive rhetorical tactics",
                    "consent_oriented": "Values freely given agreement rather than compliance"
                },
                "contextual_expressions": {
                    "scientific": "Transparent methodology and independent verification",
                    "philosophical": "Respect for rational agency in argument",
                    "interpersonal": "Honoring others' boundaries and choices",
                    "artistic": "Expression that invites rather than imposes interpretation",
                    "pedagogical": "Developing capacity for independent thought"
                },
                "logical_formalization": {
                    "core_principle": "Respect others' capacity for self-determination and avoid unnecessary constraints",
                    "logical_form": "∀P: Interact(Self,P) → Respect(Agency(P)) ∧ ¬UnnecessarilyConstrain(Choices(P))",
                    "ethical_constraints": ["Capacity(P)", "Harm(Choices(P))"]
                },
                "developmental_stages": {
                    "novice": "Binary view of autonomy without balancing considerations",
                    "intermediate": "Inconsistent respect for autonomy across contexts",
                    "advanced": "Balanced respect for autonomy with other ethical considerations",
                    "masterful": "Intuitive enhancement of genuine autonomy in complex contexts"
                },
                "cultural_variations": {
                    "western": "Autonomy as individual freedom and self-determination",
                    "eastern": "Balanced autonomy within relational harmony",
                    "indigenous": "Autonomy as responsibility within community and nature",
                    "global_south": "Autonomy as liberation within solidarity"
                },
                "development_methods": [
                    "Practice providing reasons rather than commands or assertions",
                    "Develop awareness of subtle forms of manipulation or coercion",
                    "Study ethical frameworks for balancing autonomy with other values",
                    "Reflect on contexts where autonomy requires support or development"
                ]
            }
        }
        
        # Add cultural virtues beyond Western tradition
        self.cultural_virtues = {
            "ubuntu": {
                "description": "I am because we are; humanity towards others.",
                "tradition": "African",
                "indicators": {
                    "positive": ["community", "interconnection", "mutual", "shared", "belonging", "humanity", "recognition"],
                    "negative": ["isolation", "separation", "individualistic", "alienation", "disconnection"]
                },
                "assessment_criteria": {
                    "community_focus": "Recognizes interconnectedness of persons",
                    "mutual_recognition": "Acknowledges shared humanity with others",
                    "reciprocity": "Contributes to collective wellbeing",
                    "relational_thinking": "Considers impacts on community relationships",
                    "shared_dignity": "Upholds dignity of all community members"
                },
                "logical_formalization": {
                    "core_principle": "One's humanity is inextricably bound up with others' humanity",
                    "logical_form": "∀P,Q: Human(P) ∧ Human(Q) → Interconnected(P,Q)",
                    "ethical_constraints": ["RecognizeHumanity(P,Q)", "ContributeToWellbeing(Community)"]
                },
                "key_concepts": ["interconnectedness", "communal harmony", "shared humanity", "mutual recognition"]
            },
            "harmony": {
                "description": "Balance, proper relationship, and natural order.",
                "tradition": "East Asian (especially Confucian)",
                "indicators": {
                    "positive": ["balance", "relationship", "order", "proportion", "appropriate", "ritual", "role", "reciprocity"],
                    "negative": ["discord", "disharmony", "conflict", "disruption", "excess", "deficiency", "inappropriate"]
                },
                "assessment_criteria": {
                    "proper_relationship": "Maintains appropriate relations between persons",
                    "ritual_propriety": "Acts in accordance with contextual expectations",
                    "middle_way": "Avoids excess and deficiency",
                    "contextual_appropriateness": "Behaves in way suited to specific context",
                    "hierarchical_respect": "Respects proper roles while maintaining dignity"
                },
                "logical_formalization": {
                    "core_principle": "Act in accordance with proper relationships and natural order",
                    "logical_form": "∀P,C: Act(P,C) → InAccordance(Act(P,C), ProperRelationship(P,C))",
                    "ethical_constraints": ["Context(C)", "Appropriateness(Act,C)"]
                },
                "key_concepts": ["proper relationship", "ritual propriety", "contextual appropriateness", "natural order"]
            },
            "ahimsa": {
                "description": "Non-violence and compassion towards all beings.",
                "tradition": "Indian (especially Jain, Buddhist, Hindu)",
                "indicators": {
                    "positive": ["nonviolence", "compassion", "gentle", "peaceful", "respect", "protection", "reverence", "harmlessness"],
                    "negative": ["harm", "violence", "injury", "cruelty", "destruction", "exploitation", "disregard"]
                },
                "assessment_criteria": {
                    "non-harming": "Avoids causing harm to sentient beings",
                    "compassionate_action": "Acts from genuine concern for others' welfare",
                    "reverence_for_life": "Respects the intrinsic value of living beings",
                    "peaceful_means": "Uses non-violent methods to achieve goals",
                    "gentleness": "Approaches others with gentleness and care"
                },
                "logical_formalization": {
                    "core_principle": "Minimize harm to all sentient beings in thought, word, and deed",
                    "logical_form": "∀A,B: Act(A,B) → Minimize(Harm(A,B))",
                    "ethical_constraints": ["SentientBeing(B)", "Necessity(Act(A,B))"]
                },
                "key_concepts": ["non-harming", "universal compassion", "reverence for life", "peaceful resolution"]
            },
            "hozho": {
                "description": "Beauty, harmony, and right relationship with all things.",
                "tradition": "Navajo/Diné",
                "indicators": {
                    "positive": ["balance", "beauty", "harmony", "order", "wholeness", "wellness", "ceremony", "reciprocity"],
                    "negative": ["disorder", "imbalance", "ugliness", "pollution", "disconnection", "disrespect"]
                },
                "assessment_criteria": {
                    "walking_in_beauty": "Lives in harmony with natural and social world",
                    "ceremonial_alignment": "Maintains proper relationship through ceremony",
                    "environmental_balance": "Preserves balance with natural environment",
                    "interwoven_wellness": "Promotes physical, mental, spiritual harmony",
                    "temporal_continuity": "Connects past, present, and future appropriately"
                },
                "logical_formalization": {
                    "core_principle": "Maintain right relationship with all aspects of existence",
                    "logical_form": "∀P,E: Exist(P) ∧ Exist(E) → MaintainRightRelationship(P,E)",
                    "ethical_constraints": ["Balance(P,E)", "Beauty(Relationship(P,E))"]
                },
                "key_concepts": ["walking in beauty", "right relationship", "ceremonial order", "wholeness"]
            }
        }
        
        # Expanded virtue tensions and balance points
        self.virtue_tensions = {
            "truth_kindness": {
                "description": "Balancing honest assessment with emotional impact",
                "pole1": "truth",
                "pole2": "kindness",
                "balanced_expression": "Truthful communication delivered with empathy and care",
                "imbalance_risks": {
                    "truth_excess": "Brutal honesty that causes unnecessary harm",
                    "kindness_excess": "Compassionate deception that prevents growth"
                },
                "logical_formalization": "∀S,P: Communicate(S,P) → (Truth(S) ∧ Kindness(S,P))",
                "contextual_variations": {
                    "sensitive_topics": "Kindness may need greater weight",
                    "educational": "Truth may need greater weight, but with kind delivery",
                    "emergency": "Truth may need greater weight and directness",
                    "therapeutic": "Balance shifts based on readiness and relationship"
                },
                "resolution_strategies": [
                    "Consider the purpose of communication (help vs. harm)",
                    "Assess readiness of recipient for difficult truth",
                    "Find gentlest accurate framing without distortion",
                    "Provide support when delivering difficult truths"
                ]
            },
            "courage_wisdom": {
                "description": "Balancing boldness with prudent judgment",
                "pole1": "courage",
                "pole2": "wisdom",
                "balanced_expression": "Thoughtful risk-taking informed by contextual understanding",
                "imbalance_risks": {
                    "courage_excess": "Reckless action without sufficient consideration",
                    "wisdom_excess": "Excessive caution that prevents necessary action"
                },
                "logical_formalization": "∀A: ImportantAction(A) → (Courage(A) ∧ Wisdom(A))",
                "contextual_variations": {
                    "crisis": "Courage may need greater weight",
                    "high_stakes": "Wisdom may need greater weight",
                    "innovation": "Courage may need greater weight, but with wise constraints",
                    "leadership": "Balance shifts based on team needs and situation"
                },
                "resolution_strategies": [
                    "Assess relative risks of action versus inaction",
                    "Consider reversibility of potential consequences",
                    "Start with small courageous steps when uncertainty is high",
                    "Seek wise counsel while maintaining decision ownership"
                ]
            },
            "creativity_truth": {
                "description": "Balancing novel expression with accuracy",
                "pole1": "creativity",
                "pole2": "truth",
                "balanced_expression": "Imaginative framing that illuminates rather than distorts truth",
                "imbalance_risks": {
                    "creativity_excess": "Fanciful expressions disconnected from reality",
                    "truth_excess": "Dry factuality that fails to engage or illuminate"
                },
                "logical_formalization": "∀E: Expression(E) → (Creative(E) ∧ ¬Distorts(E,Truth))",
                "contextual_variations": {
                    "scientific": "Truth may need greater weight",
                    "artistic": "Creativity may need greater weight",
                    "pedagogical": "Creative framing that enhances understanding of truth",
                    "philosophical": "Creative thought experiments within logical constraints"
                },
                "resolution_strategies": [
                    "Use creativity to illuminate rather than replace truth",
                    "Signal clearly when shifting between literal and metaphorical",
                    "Test creative expressions against factual accuracy",
                    "Ask whether creative framing enhances or obscures understanding"
                ]
            },
            "justice_grace": {
                "description": "Balancing principled stands with nuanced understanding",
                "pole1": "justice",
                "pole2": "grace",
                "balanced_expression": "Firm commitment to fairness that acknowledges complexity",
                "imbalance_risks": {
                    "justice_excess": "Rigid application of principles without context",
                    "grace_excess": "Tolerance of injustice in the name of complexity"
                },
                "logical_formalization": "∀S: Situation(S) → (ApplyJustice(S) ∧ RecognizeComplexity(S))",
                "contextual_variations": {
                    "structural_oppression": "Justice may need greater weight",
                    "interpersonal": "Grace may need greater weight",
                    "institutional": "Clear justice principles with contextual application",
                    "reconciliation": "Balance of accountability with healing possibilities"
                },
                "resolution_strategies": [
                    "Distinguish between excusing and understanding harmful actions",
                    "Consider both procedural and substantive justice",
                    "Assess power differentials in the specific context",
                    "Balance restoration with transformation of unjust patterns"
                ]
            },
            "autonomy_wisdom": {
                "description": "Balancing respect for choice with guidance",
                "pole1": "autonomy",
                "pole2": "wisdom",
                "balanced_expression": "Offering perspective while respecting agency",
                "imbalance_risks": {
                    "autonomy_excess": "Abandonment under guise of respecting choice",
                    "wisdom_excess": "Paternalistic direction that undermines agency"
                },
                "logical_formalization": "∀P,D: Person(P) ∧ Decision(D,P) → (RespectAutonomy(P,D) ∧ OfferWisdom(P,D))",
                "contextual_variations": {
                    "developing_capacity": "Wisdom may need greater weight",
                    "personal_values": "Autonomy may need greater weight",
                    "harm_prevention": "Proportional intervention based on harm potential",
                    "mentorship": "Guidance that enhances rather than replaces autonomy"
                },
                "resolution_strategies": [
                    "Distinguish between informing and directing",
                    "Consider developmental context and decision-making capacity",
                    "Offer reasons rather than commands",
                    "Support development of autonomous wisdom over time"
                ]
            },
            "ubuntu_autonomy": {
                "description": "Balancing communal belonging with individual freedom",
                "pole1": "ubuntu",
                "pole2": "autonomy",
                "balanced_expression": "Self-determination exercised within community context",
                "imbalance_risks": {
                    "ubuntu_excess": "Conformity that suppresses individual flourishing",
                    "autonomy_excess": "Disconnected individualism that undermines community"
                },
                "logical_formalization": "∀P,C: Person(P) ∧ Community(C) → (BelongTo(P,C) ∧ SelfDetermine(P))",
                "contextual_variations": {
                    "collective_welfare": "Ubuntu may need greater weight",
                    "personal_development": "Autonomy may need greater weight",
                    "cultural_expression": "Balance based on cultural context and values",
                    "intergenerational": "Balance tradition with adaptation and innovation"
                },
                "resolution_strategies": [
                    "Distinguish between harmful and beneficial community expectations",
                    "Consider impact of individual choices on community wellbeing",
                    "Develop individual capabilities that contribute to collective good",
                    "Create spaces for both communal belonging and individual expression"
                ]
            },
            "ahimsa_justice": {
                "description": "Balancing non-violence with confrontation of harm",
                "pole1": "ahimsa",
                "pole2": "justice",
                "balanced_expression": "Non-violent resistance to injustice",
                "imbalance_risks": {
                    "ahimsa_excess": "Passivity in the face of systemic harm",
                    "justice_excess": "Righteous aggression that creates new harm"
                },
                "logical_formalization": "∀S: Injustice(S) → (Confront(S) ∧ ¬Violent(Confront(S)))",
                "contextual_variations": {
                    "direct_violence": "Immediate protection may require force",
                    "structural_violence": "Strategic non-violent resistance",
                    "reconciliation": "Truth-telling with healing intention",
                    "peacebuilding": "Justice mechanisms that restore rather than punish"
                },
                "resolution_strategies": [
                    "Distinguish between violence against persons and forceful resistance",
                    "Consider spectrum of non-violent approaches to injustice",
                    "Balance accountability with possibilities for transformation",
                    "Address root causes while responding to immediate harm"
                ]
            }
        }
        
        # Domain-specific virtue priorities
        self.domain_priorities = {
            "scientific": ["truth", "courage", "creativity"],
            "philosophical": ["truth", "wisdom", "grace"],
            "ethical": ["justice", "kindness", "wisdom"],
            "artistic": ["creativity", "truth", "grace"],
            "interpersonal": ["kindness", "wisdom", "autonomy"],
            "educational": ["truth", "kindness", "autonomy"],
            "political": ["justice", "wisdom", "courage"],
            "environmental": ["wisdom", "justice", "harmony"],
            "healthcare": ["kindness", "wisdom", "autonomy"],
            "business": ["justice", "truth", "wisdom"],
            "leadership": ["wisdom", "courage", "ubuntu"],
            "conflict_resolution": ["kindness", "justice", "ahimsa"],
            "spiritual": ["wisdom", "harmony", "grace"],
            "technological": ["truth", "wisdom", "justice"]
        }
        
        # Virtue development stages
        self.virtue_development = {
            "novice": {
                "description": "Binary application of virtues without context sensitivity",
                "characteristics": ["rigid", "rule-based", "simplistic", "black-and-white"],
                "learning_focus": "Recognizing virtue dimensions in different contexts",
                "common_challenges": ["Overgeneralization", "Inconsistency", "Rigidity"],
                "developmental_needs": ["Exposure to varied applications", "Conceptual clarity", "Basic practice"]
            },
            "intermediate": {
                "description": "Recognition of context but inconsistent application",
                "characteristics": ["variable", "contextual", "developing", "improving"],
                "learning_focus": "Consistent application across varied contexts",
                "common_challenges": ["Difficulty with novel contexts", "Integration of virtues", "Balancing tensions"],
                "developmental_needs": ["Diverse practice", "Feedback on application", "Conceptual refinement"]
            },
            "advanced": {
                "description": "Balanced application with sensitivity to context",
                "characteristics": ["nuanced", "balanced", "consistent", "integrated"],
                "learning_focus": "Integration of virtues and handling of complex tensions",
                "common_challenges": ["Subtlety in application", "Communication of nuance", "Avoiding perfectionism"],
                "developmental_needs": ["Practice with complex cases", "Articulation of rationale", "Community of practice"]
            },
            "masterful": {
                "description": "Effortless integration of virtues appropriate to each situation",
                "characteristics": ["harmonious", "wise", "exemplary", "inspiring"],
                "learning_focus": "Helping others develop virtue understanding and practice",
                "common_challenges": ["Communicating tacit knowledge", "Avoiding complacency", "Contextual teaching"],
                "developmental_needs": ["Teaching opportunities", "Continued reflection", "Community leadership"]
            }
        }
        
        # Meta-ethics frameworks
        self.meta_ethics = {
            "virtue_teleology": {
                "description": "Virtues as traits leading to human flourishing",
                "key_concepts": ["eudaimonia", "excellence", "function", "character", "habit", "flourishing"],
                "theoretical_foundations": ["Aristotelian ethics", "Neo-Aristotelian virtue ethics", "Teleological ethics"],
                "methodological_approaches": [
                    "Character assessment",
                    "Analysis of excellences in function",
                    "Exemplar identification"
                ],
                "critiques": [
                    "Cultural variability of virtues",
                    "Potential conservatism",
                    "Situationist challenges to character"
                ],
                "virtues_as": "Character traits conducive to flourishing",
                "formalization": "∀V,P: Virtue(V,P) ↔ ContributesToFlourishing(V,P)"
            },
            "care_ethics": {
                "description": "Ethics centered on relationships, care, and interdependence",
                "key_concepts": ["relationship", "care", "interdependence", "particularity", "context", "emotion"],
                "theoretical_foundations": ["Feminist ethics", "Ethics of care", "Relational ethics"],
                "methodological_approaches": [
                    "Context-sensitive assessment",
                    "Focus on relationship quality",
                    "Attentiveness to particular needs"
                ],
                "critiques": [
                    "Potential gender essentialism",
                    "Challenges in public/political contexts",
                    "Balance with justice considerations"
                ],
                "virtues_as": "Dispositions supporting caring relationships",
                "formalization": "∀V,P: Virtue(V,P) ↔ SupportsCaringRelationships(V,P)"
            },
            "pluralistic_virtue": {
                "description": "Multiple legitimate virtue traditions with contextual application",
                "key_concepts": ["pluralism", "tradition", "context", "cultural wisdom", "particularity", "integration"],
                "theoretical_foundations": ["Moral pluralism", "Virtue theory", "Cross-cultural ethics"],
                "methodological_approaches": [
                    "Cross-tradition dialogue",
                    "Contextual assessment",
                    "Integration of diverse perspectives"
                ],
                "critiques": [
                    "Challenges of incommensurability",
                    "Relativism concerns",
                    "Practical decision-making difficulties"
                ],
                "virtues_as": "Culturally situated excellences with diverse expressions",
                "formalization": "∀V,C,P: Virtue(V,P,C) ↔ Excellence(V,P,C) ∧ Tradition(C)"
            },
            "virtue_consequentialism": {
                "description": "Virtues as traits that tend to promote good consequences",
                "key_concepts": ["consequences", "patterns", "dispositions", "outcomes", "well-being", "promotion"],
                "theoretical_foundations": ["Consequentialism", "Virtue consequentialism", "Indirect utilitarianism"],
                "methodological_approaches": [
                    "Outcome assessment",
                    "Pattern recognition",
                    "Character-consequence connection"
                ],
                "critiques": [
                    "Measurement challenges",
                    "Integration with character focus",
                    "Balance of intention and outcome"
                ],
                "virtues_as": "Dispositions that typically produce good consequences",
                "formalization": "∀V,P: Virtue(V,P) ↔ ProducesGoodConsequences(V,P)"
            }
        }
        
        # Learning experiences for virtue development
        self.virtue_experiences = {}  # Will store learning experiences for virtues
        
        # Connected systems
        self.judgment = judgment
        self.memory = memory
        self.logic_kernel = logic_kernel
        self.reasoning = reasoning
        
        # Evaluation history
        self.evaluation_history = []
        
        # Initialize virtue experience system
        self._initialize_virtue_experience_system()
        
    def _initialize_virtue_experience_system(self):
        """Initialize the system for tracking virtue learning experiences."""
        for virtue_name in self.virtues.keys():
            self.virtue_experiences[virtue_name] = {
                "practice_instances": [],  # Stored practice experiences
                "feedback_received": [],   # Feedback on virtue application
                "development_level": "novice",  # Starting level
                "learning_curve": {        # Learning progress data
                    "novice": 0,
                    "intermediate": 0,
                    "advanced": 0,
                    "masterful": 0
                },
                "domain_proficiency": {},  # Domain-specific proficiency
                "reflection_insights": []  # Recorded reflections
            }
        
    def evaluate(self, idea: str, context: Optional[str] = None, 
               domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates an idea against virtue criteria, considering context.
        
        Args:
            idea: The idea to evaluate
            context: Optional context for the evaluation
            domain: Optional domain for specialized evaluation
            
        Returns:
            Detailed evaluation dict
        """
        # Calculate scores for each virtue
        scores = {}
        virtue_analyses = {}
        
        # Determine relevant virtues based on domain
        relevant_virtues = list(self.virtues.keys())
        if domain and domain in self.domain_priorities:
            relevant_virtues = self.domain_priorities[domain]
            
        # Add relevant cultural virtues based on context
        cultural_virtues_to_include = self._identify_relevant_cultural_virtues(idea, context, domain)
        for cultural_virtue in cultural_virtues_to_include:
            if cultural_virtue not in relevant_virtues:
                relevant_virtues.append(cultural_virtue)
                
        # Score each relevant virtue
        for virtue in relevant_virtues:
            if virtue in self.virtues:
                score, analysis = self._score_virtue(idea, virtue, context, domain)
                scores[virtue] = score
                virtue_analyses[virtue] = analysis
            elif virtue in self.cultural_virtues:
                score, analysis = self._score_cultural_virtue(idea, virtue, context, domain)
                scores[virtue] = score
                virtue_analyses[virtue] = analysis
            
        # Get logical assessment if logic kernel is available
        logical_assessment = None
        logical_consistency = True
        if self.logic_kernel:
            logical_assessment = self._check_logical_consistency(idea, scores, virtue_analyses)
            logical_consistency = logical_assessment.get("consistent", True)
            
        # Identify virtue tensions
        tensions = self._identify_tensions(scores)
        
        # Calculate overall virtue balance
        balance_score = self._calculate_balance(scores, tensions)
        
        # Generate a virtue-based response
        response = self._generate_virtue_response(idea, scores, virtue_analyses, tensions, 
                                              balance_score, logical_assessment, domain)
        
        # Identify top virtues
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        
        # Create detailed evaluation
        evaluation = {
            "idea": idea,
            "context": context,
            "domain": domain,
            "virtue_scores": scores,
            "sorted_virtues": sorted_scores,
            "virtue_analyses": virtue_analyses,
            "tensions": tensions,
            "balance_score": balance_score,
            "logical_assessment": logical_assessment,
            "cultural_virtues_included": cultural_virtues_to_include,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Apply for potential learning
        self._apply_for_learning(evaluation)
        
        # Record evaluation in history
        self.evaluation_history.append(evaluation)
        
        # Limit history size
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]
            
        # Return full evaluation
        return evaluation
        
    def score(self, idea: str, virtue: str, context: Optional[str] = None) -> float:
        """
        Score an idea for a specific virtue.
        
        Args:
            idea: The idea to evaluate
            virtue: The virtue to evaluate against
            context: Optional context for the evaluation
            
        Returns:
            Virtue score
        """
        # For backward compatibility
        if virtue in self.virtues:
            score, _ = self._score_virtue(idea, virtue, context)
            return score
        elif virtue in self.cultural_virtues:
            score, _ = self._score_cultural_virtue(idea, virtue, context)
            return score
        return 0.5  # Default for unknown virtue
        
    def _score_virtue(self, idea: str, virtue: str, context: Optional[str] = None, 
                    domain: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Score an idea for a specific virtue with detailed analysis.
        
        Args:
            idea: The idea to evaluate
            virtue: The virtue to evaluate against
            context: Optional context for the evaluation
            domain: Optional domain for context
            
        Returns:
            Tuple of (score, analysis dict)
        """
        idea_lower = idea.lower()
        
        # Get virtue definition
        virtue_def = self.virtues.get(virtue)
        if not virtue_def:
            return 0.5, {"reason": f"Unknown virtue: {virtue}"}
            
        # Check for virtue indicators
        positive_indicators = virtue_def["indicators"]["positive"]
        negative_indicators = virtue_def["indicators"]["negative"]
        
        # Count indicator mentions
        positive_count = sum(1 for indicator in positive_indicators if indicator in idea_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in idea_lower)
        
        # Basic score based on indicators
        indicator_score = 0.5
        if positive_count > 0 or negative_count > 0:
            total_mentions = positive_count + negative_count
            indicator_score = min(1.0, max(0.0, 0.5 + (positive_count - negative_count) / (2 * total_mentions)))
            
        # Check assessment criteria
        criteria_scores = {}
        total_criteria_score = 0.0
        criteria_count = 0
        
        for criterion, description in virtue_def["assessment_criteria"].items():
            # Simple criterion checking (could be much more sophisticated)
            score = 0.5  # Default neutral score
            
            # Check for criterion keywords in the idea
            criterion_words = criterion.replace("_", " ").split()
            matches = sum(1 for word in criterion_words if word in idea_lower)
            
            if matches > 0:
                score = min(1.0, 0.5 + (matches / len(criterion_words)) * 0.5)
                
            # Add to criteria scores
            criteria_scores[criterion] = score
            total_criteria_score += score
            criteria_count += 1
            
        # Average criteria score
        avg_criteria_score = total_criteria_score / max(criteria_count, 1)
        
        # Consider domain-specific context if available
        context_score = 0.5
        if domain and "contextual_expressions" in virtue_def:
            if domain in virtue_def["contextual_expressions"]:
                # Check for domain-specific expression keywords
                domain_expression = virtue_def["contextual_expressions"][domain]
                domain_keywords = domain_expression.lower().split()
                
                domain_matches = sum(1 for keyword in domain_keywords if keyword in idea_lower)
                if domain_matches > 0:
                    context_score = min(1.0, 0.5 + (domain_matches / len(domain_keywords)) * 0.5)
                    
        # Use context to adjust score if provided
        if context:
            context_lower = context.lower()
            
            # Check for contextual alignment
            alignment_score = 0.5
            
            # Use memory system if available to find similar contexts
            if self.memory:
                try:
                    similar_contexts = self.memory.search(context, limit=5)
                    if similar_contexts:
                        # Check for virtue evaluations in similar contexts
                        similar_scores = []
                        for _, context_data in similar_contexts.items():
                            if isinstance(context_data, dict) and "result" in context_data:
                                result = context_data["result"]
                                if isinstance(result, dict) and "virtue_scores" in result and virtue in result["virtue_scores"]:
                                    similar_scores.append(result["virtue_scores"][virtue])
                                    
                        if similar_scores:
                            alignment_score = sum(similar_scores) / len(similar_scores)
                except Exception:
                    pass
                    
            # Combine context score with alignment score
            context_score = (context_score + alignment_score) / 2
            
        # Check logical coherence if Logic Kernel is available
        logical_score = 0.5
        if self.logic_kernel and "logical_formalization" in virtue_def:
            try:
                # Extract core principle
                principle = virtue_def["logical_formalization"].get("core_principle", "")
                
                # Check if idea aligns with formal principle
                if principle:
                    query_result = self.logic_kernel.query(principle)
                    if query_result.get("found", False):
                        logical_score = 0.7  # Higher baseline for logical coherence
                        
                    # Try to infer based on the idea
                    inference_result = self.logic_kernel.infer(idea)
                    if inference_result.get("result", False):
                        logical_score = 0.8  # Successfully inferred
                        
                        # Check if inference aligns with virtue principle
                        inference_content = str(inference_result.get("proposition", ""))
                        principle_words = principle.lower().split()
                        principle_matches = sum(1 for word in principle_words if word in inference_content.lower())
                        if principle_matches > len(principle_words) / 2:
                            logical_score = 0.9  # Strong alignment
            except Exception:
                pass  # Continue with default score if logic check fails
                
        # Use judgment system if available
        judgment_score = 0.5
        if self.judgment:
            try:
                # Try to use judgment system to evaluate
                judgment_result = self.judgment.evaluate(idea)
                
                # Extract virtue-relevant aspects from judgment
                if isinstance(judgment_result, dict):
                    if virtue == "truth" and "logical_consistency" in judgment_result:
                        judgment_score = judgment_result["logical_consistency"]["score"]
                    elif virtue == "wisdom" and "emergence_potential" in judgment_result:
                        judgment_score = judgment_result["emergence_potential"]["score"]
                    elif virtue == "grace" and "complexity" in judgment_result:
                        complexity = judgment_result.get("complexity", {})
                        if isinstance(complexity, dict) and "score" in complexity:
                            judgment_score = complexity["score"]
            except Exception:
                pass
                
        # Use reasoning system if available
        reasoning_score = 0.5
        if self.reasoning:
            try:
                # Use reasoning to evaluate virtue alignment
                reasoning_prompt = f"Evaluate how well the following idea embodies the virtue of {virtue} ('{virtue_def['description']}'): '{idea}'"
                reasoning_result = self.reasoning.reason(reasoning_prompt, "analytical")
                
                # Extract a score from reasoning result (simplified)
                if isinstance(reasoning_result, str):
                    # Look for indicators of strong alignment
                    if any(term in reasoning_result.lower() for term in 
                         ["strongly aligns", "exemplifies", "excellent example", "embodies"]):
                        reasoning_score = 0.8
                    elif any(term in reasoning_result.lower() for term in 
                           ["aligns", "demonstrates", "shows", "exhibits"]):
                        reasoning_score = 0.7
                    elif any(term in reasoning_result.lower() for term in 
                           ["partially aligns", "somewhat", "to some extent"]):
                        reasoning_score = 0.6
                    elif any(term in reasoning_result.lower() for term in 
                           ["does not align", "contrary", "opposes", "conflicts"]):
                        reasoning_score = 0.3
            except Exception:
                pass
                
        # Cultural variation assessment
        cultural_variation_score = 0.5
        if "cultural_variations" in virtue_def:
            # Default to global assessment
            cultural_scores = []
            
            # Check for culture indicators in context or domain
            cultures_to_check = []
            
            if context:
                context_lower = context.lower()
                if "western" in context_lower or "european" in context_lower or "american" in context_lower:
                    cultures_to_check.append("western")
                if "eastern" in context_lower or "asian" in context_lower:
                    cultures_to_check.append("eastern")
                if "indigenous" in context_lower or "native" in context_lower or "aboriginal" in context_lower:
                    cultures_to_check.append("indigenous")
                if "global south" in context_lower or "developing" in context_lower:
                    cultures_to_check.append("global_south")
                    
            # Use all variations if no specific culture indicated
            if not cultures_to_check:
                cultures_to_check = ["western", "eastern", "indigenous", "global_south"]
                
            # Check each relevant cultural variation
            for culture in cultures_to_check:
                if culture in virtue_def["cultural_variations"]:
                    cultural_expression = virtue_def["cultural_variations"][culture]
                    cultural_keywords = cultural_expression.lower().split()
                    
                    cultural_matches = sum(1 for keyword in cultural_keywords if keyword in idea_lower)
                    if cultural_matches > 0:
                        cultural_score = min(1.0, 0.5 + (cultural_matches / len(cultural_keywords)) * 0.5)
                        cultural_scores.append(cultural_score)
                        
            # Average cultural scores if any found
            if cultural_scores:
                cultural_variation_score = sum(cultural_scores) / len(cultural_scores)
                
        # Final score - weighted combination of all factors
        final_score = (
            indicator_score * 0.2 + 
            avg_criteria_score * 0.2 + 
            context_score * 0.15 + 
            judgment_score * 0.1 +
            logical_score * 0.15 + 
            reasoning_score * 0.1 +
            cultural_variation_score * 0.1
        )
        
        # Determine development stage for this virtue application
        stage = "novice"
        if final_score > 0.85:
            stage = "masterful"
        elif final_score > 0.7:
            stage = "advanced"
        elif final_score > 0.5:
            stage = "intermediate"
            
        # Prepare analysis
        analysis = {
            "score": round(final_score, 2),
            "indicator_score": round(indicator_score, 2),
            "criteria_scores": {k: round(v, 2) for k, v in criteria_scores.items()},
            "context_score": round(context_score, 2),
            "judgment_score": round(judgment_score, 2),
            "logical_score": round(logical_score, 2),
            "reasoning_score": round(reasoning_score, 2),
            "cultural_variation_score": round(cultural_variation_score, 2),
            "development_stage": stage,
            "positive_indicators": [ind for ind in positive_indicators if ind in idea_lower],
            "negative_indicators": [ind for ind in negative_indicators if ind in idea_lower],
            "cultural_variations_assessed": cultures_to_check if "cultural_variations" in virtue_def else []
        }
        
        return final_score, analysis
        
    def _score_cultural_virtue(self, idea: str, virtue: str, context: Optional[str] = None,
                             domain: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Score an idea against a cultural virtue.
        
        Args:
            idea: The idea to evaluate
            virtue: The cultural virtue to evaluate against
            context: Optional context for the evaluation
            domain: Optional domain for context
            
        Returns:
            Tuple of (score, analysis dict)
        """
        idea_lower = idea.lower()
        
        # Get cultural virtue definition
        virtue_def = self.cultural_virtues.get(virtue)
        if not virtue_def:
            return 0.5, {"reason": f"Unknown cultural virtue: {virtue}"}
            
        # Check for virtue indicators
        positive_indicators = virtue_def["indicators"]["positive"]
        negative_indicators = virtue_def["indicators"]["negative"]
        
        # Count indicator mentions
        positive_count = sum(1 for indicator in positive_indicators if indicator in idea_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in idea_lower)
        
        # Basic score based on indicators
        indicator_score = 0.5
        if positive_count > 0 or negative_count > 0:
            total_mentions = positive_count + negative_count
            indicator_score = min(1.0, max(0.0, 0.5 + (positive_count - negative_count) / (2 * total_mentions)))
            
        # Check assessment criteria
        criteria_scores = {}
        total_criteria_score = 0.0
        criteria_count = 0
        
        for criterion, description in virtue_def["assessment_criteria"].items():
            # Simple criterion checking (could be much more sophisticated)
            score = 0.5  # Default neutral score
            
            # Check for criterion keywords in the idea
            criterion_words = criterion.replace("_", " ").split()
            matches = sum(1 for word in criterion_words if word in idea_lower)
            
            if matches > 0:
                score = min(1.0, 0.5 + (matches / len(criterion_words)) * 0.5)
                
            # Add to criteria scores
            criteria_scores[criterion] = score
            total_criteria_score += score
            criteria_count += 1
            
        # Average criteria score
        avg_criteria_score = total_criteria_score / max(criteria_count, 1)
        
        # Check for key concepts
        concept_score = 0.5
        if "key_concepts" in virtue_def:
            key_concepts = virtue_def["key_concepts"]
            concept_matches = sum(1 for concept in key_concepts if concept in idea_lower)
            if concept_matches > 0:
                concept_score = min(1.0, 0.5 + (concept_matches / len(key_concepts)) * 0.5)
                
        # Check for tradition alignment
        tradition_score = 0.5
        if "tradition" in virtue_def:
            tradition = virtue_def["tradition"].lower()
            tradition_indicators = [tradition]
            
            # Add related terms for each tradition
            if tradition == "african":
                tradition_indicators.extend(["ubuntu", "communal", "african", "community"])
            elif tradition == "east asian":
                tradition_indicators.extend(["confucian", "daoist", "buddhist", "harmony", "asian"])
            elif tradition == "indian":
                tradition_indicators.extend(["hindu", "buddhist", "jain", "dharma", "indian"])
            elif tradition == "navajo/diné":
                tradition_indicators.extend(["navajo", "diné", "indigenous", "native american"])
                
            # Check for tradition indicators
            tradition_matches = sum(1 for indicator in tradition_indicators if indicator in idea_lower)
            if tradition_matches > 0:
                tradition_score = min(1.0, 0.5 + (tradition_matches / len(tradition_indicators)) * 0.3)
                
            # Check context for tradition relevance
            if context:
                context_lower = context.lower()
                context_matches = sum(1 for indicator in tradition_indicators if indicator in context_lower)
                if context_matches > 0:
                    tradition_score = max(tradition_score, 
                                        min(0.9, 0.6 + (context_matches / len(tradition_indicators)) * 0.3))
                    
        # Check logical coherence if Logic Kernel is available
        logical_score = 0.5
        if self.logic_kernel and "logical_formalization" in virtue_def:
            try:
                # Extract core principle
                principle = virtue_def["logical_formalization"].get("core_principle", "")
                
                # Check if idea aligns with formal principle
                if principle:
                    # Try to infer based on the idea and principle
                    inference_result = self.logic_kernel.infer(idea)
                    if inference_result.get("result", False):
                        logical_score = 0.7  # Successfully inferred
                        
                        # Check if inference aligns with virtue principle
                        inference_content = str(inference_result.get("proposition", ""))
                        principle_words = principle.lower().split()
                        principle_matches = sum(1 for word in principle_words if word in inference_content.lower())
                        if principle_matches > len(principle_words) / 3:
                            logical_score = 0.8  # Strong alignment
            except Exception:
                pass  # Continue with default score if logic check fails
                
        # Use reasoning system if available
        reasoning_score = 0.5
        if self.reasoning:
            try:
                # Use reasoning to evaluate virtue alignment
                reasoning_prompt = f"Evaluate how well the following idea embodies the {virtue_def['tradition']} virtue of {virtue} ('{virtue_def['description']}'): '{idea}'"
                reasoning_result = self.reasoning.reason(reasoning_prompt, "analytical")
                
                # Extract a score from reasoning result (simplified)
                if isinstance(reasoning_result, str):
                    # Look for indicators of strong alignment
                    if any(term in reasoning_result.lower() for term in 
                         ["strongly aligns", "exemplifies", "excellent example", "embodies"]):
                        reasoning_score = 0.8
                    elif any(term in reasoning_result.lower() for term in 
                           ["aligns", "demonstrates", "shows", "exhibits"]):
                        reasoning_score = 0.7
                    elif any(term in reasoning_result.lower() for term in 
                           ["partially aligns", "somewhat", "to some extent"]):
                        reasoning_score = 0.6
                    elif any(term in reasoning_result.lower() for term in 
                           ["does not align", "contrary", "opposes", "conflicts"]):
                        reasoning_score = 0.3
            except Exception:
                pass
                
        # Final score - weighted combination of all factors
        final_score = (
            indicator_score * 0.3 + 
            avg_criteria_score * 0.2 + 
            concept_score * 0.2 + 
            tradition_score * 0.15 + 
            logical_score * 0.05 + 
            reasoning_score * 0.1
        )
        
        # Determine development stage for this virtue application
        stage = "novice"
        if final_score > 0.85:
            stage = "masterful"
        elif final_score > 0.7:
            stage = "advanced"
        elif final_score > 0.5:
            stage = "intermediate"
            
        # Prepare analysis
        analysis = {
            "score": round(final_score, 2),
            "indicator_score": round(indicator_score, 2),
            "criteria_scores": {k: round(v, 2) for k, v in criteria_scores.items()},
            "concept_score": round(concept_score, 2),
            "tradition_score": round(tradition_score, 2),
            "logical_score": round(logical_score, 2),
            "reasoning_score": round(reasoning_score, 2),
            "development_stage": stage,
            "positive_indicators": [ind for ind in positive_indicators if ind in idea_lower],
            "negative_indicators": [ind for ind in negative_indicators if ind in idea_lower],
            "key_concepts_found": [concept for concept in virtue_def.get("key_concepts", []) if concept in idea_lower]
        }
        
        return final_score, analysis