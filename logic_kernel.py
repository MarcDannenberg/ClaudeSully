class LogicKernel:
    """
    Comprehensive symbolic logic system for Sully.
    
    This is the main interface to Sully's logical reasoning capabilities,
    integrating knowledge representation, inference, belief revision,
    and formal logic into a cohesive system.
    
    Attributes:
        kb: Knowledge base containing logical knowledge
        inference: Inference engine for reasoning
        belief_revision: System for maintaining consistent beliefs
        formula_parser: Parser for formalizing statements
    """
    def __init__(self):
        self.kb = KnowledgeBase()
        self.inference = InferenceEngine(self.kb)
        self.formula_parser = FormulaParser()
        self.belief_revision = BeliefRevisionSystem(self.kb, self.inference)
        
        # Statistics and metadata
        self.inference_counts = {method.name: 0 for method in InferenceMethod}
        self.creation_time = datetime.now()
        self.last_operation_time = self.creation_time
        
        self.description = """
        Sully Logic Kernel v3.1.2
        
        A sophisticated symbolic reasoning system supporting:
        - Propositional and first-order predicate logic
        - Deductive, abductive, and defeasible reasoning
        - Belief revision and consistency maintenance
        - Formal verification and proof generation
        - Integration with Sully's memory and conceptual systems
        
        For questions about usage and capabilities, see documentation.
        """
    
    def assert_fact(self, statement: str, truth: Union[bool, float, None] = True) -> Dict[str, Any]:
        """
        Assert a new fact into the knowledge base.
        
        Args:
            statement: The logical statement
            truth: Truth value for the statement
        
        Returns:
            Result of the assertion
        """
        self.last_operation_time = datetime.now()
        
        # Try to formalize the statement
        proposition = Proposition(
            id=str(uuid.uuid4()),
            statement=statement,
            truth=TruthValue.from_value(truth)
        ).formalize(self.formula_parser)
        
        # Add to knowledge base with belief revision
        result = self.belief_revision.add(proposition)
        
        return result
    
    def assert_rule(self, premises: List[str], conclusion: str, 
                  name: Optional[str] = None, priority: float = 1.0) -> Dict[str, Any]:
        """
        Assert a logical rule into the knowledge base.
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
            name: Optional name for the rule
            priority: Rule priority for conflict resolution
        
        Returns:
            Result of the assertion
        """
        self.last_operation_time = datetime.now()
        
        # Add the rule to the knowledge base
        rule_id = self.kb.add_rule(premises, conclusion, name, priority)
        rule = self.kb.get_rule(rule_id)
        
        return {
            "action": "assert_rule",
            "result": "success",
            "rule_id": rule_id,
            "rule": rule
        }
    
    def infer(self, statement: str, method: str = "DEDUCTION") -> Dict[str, Any]:
        """
        Try to infer a conclusion from the knowledge base.
        
        Args:
            statement: Statement to infer
            method: Inference method to use
        
        Returns:
            Inference result
        """
        self.last_operation_time = datetime.now()
        
        # Convert method string to enum
        try:
            inference_method = InferenceMethod[method]
        except KeyError:
            inference_method = InferenceMethod.DEDUCTION
        
        # Increment inference count
        self.inference_counts[inference_method.name] += 1
        
        # Perform inference
        result = self.inference.infer(statement, inference_method)
        
        return result
    
    def get_reasoning_chain(self, statement: str) -> Dict[str, Any]:
        """
        Get the reasoning chain for a proposition.
        
        Args:
            statement: Statement to explain
        
        Returns:
            Explanation of the reasoning
        """
        self.last_operation_time = datetime.now()
        
        # Find the proposition
        proposition = self.kb.get_proposition_by_statement(statement)
        
        if not proposition:
            return {
                "result": "not_found",
                "statement": statement
            }
        
        # Get explanation
        explanation = self.inference.explain(proposition.id)
        
        return {
            "result": "success",
            "statement": statement,
            "explanation": explanation
        }
    
    def contradictions(self) -> Dict[str, Any]:
        """
        Find contradictions in the knowledge base.
        
        Returns:
            List of contradictions
        """
        self.last_operation_time = datetime.now()
        
        contradictions = self.inference.detect_contradictions()
        
        return {
            "result": "success",
            "contradictions": contradictions,
            "count": len(contradictions)
        }
    
    def verify_consistency(self) -> Dict[str, Any]:
        """
        Check the consistency of the knowledge base.
        
        Returns:
            Consistency check results
        """
        self.last_operation_time = datetime.now()
        
        consistency = self.inference.check_consistency()
        
        return {
            "result": "success",
            "consistent": consistency["consistent"],
            "contradictions": consistency["contradictions"]
        }
    
    def query(self, statement: str) -> Dict[str, Any]:
        """
        Query the knowledge base for a statement.
        
        Args:
            statement: Statement to query
        
        Returns:
            Query result
        """
        self.last_operation_time = datetime.now()
        
        result = self.kb.query(statement)
        
        return result
    
    def revise_belief(self, statement: str, truth: Union[bool, float, None] = True) -> Dict[str, Any]:
        """
        Revise beliefs to accommodate a new fact.
        
        Args:
            statement: Statement to revise
            truth: New truth value
        
        Returns:
            Revision result
        """
        self.last_operation_time = datetime.now()
        
        # Create proposition
        proposition = Proposition(
            id=str(uuid.uuid4()),
            statement=statement,
            truth=TruthValue.from_value(truth)
        ).formalize(self.formula_parser)
        
        # Perform belief revision
        result = self.belief_revision.revise(proposition)
        
        return result
    
    def retract_belief(self, statement: str) -> Dict[str, Any]:
        """
        Retract a belief from the knowledge base.
        
        Args:
            statement: Statement to retract
        
        Returns:
            Retraction result
        """
        self.last_operation_time = datetime.now()
        
        result = self.belief_revision.contract(statement)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the logic kernel.
        
        Returns:
            Statistics
        """
        self.last_operation_time = datetime.now()
        
        return {
            "proposition_count": len(self.kb.propositions),
            "rule_count": len(self.kb.rules),
            "inference_counts": self.inference_counts,
            "creation_time": self.creation_time.isoformat(),
            "last_operation_time": self.last_operation_time.isoformat(),
            "uptime_seconds": (self.last_operation_time - self.creation_time).total_seconds()
        }
    
    def generate_proof(self, statement: str, method: str = "DEDUCTION") -> Dict[str, Any]:
        """
        Generate a formal proof for a statement.
        
        Args:
            statement: Statement to prove
            method: Inference method to use
        
        Returns:
            Proof result
        """
        self.last_operation_time = datetime.now()
        
        # First, try to infer the statement
        infer_result = self.infer(statement, method)
        
        if not infer_result.get("result", False):
            return {
                "result": "unprovable",
                "statement": statement,
                "explanation": infer_result.get("explanation", "Could not prove statement")
            }
        
        # Get the formal proof
        proof_result = self.get_reasoning_chain(statement)
        
        return {
            "result": "success",
            "statement": statement,
            "proof": proof_result.get("explanation", {})
        }
    
    def check_equivalence(self, statement1: str, statement2: str) -> Dict[str, Any]:
        """
        Check if two statements are logically equivalent.
        
        Args:
            statement1: First statement
            statement2: Second statement
        
        Returns:
            Equivalence check result
        """
        self.last_operation_time = datetime.now()
        
        # Create temporary KBs for bidirectional implication
        temp_kb1 = KnowledgeBase()
        temp_kb2 = KnowledgeBase()
        
        # Copy all propositions and rules
        for prop in self.kb.get_all_propositions():
            temp_kb1.add_proposition(prop)
            temp_kb2.add_proposition(prop)
        
        for rule in self.kb.get_all_rules():
            temp_kb1.add_rule(rule.premises, rule.conclusion, rule.name, rule.priority)
            temp_kb2.add_rule(rule.premises, rule.conclusion, rule.name, rule.priority)
        
        # Add statement1 to KB1 and try to infer statement2
        prop1 = Proposition(id=str(uuid.uuid4()), statement=statement1, truth=TruthValue.TRUE)
        temp_kb1.add_proposition(prop1)
        
        inference1 = InferenceEngine(temp_kb1)
        result1 = inference1.infer(statement2)
        
        # Add statement2 to KB2 and try to infer statement1
        prop2 = Proposition(id=str(uuid.uuid4()), statement=statement2, truth=TruthValue.TRUE)
        temp_kb2.add_proposition(prop2)
        
        inference2 = InferenceEngine(temp_kb2)
        result2 = inference2.infer(statement1)
        
        # Check results
        equivalent = result1.get("result", False) and result2.get("result", False)
        
        return {
            "result": "success",
            "equivalent": equivalent,
            "statement1": statement1,
            "statement2": statement2,
            "forward_inference": result1,
            "backward_inference": result2
        }
    
    def find_paradoxes(self) -> Dict[str, Any]:
        """
        Search for logical paradoxes in the knowledge base.
        
        Paradoxes differ from contradictions in that they represent
        self-referential or circular reasoning patterns.
        
        Returns:
            List of detected paradoxes
        """
        self.last_operation_time = datetime.now()
        
        paradoxes = []
        
        # Check for self-referential statements
        for prop_id, prop in self.kb.propositions.items():
            # Skip non-derived propositions
            if not prop.derived:
                continue
            
            # Get the derivation chain
            explanation = self.inference.explain(prop_id)
            proof = explanation.get("proof", {})
            
            # Check if this proposition appears in its own proof chain
            self._check_proof_chain_for_cycles(prop_id, proof, [], paradoxes)
        
        return {
            "result": "success",
            "paradoxes": paradoxes,
            "count": len(paradoxes)
        }
    
    def _check_proof_chain_for_cycles(self, prop_id: str, proof_node: Dict[str, Any], 
                                     chain: List[str], paradoxes: List[Dict[str, Any]]) -> None:
        """Recursively check a proof chain for cycles that indicate paradoxes."""
        # Get proposition from the proof node
        prop_str = proof_node.get("proposition", "")
        
        # Check if this proposition is already in the chain
        if prop_str in chain:
            # Found a cycle
            cycle_start = chain.index(prop_str)
            cycle = chain[cycle_start:] + [prop_str]
            
            paradoxes.append({
                "type": "cyclic_reasoning",
                "cycle": cycle,
                "proposition_id": prop_id
            })
            return
        
        # Add this proposition to the chain
        new_chain = chain + [prop_str]
        
        # Recurse through premises
        for premise in proof_node.get("premises", []):
            self._check_proof_chain_for_cycles(prop_id, premise, new_chain, paradoxes)
    
    def detect_undecidable(self, statement: str) -> Dict[str, Any]:
        """
        Attempt to detect if a statement is potentially undecidable.
        
        Undecidable statements cannot be proven true or false within
        the current knowledge base and rules.
        
        Args:
            statement: Statement to check
        
        Returns:
            Analysis result
        """
        self.last_operation_time = datetime.now()
        
        # First, try to prove the statement
        proof_result = self.infer(statement)
        
        # If it can be proven, it's decidable
        if proof_result.get("result", False):
            return {
                "result": "success",
                "decidable": True,
                "statement": statement,
                "proof": proof_result
            }
        
        # Try to prove the negation
        negation = f"not ({statement})"
        negation_result = self.infer(negation)
        
        # If the negation can be proven, it's decidable
        if negation_result.get("result", False):
            return {
                "result": "success",
                "decidable": True,
                "statement": statement,
                "negation_proof": negation_result
            }
        
        # Analyze statement structure for potential undecidability
        formalized = self.formula_parser.parse_statement(statement)
        self_reference = self._check_self_reference(statement)
        
        # Check for patterns that suggest undecidability
        potentially_undecidable = self_reference or self._has_undecidable_pattern(formalized)
        
        return {
            "result": "success",
            "decidable": False,
            "potentially_undecidable": potentially_undecidable,
            "statement": statement,
            "self_referential": self_reference,
            "explanation": "Could not prove the statement or its negation"
        }
    
    def _check_self_reference(self, statement: str) -> bool:
        """Check if a statement refers to itself or contains circular references."""
        # Simple check for direct self-reference
        # A more sophisticated implementation would parse the statement structure
        return "this statement" in statement.lower() or "itself" in statement.lower()
    
    def _has_undecidable_pattern(self, formula: Formula) -> bool:
        """Check if a formula exhibits patterns associated with undecidable statements."""
        # This is a simplified placeholder for a more complex analysis
        # A full implementation would check for known undecidable patterns
        # like certain forms of quantifier alternation or recursive definitions
        
        # Check for universal quantification followed by existential
        if (formula.type == "quantified" and formula.quantifier == Quantifier.UNIVERSAL and
            formula.subformulas and formula.subformulas[0].type == "quantified" and
            formula.subformulas[0].quantifier == Quantifier.EXISTENTIAL):
            return True
        
        return False
    
    def godel_numbering(self, formula: Formula) -> int:
        """
        Generate a Gödel number for a logical formula.
        
        This is a unique integer representation of a formula,
        useful for meta-logical reasoning.
        
        Args:
            formula: The formula to encode
        
        Returns:
            Gödel number
        """
        # This is a simplified implementation of Gödel numbering
        # A complete implementation would follow formal Gödel encoding rules
        
        formula_str = str(formula)
        
        # Simple hash-based encoding
        prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        char_codes = [ord(c) for c in formula_str]
        
        godel_number = 1
        for i, code in enumerate(char_codes):
            prime = prime_numbers[i % len(prime_numbers)]
            godel_number *= prime ** code
        
        return hash(godel_number)  # Use hash to keep the number manageable
    
    def apply_law(self, law_name: str, formula: Formula) -> Formula:
        """
        Apply a logical law or transformation to a formula.
        
        Args:
            law_name: Name of the logical law to apply
            formula: Formula to transform
        
        Returns:
            Transformed formula
        """
        # Laws of propositional logic
        if law_name == "double_negation":
            if formula.type == "negation" and formula.subformulas[0].type == "negation":
                return formula.subformulas[0].subformulas[0]
        
        elif law_name == "modus_ponens":
            if formula.type == "implication":
                antecedent = formula.subformulas[0]
                consequent = formula.subformulas[1]
                
                # Check if antecedent is TRUE in knowledge base
                antecedent_str = str(antecedent)
                antecedent_prop = self.kb.get_proposition_by_statement(antecedent_str)
                
                if antecedent_prop and antecedent_prop.truth == TruthValue.TRUE:
                    return consequent
        
        elif law_name == "de_morgan_not_and":
            if formula.type == "negation" and formula.subformulas[0].type == "conjunction":
                conjuncts = formula.subformulas[0].subformulas
                negated_conjuncts = [Formula.negation(c) for c in conjuncts]
                return Formula.disjunction(negated_conjuncts)
        
        elif law_name == "de_morgan_not_or":
            if formula.type == "negation" and formula.subformulas[0].type == "disjunction":
                disjuncts = formula.subformulas[0].subformulas
                negated_disjuncts = [Formula.negation(d) for d in disjuncts]
                return Formula.conjunction(negated_disjuncts)
        
        # Return original formula if no transformation applies
        return formula
    
    def theorem_prove(self, statement: str) -> Dict[str, Any]:
        """
        Attempt to prove a statement as a theorem using formal methods.
        
        This implements a more rigorous proof process than standard inference,
        using formal logical laws and transformations.
        
        Args:
            statement: Statement to prove
        
        Returns:
            Proof result
        """
        self.last_operation_time = datetime.now()
        
        # Parse the statement into a formal representation
        try:
            formula = self.formula_parser.parse_statement(statement)
        except Exception as e:
            return {
                "result": "error",
                "error": f"Failed to parse statement: {e}",
                "statement": statement
            }
        
        # Try to prove using natural deduction
        proof_steps = []
        proven = self._natural_deduction_prove(formula, proof_steps)
        
        if proven:
            return {
                "result": "success",
                "proven": True,
                "statement": statement,
                "proof_steps": proof_steps
            }
        
        # Try alternate proof methods if natural deduction fails
        tableau_result = self._tableau_prove(formula)
        
        return {
            "result": "success",
            "proven": tableau_result["proven"],
            "statement": statement,
            "natural_deduction": {
                "proven": False,
                "steps": proof_steps
            },
            "tableau": tableau_result
        }
    
    def _natural_deduction_prove(self, formula: Formula, proof_steps: List[Dict[str, Any]]) -> bool:
        """Prove a formula using natural deduction."""
        # This is a simplified implementation of natural deduction
        # A full implementation would include all natural deduction rules
        
        # Try to find a direct proof in the knowledge base
        formula_str = str(formula)
        prop = self.kb.get_proposition_by_statement(formula_str)
        
        if prop and prop.truth == TruthValue.TRUE:
            proof_steps.append({
                "rule": "knowledge_base_lookup",
                "formula": formula_str,
                "justification": "Statement directly found in knowledge base"
            })
            return True
        
        # Try to apply modus ponens
        for rule in self.kb.get_all_rules():
            conclusion = rule.conclusion
            conclusion_str = conclusion if isinstance(conclusion, str) else conclusion.statement
            
            if conclusion_str == formula_str:
                # Found a rule with the target as conclusion
                all_premises_proven = True
                premise_proofs = []
                
                for premise in rule.premises:
                    premise_str = premise if isinstance(premise, str) else premise.statement
                    premise_formula = self.formula_parser.parse_statement(premise_str)
                    
                    premise_steps = []
                    premise_proven = self._natural_deduction_prove(premise_formula, premise_steps)
                    
                    if not premise_proven:
                        all_premises_proven = False
                        break
                    
                    premise_proofs.append({
                        "premise": premise_str,
                        "proof": premise_steps
                    })
                
                if all_premises_proven:
                    proof_steps.append({
                        "rule": "modus_ponens",
                        "premises": [p if isinstance(p, str) else p.statement for p in rule.premises],
                        "conclusion": formula_str,
                        "premise_proofs": premise_proofs,
                        "justification": f"Applied rule: {rule.name if rule.name else 'unnamed'}"
                    })
                    return True
        
        # Try logical equivalences and transformations
        tried_transformations = []
        
        # Try double negation
        if formula.type == "negation" and formula.subformulas[0].type == "negation":
            inner_formula = formula.subformulas[0].subformulas[0]
            inner_steps = []
            
            if self._natural_deduction_prove(inner_formula, inner_steps):
                proof_steps.append({
                    "rule": "double_negation",
                    "original": f"¬¬({inner_formula})",
                    "transformed": str(inner_formula),
                    "subproof": inner_steps,
                    "justification": "Double negation elimination"
                })
                return True
            
            tried_transformations.append("double_negation")
        
        # Could add more transformations here...
        
        # No successful proof found
        if tried_transformations:
            proof_steps.append({
                "rule": "failed_transformations",
                "transformations_tried": tried_transformations,
                "justification": "Tried logical transformations but could not complete proof"
            })
        else:
            proof_steps.append({
                "rule": "no_applicable_rules",
                "justification": "No applicable deduction rules found"
            })
        
        return False
    
    def _tableau_prove(self, formula: Formula) -> Dict[str, Any]:
        """Prove a formula using the tableau method."""
        # This is a placeholder for a full tableau implementation
        # The tableau method builds a tree of formula decompositions
        # and checks for contradictions in all branches
        
        return {
            "proven": False,
            "method": "tableau",
            "explanation": "Tableau proof method not fully implemented"
        }
    
    def export_knowledge(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Export the knowledge base to a specified format.
        
        Args:
            format_type: Format to export ("json", "prolog", etc.)
        
        Returns:
            Exported knowledge
        """
        self.last_operation_time = datetime.now()
        
        if format_type == "json":
            # Export to JSON format
            propositions = {}
            for prop_id, prop in self.kb.propositions.items():
                truth_val = prop.truth
                if isinstance(truth_val, TruthValue):
                    truth_val = truth_val.name
                
                propositions[prop_id] = {
                    "statement": prop.statement,
                    "truth": truth_val,
                    "confidence": prop.confidence,
                    "derived": prop.derived,
                    "framework": prop.framework.name,
                    "source": prop.source,
                    "timestamp": prop.timestamp.isoformat()
                }
            
            rules = {}
            for rule_id, rule in self.kb.rules.items():
                premises = []
                for premise in rule.premises:
                    if isinstance(premise, str):
                        premises.append(premise)
                    else:
                        premises.append(premise.statement)
                
                conclusion = rule.conclusion
                if not isinstance(conclusion, str):
                    conclusion = conclusion.statement
                
                rules[rule_id] = {
                    "name": rule.name,
                    "premises": premises,
                    "conclusion": conclusion,
                    "priority": rule.priority,
                    "framework": rule.framework.name
                }
            
            return {
                "result": "success",
                "format": "json",
                "knowledge_base": {
                    "propositions": propositions,
                    "rules": rules,
                    "export_time": datetime.now().isoformat(),
                    "statistics": self.get_statistics()
                }
            }
        
        elif format_type == "prolog":
            # Export to Prolog format
            prolog_facts = []
            prolog_rules = []
            
            # Export propositions as facts
            for prop in self.kb.get_all_propositions():
                # Only export TRUE propositions
                if prop.truth == TruthValue.TRUE:
                    # Sanitize for Prolog
                    statement = prop.statement
                    statement = statement.replace(" ", "_").lower()
                    if "(" not in statement:
                        statement += "()"
                    
                    prolog_facts.append(f"{statement}.")
            
            # Export rules
            for rule in self.kb.get_all_rules():
                # Sanitize conclusion for Prolog
                conclusion = rule.conclusion
                if isinstance(conclusion, Proposition):
                    conclusion = conclusion.statement
                conclusion = conclusion.replace(" ", "_").lower()
                if "(" not in conclusion:
                    conclusion += "()"
                
                # Sanitize premises
                premises = []
                for premise in rule.premises:
                    if isinstance(premise, Proposition):
                        premise = premise.statement
                    premise = premise.replace(" ", "_").lower()
                    if "(" not in premise:
                        premise += "()"
                    premises.append(premise)
                
                premises_str = ", ".join(premises)
                prolog_rules.append(f"{conclusion} :- {premises_str}.")
            
            # Combine facts and rules
            prolog_program = "\n".join(prolog_facts + prolog_rules)
            
            return {
                "result": "success",
                "format": "prolog",
                "prolog_program": prolog_program
            }
        
        return {
            "result": "error",
            "error": f"Unsupported export format: {format_type}"
        }
    
    def import_knowledge(self, data: Dict[str, Any], format_type: str = "json") -> Dict[str, Any]:
        """
        Import knowledge from an external source.
        
        Args:
            data: Data to import
            format_type: Format of the imported data
        
        Returns:
            Import result
        """
        self.last_operation_time = datetime.now()
        
        if format_type == "json":
            imported_props = 0
            imported_rules = 0
            
            try:
                kb_data = data.get("knowledge_base", {})
                
                # Import propositions
                for prop_id, prop_data in kb_data.get("propositions", {}).items():
                    # Convert truth value
                    truth_val = prop_data.get("truth")
                    if isinstance(truth_val, str):
                        try:
                            truth_val = TruthValue[truth_val]
                        except:
                            truth_val = TruthValue.UNKNOWN
                    
                    # Create proposition
                    prop = Proposition(
                        id=prop_id,
                        statement=prop_data.get("statement", ""),
                        truth=truth_val,
                        confidence=prop_data.get("confidence", 1.0),
                        derived=prop_data.get("derived", False),
                        framework=LogicFramework[prop_data.get("framework", "PROPOSITIONAL")],
                        source=prop_data.get("source", "imported")
                    )
                    
                    # Add to KB
                    self.kb.add_proposition(prop)
                    imported_props += 1
                
                # Import rules
                for rule_id, rule_data in kb_data.get("rules", {}).items():
                    # Create rule
                    rule = Rule(
                        id=rule_id,
                        premises=rule_data.get("premises", []),
                        conclusion=rule_data.get("conclusion", ""),
                        name=rule_data.get("name"),
                        priority=rule_data.get("priority", 1.0),
                        framework=LogicFramework[rule_data.get("framework", "PROPOSITIONAL")]
                    )
                    
                    # Formalize and add to KB
                    rule.formalize(self.kb)
                    self.kb.rules[rule_id] = rule
                    imported_rules += 1
                
                return {
                    "result": "success",
                    "format": "json",
                    "imported_propositions": imported_props,
                    "imported_rules": imported_rules
                }
            
            except Exception as e:
                return {
                    "result": "error",
                    "error": f"Failed to import knowledge: {str(e)}"
                }
        
        return {
            "result": "error",
            "error": f"Unsupported import format: {format_type}"
        }
    
    def analyze_arguments(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """
        Analyze the logical structure and validity of an argument.
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
        
        Returns:
            Analysis result
        """
        self.last_operation_time = datetime.now()
        
        # Create a temporary KB with just these premises
        temp_kb = KnowledgeBase()
        
        for premise in premises:
            temp_kb.add_proposition(
                Proposition(
                    id=str(uuid.uuid4()),
                    statement=premise,
                    truth=TruthValue.TRUE
                )
            )
        
        # Try to infer the conclusion
        temp_inference = InferenceEngine(temp_kb)
        result = temp_inference.infer(conclusion)
        
        # Determine if the argument is valid
        valid = result.get("result", False)
        
        # Identify fallacies (simplified)
        fallacies = []
        
        # Check for circular reasoning
        circular = False
        for premise in premises:
            if premise == conclusion:
                circular = True
                fallacies.append({
                    "type": "circular_reasoning",
                    "explanation": "Conclusion appears as a premise"
                })
        
        # Check for affirming the consequent
        # This is a simplified check - would need proper implication parsing
        for premise in premises:
            if "if" in premise and "then" in premise:
                parts = premise.split("then", 1)
                if len(parts) == 2:
                    consequent = parts[1].strip()
                    if consequent in premises:
                        fallacies.append({
                            "type": "affirming_consequent",
                            "explanation": f"Premises contain both 'if A then B' and 'B'"
                        })
        
        return {
            "result": "success",
            "valid": valid,
            "premises": premises,
            "conclusion": conclusion,
            "fallacies": fallacies,
            "inference_result": result
        }
    
    def simplify_formula(self, formula: Formula) -> Formula:
        """
        Simplify a logical formula using logical equivalences.
        
        Args:
            formula: Formula to simplify
        
        Returns:
            Simplified formula
        """
        # Handle negations first
        if formula.type == "negation":
            # Double negation
            if formula.subformulas[0].type == "negation":
                return self.simplify_formula(formula.subformulas[0].subformulas[0])
            
            # De Morgan's laws
            if formula.subformulas[0].type == "conjunction":
                conjuncts = formula.subformulas[0].subformulas
                negated_conjuncts = [Formula.negation(c) for c in conjuncts]
                simplified_disjuncts = [self.simplify_formula(nc) for nc in negated_conjuncts]
                return Formula.disjunction(simplified_disjuncts)
            
            if formula.subformulas[0].type == "disjunction":
                disjuncts = formula.subformulas[0].subformulas
                negated_disjuncts = [Formula.negation(d) for d in disjuncts]
                simplified_conjuncts = [self.simplify_formula(nd) for nd in negated_disjuncts]
                return Formula.conjunction(simplified_conjuncts)
            
            # Simplify negation subformula
            inner_simplified = self.simplify_formula(formula.subformulas[0])
            return Formula.negation(inner_simplified)
        
        # Handle conjunctions and disjunctions
        elif formula.type == "conjunction" or formula.type == "disjunction":
            # Recursively simplify subformulas
            simplified_subformulas = [self.simplify_formula(f) for f in formula.subformulas]
            
            # Remove duplicates
            unique_subformulas = []
            for sf in simplified_subformulas:
                if not any(str(sf) == str(existing) for existing in unique_subformulas):
                    unique_subformulas.append(sf)
            
            # Create simplified formula
            if formula.type == "conjunction":
                return Formula.conjunction(unique_subformulas)
            else:
                return Formula.disjunction(unique_subformulas)
        
        # Handle implications
        elif formula.type == "implication":
            # A → B is equivalent to ¬A ∨ B
            antecedent = formula.subformulas[0]
            consequent = formula.subformulas[1]
            
            negated_antecedent = Formula.negation(antecedent)
            simplified_negated_antecedent = self.simplify_formula(negated_antecedent)
            simplified_consequent = self.simplify_formula(consequent)
            
            return Formula.disjunction([simplified_negated_antecedent, simplified_consequent])
        
        # Handle equivalences
        elif formula.type == "equivalence":
            # A ↔ B is equivalent to (A → B) ∧ (B → A)
            left = formula.subformulas[0]
            right = formula.subformulas[1]
            
            # Create implications in both directions
            implication1 = Formula.implication(left, right)
            implication2 = Formula.implication(right, left)
            
            # Simplify each implication
            simplified_impl1 = self.simplify_formula(implication1)
            simplified_impl2 = self.simplify_formula(implication2)
            
            return Formula.conjunction([simplified_impl1, simplified_impl2])
        
        # Return the formula unchanged if no simplification applies
        return formula


# Integration with Sully system
def integrate_with_sully(sully_system):
    """
    Integrate the logic kernel with a Sully system instance.
    
    Args:
        sully_system: Sully system instance
    
    Returns:
        Configured LogicKernel instance
    """
    logic_kernel = LogicKernel()
    
    # Connect to Sully's memory systems if available
    if hasattr(sully_system, 'codex'):
        # Import knowledge from codex
        for concept in sully_system.codex.get_all_concepts():
            statement = f"{concept} is a known concept"
            logic_kernel.assert_fact(statement)
    
    # Connect to formal knowledge if available
    if hasattr(sully_system, 'knowledge'):
        for knowledge_item in sully_system.knowledge:
            # Create formal rules from knowledge statements
            if isinstance(knowledge_item, str):
                # Simple heuristic to detect rule-like statements
                if " implies " in knowledge_item or " if " in knowledge_item:
                    # Extract premise and conclusion
                    if " implies " in knowledge_item:
                        premise, conclusion = knowledge_item.split(" implies ", 1)
                        logic_kernel.assert_rule([premise.strip()], conclusion.strip())
                    elif " if " in knowledge_item:
                        parts = knowledge_item.split(" if ", 1)
                        if len(parts) == 2:
                            conclusion, premise = parts
                            logic_kernel.assert_rule([premise.strip()], conclusion.strip())
                else:
                    # Treat as a simple fact
                    logic_kernel.assert_fact(knowledge_item)
    
    # Set up callback connection to Sully's reasoning system
    if hasattr(sully_system, 'reasoning_node'):
        # Logic kernel can provide formal verification for the reasoning system
        pass
    
    return logic_kernel


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sully Logic Kernel")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    args = parser.parse_args()
    
    if args.test:
        print("Initializing Logic Kernel...")
        kernel = LogicKernel()
        
        print("\n1. Adding basic knowledge...")
        kernel.assert_fact("Socrates is a human")
        kernel.assert_fact("All humans are mortal")
        kernel.assert_rule(["X is a human"], "X is mortal", "mortality_rule")
        
        print("\n2. Testing inference...")
        result = kernel.infer("Socrates is mortal")
        print(f"Inference result: {result}")
        
        print("\n3. Testing contradiction detection...")
        kernel.assert_fact("Socrates is immortal")
        contradictions = kernel.contradictions()
        print(f"Contradictions: {contradictions}")
        
        print("\n4. Testing belief revision...")
        kernel.revise_belief("Socrates is immortal", False)
        contradictions_after = kernel.contradictions()
        print(f"Contradictions after revision: {contradictions_after}")
        
        print("\n5. Testing paradox detection...")
        kernel.assert_fact("This statement is false")
        paradoxes = kernel.find_paradoxes()
        print(f"Paradoxes: {paradoxes}")
        
        print("\nLogic Kernel test complete!")
    else:
        print("Sully Logic Kernel initialized. Use --test to run tests.")"""
logic_kernel.py 🧠🔬
------------------
Sully's Advanced Symbolic Logic System — A sophisticated engine for formal reasoning,
inference, theorem proving, and paradox detection across multiple logical frameworks.

Core capabilities:
- First-order predicate logic with quantifiers and variables
- Modal logic extensions for possibility, necessity, and temporal reasoning
- Non-monotonic reasoning with defeasible inference
- Belief revision and truth maintenance
- Contradiction resolution through assumption tracking
- Automatic proof generation with explanation capabilities
- Seamless integration with memory, codex, and reasoning subsystems

Author: Cognitive Architecture Team (2025)
Version: 3.1.2
"""

from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Any, Iterator
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque
import re
import uuid
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logic_kernel")


class LogicFramework(Enum):
    """Available logical frameworks for reasoning."""
    PROPOSITIONAL = auto()  # Simple propositional logic
    FIRST_ORDER = auto()    # First-order predicate logic
    MODAL = auto()          # Modal logic (possibility/necessity)
    TEMPORAL = auto()       # Temporal logic
    DEFEASIBLE = auto()     # Non-monotonic/defeasible logic
    FUZZY = auto()          # Fuzzy logic with degrees of truth
    PROBABILITY = auto()    # Probabilistic logic


class TruthValue(Enum):
    """Extended truth values for multi-valued logics."""
    TRUE = 1.0
    FALSE = 0.0
    UNKNOWN = None
    CONTRADICTION = "contradiction"
    PARADOXICAL = "paradoxical"
    UNDECIDABLE = "undecidable"
    
    @classmethod
    def from_value(cls, value: Union[bool, float, None, str]) -> 'TruthValue':
        """Convert Python types to TruthValue enum."""
        if value is True:
            return cls.TRUE
        elif value is False:
            return cls.FALSE
        elif value is None:
            return cls.UNKNOWN
        elif isinstance(value, float):
            if value == 1.0:
                return cls.TRUE
            elif value == 0.0:
                return cls.FALSE
            else:
                return value  # Fuzzy truth degree
        elif isinstance(value, str):
            if value.lower() in ["contradiction", "paradoxical", "undecidable"]:
                return getattr(cls, value.upper())
        return cls.UNKNOWN


class InferenceMethod(Enum):
    """Available inference techniques."""
    DEDUCTION = auto()       # Traditional logical deduction
    INDUCTION = auto()       # Inferring general rules from examples
    ABDUCTION = auto()       # Inferring explanations for observations
    ANALOGY = auto()         # Reasoning by similarity
    STATISTICAL = auto()     # Probabilistic inference


@dataclass
class LogicalTerm:
    """
    Represents a term in predicate logic (constant, variable, or function).
    
    Attributes:
        name: The identifier for this term
        term_type: "constant", "variable", or "function"
        args: For functions, the argument terms
    """
    name: str
    term_type: str = "constant"  # "constant", "variable", or "function"
    args: List['LogicalTerm'] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.term_type == "function":
            args_str = ", ".join(str(arg) for arg in self.args)
            return f"{self.name}({args_str})"
        return self.name
    
    def variables(self) -> Set[str]:
        """Return all variables in this term."""
        if self.term_type == "variable":
            return {self.name}
        elif self.term_type == "function":
            return set().union(*[arg.variables() for arg in self.args]) if self.args else set()
        return set()
    
    @staticmethod
    def parse(term_str: str) -> 'LogicalTerm':
        """Parse a string into a LogicalTerm object."""
        term_str = term_str.strip()
        
        # Check if it's a function
        if '(' in term_str and term_str.endswith(')'):
            name, args_str = term_str.split('(', 1)
            args_str = args_str[:-1]  # Remove closing parenthesis
            
            # Parse arguments recursively
            args = []
            current_arg = ""
            paren_depth = 0
            
            for char in args_str:
                if char == ',' and paren_depth == 0:
                    if current_arg.strip():
                        args.append(LogicalTerm.parse(current_arg.strip()))
                    current_arg = ""
                else:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    current_arg += char
            
            if current_arg.strip():
                args.append(LogicalTerm.parse(current_arg.strip()))
                
            return LogicalTerm(name.strip(), "function", args)
        
        # Check if it's a variable (uppercase by convention)
        if term_str and term_str[0].isupper():
            return LogicalTerm(term_str, "variable")
        
        # Otherwise, it's a constant
        return LogicalTerm(term_str, "constant")


@dataclass
class Predicate:
    """
    Represents a predicate in first-order logic.
    
    Attributes:
        name: The predicate name
        args: List of logical terms as arguments
        negated: Whether this predicate is negated
    """
    name: str
    args: List[LogicalTerm] = field(default_factory=list)
    negated: bool = False
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        predicate_str = f"{self.name}({args_str})"
        return f"¬{predicate_str}" if self.negated else predicate_str
    
    def variables(self) -> Set[str]:
        """Return all variables in this predicate."""
        return set().union(*[arg.variables() for arg in self.args]) if self.args else set()
    
    def negate(self) -> 'Predicate':
        """Return a new predicate with the negation flipped."""
        return Predicate(
            self.name,
            self.args.copy(),
            not self.negated
        )
    
    @staticmethod
    def parse(pred_str: str) -> 'Predicate':
        """Parse a string into a Predicate object."""
        pred_str = pred_str.strip()
        negated = False
        
        # Check for negation
        if pred_str.startswith('¬') or pred_str.startswith('~') or pred_str.startswith('not '):
            negated = True
            if pred_str.startswith('not '):
                pred_str = pred_str[4:].strip()
            else:
                pred_str = pred_str[1:].strip()
        
        # Extract name and arguments
        if '(' in pred_str and pred_str.endswith(')'):
            name, args_str = pred_str.split('(', 1)
            args_str = args_str[:-1]  # Remove closing parenthesis
            
            # Parse arguments
            args = []
            current_arg = ""
            paren_depth = 0
            
            for char in args_str:
                if char == ',' and paren_depth == 0:
                    if current_arg.strip():
                        args.append(LogicalTerm.parse(current_arg.strip()))
                    current_arg = ""
                else:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    current_arg += char
            
            if current_arg.strip():
                args.append(LogicalTerm.parse(current_arg.strip()))
                
            return Predicate(name.strip(), args, negated)
        
        # Predicate without arguments
        return Predicate(pred_str, [], negated)


class Quantifier(Enum):
    """Logic quantifiers."""
    UNIVERSAL = "∀"        # For all
    EXISTENTIAL = "∃"      # There exists
    UNIQUE = "∃!"          # There exists exactly one


@dataclass
class Formula:
    """
    Represents a logical formula in first-order logic.
    
    Attributes:
        type: Formula type (atomic, conjunction, disjunction, implication, etc.)
        predicates: List of predicates for atomic formulas
        subformulas: List of subformulas for compound formulas
        quantifier: Optional quantifier for quantified formulas
        variables: Variables bound by a quantifier
    """
    type: str  # atomic, conjunction, disjunction, implication, equivalence, quantified
    predicates: List[Predicate] = field(default_factory=list)
    subformulas: List['Formula'] = field(default_factory=list)
    quantifier: Optional[Quantifier] = None
    variables: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.type == "atomic":
            return str(self.predicates[0]) if self.predicates else "⊤"  # Default to True if empty
        
        elif self.type == "conjunction":
            return "(" + " ∧ ".join(str(f) for f in self.subformulas) + ")"
        
        elif self.type == "disjunction":
            return "(" + " ∨ ".join(str(f) for f in self.subformulas) + ")"
        
        elif self.type == "implication":
            return f"({self.subformulas[0]} → {self.subformulas[1]})"
        
        elif self.type == "equivalence":
            return f"({self.subformulas[0]} ↔ {self.subformulas[1]})"
        
        elif self.type == "negation":
            return f"¬({self.subformulas[0]})"
        
        elif self.type == "quantified":
            vars_str = ", ".join(self.variables)
            return f"{self.quantifier.value}{vars_str}. ({self.subformulas[0]})"
        
        return "⊥"  # Default to False for unknown types
    
    def get_free_variables(self) -> Set[str]:
        """Return all free variables in this formula."""
        if self.type == "atomic":
            return set().union(*[p.variables() for p in self.predicates]) if self.predicates else set()
        
        elif self.type in ["conjunction", "disjunction", "implication", "equivalence", "negation"]:
            return set().union(*[f.get_free_variables() for f in self.subformulas]) if self.subformulas else set()
        
        elif self.type == "quantified":
            # Remove bound variables from the free variables of the subformula
            subformula_vars = self.subformulas[0].get_free_variables()
            return subformula_vars - set(self.variables)
        
        return set()
    
    def is_ground(self) -> bool:
        """Return whether this formula has no free variables."""
        return len(self.get_free_variables()) == 0
    
    @staticmethod
    def atomic(predicate: Predicate) -> 'Formula':
        """Create an atomic formula from a predicate."""
        return Formula("atomic", [predicate])
    
    @staticmethod
    def conjunction(formulas: List['Formula']) -> 'Formula':
        """Create a conjunction formula."""
        return Formula("conjunction", subformulas=formulas)
    
    @staticmethod
    def disjunction(formulas: List['Formula']) -> 'Formula':
        """Create a disjunction formula."""
        return Formula("disjunction", subformulas=formulas)
    
    @staticmethod
    def implication(antecedent: 'Formula', consequent: 'Formula') -> 'Formula':
        """Create an implication formula."""
        return Formula("implication", subformulas=[antecedent, consequent])
    
    @staticmethod
    def equivalence(left: 'Formula', right: 'Formula') -> 'Formula':
        """Create an equivalence formula."""
        return Formula("equivalence", subformulas=[left, right])
    
    @staticmethod
    def negation(formula: 'Formula') -> 'Formula':
        """Create a negation formula."""
        return Formula("negation", subformulas=[formula])
    
    @staticmethod
    def quantified(quantifier: Quantifier, variables: List[str], formula: 'Formula') -> 'Formula':
        """Create a quantified formula."""
        return Formula("quantified", subformulas=[formula], quantifier=quantifier, variables=variables)


@dataclass
class ModalFormula(Formula):
    """
    Extends Formula with modal operators (possibility and necessity).
    
    Attributes:
        modal_operator: "□" (necessity) or "◇" (possibility)
    """
    modal_operator: Optional[str] = None  # "□" or "◇"
    
    def __str__(self) -> str:
        if self.modal_operator:
            return f"{self.modal_operator}({self.subformulas[0]})"
        return super().__str__()
    
    @staticmethod
    def necessity(formula: Formula) -> 'ModalFormula':
        """Create a necessity formula."""
        return ModalFormula(
            "modal",
            subformulas=[formula],
            modal_operator="□"
        )
    
    @staticmethod
    def possibility(formula: Formula) -> 'ModalFormula':
        """Create a possibility formula."""
        return ModalFormula(
            "modal",
            subformulas=[formula],
            modal_operator="◇"
        )


@dataclass
class TemporalFormula(Formula):
    """
    Extends Formula with temporal operators.
    
    Attributes:
        temporal_operator: One of "G" (always), "F" (eventually), "X" (next), "U" (until)
    """
    temporal_operator: Optional[str] = None  # "G", "F", "X", or "U"
    
    def __str__(self) -> str:
        if self.temporal_operator:
            if self.temporal_operator == "U":
                return f"({self.subformulas[0]} U {self.subformulas[1]})"
            return f"{self.temporal_operator}({self.subformulas[0]})"
        return super().__str__()
    
    @staticmethod
    def always(formula: Formula) -> 'TemporalFormula':
        """Create an 'always' formula."""
        return TemporalFormula(
            "temporal",
            subformulas=[formula],
            temporal_operator="G"
        )
    
    @staticmethod
    def eventually(formula: Formula) -> 'TemporalFormula':
        """Create an 'eventually' formula."""
        return TemporalFormula(
            "temporal",
            subformulas=[formula],
            temporal_operator="F"
        )
    
    @staticmethod
    def next(formula: Formula) -> 'TemporalFormula':
        """Create a 'next' formula."""
        return TemporalFormula(
            "temporal",
            subformulas=[formula],
            temporal_operator="X"
        )
    
    @staticmethod
    def until(left: Formula, right: Formula) -> 'TemporalFormula':
        """Create an 'until' formula."""
        return TemporalFormula(
            "temporal",
            subformulas=[left, right],
            temporal_operator="U"
        )


@dataclass
class Proposition:
    """
    Represents a logical statement with truth value and metadata.
    
    Attributes:
        id: Unique identifier
        statement: String representation
        formula: Formalized representation
        truth: Truth value
        confidence: Confidence in the truth value (0.0-1.0)
        derived: Whether this was inferred rather than asserted
        framework: Logical framework this belongs to
        source: Where this proposition came from
        timestamp: When this was added
        metadata: Additional information
    """
    id: str
    statement: str
    formula: Optional[Formula] = None
    truth: Union[TruthValue, float] = TruthValue.UNKNOWN
    confidence: float = 1.0
    derived: bool = False
    framework: LogicFramework = LogicFramework.PROPOSITIONAL
    source: str = "assertion"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        truth_str = "UNKNOWN"
        if isinstance(self.truth, TruthValue):
            truth_str = self.truth.name
        elif isinstance(self.truth, float):
            truth_str = f"{self.truth:.2f}"
        
        return (f"[{truth_str}, {self.confidence:.2f}] {self.statement}" +
                (" (derived)" if self.derived else ""))
    
    def formalize(self, parser) -> 'Proposition':
        """Convert a natural language statement to a formal representation."""
        if not self.formula and parser:
            try:
                self.formula = parser.parse_statement(self.statement)
            except Exception as e:
                logger.warning(f"Failed to formalize proposition: {e}")
        return self
    
    def negate(self) -> 'Proposition':
        """Return the negation of this proposition."""
        negated_stmt = f"not ({self.statement})"
        
        negated_truth = TruthValue.UNKNOWN
        if isinstance(self.truth, TruthValue):
            if self.truth == TruthValue.TRUE:
                negated_truth = TruthValue.FALSE
            elif self.truth == TruthValue.FALSE:
                negated_truth = TruthValue.TRUE
            # Other cases remain UNKNOWN
        elif isinstance(self.truth, float):
            # Fuzzy logic negation
            negated_truth = 1.0 - self.truth
        
        negated_formula = None
        if self.formula:
            negated_formula = Formula.negation(self.formula)
        
        return Proposition(
            id=str(uuid.uuid4()),
            statement=negated_stmt,
            formula=negated_formula,
            truth=negated_truth,
            confidence=self.confidence,
            derived=True,
            framework=self.framework,
            source=f"negation_of_{self.id}",
            metadata={**self.metadata, "negation_of": self.id}
        )


@dataclass
class Rule:
    """
    Represents a logical rule with premises and conclusion.
    
    Attributes:
        id: Unique identifier
        premises: List of propositions required for the rule to apply
        conclusion: Resulting proposition
        name: Optional name for this rule
        priority: For defeasible reasoning and conflict resolution
        framework: Logical framework this belongs to
        metadata: Additional information
    """
    id: str
    premises: List[Union[str, Proposition]]
    conclusion: Union[str, Proposition]
    name: Optional[str] = None
    priority: float = 1.0
    framework: LogicFramework = LogicFramework.PROPOSITIONAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        premises_str = ", ".join(str(p) for p in self.premises)
        rule_name = f"[{self.name}] " if self.name else ""
        return f"{rule_name}{premises_str} ⊢ {self.conclusion}"
    
    def formalize(self, kb) -> 'Rule':
        """Ensure all premises and conclusion are Proposition objects."""
        formalized_premises = []
        for premise in self.premises:
            if isinstance(premise, str):
                # Try to find existing proposition
                prop = kb.get_proposition_by_statement(premise)
                if not prop:
                    # Create new proposition
                    prop = Proposition(
                        id=str(uuid.uuid4()),
                        statement=premise,
                        framework=self.framework
                    )
                formalized_premises.append(prop)
            else:
                formalized_premises.append(premise)
        
        formalized_conclusion = self.conclusion
        if isinstance(self.conclusion, str):
            # Try to find existing proposition
            prop = kb.get_proposition_by_statement(self.conclusion)
            if not prop:
                # Create new proposition
                prop = Proposition(
                    id=str(uuid.uuid4()),
                    statement=self.conclusion,
                    framework=self.framework
                )
            formalized_conclusion = prop
        
        self.premises = formalized_premises
        self.conclusion = formalized_conclusion
        return self


class UnificationError(Exception):
    """Exception raised when unification fails."""
    pass


class Substitution:
    """
    Manages variable substitutions for unification and resolution.
    
    Attributes:
        mappings: Dictionary mapping variable names to terms
    """
    def __init__(self, mappings: Optional[Dict[str, LogicalTerm]] = None):
        self.mappings = mappings or {}
    
    def apply(self, term: LogicalTerm) -> LogicalTerm:
        """Apply this substitution to a term."""
        if term.term_type == "variable" and term.name in self.mappings:
            return self.mappings[term.name]
        
        if term.term_type == "function":
            return LogicalTerm(
                term.name,
                "function",
                [self.apply(arg) for arg in term.args]
            )
        
        return term
    
    def apply_to_predicate(self, predicate: Predicate) -> Predicate:
        """Apply this substitution to a predicate."""
        return Predicate(
            predicate.name,
            [self.apply(arg) for arg in predicate.args],
            predicate.negated
        )
    
    def apply_to_formula(self, formula: Formula) -> Formula:
        """Apply this substitution to a formula."""
        if formula.type == "atomic":
            return Formula(
                "atomic",
                [self.apply_to_predicate(p) for p in formula.predicates]
            )
        
        elif formula.type in ["conjunction", "disjunction", "implication", "equivalence", "negation"]:
            return Formula(
                formula.type,
                subformulas=[self.apply_to_formula(f) for f in formula.subformulas]
            )
        
        elif formula.type == "quantified":
            # Don't substitute variables bound by the quantifier
            temp_mappings = self.mappings.copy()
            for var in formula.variables:
                if var in temp_mappings:
                    del temp_mappings[var]
            
            temp_subst = Substitution(temp_mappings)
            return Formula(
                "quantified",
                subformulas=[temp_subst.apply_to_formula(formula.subformulas[0])],
                quantifier=formula.quantifier,
                variables=formula.variables
            )
        
        return formula
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose this substitution with another one."""
        result = {}
        
        # Apply other to each mapping in self
        for var, term in self.mappings.items():
            result[var] = other.apply(term)
        
        # Add mappings from other that don't override self
        for var, term in other.mappings.items():
            if var not in result:
                result[var] = term
        
        return Substitution(result)
    
    def __str__(self) -> str:
        mappings_str = ", ".join(f"{var} ↦ {term}" for var, term in self.mappings.items())
        return f"{{{mappings_str}}}"


class Unifier:
    """Static methods for unification in first-order logic."""
    
    @staticmethod
    def unify(term1: LogicalTerm, term2: LogicalTerm) -> Substitution:
        """Unify two terms and return a substitution that makes them equal."""
        # Case 1: Both are constants
        if term1.term_type == "constant" and term2.term_type == "constant":
            if term1.name == term2.name:
                return Substitution()
            raise UnificationError(f"Cannot unify constants {term1} and {term2}")
        
        # Case 2: First term is a variable
        if term1.term_type == "variable":
            return Unifier._unify_variable(term1.name, term2)
        
        # Case 3: Second term is a variable
        if term2.term_type == "variable":
            return Unifier._unify_variable(term2.name, term1)
        
        # Case 4: Both are functions
        if term1.term_type == "function" and term2.term_type == "function":
            if term1.name != term2.name or len(term1.args) != len(term2.args):
                raise UnificationError(f"Cannot unify functions {term1} and {term2}")
            
            substitution = Substitution()
            for arg1, arg2 in zip(term1.args, term2.args):
                # Apply current substitution to both arguments before unifying
                arg1_subst = substitution.apply(arg1)
                arg2_subst = substitution.apply(arg2)
                
                # Unify arguments and compose with current substitution
                arg_subst = Unifier.unify(arg1_subst, arg2_subst)
                substitution = substitution.compose(arg_subst)
            
            return substitution
        
        raise UnificationError(f"Cannot unify {term1} and {term2}")
    
    @staticmethod
    def _unify_variable(var: str, term: LogicalTerm) -> Substitution:
        """Helper method to unify a variable with a term."""
        # Check for occurs check
        if term.term_type == "variable" and term.name == var:
            return Substitution()  # var = var is a valid unification
        
        if term.variables() and var in term.variables():
            raise UnificationError(f"Occurs check failed: {var} occurs in {term}")
        
        return Substitution({var: term})
    
    @staticmethod
    def unify_predicates(pred1: Predicate, pred2: Predicate) -> Substitution:
        """Unify two predicates."""
        if pred1.name != pred2.name or len(pred1.args) != len(pred2.args) or pred1.negated != pred2.negated:
            raise UnificationError(f"Cannot unify predicates {pred1} and {pred2}")
        
        substitution = Substitution()
        for arg1, arg2 in zip(pred1.args, pred2.args):
            # Apply current substitution to both arguments before unifying
            arg1_subst = substitution.apply(arg1)
            arg2_subst = substitution.apply(arg2)
            
            # Unify arguments and compose with current substitution
            arg_subst = Unifier.unify(arg1_subst, arg2_subst)
            substitution = substitution.compose(arg_subst)
        
        return substitution


class KnowledgeBase:
    """
    Advanced knowledge base for storing and querying logical knowledge.
    
    Attributes:
        propositions: Dictionary mapping proposition IDs to Proposition objects
        rules: Dictionary mapping rule IDs to Rule objects
        statement_index: Index for quick lookup of propositions by statement
        predicate_index: Index for quick lookup of propositions by predicate name
    """
    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
        self.rules: Dict[str, Rule] = {}
        self.statement_index: Dict[str, str] = {}  # statement -> proposition_id
        self.predicate_index: Dict[str, List[str]] = defaultdict(list)  # predicate name -> proposition_ids
    
    def add_proposition(self, proposition: Union[Proposition, str], 
                      truth: Union[TruthValue, float, bool, None] = None) -> str:
        """
        Add a proposition to the knowledge base.
        
        Args:
            proposition: Proposition object or statement string
            truth: Optional truth value to assign
        
        Returns:
            ID of the added proposition
        """
        if isinstance(proposition, str):
            # Create a new Proposition object
            prop_id = str(uuid.uuid4())
            proposition = Proposition(
                id=prop_id,
                statement=proposition,
                truth=TruthValue.from_value(truth) if truth is not None else TruthValue.UNKNOWN
            )
        else:
            # Update truth value if provided
            if truth is not None:
                proposition.truth = TruthValue.from_value(truth)
        
        # Add to the knowledge base
        self.propositions[proposition.id] = proposition
        self.statement_index[proposition.statement] = proposition.id
        
        # Update predicate index if formula is available
        if proposition.formula and proposition.formula.type == "atomic":
            for predicate in proposition.formula.predicates:
                self.predicate_index[predicate.name].append(proposition.id)
        
        return proposition.id
    
    def add_rule(self, premises: List[Union[str, Proposition]], 
                conclusion: Union[str, Proposition], 
                name: Optional[str] = None,
                priority: float = 1.0) -> str:
        """
        Add a rule to the knowledge base.
        
        Args:
            premises: List of premise propositions or statements
            conclusion: Conclusion proposition or statement
            name: Optional name for the rule
            priority: Rule priority for conflict resolution
        
        Returns:
            ID of the added rule
        """
        rule_id = str(uuid.uuid4())
        rule = Rule(
            id=rule_id,
            premises=premises,
            conclusion=conclusion,
            name=name,
            priority=priority
        )
        
        # Formalize rule to ensure all components are Proposition objects
        rule.formalize(self)
        
        # Add to the knowledge base
        self.rules[rule_id] = rule
        
        return rule_id
    
    def get_proposition(self, prop_id: str) -> Optional[Proposition]:
        """Get a proposition by ID."""
        return self.propositions.get(prop_id)
    
    def get_proposition_by_statement(self, statement: str) -> Optional[Proposition]:
        """Get a proposition by its statement text."""
        prop_id = self.statement_index.get(statement)
        return self.propositions.get(prop_id) if prop_id else None
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)
    
    def get_rules_by_conclusion(self, conclusion: str) -> List[Rule]:
        """Get all rules that have the given conclusion."""
        result = []
        
        for rule in self.rules.values():
            conclusion_stmt = rule.conclusion if isinstance(rule.conclusion, str) else rule.conclusion.statement
            if conclusion_stmt == conclusion:
                result.append(rule)
        
        return result
    
    def get_propositions_by_predicate(self, predicate_name: str) -> List[Proposition]:
        """Get all propositions that use the given predicate."""
        prop_ids = self.predicate_index.get(predicate_name, [])
        return [self.propositions[pid] for pid in prop_ids if pid in self.propositions]
    
    def get_all_propositions(self) -> List[Proposition]:
        """Get all propositions in the knowledge base."""
        return list(self.propositions.values())
    
    def get_all_rules(self) -> List[Rule]:
        """Get all rules in the knowledge base."""
        return list(self.rules.values())
    
    def query(self, statement: str) -> Dict[str, Any]:
        """
        Query the knowledge base for a proposition.
        
        Args:
            statement: The statement to query
        
        Returns:
            Dictionary with query results
        """
        proposition = self.get_proposition_by_statement(statement)
        
        if proposition:
            return {
                "found": True,
                "proposition": proposition,
                "truth": proposition.truth,
                "confidence": proposition.confidence
            }
        
        return {"found": False}


class ProofNode:
    """
    Node in a proof tree for tracking inference steps.
    
    Attributes:
        proposition: The proposition at this node
        rule: Rule used to derive this node (if any)
        premises: List of premise nodes used in the derivation
        justification: Text explaining the inference step
    """
    def __init__(self, proposition: Proposition, 
                rule: Optional[Rule] = None,
                premises: List['ProofNode'] = None,
                justification: str = ""):
        self.proposition = proposition
        self.rule = rule
        self.premises = premises or []
        self.justification = justification
    
    def __str__(self) -> str:
        return str(self.proposition)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "proposition": str(self.proposition),
            "rule": str(self.rule) if self.rule else None,
            "justification": self.justification,
            "premises": [p.to_dict() for p in self.premises]
        }


class InferenceEngine:
    """
    Advanced inference engine for logical reasoning.
    
    This engine implements multiple reasoning methods, including deduction,
    abduction, and non-monotonic reasoning. It produces detailed proof trees
    and can handle complex logical formulas.
    
    Attributes:
        kb: Knowledge base containing facts and rules
        proof_trace: Dictionary mapping proposition IDs to proof nodes
        inferred_propositions: Set of proposition IDs that have been derived
        max_inference_depth: Maximum depth for inference to prevent infinite loops
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.proof_trace: Dict[str, ProofNode] = {}
        self.inferred_propositions: Set[str] = set()
        self.max_inference_depth: int = 10
        self.current_depth: int = 0
    
    def infer(self, goal: Union[str, Proposition], 
             method: InferenceMethod = InferenceMethod.DEDUCTION) -> Dict[str, Any]:
        """
        Try to prove a goal using the knowledge base.
        
        Args:
            goal: Goal proposition or statement
            method: Inference method to use
        
        Returns:
            Dictionary with inference results
        """
        # Reset state for new inference
        self.proof_trace = {}
        self.inferred_propositions = set()
        self.current_depth = 0
        
        # Convert string goal to proposition if needed
        goal_prop = goal
        if isinstance(goal, str):
            existing_prop = self.kb.get_proposition_by_statement(goal)
            if existing_prop:
                goal_prop = existing_prop
            else:
                goal_prop = Proposition(id=str(uuid.uuid4()), statement=goal)
        
        # Check if the goal is already known
        existing = self.kb.get_proposition_by_statement(goal_prop.statement)
        if existing and isinstance(existing.truth, TruthValue) and existing.truth == TruthValue.TRUE:
            self.proof_trace[existing.id] = ProofNode(
                existing,
                justification="Known fact in knowledge base"
            )
            return {
                "result": True,
                "proposition": existing,
                "proof": self.proof_trace[existing.id],
                "derived": False
            }
        
        # Select inference method
        if method == InferenceMethod.DEDUCTION:
            result = self._backward_chaining(goal_prop)
        elif method == InferenceMethod.ABDUCTION:
            result = self._abductive_inference(goal_prop)
        else:
            # Default to deduction for now
            result = self._backward_chaining(goal_prop)
        
        return result
    
    def _backward_chaining(self, goal: Proposition) -> Dict[str, Any]:
        """
        Backward chaining algorithm for logical inference.
        
        Args:
            goal: Goal proposition to prove
        
        Returns:
            Dictionary with inference results
        """
        if self.current_depth >= self.max_inference_depth:
            return {
                "result": False,
                "explanation": f"Maximum inference depth ({self.max_inference_depth}) reached"
            }
        
        self.current_depth += 1
        
        # Get rules that might derive the goal
        potential_rules = self.kb.get_rules_by_conclusion(goal.statement)
        
        for rule in potential_rules:
            # Try to prove all premises
            all_premises_proven = True
            premise_proofs = []
            
            for premise in rule.premises:
                premise_prop = premise
                if isinstance(premise, str):
                    premise_prop = self.kb.get_proposition_by_statement(premise)
                    if not premise_prop:
                        premise_prop = Proposition(id=str(uuid.uuid4()), statement=premise)
                
                # Recursively try to prove the premise
                premise_result = self._backward_chaining(premise_prop)
                
                if not premise_result.get("result", False):
                    all_premises_proven = False
                    break
                
                premise_proofs.append(premise_result.get("proof"))
            
            if all_premises_proven:
                # All premises are proven, so the conclusion is proven
                derived_prop = Proposition(
                    id=str(uuid.uuid4()),
                    statement=goal.statement,
                    truth=TruthValue.TRUE,
                    derived=True,
                    source=f"derived_from_rule_{rule.id}"
                )
                
                # Add to knowledge base and inference trace
                self.kb.add_proposition(derived_prop)
                self.inferred_propositions.add(derived_prop.id)
                
                # Create proof node
                self.proof_trace[derived_prop.id] = ProofNode(
                    derived_prop,
                    rule,
                    premise_proofs,
                    justification=f"Derived using rule: {rule}"
                )
                
                self.current_depth -= 1
                return {
                    "result": True,
                    "proposition": derived_prop,
                    "proof": self.proof_trace[derived_prop.id],
                    "derived": True
                }
        
        self.current_depth -= 1
        return {"result": False, "explanation": "No applicable rules found"}
    
    def _abductive_inference(self, goal: Proposition) -> Dict[str, Any]:
        """
        Abductive inference: find hypotheses that would explain the goal.
        
        Args:
            goal: Goal proposition to explain
        
        Returns:
            Dictionary with inference results
        """
        # Find rules that could have the goal as conclusion
        potential_rules = self.kb.get_rules_by_conclusion(goal.statement)
        
        if not potential_rules:
            return {"result": False, "explanation": "No rules found with this conclusion"}
        
        # For each rule, consider its premises as hypotheses
        hypotheses = []
        
        for rule in potential_rules:
            rule_hypotheses = []
            
            for premise in rule.premises:
                premise_stmt = premise if isinstance(premise, str) else premise.statement
                existing = self.kb.get_proposition_by_statement(premise_stmt)
                
                if not existing:
                    # This premise doesn't exist yet, so it's a potential hypothesis
                    rule_hypotheses.append({
                        "statement": premise_stmt,
                        "rule": rule.id
                    })
                elif existing.truth != TruthValue.TRUE:
                    # This premise exists but isn't proven, so it's a potential hypothesis
                    rule_hypotheses.append({
                        "statement": premise_stmt,
                        "proposition": existing.id,
                        "rule": rule.id
                    })
            
            if rule_hypotheses:
                hypotheses.append({
                    "rule": rule.id,
                    "hypotheses": rule_hypotheses
                })
        
        return {
            "result": "abductive",
            "explanation": "Found potential explanations",
            "hypotheses": hypotheses
        }
    
    def explain(self, proposition_id: str) -> Dict[str, Any]:
        """
        Explain how a proposition was derived.
        
        Args:
            proposition_id: ID of the proposition to explain
        
        Returns:
            Dictionary with explanation
        """
        if proposition_id in self.proof_trace:
            return {
                "explanation": "Proof found",
                "proof": self.proof_trace[proposition_id].to_dict()
            }
        
        proposition = self.kb.get_proposition(proposition_id)
        if not proposition:
            return {"explanation": "Proposition not found"}
        
        if not proposition.derived:
            return {"explanation": "This is an asserted fact, not a derived proposition"}
        
        return {"explanation": "No proof trace available for this proposition"}
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """Identify contradictions in the knowledge base."""
        contradictions = []
        
        # Check for direct contradictions (P and not P)
        for prop_id, prop in self.kb.propositions.items():
            # Skip non-TRUE propositions
            if prop.truth != TruthValue.TRUE:
                continue
            
            # Look for negation of this proposition
            if prop.statement.startswith("not "):
                positive_stmt = prop.statement[4:].strip()
                positive = self.kb.get_proposition_by_statement(positive_stmt)
                
                if positive and positive.truth == TruthValue.TRUE:
                    contradictions.append({
                        "type": "direct",
                        "proposition1": prop.id,
                        "proposition2": positive.id,
                        "statement1": prop.statement,
                        "statement2": positive.statement
                    })
            else:
                negated_stmt = f"not {prop.statement}"
                negated = self.kb.get_proposition_by_statement(negated_stmt)
                
                if negated and negated.truth == TruthValue.TRUE:
                    contradictions.append({
                        "type": "direct",
                        "proposition1": prop.id,
                        "proposition2": negated.id,
                        "statement1": prop.statement,
                        "statement2": negated.statement
                    })
        
        # Check for more complex contradictions using formal representations
        # (This would be more complex and depends on the formal logic system)
        
        return contradictions
    
    def check_consistency(self) -> Dict[str, Any]:
        """
        Check the consistency of the knowledge base.
        
        Returns:
            Dictionary with consistency check results
        """
        contradictions = self.detect_contradictions()
        
        return {
            "consistent": len(contradictions) == 0,
            "contradictions": contradictions
        }


class FormulaParser:
    """
    Parser for converting logical statements to formal representations.
    
    This parser supports propositional and first-order logic formulas,
    including quantifiers, predicates, and complex formulas.
    """
    def __init__(self):
        pass
    
    def parse_statement(self, statement: str) -> Formula:
        """
        Parse a statement string into a Formula object.
        
        Args:
            statement: The statement to parse
        
        Returns:
            Formula representation
        """
        # This is a simplified version - a real implementation would be more complex
        # and would handle the full syntax of first-order logic
        
        # Check for quantifiers
        quantifier_match = re.match(r'(forall|exists|exists!)\s+([A-Z][a-zA-Z0-9_]*(?:,\s*[A-Z][a-zA-Z0-9_]*)*)\.\s*(.+)', statement)
        if quantifier_match:
            quantifier_str, variables_str, subformula_str = quantifier_match.groups()
            
            # Convert quantifier string to Quantifier enum
            quantifier_map = {
                "forall": Quantifier.UNIVERSAL,
                "exists": Quantifier.EXISTENTIAL,
                "exists!": Quantifier.UNIQUE
            }
            quantifier = quantifier_map.get(quantifier_str, Quantifier.UNIVERSAL)
            
            # Parse variables
            variables = [v.strip() for v in variables_str.split(',')]
            
            # Parse subformula
            subformula = self.parse_statement(subformula_str)
            
            return Formula.quantified(quantifier, variables, subformula)
        
        # Check for implications
        if " implies " in statement:
            antecedent_str, consequent_str = statement.split(" implies ", 1)
            antecedent = self.parse_statement(antecedent_str)
            consequent = self.parse_statement(consequent_str)
            
            return Formula.implication(antecedent, consequent)
        
        # Check for equivalence
        if " iff " in statement:
            left_str, right_str = statement.split(" iff ", 1)
            left = self.parse_statement(left_str)
            right = self.parse_statement(right_str)
            
            return Formula.equivalence(left, right)
        
        # Check for disjunction
        if " or " in statement:
            subformulas = []
            for subformula_str in statement.split(" or "):
                subformulas.append(self.parse_statement(subformula_str))
            
            return Formula.disjunction(subformulas)
        
        # Check for conjunction
        if " and " in statement:
            subformulas = []
            for subformula_str in statement.split(" and "):
                subformulas.append(self.parse_statement(subformula_str))
            
            return Formula.conjunction(subformulas)
        
        # Check for negation
        if statement.startswith("not "):
            subformula = self.parse_statement(statement[4:])
            return Formula.negation(subformula)
        
        # Atomic formula
        try:
            predicate = Predicate.parse(statement)
            return Formula.atomic(predicate)
        except Exception as e:
            # Handle parsing errors
            logger.warning(f"Failed to parse predicate: {e}")
            # Return a placeholder predicate for unparseable statements
            placeholder = Predicate(statement, [])
            return Formula.atomic(placeholder)


class BeliefRevisionSystem:
    """
    Manages belief revision and truth maintenance.
    
    This system implements the AGM belief revision model, handling
    additions, revisions, and contractions to the knowledge base while
    maintaining consistency.
    
    Attributes:
        kb: Knowledge base to maintain
        inference: Inference engine for consistency checking
    """
    def __init__(self, kb: KnowledgeBase, inference: InferenceEngine):
        self.kb = kb
        self.inference = inference
    
    def add(self, proposition: Union[str, Proposition]) -> Dict[str, Any]:
        """
        Add a new belief, maintaining consistency.
        
        Args:
            proposition: Proposition to add
        
        Returns:
            Result of the operation
        """
        # Convert to Proposition if needed
        if isinstance(proposition, str):
            existing = self.kb.get_proposition_by_statement(proposition)
            if existing:
                return {
                    "action": "add",
                    "result": "already_exists",
                    "proposition": existing
                }
            
            proposition = Proposition(
                id=str(uuid.uuid4()),
                statement=proposition,
                truth=TruthValue.TRUE
            )
        
        # Check for contradictions
        temp_kb = KnowledgeBase()
        for prop in self.kb.get_all_propositions():
            temp_kb.add_proposition(prop)
        temp_kb.add_proposition(proposition)
        
        temp_inference = InferenceEngine(temp_kb)
        consistency = temp_inference.check_consistency()
        
        if not consistency["consistent"]:
            # Resolve the contradiction by revising beliefs
            return self._resolve_contradiction(proposition, consistency["contradictions"])
        
        # No contradictions, add directly
        self.kb.add_proposition(proposition)
        
        return {
            "action": "add",
            "result": "success",
            "proposition": proposition
        }
    
    def revise(self, proposition: Union[str, Proposition]) -> Dict[str, Any]:
        """
        Revise beliefs to accommodate a new proposition.
        
        Args:
            proposition: Proposition to add
        
        Returns:
            Result of the operation
        """
        # First remove any contradicting beliefs
        contraction_result = self.contract(proposition.statement if isinstance(proposition, Proposition) else proposition)
        
        # Then add the new belief
        addition_result = self.add(proposition)
        
        return {
            "action": "revise",
            "contraction": contraction_result,
            "addition": addition_result
        }
    
    def contract(self, proposition: Union[str, Proposition]) -> Dict[str, Any]:
        """
        Remove a belief and its dependents from the knowledge base.
        
        Args:
            proposition: Proposition to remove
        
        Returns:
            Result of the operation
        """
        # Find the proposition
        prop_stmt = proposition if isinstance(proposition, str) else proposition.statement
        prop = self.kb.get_proposition_by_statement(prop_stmt)
        
        if not prop:
            return {
                "action": "contract",
                "result": "not_found",
                "statement": prop_stmt
            }
        
        # Find all propositions that depend on this one
        dependents = self._find_dependents(prop.id)
        
        # Remove the proposition and its dependents
        removed = []
        for prop_id in [prop.id] + dependents:
            removed.append(self.kb.get_proposition(prop_id))
            del self.kb.propositions[prop_id]
            
            # Also remove from statement index
            stmt = removed[-1].statement
            if self.kb.statement_index.get(stmt) == prop_id:
                del self.kb.statement_index[stmt]
        
        return {
            "action": "contract",
            "result": "success",
            "removed": removed
        }
    
    def _find_dependents(self, prop_id: str) -> List[str]:
        """Find all propositions that depend on the given proposition."""
        dependents = []
        
        # Find all propositions that were derived using rules
        for pid, prop in self.kb.propositions.items():
            if not prop.derived:
                continue
            
            # Check if this proposition depends on the target
            if prop_id in prop.metadata.get("derived_from", []):
                dependents.append(pid)
        
        # Recursively find dependents of dependents
        for dep_id in dependents.copy():
            sub_deps = self._find_dependents(dep_id)
            for sub_dep in sub_deps:
                if sub_dep not in dependents:
                    dependents.append(sub_dep)
        
        return dependents
    
    def _resolve_contradiction(self, new_prop: Proposition, contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve a contradiction by deciding which beliefs to keep.
        
        Current strategy: prefer newer information (the new proposition)
        over existing beliefs.
        
        Args:
            new_prop: New proposition causing the contradiction
            contradictions: List of detected contradictions
        
        Returns:
            Result of the resolution
        """
        removed = []
        
        for contradiction in contradictions:
            # Get the conflicting proposition
            conflict_id = contradiction["proposition1"]
            if conflict_id == new_prop.id:
                conflict_id = contradiction["proposition2"]
            
            conflict_prop = self.kb.get_proposition(conflict_id)
            
            # Remove the conflicting proposition and its dependents
            contraction_result = self.contract(conflict_prop)
            if contraction_result["result"] == "success":
                removed.extend(contraction_result["removed"])
        
        # Now add the new proposition
        self.kb.add_proposition(new_prop)
        
        return {
            "action": "resolve_contradiction",
            "result": "success",
            "added": new_prop,
            "removed": removed
        }