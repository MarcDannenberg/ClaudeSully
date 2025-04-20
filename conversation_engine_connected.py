class ConversationEngine:
    def __init__(self, logic_kernel, memory_system=None):
        self.logic_kernel = logic_kernel
        self.memory_system = memory_system
        self.topics = []
        self.depth = 0
        self.reasoning_mode = "narrative"

    def generate_response(self, user_input):
        if user_input.strip().endswith("?"):
            return self.handle_question(user_input)
        else:
            return self.handle_statement(user_input)

    def handle_statement(self, statement):
        try:
            self.logic_kernel.assert_fact(statement)
            return f"Okay, I've recorded: '{statement}' as a belief."
        except Exception as e:
            return f"I couldn't process that fully. Here's what I got: {e}"

    def handle_question(self, question):
        symbolic = f"→Symbolic(You: {question})"
        return self.resolve_symbolic(symbolic)

    def resolve_symbolic(self, symbolic):
        if symbolic.startswith("→Symbolic("):
            inner_text = symbolic[len("→Symbolic("):-1]
            try:
                interpretation = self.logic_kernel.interpret(inner_text)
                return interpretation
            except Exception as e:
                return f"I'm having trouble interpreting that symbolically: {e}"
        return symbolic
