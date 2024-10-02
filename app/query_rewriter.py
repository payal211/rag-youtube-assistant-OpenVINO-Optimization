import ollama

class QueryRewriter:
    def __init__(self):
        self.model = "phi"  # Using Phi-3.5 model

    def rewrite_cot(self, query):
        prompt = f"""
        Rewrite the following query using Chain-of-Thought reasoning:
        Query: {query}
        
        Rewritten query:
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response'].strip()

    def rewrite_react(self, query):
        prompt = f"""
        Rewrite the following query using the ReAct framework (Reasoning and Acting):
        Query: {query}
        
        Thought 1:
        Action 1:
        Observation 1:
        
        Thought 2:
        Action 2:
        Observation 2:
        
        Final rewritten query:
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response'].strip()