import ollama

class RAGSystem:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = "phi3.5"  # Using Phi-3.5 model

    def query(self, user_query, top_k=3, search_method='hybrid'):
        # Retrieve relevant documents using the specified search method
        relevant_docs = self.data_processor.search(user_query, num_results=top_k, method=search_method)
        
        # Construct the prompt
        context = "\n".join([doc['content'] for doc in relevant_docs])
        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
        
        # Generate response using Ollama
        response = ollama.generate(model=self.model, prompt=prompt)
        
        return response['response']

    def rerank_documents(self, documents, query):
        # Implement a simple re-ranking strategy
        # This could be improved with more sophisticated methods
        reranked = sorted(documents, key=lambda doc: self.calculate_relevance(doc['content'], query), reverse=True)
        return reranked

    def calculate_relevance(self, document, query):
        # Simple relevance calculation based on word overlap
        doc_words = set(document.lower().split())
        query_words = set(query.lower().split())
        return len(doc_words.intersection(query_words)) / len(query_words)