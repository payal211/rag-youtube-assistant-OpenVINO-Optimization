import os
from dotenv import load_dotenv
import ollama
import logging
import time

load_dotenv()

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = os.getenv('OLLAMA_MODEL', 'phi3')
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.timeout = int(os.getenv('OLLAMA_TIMEOUT', 240))
        self.max_retries = int(os.getenv('OLLAMA_MAX_RETRIES', 3))
        
        self.check_ollama_service()

    def check_ollama_service(self):
        try:
            ollama.list()
            logger.info("Ollama service is accessible.")
            self.pull_model()
        except Exception as e:
            logger.error(f"An error occurred while connecting to Ollama: {e}")
            logger.error(f"Please ensure Ollama is running and accessible at {self.ollama_host}")

    def pull_model(self):
        try:
            ollama.pull(self.model)
            logger.info(f"Successfully pulled model {self.model}.")
        except Exception as e:
            logger.error(f"Error pulling model {self.model}: {e}")

    def generate(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                print("Printing the response from OLLAMA: "+response['message']['content'])
                return response['message']['content']
            except Exception as e:
                logger.error(f"Error generating response on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("All retries exhausted. Unable to generate response.")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_prompt(self, user_query, relevant_docs):
        context = "\n".join([doc['content'] for doc in relevant_docs])
        prompt = f"""You are AI Youtube transcript assistant that analyses youtube transcripts and responds back to the user query based on the Context shared with you. Please ensure that the answers are correct, meaningful, and help in answering the query.

Context: {context}

Question: {user_query}

Answer:"""
        return prompt

    def query(self, user_query, search_method='hybrid', index_name=None):
        try:
            if not index_name:
                raise ValueError("No index name provided. Please select a video and ensure it has been processed.")

            relevant_docs = self.data_processor.search(user_query, num_results=3, method=search_method, index_name=index_name)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for the query.")
                return "I couldn't find any relevant information to answer your query.", ""

            prompt = self.get_prompt(user_query, relevant_docs)
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response['message']['content']
            return answer, prompt
        except Exception as e:
            logger.error(f"An error occurred in the RAG system: {e}")
            return f"An error occurred: {str(e)}", ""
        
    def rewrite_cot(self, query):
        prompt = f"""Rewrite the following query using chain-of-thought reasoning:

Query: {query}

Rewritten query:"""
        response = self.generate(prompt)
        if response:
            return response, prompt
        return query, prompt  # Return original query if rewriting fails

    def rewrite_react(self, query):
        prompt = f"""Rewrite the following query using ReAct (Reasoning and Acting) approach:

Query: {query}

Rewritten query:"""
        response = self.generate(prompt)
        if response:
            return response, prompt
        return query, prompt  # Return original query if rewriting fails