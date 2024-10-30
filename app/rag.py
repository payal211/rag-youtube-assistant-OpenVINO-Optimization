import os
from dotenv import load_dotenv
import logging
import time
import openvino_genai as ov_genai

load_dotenv()

logger = logging.getLogger(__name__)

# Define the RAG prompt template
RAG_PROMPT_TEMPLATE = """
You are an AI assistant analyzing YouTube video transcripts. Your task is to answer questions based on the provided transcript context.

Context from transcript:
{context}

User Question: {question}

Please provide a clear, concise answer based only on the information given in the context. If the context doesn't contain enough information to fully answer the question, acknowledge this in your response.

Guidelines:
1. Use only information from the provided context
2. Be specific and direct in your answer
3. If context is insufficient, say so
4. Maintain accuracy and avoid speculation
5. Use natural, conversational language
""".strip()

class RAGSystem:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model_path = os.getenv('OPENVINO_MODEL_PATH', 'Phi-3-mini-128k-instruct-int8-ov')
        self.device = os.getenv('OPENVINO_DEVICE', 'CPU')
        
        # Initialize the OpenVINO pipeline
        self.pipe = ov_genai.LLMPipeline(self.model_path, self.device)
        logger.info(f"OpenVINO model loaded from {self.model_path} on device {self.device}")

    def generate(self, prompt):
        try:
            response = self.pipe.generate(prompt, max_length=2000)
            logger.info("Generated response from OpenVINO model.")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    def get_prompt(self, user_query, relevant_docs):
        context = "\n".join([doc['content'] for doc in relevant_docs])
        return RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=user_query
        )

    def query(self, user_query, search_method='hybrid', index_name=None):
        try:
            if not index_name:
                raise ValueError("No index name provided. Please select a video and ensure it has been processed.")

            relevant_docs = self.data_processor.search(user_query, num_results=3, method=search_method, index_name=index_name)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for the query.")
                return "I couldn't find any relevant information to answer your query.", ""

            prompt = self.get_prompt(user_query, relevant_docs)
            answer = self.generate(prompt)

            if answer is not None:
                return answer, prompt
            else:
                return "An error occurred while generating the answer.", ""
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
