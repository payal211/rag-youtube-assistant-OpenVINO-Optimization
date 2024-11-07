import os
from dotenv import load_dotenv
import logging
import time
import openvino_genai as ov_genai
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO)
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
        """Initialize the RAG system with model loading and error handling"""
        try:
            self.data_processor = data_processor
            
            # Get model path from environment or use default
            model_base = os.getenv('OPENVINO_MODEL_PATH', '/app/models/Phi-3-mini-128k-instruct-int4-ov')
            self.model_path = self._verify_model_path(model_base)
            
            # Get device from environment or use default
            self.device = os.getenv('OPENVINO_DEVICE', 'CPU')
            
            logger.info(f"Initializing OpenVINO pipeline with model: {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Initialize the OpenVINO pipeline with retry logic
            self.pipe = self._initialize_pipeline()
            
            logger.info("RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def _verify_model_path(self, base_path):
        """Verify model path and required files exist"""
        try:
            model_path = Path(base_path)
            
            # Check if path exists
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
            
            # Check for required files
            required_files = ['openvino_tokenizer.xml', 'openvino_tokenizer.bin']
            missing_files = [f for f in required_files 
                           if not (model_path / f).exists()]
            
            if missing_files:
                raise ValueError(f"Missing model files: {missing_files}")
            
            logger.info(f"Verified model path: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error verifying model path: {str(e)}")
            raise

    def _initialize_pipeline(self, max_retries=3):
        """Initialize OpenVINO pipeline with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                pipe = ov_genai.LLMPipeline(self.model_path, self.device)
                logger.info(f"Successfully initialized pipeline on attempt {attempt + 1}")
                return pipe
            except Exception as e:
                last_error = e
                logger.warning(f"Pipeline initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"Failed to initialize pipeline after {max_retries} attempts: {str(last_error)}")

    def generate(self, prompt, max_retries=3):
        """Generate response with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.pipe.generate(
                    prompt,
                    max_length=2000,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
                logger.info("Successfully generated response")
                return response
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All generation attempts failed")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_prompt(self, user_query, relevant_docs):
        """Format prompt with context and query"""
        try:
            context = "\n".join([doc.get('content', '') for doc in relevant_docs])
            return RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=user_query
            )
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise

    def query(self, user_query, search_method='hybrid', index_name=None):
        """Process query and generate response"""
        try:
            if not index_name:
                raise ValueError("No index name provided. Please select a video and ensure it has been processed.")

            # Get relevant documents
            relevant_docs = self.data_processor.search(
                user_query, 
                num_results=3, 
                method=search_method, 
                index_name=index_name
            )
            
            if not relevant_docs:
                logger.warning("No relevant documents found for the query")
                return "I couldn't find any relevant information to answer your query.", ""

            # Generate and validate response
            prompt = self.get_prompt(user_query, relevant_docs)
            answer = self.generate(prompt)
            
            if answer is not None:
                return answer, prompt
            else:
                logger.error("Failed to generate response")
                return "An error occurred while generating the answer.", ""
                
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return f"An error occurred: {str(e)}", ""

    def rewrite_cot(self, query):
        """Rewrite query using Chain of Thought reasoning"""
        try:
            prompt = f"""Rewrite the following query using chain-of-thought reasoning:

Query: {query}

Let's think about this step by step:
1. Understand the core question
2. Break down the components
3. Consider relevant context
4. Reformulate the question

Rewritten query:"""
            
            response = self.generate(prompt)
            if response:
                return response, prompt
            return query, prompt
        except Exception as e:
            logger.error(f"Error in CoT rewriting: {str(e)}")
            return query, prompt

    def rewrite_react(self, query):
        """Rewrite query using ReAct approach"""
        try:
            prompt = f"""Rewrite the following query using ReAct (Reasoning and Acting) approach:

Query: {query}

Thought: Let's break this down
Action: Analyze the query components
Observation: Identify key elements
Thought: Reformulate for better context
Action: Construct improved query

Rewritten query:"""
            
            response = self.generate(prompt)
            if response:
                return response, prompt
            return query, prompt
        except Exception as e:
            logger.error(f"Error in ReAct rewriting: {str(e)}")
            return query, prompt
