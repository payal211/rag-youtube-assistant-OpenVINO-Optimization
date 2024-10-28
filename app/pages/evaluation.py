import streamlit as st

st.set_page_config(
    page_title="04_Evaluation",  # Use this format for ordering
    page_icon="📊",
    layout="wide"
)

import pandas as pd
from database import DatabaseHandler
from data_processor import DataProcessor
from rag import RAGSystem
from evaluation import EvaluationSystem
from generate_ground_truth import get_evaluation_display_data
import logging

logger = logging.getLogger(__name__)

# Define evaluation prompt template
EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for a Youtube transcript assistant.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in the following JSON format:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "Your explanation for the relevance classification"
}}

Requirements:
1. Relevance must be one of the three exact values
2. Provide clear reasoning in the explanation
3. Consider accuracy and completeness of the answer
4. Return valid JSON only
""".strip()

@st.cache_resource
def init_components():
    db_handler = DatabaseHandler()
    data_processor = DataProcessor()
    rag_system = RAGSystem(data_processor)
    evaluation_system = EvaluationSystem(data_processor, db_handler)
    return db_handler, data_processor, rag_system, evaluation_system

def main():
    st.title("RAG Evaluation 📊")
    
    db_handler, data_processor, rag_system, evaluation_system = init_components()
    
    try:
        # Check for ground truth data
        ground_truth_df = pd.read_csv('data/ground-truth-retrieval.csv')
        ground_truth_available = True
        
        # Display existing evaluations
        existing_evaluations = get_evaluation_display_data()
        if not existing_evaluations.empty:
            st.subheader("Existing Evaluation Results")
            st.dataframe(existing_evaluations)
            
            # Download button for evaluation results
            csv = existing_evaluations.to_csv(index=False)
            st.download_button(
                label="Download Evaluation Results",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
            )
        
        # Run evaluation
        if ground_truth_available:
            if st.button("Run Full Evaluation"):
                with st.spinner("Running evaluation..."):
                    try:
                        evaluation_results = evaluation_system.run_full_evaluation(
                            rag_system,
                            'data/ground-truth-retrieval.csv',
                            EVALUATION_PROMPT_TEMPLATE
                        )
                        
                        if evaluation_results:
                            # Display RAG evaluations
                            st.subheader("RAG Evaluations")
                            rag_eval_df = pd.DataFrame(evaluation_results["rag_evaluations"])
                            st.dataframe(rag_eval_df)
                            
                            # Display search performance
                            st.subheader("Search Performance")
                            search_perf_df = pd.DataFrame([evaluation_results["search_performance"]])
                            st.dataframe(search_perf_df)
                            
                            # Display optimized parameters
                            st.subheader("Optimized Search Parameters")
                            params_df = pd.DataFrame([{
                                'parameter': k,
                                'value': v,
                                'score': evaluation_results['best_score']
                            } for k, v in evaluation_results['best_params'].items()])
                            st.dataframe(params_df)
                            
                            # Save results
                            for video_id in rag_eval_df['video_id'].unique():
                                db_handler.save_search_performance(
                                    video_id,
                                    evaluation_results["search_performance"]['hit_rate'],
                                    evaluation_results["search_performance"]['mrr']
                                )
                                db_handler.save_search_parameters(
                                    video_id,
                                    evaluation_results['best_params'],
                                    evaluation_results['best_score']
                                )
                            
                            st.success("Evaluation complete. Results saved to database and CSV.")
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        logger.error(f"Error in evaluation: {str(e)}")
        
    except FileNotFoundError:
        st.warning("No ground truth data available. Please generate ground truth data in the Ground Truth Generation page first.")
        if st.button("Go to Ground Truth Generation"):
            st.switch_page("pages/3_Ground_Truth.py")

if __name__ == "__main__":
    main()