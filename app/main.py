import streamlit as st
import pandas as pd
from transcript_extractor import get_transcript, get_youtube_client, extract_video_id, get_channel_videos, test_api_key, initialize_youtube_api
from data_processor import DataProcessor
from database import DatabaseHandler
from rag import RAGSystem
from query_rewriter import QueryRewriter
from evaluation import EvaluationSystem
from generate_ground_truth import generate_ground_truth, generate_ground_truth_for_all_videos
from sentence_transformers import SentenceTransformer
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def init_components():
    try:
        db_handler = DatabaseHandler()
        data_processor = DataProcessor()
        rag_system = RAGSystem(data_processor)
        query_rewriter = QueryRewriter()
        evaluation_system = EvaluationSystem(data_processor, db_handler)
        logger.info("Components initialized successfully")
        return db_handler, data_processor, rag_system, query_rewriter, evaluation_system
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        st.error("Please check your configuration and ensure all services are running.")
        return None, None, None, None, None


def check_api_key():
    if test_api_key():
        st.success("YouTube API key is valid and working.")
    else:
        st.error("YouTube API key is invalid or not set. Please check your .env file.")
        new_api_key = st.text_input("Enter your YouTube API key:")
        if new_api_key:
            os.environ['YOUTUBE_API_KEY'] = new_api_key
            with open('.env', 'a') as f:
                f.write(f"\nYOUTUBE_API_KEY={new_api_key}")
            st.success("API key saved. Reinitializing YouTube client...")
            get_youtube_client.cache_clear()  # Clear the cache to force reinitialization
            if test_api_key():
                st.success("YouTube client reinitialized successfully.")
            else:
                st.error("Failed to reinitialize YouTube client. Please check your API key.")
            st.experimental_rerun()

# LLM-as-a-judge prompt template
prompt_template = """
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
  "Relevance": "NON_RELEVANT",
  "Explanation": "Your explanation here"
}}

OR

{{
  "Relevance": "PARTLY_RELEVANT",
  "Explanation": "Your explanation here"
}}

OR

{{
  "Relevance": "RELEVANT",
  "Explanation": "Your explanation here"
}}

Ensure your response is a valid JSON object with these exact keys and one of the three exact values for "Relevance".
Do not include any text outside of this JSON object.
"""

def process_single_video(db_handler, data_processor, video_id, embedding_model):
    existing_index = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
    if existing_index:
        logger.info(f"Video {video_id} has already been processed with {embedding_model}. Using existing index: {existing_index}")
        return existing_index

    transcript_data = get_transcript(video_id)
    if transcript_data is None:
        logger.error(f"Failed to retrieve transcript for video {video_id}")
        st.error(f"Failed to retrieve transcript for video {video_id}. Please check if the video ID is correct and the video has captions available.")
        return None

    # Process the transcript
    processed_data = data_processor.process_transcript(video_id, transcript_data)
    if processed_data is None:
        logger.error(f"Failed to process transcript for video {video_id}")
        return None

    # Prepare video data for database insertion
    video_data = {
        'video_id': video_id,
        'title': transcript_data['metadata'].get('title', 'Unknown Title'),
        'author': transcript_data['metadata'].get('author', 'Unknown Author'),
        'upload_date': transcript_data['metadata'].get('upload_date', 'Unknown Date'),
        'view_count': int(transcript_data['metadata'].get('view_count', 0)),
        'like_count': int(transcript_data['metadata'].get('like_count', 0)),
        'comment_count': int(transcript_data['metadata'].get('comment_count', 0)),
        'video_duration': transcript_data['metadata'].get('duration', 'Unknown Duration'),
        'transcript_content': processed_data['content']  # Add this line to include the transcript content
    }

    try:
        db_handler.add_video(video_data)
    except Exception as e:
        logger.error(f"Error adding video to database: {str(e)}")
        st.error(f"Error adding video {video_id} to database: {str(e)}")
        return None

    index_name = f"video_{video_id}_{embedding_model}".lower()
    try:
        index_name = data_processor.build_index(index_name)
        logger.info(f"Successfully built index: {index_name}")
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        st.error(f"Error building index for video {video_id}: {str(e)}")
        return None
    
    embedding_model_id = db_handler.add_embedding_model(embedding_model, "Description of the model")
    
    video_db_record = db_handler.get_video_by_youtube_id(video_id)
    if video_db_record is None:
        logger.error(f"Failed to retrieve video record from database for video {video_id}")
        st.error(f"Failed to retrieve video record from database for video {video_id}")
        return None
    video_db_id = video_db_record[0]
    
    db_handler.add_elasticsearch_index(video_db_id, index_name, embedding_model_id)
    
    logger.info(f"Processed and indexed transcript for video {video_id}")
    st.success(f"Successfully processed and indexed transcript for video {video_id}")
    return index_name

def process_multiple_videos(db_handler, data_processor, video_ids, embedding_model):
    indices = []
    for video_id in video_ids:
        index = process_single_video(db_handler, data_processor, video_id, embedding_model)
        if index:
            indices.append(index)
    logger.info(f"Processed and indexed transcripts for {len(indices)} videos")
    st.success(f"Processed and indexed transcripts for {len(indices)} videos")
    return indices

def ensure_video_processed(db_handler, data_processor, video_id, embedding_model):
    index_name = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
    if not index_name:
        st.warning(f"Video {video_id} has not been processed yet. Processing now...")
        index_name = process_single_video(db_handler, data_processor, video_id, embedding_model)
        if not index_name:
            st.error(f"Failed to process video {video_id}. Please check the logs for more information.")
            return False
    return True

def main():
    st.title("YouTube Transcript RAG System")

    check_api_key()

    components = init_components()
    if components:
        db_handler, data_processor, rag_system, query_rewriter, evaluation_system = components
    else:
        st.stop()
        
    tab1, tab2, tab3 = st.tabs(["RAG System", "Ground Truth Generation", "Evaluation"])

    with tab1:
        st.header("RAG System")
        
        embedding_model = st.selectbox("Select embedding model:", ["multi-qa-MiniLM-L6-cos-v1", "all-mpnet-base-v2"])
        
        st.subheader("Select a Video")
        videos = db_handler.get_all_videos()
        if not videos:
            st.warning("No videos available. Please process some videos first.")
        else:
            video_df = pd.DataFrame(videos, columns=['youtube_id', 'title', 'channel_name', 'upload_date'])
            
            channels = sorted(video_df['channel_name'].unique())
            selected_channel = st.selectbox("Filter by Channel", ["All"] + channels)
            
            if selected_channel != "All":
                video_df = video_df[video_df['channel_name'] == selected_channel]
            
            st.dataframe(video_df)
            selected_video_id = st.selectbox("Select a Video", video_df['youtube_id'].tolist(), format_func=lambda x: video_df[video_df['youtube_id'] == x]['title'].iloc[0])
            
            index_name = db_handler.get_elasticsearch_index_by_youtube_id(selected_video_id)
            
            if index_name:
                st.success(f"Using index: {index_name}")
            else:
                st.warning("No index found for the selected video and embedding model. The index will be built when you search.")
        
        st.subheader("Process New Video")
        input_type = st.radio("Select input type:", ["Video URL", "Channel URL", "YouTube ID"])
        input_value = st.text_input("Enter the URL or ID:")
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                data_processor.set_embedding_model(embedding_model)
                if input_type == "Video URL":
                    video_id = extract_video_id(input_value)
                    if video_id:
                        index_name = process_single_video(db_handler, data_processor, video_id, embedding_model)
                        if index_name is None:
                            st.error(f"Failed to process video {video_id}")
                        else:
                            st.success(f"Successfully processed video {video_id}")
                    else:
                        st.error("Failed to extract video ID from the URL")
                elif input_type == "Channel URL":
                    channel_videos = get_channel_videos(input_value)
                    if channel_videos:
                        index_names = process_multiple_videos(db_handler, data_processor, [video['video_id'] for video in channel_videos], embedding_model)
                        if not index_names:
                            st.error("Failed to process any videos from the channel")
                        else:
                            st.success(f"Successfully processed {len(index_names)} videos from the channel")
                    else:
                        st.error("Failed to retrieve videos from the channel")
                else:
                    index_name = process_single_video(db_handler, data_processor, input_value, embedding_model)
                    if index_name is None:
                        st.error(f"Failed to process video {input_value}")
                    else:
                        st.success(f"Successfully processed video {input_value}")
        
        st.subheader("Query the RAG System")
        query = st.text_input("Enter your query:")
        rewrite_method = st.radio("Query rewriting method:", ["None", "Chain of Thought", "ReAct"])
        search_method = st.radio("Search method:", ["Hybrid", "Text-only", "Embedding-only"])

        if st.button("Search"):
            if not selected_video_id:
                st.error("Please select a video before searching.")
            else:
                with st.spinner("Searching..."):
                    rewritten_query = query
                    rewrite_prompt = ""
                    if rewrite_method == "Chain of Thought":
                        rewritten_query, rewrite_prompt = query_rewriter.rewrite_cot(query)
                    elif rewrite_method == "ReAct":
                        rewritten_query, rewrite_prompt = query_rewriter.rewrite_react(query)

                    st.subheader("Query Processing")
                    st.write("Original query:", query)
                    if rewrite_method != "None":
                        st.write("Rewritten query:", rewritten_query)
                        st.text_area("Query rewriting prompt:", rewrite_prompt, height=100)
                        if rewritten_query == query:
                            st.warning("Query rewriting failed. Using original query.")

                    search_method_map = {"Hybrid": "hybrid", "Text-only": "text", "Embedding-only": "embedding"}
                    try:
                        if not index_name:
                            st.info("Building index for the selected video...")
                            index_name = process_single_video(db_handler, data_processor, selected_video_id, embedding_model)
                            if not index_name:
                                st.error("Failed to build index for the selected video.")
                                return

                        response, final_prompt = rag_system.query(rewritten_query, search_method=search_method_map[search_method], index_name=index_name)
                        
                        st.subheader("RAG System Prompt")
                        if final_prompt:
                            st.text_area("Prompt sent to LLM:", final_prompt, height=300)
                        else:
                            st.warning("No prompt was generated. This might indicate an issue with the RAG system.")
                        
                        st.subheader("Response")
                        if response:
                            st.write(response)
                        else:
                            st.error("No response generated. Please try again or check the system logs for errors.")
                    except ValueError as e:
                        logger.error(f"Error during search: {str(e)}")
                        st.error(f"Error during search: {str(e)}")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred: {str(e)}")
                        st.error(f"An unexpected error occurred: {str(e)}")

    with tab2:
        st.header("Ground Truth Generation")
        
        videos = db_handler.get_all_videos()
        if not videos:
            st.warning("No videos available. Please process some videos first.")
        else:
            video_df = pd.DataFrame(videos, columns=['youtube_id', 'title', 'channel_name', 'upload_date'])
            
            st.dataframe(video_df)
            selected_video_id = st.selectbox("Select a Video", video_df['youtube_id'].tolist(), 
                                             format_func=lambda x: video_df[video_df['youtube_id'] == x]['title'].iloc[0],
                                             key="gt_video_select")
            
            if st.button("Generate Ground Truth for Selected Video"):
                if ensure_video_processed(db_handler, data_processor, selected_video_id, embedding_model):
                    with st.spinner("Generating ground truth..."):
                        ground_truth_df = generate_ground_truth(db_handler, data_processor, selected_video_id)
                        if ground_truth_df is not None:
                            st.dataframe(ground_truth_df)
                            csv = ground_truth_df.to_csv(index=False)
                            st.download_button(
                                label="Download Ground Truth CSV",
                                data=csv,
                                file_name=f"ground_truth_{selected_video_id}.csv",
                                mime="text/csv",
                            )
            if st.button("Generate Ground Truth for All Videos"):
                with st.spinner("Processing videos and generating ground truth..."):
                    for video_id in video_df['youtube_id']:
                        ensure_video_processed(db_handler, data_processor, video_id, embedding_model)
                    ground_truth_df = generate_ground_truth_for_all_videos(db_handler, data_processor)
                    if ground_truth_df is not None:
                        st.dataframe(ground_truth_df)
                        csv = ground_truth_df.to_csv(index=False)
                        st.download_button(
                            label="Download Ground Truth CSV (All Videos)",
                            data=csv,
                            file_name="ground_truth_all_videos.csv",
                            mime="text/csv",
                        )

    with tab3:
        st.header("RAG Evaluation")

        try:
            ground_truth_df = pd.read_csv('data/ground-truth-retrieval.csv')
            ground_truth_available = True
        except FileNotFoundError:
            ground_truth_available = False

        if ground_truth_available:
            st.write("Evaluation will be run on the following ground truth data:")
            st.dataframe(ground_truth_df)
            st.info("The evaluation will use this ground truth data to assess the performance of the RAG system.")

            sample_size = st.number_input("Enter sample size for evaluation:", min_value=1, max_value=len(ground_truth_df), value=min(200, len(ground_truth_df)))
            
            if st.button("Run Evaluation"):
                with st.spinner("Running evaluation..."):
                    evaluation_results = evaluation_system.evaluate_rag(rag_system, 'data/ground-truth-retrieval.csv', sample_size, prompt_template)
                    if evaluation_results:
                        st.write("Evaluation Results:")
                        st.dataframe(pd.DataFrame(evaluation_results, columns=['Video ID', 'Question', 'Answer', 'Relevance', 'Explanation']))
        else:
            st.warning("No ground truth data available. Please generate ground truth data first.")
            st.button("Run Evaluation", disabled=True)

        if not ground_truth_available:
            st.subheader("Generate Ground Truth")
            st.write("You need to generate ground truth data before running the evaluation.")
            if st.button("Go to Ground Truth Generation"):
                st.session_state.active_tab = "Ground Truth Generation"
                st.experimental_rerun()

if __name__ == "__main__":
    if not initialize_youtube_api():
        logger.error("Failed to initialize YouTube API. Exiting.")
        sys.exit(1)
    main()