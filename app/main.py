import streamlit as st
import pandas as pd
from transcript_extractor import extract_video_id, get_transcript, get_channel_videos, process_videos
from data_processor import DataProcessor
from database import DatabaseHandler
from rag import RAGSystem
from query_rewriter import QueryRewriter
from evaluation import EvaluationSystem
from sentence_transformers import SentenceTransformer
import os
import json
import requests
from tqdm import tqdm
import sqlite3

# Initialize components
@st.cache_resource
def init_components():
    db_handler = DatabaseHandler()
    data_processor = DataProcessor()
    rag_system = RAGSystem(data_processor)
    query_rewriter = QueryRewriter()
    evaluation_system = EvaluationSystem(data_processor, db_handler)
    return db_handler, data_processor, rag_system, query_rewriter, evaluation_system

db_handler, data_processor, rag_system, query_rewriter, evaluation_system = init_components()

# Ground Truth Generation
def generate_questions(transcript):
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
    OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
    prompt_template = """
    You are an AI assistant tasked with generating questions based on a YouTube video transcript.
    Formulate 10 questions that a user might ask based on the provided transcript.
    Make the questions specific to the content of the transcript.
    The questions should be complete and not too short. Use as few words as possible from the transcript.

    The transcript:

    {transcript}

    Provide the output in parsable JSON without using code blocks:

    {{"questions": ["question1", "question2", ..., "question10"]}}
    """.strip()

    prompt = prompt_template.format(transcript=transcript)

    try:
        response = requests.post(f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate', json={
            'model': 'phi3.5',
            'prompt': prompt
        })
        response.raise_for_status()
        return json.loads(response.json()['response'])
    except requests.RequestException as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

def generate_ground_truth(video_id):
    transcript_data = get_transcript(video_id)
    
    if transcript_data and 'transcript' in transcript_data:
        full_transcript = " ".join([entry['text'] for entry in transcript_data['transcript']])
        questions = generate_questions(full_transcript)
        
        if questions and 'questions' in questions:
            df = pd.DataFrame([(video_id, q) for q in questions['questions']], columns=['video_id', 'question'])
            
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/ground-truth-retrieval.csv', index=False)
            st.success("Ground truth data generated and saved to data/ground-truth-retrieval.csv")
            return df
        else:
            st.error("Failed to generate questions.")
    else:
        st.error("Failed to generate ground truth data due to transcript retrieval error.")
    return None

# RAG Evaluation
def evaluate_rag(sample_size=200):
    try:
        ground_truth = pd.read_csv('data/ground-truth-retrieval.csv')
    except FileNotFoundError:
        st.error("Ground truth file not found. Please generate ground truth data first.")
        return None

    sample = ground_truth.sample(n=min(sample_size, len(ground_truth)), random_state=1)
    evaluations = []
    
    prompt_template = """
    You are an expert evaluator for a Youtube transcript assistant.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer_llm}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    progress_bar = st.progress(0)
    for i, (_, row) in enumerate(sample.iterrows()):
        question = row['question']
        answer_llm = rag_system.query(question)
        prompt = prompt_template.format(question=question, answer_llm=answer_llm)
        evaluation = rag_system.query(prompt)  # Assuming rag_system can handle this type of query
        try:
            evaluation_json = json.loads(evaluation)
            evaluations.append((row['video_id'], question, answer_llm, evaluation_json['Relevance'], evaluation_json['Explanation']))
        except json.JSONDecodeError:
            st.warning(f"Failed to parse evaluation for question: {question}")
        progress_bar.progress((i + 1) / len(sample))

    # Store RAG evaluations in the database
    conn = sqlite3.connect('data/sqlite.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rag_evaluations (
        video_id TEXT,
        question TEXT,
        answer TEXT,
        relevance TEXT,
        explanation TEXT
    )
    ''')
    cursor.executemany('''
    INSERT INTO rag_evaluations (video_id, question, answer, relevance, explanation)
    VALUES (?, ?, ?, ?, ?)
    ''', evaluations)
    conn.commit()
    conn.close()

    st.success("Evaluation complete. Results stored in the database.")
    return evaluations

def main():
    st.title("YouTube Transcript RAG System")

    tab1, tab2, tab3 = st.tabs(["RAG System", "Ground Truth Generation", "Evaluation"])

    with tab1:
        st.header("RAG System")
        # Input section
        input_type = st.radio("Select input type:", ["Video URL", "Channel URL", "YouTube ID"])
        input_value = st.text_input("Enter the URL or ID:")
        embedding_model = st.selectbox("Select embedding model:", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])

        if st.button("Process"):
            with st.spinner("Processing..."):
                data_processor.embedding_model = SentenceTransformer(embedding_model)
                if input_type == "Video URL":
                    video_id = extract_video_id(input_value)
                    if video_id:
                        process_single_video(video_id, embedding_model)
                    else:
                        st.error("Failed to extract video ID from the URL")
                elif input_type == "Channel URL":
                    channel_videos = get_channel_videos(input_value)
                    if channel_videos:
                        process_multiple_videos([video['video_id'] for video in channel_videos], embedding_model)
                    else:
                        st.error("Failed to retrieve videos from the channel")
                else:
                    process_single_video(input_value, embedding_model)

        # Query section
        st.subheader("Query the RAG System")
        query = st.text_input("Enter your query:")
        rewrite_method = st.radio("Query rewriting method:", ["None", "Chain of Thought", "ReAct"])
        search_method = st.radio("Search method:", ["Hybrid", "Text-only", "Embedding-only"])

        if st.button("Search"):
            with st.spinner("Searching..."):
                if rewrite_method == "Chain of Thought":
                    query = query_rewriter.rewrite_cot(query)
                elif rewrite_method == "ReAct":
                    query = query_rewriter.rewrite_react(query)

                search_method_map = {"Hybrid": "hybrid", "Text-only": "text", "Embedding-only": "embedding"}
                response = rag_system.query(query, search_method=search_method_map[search_method])
                st.write("Response:", response)

            # Feedback
            feedback = st.radio("Provide feedback:", ["+1", "-1"])
            if st.button("Submit Feedback"):
                db_handler.add_user_feedback("all_videos", query, 1 if feedback == "+1" else -1)
                st.success("Feedback submitted successfully!")

    with tab2:
        st.header("Ground Truth Generation")
        video_id = st.text_input("Enter YouTube Video ID for ground truth generation:")
        if st.button("Generate Ground Truth"):
            with st.spinner("Generating ground truth..."):
                ground_truth_df = generate_ground_truth(video_id)
                if ground_truth_df is not None:
                    st.dataframe(ground_truth_df)
                    csv = ground_truth_df.to_csv(index=False)
                    st.download_button(
                        label="Download Ground Truth CSV",
                        data=csv,
                        file_name="ground_truth.csv",
                        mime="text/csv",
                    )

    with tab3:
        st.header("RAG Evaluation")
        sample_size = st.number_input("Enter sample size for evaluation:", min_value=1, max_value=1000, value=200)
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                evaluation_results = evaluate_rag(sample_size)
                if evaluation_results:
                    st.write("Evaluation Results:")
                    st.dataframe(pd.DataFrame(evaluation_results, columns=['Video ID', 'Question', 'Answer', 'Relevance', 'Explanation']))

@st.cache_data
def process_single_video(video_id, embedding_model):
    # Check if the video has already been processed with the current embedding model
    existing_index = db_handler.get_elasticsearch_index(video_id, embedding_model)
    if existing_index:
        st.info(f"Video {video_id} has already been processed with {embedding_model}. Using existing index: {existing_index}")
        return existing_index

    transcript_data = get_transcript(video_id)
    if transcript_data:
        # Store video metadata in the database
        video_data = {
            'video_id': video_id,
            'title': transcript_data['metadata'].get('title', 'Unknown Title'),
            'author': transcript_data['metadata'].get('author', 'Unknown Author'),
            'upload_date': transcript_data['metadata'].get('upload_date', 'Unknown Date'),
            'view_count': int(transcript_data['metadata'].get('view_count', 0)),
            'like_count': int(transcript_data['metadata'].get('like_count', 0)),
            'comment_count': int(transcript_data['metadata'].get('comment_count', 0)),
            'video_duration': transcript_data['metadata'].get('duration', 'Unknown Duration')
        }
        db_handler.add_video(video_data)

        # Store transcript segments in the database
        for i, segment in enumerate(transcript_data['transcript']):
            segment_data = {
                'segment_id': f"{video_id}_{i}",
                'video_id': video_id,
                'content': segment.get('text', ''),
                'start_time': segment.get('start', 0),
                'duration': segment.get('duration', 0)
            }
            db_handler.add_transcript_segment(segment_data)

        # Process transcript for RAG system
        data_processor.process_transcript(video_id, transcript_data)
        
        # Create Elasticsearch index
        index_name = f"video_{video_id}_{embedding_model}"
        data_processor.build_index(index_name)
        
        # Store Elasticsearch index information
        db_handler.add_elasticsearch_index(video_id, index_name, embedding_model)
        
        st.success(f"Processed and indexed transcript for video {video_id}")
        st.write("Metadata:", transcript_data['metadata'])
        return index_name
    else:
        st.error(f"Failed to retrieve transcript for video {video_id}")
        return None

@st.cache_data
def process_multiple_videos(video_ids, embedding_model):
    indices = []
    for video_id in video_ids:
        index = process_single_video(video_id, embedding_model)
        if index:
            indices.append(index)
    st.success(f"Processed and indexed transcripts for {len(indices)} videos")
    return indices

if __name__ == "__main__":
    main()