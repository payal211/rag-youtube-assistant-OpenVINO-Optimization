import pandas as pd
import json
from tqdm import tqdm
# import ollama
import openvino_genai as ov_genai
from elasticsearch import Elasticsearch
import sqlite3
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_model_name(index_name):
    # Extract the model name from the index name
    match = re.search(r'video_[^_]+_(.+)$', index_name)
    if match:
        return match.group(1)
    return None

def get_transcript_from_elasticsearch(es, index_name, video_id):
    try:
        result = es.search(index=index_name, body={
            "query": {
                "match": {
                    "video_id": video_id
                }
            }
        })
        if result['hits']['hits']:
            return result['hits']['hits'][0]['_source']['content']
    except Exception as e:
        logger.error(f"Error retrieving transcript from Elasticsearch: {str(e)}")
    return None

def get_transcript_from_sqlite(db_path, video_id):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT transcript_content FROM videos WHERE youtube_id = ?", (video_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
    except Exception as e:
        logger.error(f"Error retrieving transcript from SQLite: {str(e)}")
    return None

def generate_questions(transcript, max_retries=3):
    prompt_template = """
    You are an AI assistant tasked with generating questions based on a YouTube video transcript.
    Formulate EXACTLY 10 questions that a user might ask based on the provided transcript.
    Make the questions specific to the content of the transcript.
    The questions should be complete and not too short. Use as few words as possible from the transcript.
    Ensure that all 10 questions are unique and not repetitive.

    The transcript:

    {transcript}

    Provide the output in parsable JSON without using code blocks:

    {{"questions": ["question1", "question2", ..., "question10"]}}
    """.strip()

    all_questions = set()
    retries = 0

    while len(all_questions) < 10 and retries < max_retries:
        prompt = prompt_template.format(transcript=transcript)
        try:
            # response = ollama.chat(
            #     model='phi3.5',
            #     messages=[{"role": "user", "content": prompt}]
            # )
         
            # Corrected: Using OpenVINO GenAI for question generation
            model_path = os.getenv('OPENVINO_MODEL_PATH', '/app/models/Phi-3-mini-128k-instruct-int4-ov')  # Model path environment variable
            device = os.getenv('OPENVINO_DEVICE', 'CPU')  # Device to run on (e.g., 'CPU', 'GPU')
            model = ov_genai.load_model(model_path, device=device)  # Load the OpenVINO GenAI model

            # Generate the response using OpenVINO GenAI
            response = model.generate(prompt)  # Corrected: Generate questions based on the prompt using OpenVINO GenAI
                     
            questions = json.loads(response['message']['content'])['questions']
            all_questions.update(questions)
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
        retries += 1

    if len(all_questions) < 10:
        logger.warning(f"Could only generate {len(all_questions)} unique questions after {max_retries} attempts.")

    return {"questions": list(all_questions)[:10]}

def generate_ground_truth(db_handler, data_processor, video_id):
    es = Elasticsearch([f'http://{os.getenv("ELASTICSEARCH_HOST", "localhost")}:{os.getenv("ELASTICSEARCH_PORT", "9200")}'])
    
    # Get existing questions for this video to avoid duplicates
    existing_questions = set(q[1] for q in db_handler.get_ground_truth_by_video(video_id))
    
    transcript = None
    index_name = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
    
    if index_name:
        transcript = get_transcript_from_elasticsearch(es, index_name, video_id)
    
    if not transcript:
        transcript = db_handler.get_transcript_content(video_id)
    
    if not transcript:
        logger.error(f"Failed to retrieve transcript for video {video_id}")
        return None

    # Generate questions until we have 10 unique ones
    all_questions = set()
    max_attempts = 3
    attempts = 0

    while len(all_questions) < 10 and attempts < max_attempts:
        questions = generate_questions(transcript)
        if questions and 'questions' in questions:
            new_questions = set(questions['questions']) - existing_questions
            all_questions.update(new_questions)
        attempts += 1

    if not all_questions:
        logger.error("Failed to generate any unique questions.")
        return None

    # Store questions in database
    db_handler.add_ground_truth_questions(video_id, all_questions)

    # Create DataFrame and save to CSV
    df = pd.DataFrame([(video_id, q) for q in all_questions], columns=['video_id', 'question'])
    csv_path = 'data/ground-truth-retrieval.csv'
    
    # Append to existing CSV if it exists, otherwise create new
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    logger.info(f"Ground truth data saved to {csv_path}")
    return df

def get_ground_truth_display_data(db_handler, video_id=None, channel_name=None):
    """Get ground truth data from both database and CSV file"""
    import pandas as pd
    
    # Try to get data from database first
    if video_id:
        data = db_handler.get_ground_truth_by_video(video_id)
    elif channel_name:
        data = db_handler.get_ground_truth_by_channel(channel_name)
    else:
        data = []
    
    # Create DataFrame from database data
    if data:
        db_df = pd.DataFrame(data, columns=['id', 'video_id', 'question', 'generation_date', 'channel_name'])
    else:
        db_df = pd.DataFrame()
    
    # Try to get data from CSV
    try:
        csv_df = pd.read_csv('data/ground-truth-retrieval.csv')
        if video_id:
            csv_df = csv_df[csv_df['video_id'] == video_id]
        elif channel_name:
            # Join with videos table to get channel information
            videos_df = pd.DataFrame(db_handler.get_all_videos(), 
                                   columns=['youtube_id', 'title', 'channel_name', 'upload_date'])
            csv_df = csv_df.merge(videos_df, left_on='video_id', right_on='youtube_id')
            csv_df = csv_df[csv_df['channel_name'] == channel_name]
    except FileNotFoundError:
        csv_df = pd.DataFrame()
    
    # Combine data from both sources
    if not db_df.empty and not csv_df.empty:
        combined_df = pd.concat([db_df, csv_df]).drop_duplicates(subset=['video_id', 'question'])
    elif not db_df.empty:
        combined_df = db_df
    elif not csv_df.empty:
        combined_df = csv_df
    else:
        combined_df = pd.DataFrame()
    
    return combined_df

def generate_ground_truth_for_all_videos(db_handler, data_processor):
    videos = db_handler.get_all_videos()
    all_questions = []

    for video in tqdm(videos, desc="Generating ground truth"):
        video_id = video[0]  # Assuming the video ID is the first element in the tuple
        df = generate_ground_truth(db_handler, data_processor, video_id)
        if df is not None:
            all_questions.extend(df.values.tolist())

    if all_questions:
        df = pd.DataFrame(all_questions, columns=['video_id', 'question'])
        csv_path = 'data/ground-truth-retrieval.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Ground truth data for all videos saved to {csv_path}")
        return df
    else:
        logger.error("Failed to generate questions for any video.")
        return None
    
def get_evaluation_display_data(video_id=None):
    """Get evaluation data from both database and CSV file"""
    import pandas as pd
    
    # Try to get data from CSV
    try:
        csv_df = pd.read_csv('data/evaluation_results.csv')
        if video_id:
            csv_df = csv_df[csv_df['video_id'] == video_id]
    except FileNotFoundError:
        csv_df = pd.DataFrame()
    
    return csv_df
