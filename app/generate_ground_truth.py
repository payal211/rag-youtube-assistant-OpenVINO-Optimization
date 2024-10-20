import pandas as pd
import json
from tqdm import tqdm
import ollama
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

def generate_questions(transcript):
    prompt_template = """
    You are an AI assistant tasked with generating questions based on a YouTube video transcript.
    Formulate at least 10 questions that a user might ask based on the provided transcript.
    Make the questions specific to the content of the transcript.
    The questions should be complete and not too short. Use as few words as possible from the transcript.
    It is important that the questions are relevant to the content of the transcript and are at least 10 in number.

    The transcript:

    {transcript}

    Provide the output in parsable JSON without using code blocks:

    {{"questions": ["question1", "question2", ..., "question10"]}}
    """.strip()

    prompt = prompt_template.format(transcript=transcript)

    try:
        response = ollama.chat(
            model='phi3.5',
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return None

def generate_ground_truth(db_handler, data_processor, video_id):
    es = Elasticsearch([f'http://{os.getenv("ELASTICSEARCH_HOST", "localhost")}:{os.getenv("ELASTICSEARCH_PORT", "9200")}'])
    
    # Get the index name for the video
    index_name = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
    
    if not index_name:
        logger.error(f"No Elasticsearch index found for video {video_id}")
        return None
    
    # Extract the model name from the index name
    model_name = extract_model_name(index_name)
    
    if not model_name:
        logger.error(f"Could not extract model name from index name: {index_name}")
        return None
    
    transcript = None
    if index_name:
        transcript = get_transcript_from_elasticsearch(es, index_name, video_id)
        logger.info(f"Transcript to generate questions using elasticsearch is {transcript}")
    
    if not transcript:
        transcript = db_handler.get_transcript_content(video_id)
        logger.info(f"Transcript to generate questions using textual data is {transcript}")
    
    if not transcript:
        logger.error(f"Failed to retrieve transcript for video {video_id}")
        return None

    questions = generate_questions(transcript)
    
    if questions and 'questions' in questions:
        df = pd.DataFrame([(video_id, q) for q in questions['questions']], columns=['video_id', 'question'])
        
        csv_path = 'data/ground-truth-retrieval.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Ground truth data saved to {csv_path}")
        return df
    else:
        logger.error("Failed to generate questions.")
    return None

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