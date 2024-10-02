import os
import pandas as pd
import json
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import requests

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"Error extracting transcript for video {video_id}: {str(e)}")
        return None

def generate_questions(transcript):
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

    response = requests.post(f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate', json={
        'model': 'phi3.5',
        'prompt': prompt
    })
    
    if response.status_code == 200:
        return json.loads(response.json()['response'])
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    video_id = "zjkBMFhNj_g"
    transcript = get_transcript(video_id)
    
    if transcript:
        questions = generate_questions(transcript)
        
        if questions:
            df = pd.DataFrame([(video_id, q) for q in questions['questions']], columns=['video_id', 'question'])
            
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/ground-truth-retrieval.csv', index=False)
            print("Ground truth data saved to data/ground-truth-retrieval.csv")
        else:
            print("Failed to generate questions.")
    else:
        print("Failed to generate ground truth data due to transcript retrieval error.")
        
if __name__ == "__main__":
    main()