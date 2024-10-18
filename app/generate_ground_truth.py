import pandas as pd
import json
from tqdm import tqdm
import ollama
from transcript_extractor import get_transcript

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
        print(f"Error generating questions: {str(e)}")
        return None

def generate_ground_truth(db_handler, data_processor, video_id):
    transcript_data = get_transcript(video_id)
    if transcript_data and 'transcript' in transcript_data:
        full_transcript = " ".join([entry['text'] for entry in transcript_data['transcript']])
        # Process the transcript
        data_processor.process_transcript(video_id, transcript_data)
    else:
        print(f"Failed to retrieve transcript for video {video_id}")
        return None

    questions = generate_questions(full_transcript)
    
    if questions and 'questions' in questions:
        df = pd.DataFrame([(video_id, q) for q in questions['questions']], columns=['video_id', 'question'])
        
        csv_path = 'data/ground-truth-retrieval.csv'
        df.to_csv(csv_path, index=False)
        print(f"Ground truth data saved to {csv_path}")
        return df
    else:
        print("Failed to generate questions.")
    return None

def generate_ground_truth_for_all_videos(db_handler, data_processor):
    videos = db_handler.get_all_videos()
    all_questions = []

    for video in tqdm(videos, desc="Generating ground truth"):
        video_id = video[0]  # Assuming the video ID is the first element in the tuple
        transcript_data = get_transcript(video_id)
        if transcript_data and 'transcript' in transcript_data:
            full_transcript = " ".join([entry['text'] for entry in transcript_data['transcript']])
            # Process the transcript
            data_processor.process_transcript(video_id, transcript_data)
            questions = generate_questions(full_transcript)
            if questions and 'questions' in questions:
                all_questions.extend([(video_id, q) for q in questions['questions']])
        else:
            print(f"Failed to retrieve transcript for video {video_id}")

    if all_questions:
        df = pd.DataFrame(all_questions, columns=['video_id', 'question'])
        csv_path = 'data/ground-truth-retrieval.csv'
        df.to_csv(csv_path, index=False)
        print(f"Ground truth data for all videos saved to {csv_path}")
        return df
    else:
        print("Failed to generate questions for any video.")
        return None