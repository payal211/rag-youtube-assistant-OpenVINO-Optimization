import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file (one directory up from the current script)
dotenv_path = os.path.join(os.path.dirname(current_dir), '.env')
print("the .env path is :" + dotenv_path)
# Load environment variables from .env file
load_dotenv(dotenv_path)

# Get API key from environment variable
API_KEY = os.getenv('YOUTUBE_API_KEY')
print("the api key is :" + API_KEY)
if not API_KEY:
    raise ValueError("YouTube API key not found. Make sure it's set in your .env file in the parent directory of the 'app' folder.")

print(f"API_KEY: {API_KEY[:5]}...{API_KEY[-5:]}")  # Print first and last 5 characters for verification

try:
    youtube = build('youtube', 'v3', developerKey=API_KEY)
except Exception as e:
    print(f"Error initializing YouTube API client: {str(e)}")
    raise

def extract_video_id(url):
    if not url:
        return None
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def get_video_metadata(video_id):
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()

        if 'items' in response and len(response['items']) > 0:
            video = response['items'][0]
            snippet = video['snippet']
            return {
                'title': snippet['title'],
                'author': snippet['channelTitle'],
                'upload_date': snippet['publishedAt'],
                'view_count': video['statistics'].get('viewCount', '0'),
                'like_count': video['statistics'].get('likeCount', '0'),
                'comment_count': video['statistics'].get('commentCount', '0'),
                'duration': video['contentDetails']['duration']
            }
        else:
            print(f"No video found with ID: {video_id}")
            return None
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None
    except Exception as e:
        print(f"An error occurred while fetching video metadata: {str(e)}")
        return None

def get_transcript(video_id):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the .env file (one directory up from the current script)
    dotenv_path = os.path.join(os.path.dirname(current_dir), '.env')
    print("the .env path is :" + dotenv_path)
    # Load environment variables from .env file
    load_dotenv(dotenv_path)

    # Get API key from environment variable
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    print("the api key is :" + API_KEY)
    if not API_KEY:
        raise ValueError("YouTube API key not found. Make sure it's set in your .env file in the parent directory of the 'app' folder.")

    print(f"API_KEY: {API_KEY[:5]}...{API_KEY[-5:]}")  # Print first and last 5 characters for verification

    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
    except Exception as e:
        print(f"Error initializing YouTube API client: {str(e)}")
        raise
    
    if not video_id:
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        metadata = get_video_metadata(video_id)
        print(f"Metadata for video {video_id}: {metadata}")
        print(f"Transcript length for video {video_id}: {len(transcript)}")
        if not metadata:
            return None
        return {
            'transcript': transcript,
            'metadata': metadata
        }
    except Exception as e:
        print(f"Error extracting transcript for video {video_id}: {str(e)}")
        return None

def get_channel_videos(channel_url):
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        print(f"Invalid channel URL: {channel_url}")
        return []
    try:
        request = youtube.search().list(
            part="id,snippet",
            channelId=channel_id,
            type="video",
            maxResults=50  # Adjust as needed
        )
        response = request.execute()

        videos = []
        for item in response['items']:
            videos.append({
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'published_at': item['snippet']['publishedAt']
            })
        return videos
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return []
    except Exception as e:
        print(f"An error occurred while fetching channel videos: {str(e)}")
        return []

def extract_channel_id(url):
    channel_id_match = re.search(r"(?:channel\/|c\/|@)([a-zA-Z0-9-_]+)", url)
    if channel_id_match:
        return channel_id_match.group(1)
    return None

def process_videos(video_ids):
    transcripts = {}
    for video_id in video_ids:
        transcript_data = get_transcript(video_id)
        if transcript_data:
            transcripts[video_id] = transcript_data
    return transcripts