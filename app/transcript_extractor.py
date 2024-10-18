import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import re
import logging
import ssl
import certifi
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file (one directory up from the current script)
dotenv_path = os.path.join(os.path.dirname(current_dir), '.env')
logger.info(f"The .env path is: {dotenv_path}")
# Load environment variables from .env file
load_dotenv(dotenv_path)

# Get API key from environment variable
API_KEY = os.getenv('YOUTUBE_API_KEY')
logger.info(f"API_KEY: {API_KEY[:5]}...{API_KEY[-5:]}")  # Log first and last 5 characters for verification

if not API_KEY:
    raise ValueError("YouTube API key not found. Make sure it's set in your .env file in the parent directory of the 'app' folder.")

def get_youtube_client():
    try:
        # Create a custom session with SSL verification
        session = requests.Session()
        session.verify = certifi.where()

        # Create a custom HTTP object
        http = googleapiclient.http.build_http()
        http.verify = session.verify

        # Build the YouTube client with the custom HTTP object
        youtube = build('youtube', 'v3', developerKey=API_KEY, http=http)
        logger.info("YouTube API client initialized successfully")
        return youtube
    except Exception as e:
        logger.error(f"Error initializing YouTube API client: {str(e)}")
        raise

def extract_video_id(url):
    if not url:
        return None
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def get_video_metadata(video_id):
    youtube = get_youtube_client()
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            video = response['items'][0]
            snippet = video['snippet']
            
            # Get the description and set default if it's blank
            description = snippet.get('description', '').strip()
            if not description:
                description = 'Not Available'
            
            return {
                'title': snippet['title'],
                'author': snippet['channelTitle'],
                'upload_date': snippet['publishedAt'],
                'view_count': video['statistics'].get('viewCount', '0'),
                'like_count': video['statistics'].get('likeCount', '0'),
                'comment_count': video['statistics'].get('commentCount', '0'),
                'duration': video['contentDetails']['duration'],
                'description': description  # Add the description to the metadata
            }
        else:
            logger.error(f"No video found with id: {video_id}")
            return None
    except Exception as e:
        logger.error(f"An error occurred while fetching metadata for video {video_id}: {str(e)}")
        return None
    
def get_transcript(video_id):
    if not video_id:
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        metadata = get_video_metadata(video_id)
        logger.info(f"Metadata for video {video_id}: {metadata}")
        logger.info(f"Transcript length for video {video_id}: {len(transcript)}")
        if not metadata:
            return None
        return {
            'transcript': transcript,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Error extracting transcript for video {video_id}: {str(e)}")
        return None

def get_channel_videos(channel_url):
    youtube = get_youtube_client()
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        logger.error(f"Invalid channel URL: {channel_url}")
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
        logger.error(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return []
    except Exception as e:
        logger.error(f"An error occurred while fetching channel videos: {str(e)}")
        return []

def extract_channel_id(url):
    channel_id_match = re.search(r"(?:channel\/|c\/|@)([a-zA-Z0-9-_]+)", url)
    if channel_id_match:
        return channel_id_match.group(1)
    return None

def test_api_key():
    youtube = get_youtube_client()
    try:
        request = youtube.videos().list(part="snippet", id="dQw4w9WgXcQ")
        response = request.execute()
        if 'items' in response:
            logger.info("API key is valid and working")
            return True
        else:
            logger.error("API request successful but returned unexpected response")
            return False
    except Exception as e:
        logger.error(f"API key test failed: {str(e)}")
        return False

def initialize_youtube_api():
    if test_api_key():
        logger.info("YouTube API initialized successfully")
        return True
    else:
        logger.error("Failed to initialize YouTube API")
        return False