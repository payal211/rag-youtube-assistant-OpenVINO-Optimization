import os
from dotenv import load_dotenv
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
import html
import xml.etree.ElementTree as ET
from datetime import datetime
import isodate
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(os.path.dirname(current_dir), '.env')
logger.info(f"The .env path is: {dotenv_path}")
load_dotenv(dotenv_path)

API_KEY = os.getenv('YOUTUBE_API_KEY')
logger.info(f"API_KEY: {API_KEY[:5]}...{API_KEY[-5:]}" if API_KEY else "No API key found")

if not API_KEY:
    raise ValueError("YouTube API key not found. Make sure it's set in your .env file.")

def get_youtube_client():
    """Initialize the YouTube API client"""
    try:
        session = requests.Session()
        session.verify = certifi.where()
        http = googleapiclient.http.build_http()
        http.verify = session.verify
        youtube = build('youtube', 'v3', developerKey=API_KEY, http=http)
        logger.info("YouTube API client initialized successfully")
        return youtube
    except Exception as e:
        logger.error(f"Error initializing YouTube API client: {str(e)}")
        raise

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    if not url:
        return None
    # Handle various URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_metadata(video_id):
    """Get video metadata using YouTube Data API"""
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
                'description': description
            }
        else:
            logger.error(f"No video found with id: {video_id}")
            return None
    except Exception as e:
        logger.error(f"Error fetching metadata for video {video_id}: {str(e)}")
        return None

def get_transcript_from_timedtext(video_id):
    """Get transcript using YouTube's timedtext API"""
    try:
        # First get the video page to find available captions
        session = requests.Session()
        session.verify = certifi.where()
        
        # Get video webpage
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = session.get(url, headers=headers)
        html_content = response.text

        # Extract caption data
        captions_regex = r'"captions":{.*?"playerCaptionsTracklistRenderer":.*?"captionTracks":\[(.*?)\]'
        captions_match = re.search(captions_regex, html_content, re.DOTALL)
        
        if not captions_match:
            logger.warning(f"No captions found for video {video_id}")
            return None

        # Parse caption data
        caption_data = captions_match.group(1)
        caption_list = json.loads(f"[{caption_data}]")
        
        # Find English captions or fall back to first available
        caption_url = None
        for caption in caption_list:
            if caption.get('languageCode') == 'en':
                caption_url = caption.get('baseUrl')
                break
        
        if not caption_url and caption_list:
            caption_url = caption_list[0].get('baseUrl')
        
        if not caption_url:
            logger.warning(f"No suitable captions found for video {video_id}")
            return None

        # Get the transcript XML
        response = session.get(caption_url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch transcript: {response.status_code}")
            return None

        # Parse the XML
        root = ET.fromstring(response.text)
        transcript = []
        
        for text in root.findall('.//text'):
            start = float(text.get('start', 0))
            duration = float(text.get('dur', 0))
            content = text.text or ''
            
            # Clean up text
            content = html.unescape(content).strip()
            if content:
                transcript.append({
                    'text': content,
                    'start': start,
                    'duration': duration
                })

        return transcript

    except Exception as e:
        logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
        return None

def get_transcript(video_id):
    """Main function to get both video metadata and transcript"""
    if not video_id:
        return None
    try:
        # Get video metadata
        metadata = get_video_metadata(video_id)
        if not metadata:
            return None

        # Get transcript using timedtext API
        transcript = get_transcript_from_timedtext(video_id)
        
        if not transcript:
            logger.error(f"No transcript available for video {video_id}")
            return None

        logger.info(f"Metadata for video {video_id}: {metadata}")
        logger.info(f"Transcript length for video {video_id}: {len(transcript)}")

        return {
            'transcript': transcript,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
        return None

def get_channel_id_from_handle(handle):
    """Get channel ID from a YouTube handle"""
    try:
        youtube = get_youtube_client()
        request = youtube.search().list(
            part="snippet",
            q=f"@{handle}",
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if response.get('items'):
            return response['items'][0]['snippet']['channelId']
        else:
            logger.error(f"No channel found for handle: @{handle}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting channel ID from handle: {str(e)}")
        return None

def get_channel_id_from_username(username):
    """Get channel ID from a YouTube username or custom URL"""
    try:
        youtube = get_youtube_client()
        request = youtube.search().list(
            part="snippet",
            q=username,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if response.get('items'):
            return response['items'][0]['snippet']['channelId']
        else:
            logger.error(f"No channel found for username: {username}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting channel ID from username: {str(e)}")
        return None

def extract_channel_id(url):
    """Extract channel ID from various YouTube channel URL formats"""
    try:
        # Remove any trailing slashes or parameters
        url = url.split('?')[0].rstrip('/')
        
        # Handle @username format
        if '@' in url:
            handle = url.split('@')[-1]
            return get_channel_id_from_handle(handle)
            
        # Handle different URL patterns
        patterns = [
            r'youtube\.com/channel/([^/]+)',    # Channel ID
            r'youtube\.com/c/([^/]+)',          # Custom URL
            r'youtube\.com/user/([^/]+)',       # Username
            r'youtube\.com/([^/]+)'             # Direct username
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                identifier = match.group(1)
                if pattern == r'youtube\.com/channel/([^/]+)':
                    return identifier  # Direct channel ID
                else:
                    return get_channel_id_from_username(identifier)
                    
        logger.error(f"Could not extract channel identifier from URL: {url}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting channel ID: {str(e)}")
        return None

def get_channel_videos(channel_url):
    """Get list of videos from a YouTube channel"""
    try:
        youtube = get_youtube_client()
        channel_id = extract_channel_id(channel_url)
        
        if not channel_id:
            logger.error(f"Could not get channel ID from URL: {channel_url}")
            return []

        logger.info(f"Found channel ID: {channel_id}")
        
        try:
            request = youtube.search().list(
                part="id,snippet",
                channelId=channel_id,
                type="video",
                order="date",
                maxResults=50
            )
            response = request.execute()

            videos = []
            for item in response.get('items', []):
                video_data = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt']
                }
                logger.info(f"Found video: {video_data['title']}")
                videos.append(video_data)
            
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e.resp.status} - {e.content}")
            return []
            
    except Exception as e:
        logger.error(f"Error getting channel videos: {str(e)}")
        return []

def test_api_key():
    """Test if the YouTube API key is valid"""
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
    """Initialize the YouTube API and test the connection"""
    if test_api_key():
        logger.info("YouTube API initialized successfully")
        return True
    else:
        logger.error("Failed to initialize YouTube API")
        return False
