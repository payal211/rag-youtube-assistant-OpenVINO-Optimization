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
from datetime import datetime
import isodate

# Enhanced logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.getenv('LOG_DIR', '/app/logs'), 'transcript_extractor.log'))
    ]
)
logger = logging.getLogger(__name__)

def get_youtube_client():
    try:
        # Explicit SSL context configuration
        ssl_context = ssl.create_default_context(cafile=os.getenv('SSL_CERT_FILE'))
        session = requests.Session()
        session.verify = os.getenv('REQUESTS_CA_BUNDLE')
        
        # More detailed error logging for API key
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            logger.error("YouTube API key not found in environment variables")
            raise ValueError("YouTube API key not found")
        
        # Build with custom SSL configuration
        http = googleapiclient.http.build_http()
        http.verify = session.verify
        
        youtube = build('youtube', 'v3', 
                       developerKey=api_key, 
                       http=http,
                       cache_discovery=False)
        
        logger.info("YouTube API client initialized successfully")
        return youtube
    except Exception as e:
        logger.error(f"Error initializing YouTube API client: {str(e)}", exc_info=True)
        raise

def get_transcript(video_id):
    if not video_id:
        logger.error("No video ID provided")
        return None
        
    try:
        # Validate video ID format
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            logger.error(f"Invalid video ID format: {video_id}")
            return None
            
        # Get video metadata with retry logic
        retry_count = 3
        metadata = None
        for attempt in range(retry_count):
            try:
                metadata = get_video_metadata(video_id)
                if metadata:
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to get metadata: {str(e)}")
                if attempt == retry_count - 1:
                    raise
        
        if not metadata:
            logger.error(f"Failed to retrieve metadata for video {video_id}")
            return None

        # Get and parse captions with retry logic
        caption_track = None
        for attempt in range(retry_count):
            try:
                caption_track = get_caption_track(video_id)
                if caption_track:
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to get captions: {str(e)}")
                if attempt == retry_count - 1:
                    raise

        if not caption_track:
            logger.error(f"No caption track available for video {video_id}")
            return None

        transcript = parse_caption_track(caption_track)
        if not transcript:
            logger.error(f"Failed to parse caption track for video {video_id}")
            return None

        logger.info(f"Successfully retrieved transcript for video {video_id}")
        return {
            'transcript': transcript,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Error in get_transcript for video {video_id}: {str(e)}", exc_info=True)
        return None
