from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import os

# Replace with your actual API key
API_KEY = os.environ.get('YOUTUBE_API_KEY', 'YOUR_API_KEY_HERE')

youtube = build('youtube', 'v3', developerKey=API_KEY)

def extract_video_id(url):
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
                'view_count': video['statistics']['viewCount'],
                'like_count': video['statistics'].get('likeCount', 'N/A'),
                'comment_count': video['statistics'].get('commentCount', 'N/A'),
                'duration': video['contentDetails']['duration']
            }
        else:
            return None
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        metadata = get_video_metadata(video_id)
        return {
            'transcript': transcript,
            'metadata': metadata
        }
    except Exception as e:
        print(f"Error extracting transcript for video {video_id}: {str(e)}")
        return None

def get_channel_videos(channel_id):
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

def process_videos(video_ids):
    transcripts = {}
    for video_id in video_ids:
        transcript_data = get_transcript(video_id)
        if transcript_data:
            transcripts[video_id] = transcript_data
    return transcripts