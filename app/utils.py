import streamlit as st
from transcript_extractor import get_transcript
import logging

logger = logging.getLogger(__name__)

def process_single_video(db_handler, data_processor, video_id, embedding_model):
    """Process a single video for indexing"""
    try:
        # Check for existing index
        existing_index = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
        if existing_index:
            logger.info(f"Video {video_id} already processed. Using existing index.")
            return existing_index
        
        # Get transcript data
        transcript_data = get_transcript(video_id)
        if not transcript_data:
            logger.error(f"Failed to retrieve transcript for video {video_id}")
            return None

        # Process transcript
        processed_data = data_processor.process_transcript(video_id, transcript_data)
        if not processed_data:
            logger.error(f"Failed to process transcript for video {video_id}")
            return None

        # Prepare video data
        video_data = {
            'video_id': video_id,
            'title': transcript_data['metadata'].get('title', 'Unknown Title'),
            'author': transcript_data['metadata'].get('author', 'Unknown Author'),
            'upload_date': transcript_data['metadata'].get('upload_date', 'Unknown Date'),
            'view_count': int(transcript_data['metadata'].get('view_count', 0)),
            'like_count': int(transcript_data['metadata'].get('like_count', 0)),
            'comment_count': int(transcript_data['metadata'].get('comment_count', 0)),
            'video_duration': transcript_data['metadata'].get('duration', 'Unknown Duration'),
            'transcript_content': processed_data['content']
        }

        # Save to database
        db_handler.add_video(video_data)

        # Build index
        index_name = f"video_{video_id}_{embedding_model}".lower()
        index_name = data_processor.build_index(index_name)
        
        if index_name:
            # Save index information
            embedding_model_id = db_handler.add_embedding_model(embedding_model, "Description of the model")
            video_record = db_handler.get_video_by_youtube_id(video_id)
            if video_record:
                db_handler.add_elasticsearch_index(video_record[0], index_name, embedding_model_id)
                logger.info(f"Successfully processed video: {video_data['title']}")
                return index_name

        logger.error(f"Failed to process video {video_id}")
        return None

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        return None