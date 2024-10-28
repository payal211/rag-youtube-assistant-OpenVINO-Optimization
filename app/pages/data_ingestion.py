import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="01_Data_Ingestion",  # Use this format for ordering
    page_icon="📥",
    layout="wide"
)

import pandas as pd
from transcript_extractor import get_transcript, extract_video_id, get_channel_videos
from database import DatabaseHandler
from data_processor import DataProcessor
from utils import process_single_video
import logging

logger = logging.getLogger(__name__)

@st.cache_resource
def init_components():
    return DatabaseHandler(), DataProcessor()

def process_multiple_videos(db_handler, data_processor, video_ids, embedding_model):
    progress_bar = st.progress(0)
    processed = 0
    total = len(video_ids)
    
    for video_id in video_ids:
        if process_single_video(db_handler, data_processor, video_id, embedding_model):
            processed += 1
        progress_bar.progress(processed / total)
    
    st.success(f"Processed {processed} out of {total} videos")

def main():
    st.title("Data Ingestion 📥")
    
    db_handler, data_processor = init_components()
    
    # Model selection
    embedding_model = st.selectbox(
        "Select embedding model:",
        ["multi-qa-MiniLM-L6-cos-v1", "all-mpnet-base-v2"]
    )
    
    # Display existing videos
    st.header("Processed Videos")
    videos = db_handler.get_all_videos()
    if videos:
        video_df = pd.DataFrame(videos, columns=['youtube_id', 'title', 'channel_name', 'upload_date'])
        channels = sorted(video_df['channel_name'].unique())
        
        selected_channel = st.selectbox("Filter by Channel", ["All"] + channels)
        if selected_channel != "All":
            video_df = video_df[video_df['channel_name'] == selected_channel]
        
        st.dataframe(video_df)
    else:
        st.info("No videos processed yet. Use the form below to add videos.")
    
    # Process new videos
    st.header("Process New Video")
    with st.form("process_video_form"):
        input_type = st.radio("Select input type:", ["Video URL", "Channel URL", "YouTube ID"])
        input_value = st.text_input("Enter the URL or ID:")
        submit_button = st.form_submit_button("Process")
        
        if submit_button:
            data_processor.set_embedding_model(embedding_model)
            
            with st.spinner("Processing..."):
                if input_type == "Video URL":
                    video_id = extract_video_id(input_value)
                    if video_id:
                        process_single_video(db_handler, data_processor, video_id, embedding_model)
                
                elif input_type == "Channel URL":
                    channel_videos = get_channel_videos(input_value)
                    if channel_videos:
                        video_ids = [video['video_id'] for video in channel_videos]
                        process_multiple_videos(db_handler, data_processor, video_ids, embedding_model)
                    else:
                        st.error("Failed to retrieve videos from the channel")
                
                else:  # YouTube ID
                    process_single_video(db_handler, data_processor, input_value, embedding_model)

def process_single_video(db_handler, data_processor, video_id, embedding_model):
    try:
        existing_index = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
        if existing_index:
            st.info(f"Video {video_id} already processed. Using existing index.")
            return existing_index
        
        transcript_data = get_transcript(video_id)
        if not transcript_data:
            st.error("Failed to retrieve transcript.")
            return None
            
        # Process transcript and create indices
        processed_data = data_processor.process_transcript(video_id, transcript_data)
        if not processed_data:
            st.error("Failed to process transcript.")
            return None
            
        # Save to database and create index
        video_data = {
            'video_id': video_id,
            'title': transcript_data['metadata'].get('title', 'Unknown'),
            'author': transcript_data['metadata'].get('author', 'Unknown'),
            'upload_date': transcript_data['metadata'].get('upload_date', ''),
            'view_count': transcript_data['metadata'].get('view_count', 0),
            'like_count': transcript_data['metadata'].get('like_count', 0),
            'comment_count': transcript_data['metadata'].get('comment_count', 0),
            'video_duration': transcript_data['metadata'].get('duration', ''),
            'transcript_content': processed_data['content']
        }
        
        db_handler.add_video(video_data)
        
        index_name = f"video_{video_id}_{embedding_model}".lower()
        index_name = data_processor.build_index(index_name)
        
        if index_name:
            st.success(f"Successfully processed video: {video_data['title']}")
            return index_name
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        logger.error(f"Error processing video {video_id}: {str(e)}")
        return None

def process_multiple_videos(db_handler, data_processor, video_ids, embedding_model):
    progress_bar = st.progress(0)
    processed = 0
    total = len(video_ids)
    
    for video_id in video_ids:
        if process_single_video(db_handler, data_processor, video_id, embedding_model):
            processed += 1
        progress_bar.progress(processed / total)
    
    st.success(f"Processed {processed} out of {total} videos")

if __name__ == "__main__":
    main()