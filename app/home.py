import streamlit as st
import os
import sys
import logging
from transcript_extractor import test_api_key, initialize_youtube_api

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# Commented the original logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# Updated logging setup with directory creation
log_dir = '/app/logs'  # Specify the directory where log file will be stored
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create the directory if it doesn't exist

log_file = os.path.join(log_dir, 'app.log')  # Absolute path to the log file

# Configure logging with FileHandler for log file and StreamHandler for stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Use the log file in the specified directory
        logging.StreamHandler(sys.stdout)  # Also print logs to stdout (terminal)
    ]
)
logger = logging.getLogger(__name__)  # Logger instance

def main():
    st.title("YouTube Transcript RAG System 🎥")
    st.write("Welcome to the YouTube Transcript RAG System!")
    
    # Check API key
    if not test_api_key():
        st.error("YouTube API key is invalid or not set. Please check your configuration.")
        new_api_key = st.text_input("Enter your YouTube API key:")
        if new_api_key:
            os.environ['YOUTUBE_API_KEY'] = new_api_key
            if test_api_key():
                st.success("API key validated successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid API key. Please try again.")
        return
    
    st.success("System is ready! Please use the sidebar to navigate between different functions.")
    
    # Display system overview
    st.header("System Overview")
    st.write("""
    This system provides the following functionality:
    
    1. **Data Ingestion** 📥
       - Process YouTube videos and transcripts
       - Support for single videos or entire channels
       
    2. **Chat Interface** 💬
       - Interactive chat with processed videos
       - Multiple query rewriting methods
       - Various search strategies
       
    3. **Ground Truth Generation** 📝
       - Generate and manage ground truth questions
       - Export ground truth data
       
    4. **RAG Evaluation** 📊
       - Evaluate system performance
       - View detailed metrics and analytics
    """)

if __name__ == "__main__":
    main()
