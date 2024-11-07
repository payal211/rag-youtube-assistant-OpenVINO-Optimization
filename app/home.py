import streamlit as st
import os
import sys
import logging
from pathlib import Path
from transcript_extractor import test_api_key, initialize_youtube_api

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

def setup_logging():
    """Set up logging with proper error handling and fallback"""
    try:
        # Get log directory from environment or use default
        log_dir = os.getenv('LOG_DIR', '/app/logs')
        log_file = os.path.join(log_dir, 'app.log')

        # Ensure log directory exists with proper permissions
        Path(log_dir).mkdir(parents=True, mode=0o775, exist_ok=True)

        # Create log file if it doesn't exist
        if not os.path.exists(log_file):
            Path(log_file).touch(mode=0o664)
            
        # Ensure the log file is writable
        if not os.access(log_file, os.W_OK):
            os.chmod(log_file, 0o664)

        # Configure logging with both file and console handlers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Override any existing configuration
        )

        # Create logger
        logger = logging.getLogger(__name__)
        
        # Remove any existing handlers
        logger.handlers = []

        # Create handlers
        file_handler = logging.FileHandler(log_file, mode='a')
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set formatter for handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set level
        logger.setLevel(logging.INFO)

        logger.info("Logging initialized successfully")
        return logger

    except Exception as e:
        # Fallback to console-only logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to initialize file logging: {str(e)}")
        logger.info("Falling back to console-only logging")
        return logger

# Initialize logging
logger = setup_logging()

def main():
    try:
        st.title("YouTube Transcript RAG System 🎥")
        st.write("Welcome to the YouTube Transcript RAG System!")
        
        # Check API key
        if not test_api_key():
            logger.warning("YouTube API key is invalid or not set")
            st.error("YouTube API key is invalid or not set. Please check your configuration.")
            new_api_key = st.text_input("Enter your YouTube API key:")
            if new_api_key:
                os.environ['YOUTUBE_API_KEY'] = new_api_key
                if test_api_key():
                    logger.info("API key validated successfully")
                    st.success("API key validated successfully!")
                    st.experimental_rerun()
                else:
                    logger.error("Invalid API key provided")
                    st.error("Invalid API key. Please try again.")
            return
        
        logger.info("System initialized successfully")
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
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
