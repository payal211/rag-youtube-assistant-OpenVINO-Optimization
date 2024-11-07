import streamlit as st
import pandas as pd
import logging
import sqlite3
from datetime import datetime
import sys
import os

from database import DatabaseHandler
from data_processor import DataProcessor
from rag import RAGSystem
from query_rewriter import QueryRewriter
from utils import process_single_video

# Set up logging
logger = logging.getLogger(__name__)

@st.cache_resource
def init_components():
    """Initialize system components"""
    try:
        db_handler = DatabaseHandler()
        if not hasattr(db_handler, 'db_path') or db_handler.db_path is None:
            st.error("Failed to initialize database handler - database path not set")
            return None, None, None, None
            
        data_processor = DataProcessor()
        rag_system = RAGSystem(data_processor)
        query_rewriter = QueryRewriter()
        return db_handler, data_processor, rag_system, query_rewriter
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None

def get_system_status(db_handler):
    """Get system status information"""
    if db_handler is None or not hasattr(db_handler, 'db_path'):
        logger.error("Invalid database handler")
        return None
        
    try:
        with sqlite3.connect(db_handler.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total videos
            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]
            
            # Get total indices
            cursor.execute("SELECT COUNT(DISTINCT index_name) FROM elasticsearch_indices")
            total_indices = cursor.fetchone()[0]
            
            # Get available embedding models
            cursor.execute("SELECT model_name FROM embedding_models")
            models = [row[0] for row in cursor.fetchall()]
            
            return {
                "total_videos": total_videos,
                "total_indices": total_indices,
                "models": models
            }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return None

def create_chat_interface(db_handler, rag_system, video_id, index_name, rewrite_method, search_method):
    """Create the chat interface with feedback functionality"""
    if not db_handler or not hasattr(db_handler, 'db_path'):
        st.error("Database handler not properly initialized")
        return

    try:
        # Load chat history if video changed
        if 'current_video_id' not in st.session_state or st.session_state.current_video_id != video_id:
            st.session_state.chat_history = []
            db_history = db_handler.get_chat_history(video_id)
            for chat_id, user_msg, asst_msg, timestamp in db_history:
                st.session_state.chat_history.append({
                    'id': chat_id,
                    'user': user_msg,
                    'assistant': asst_msg,
                    'timestamp': timestamp
                })
            st.session_state.current_video_id = video_id

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(message['user'])
            
            with st.chat_message("assistant"):
                st.markdown(message['assistant'])
                
                # Add feedback buttons
                message_key = f"{message['id']}"
                if 'feedback_given' not in st.session_state:
                    st.session_state.feedback_given = set()
                    
                if message_key not in st.session_state.feedback_given:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍", key=f"like_{message_key}"):

                            db_handler.add_user_feedback(
                                video_id=video_id,
                                chat_id=message['id'],
                                query=message['user'],
                                response=message['assistant'],
                                feedback=1
                            )
                            st.session_state.feedback_given.add(message_key)
                            st.success("Thank you for your positive feedback!")
                            st.rerun()
                    
                    with col2:
                        if st.button("👎", key=f"dislike_{message_key}"):

                            db_handler.add_user_feedback(
                                video_id=video_id,
                                chat_id=message['id'],
                                query=message['user'],
                                response=message['assistant'],
                                feedback=-1
                            )
                            st.session_state.feedback_given.add(message_key)
                            st.success("Thank you for your feedback. We'll work to improve.")
                            st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask a question about the video..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Apply query rewriting if selected
                        rewritten_query = prompt
                        if rewrite_method == "Chain of Thought":
                            rewritten_query, _ = rag_system.rewrite_cot(prompt)
                            st.caption("Rewritten query: " + rewritten_query)
                        elif rewrite_method == "ReAct":
                            rewritten_query, _ = rag_system.rewrite_react(prompt)
                            st.caption("Rewritten query: " + rewritten_query)
                        
                        # Get response using selected search method
                        search_method_map = {
                            "Hybrid": "hybrid",
                            "Text-only": "text",
                            "Embedding-only": "embedding"
                        }
                        
                        response, _ = rag_system.query(
                            rewritten_query,
                            search_method=search_method_map[search_method],
                            index_name=index_name
                        )
                        
                        st.markdown(response)
                        
                        # Save to database and session state
                        chat_id = db_handler.add_chat_message(video_id, prompt, response)
                        st.session_state.chat_history.append({
                            'id': chat_id,
                            'user': prompt,
                            'assistant': response,
                            'timestamp': datetime.now()
                        })

                        # Add feedback buttons for new message
                        message_key = f"{chat_id}"
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍", key=f"like_{message_key}"):

                                db_handler.add_user_feedback(
                                    video_id=video_id,
                                    chat_id=chat_id,
                                    query=prompt,
                                    response=response,
                                    feedback=1
                                )
                                st.session_state.feedback_given.add(message_key)
                                st.success("Thank you for your positive feedback!")
                                st.rerun()
                        with col2:
                            if st.button("👎", key=f"dislike_{message_key}"):

                                db_handler.add_user_feedback(
                                    video_id=video_id,
                                    chat_id=chat_id,
                                    query=prompt,
                                    response=response,
                                    feedback=-1
                                )
                                st.session_state.feedback_given.add(message_key)
                                st.success("Thank you for your feedback. We'll work to improve.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        logger.error(f"Error in chat interface: {str(e)}")

    except Exception as e:  # <-- This is your outer try-except block
        st.error(f"Error in create_chat_interface: {str(e)}")
        logger.error(f"Error in create_chat_interface: {str(e)}")

def main():
    st.title("Chat Interface 💬")
    
    # Initialize components
    components = init_components()
    if not components or None in components:
        st.error("Failed to initialize components. Please check the logs.")
        return
    
    db_handler, data_processor, rag_system, query_rewriter = components
    
    try:  # Start of try block
        # Get system status
        system_status = get_system_status(db_handler)
        
        if not system_status:
            st.warning("System status unavailable. Some features may be limited.")
            system_status = {"total_videos": 0, "total_indices": 0, "models": []}
            
        # Video selection
        st.sidebar.header("Video Selection")
        
        # Get available videos
        try:  # Nested try block
            with sqlite3.connect(db_handler.db_path) as conn:
                query = """
                    SELECT DISTINCT v.youtube_id, v.title, v.channel_name, v.upload_date, 
                           GROUP_CONCAT(ei.index_name) as indices
                    FROM videos v
                    LEFT JOIN elasticsearch_indices ei ON v.id = ei.video_id
                    GROUP BY v.youtube_id
                    ORDER BY v.upload_date DESC
                """
                df = pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching videos: {str(e)}")
            st.error("Failed to fetch available videos")
            return

        if df.empty:
            st.info("No videos available. Please process some videos in the Data Ingestion page first.")
            return
        
        # Display available videos
        st.sidebar.markdown(f"**Available Videos:** {len(df)}")
        
        # Channel filter
        channels = sorted(df['channel_name'].unique())
        selected_channel = st.sidebar.selectbox(
            "Filter by Channel",
            ["All"] + channels,
            key="channel_filter"
        )
        
        filtered_df = df if selected_channel == "All" else df[df['channel_name'] == selected_channel]
        
        # Video selection
        selected_video_id = st.sidebar.selectbox(
            "Select a Video",
            filtered_df['youtube_id'].tolist(),
            format_func=lambda x: filtered_df[filtered_df['youtube_id'] == x]['title'].iloc[0],
            key="video_select"
        )

        if selected_video_id:
            # Get the index for the selected video
            index_name = db_handler.get_elasticsearch_index_by_youtube_id(selected_video_id)
            
            if not index_name:
                st.warning("This video hasn't been indexed yet. You can process it in the Data Ingestion page.")
                if st.button("Process Now"):
                    with st.spinner("Processing video..."):
                        try:
                            embedding_model = data_processor.embedding_model.__class__.__name__
                            index_name = process_single_video(db_handler, data_processor, selected_video_id, embedding_model)
                            if index_name:
                                st.success("Video processed successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error processing video: {str(e)}")
                            logger.error(f"Error processing video: {str(e)}")
            else:
                # Chat settings
                st.sidebar.header("Chat Settings")
                rewrite_method = st.sidebar.radio(
                    "Query Rewriting Method",
                    ["None", "Chain of Thought", "ReAct"],
                    key="rewrite_method"
                )
                search_method = st.sidebar.radio(
                    "Search Method",
                    ["Hybrid", "Text-only", "Embedding-only"],
                    key="search_method"
                )
                
                # Create chat interface
                create_chat_interface(
                    db_handler,
                    rag_system,
                    selected_video_id,
                    index_name,
                    rewrite_method,
                    search_method
                )
                
    except Exception as e:  # <-- Add except block
        logger.error(f"Error in main function: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
