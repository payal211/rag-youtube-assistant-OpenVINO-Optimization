import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="02_Chat_Interface",  # Use this format for ordering
    page_icon="💬",
    layout="wide"
)

# Rest of the imports
import pandas as pd
import logging
import sqlite3
from datetime import datetime
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports
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
        data_processor = DataProcessor()
        rag_system = RAGSystem(data_processor)
        query_rewriter = QueryRewriter()
        return db_handler, data_processor, rag_system, query_rewriter
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None

def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()

def create_chat_interface(db_handler, rag_system, video_id, index_name, rewrite_method, search_method):
    """Create the chat interface with feedback functionality"""
    # Load chat history if video changed
    if st.session_state.current_video_id != video_id:
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
            
            message_key = f"{message['id']}"
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

def get_system_status(db_handler, selected_video_id=None):
    """Get system status information"""
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
            
            if selected_video_id:
                # Get video details
                cursor.execute("""
                    SELECT v.id, v.title, v.channel_name, v.processed_date,
                           ei.index_name, em.model_name
                    FROM videos v
                    LEFT JOIN elasticsearch_indices ei ON v.id = ei.video_id
                    LEFT JOIN embedding_models em ON ei.embedding_model_id = em.id
                    WHERE v.youtube_id = ?
                """, (selected_video_id,))
                video_details = cursor.fetchall()
            else:
                video_details = None
            
            return {
                "total_videos": total_videos,
                "total_indices": total_indices,
                "models": models,
                "video_details": video_details
            }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return None

def display_system_status(status, selected_video_id=None):
    """Display system status in the sidebar"""
    if not status:
        st.sidebar.error("Unable to fetch system status")
        return

    st.sidebar.header("System Status")
    
    # Display general stats
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Videos", status["total_videos"])
    with col2:
        st.metric("Total Indices", status["total_indices"])
    
    st.sidebar.markdown("**Available Models:**")
    for model in status["models"]:
        st.sidebar.markdown(f"- {model}")

    # Display selected video details
    if selected_video_id and status["video_details"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Selected Video Details:**")
        for details in status["video_details"]:
            video_id, title, channel, processed_date, index_name, model = details
            st.sidebar.markdown(f"""
            - **Title:** {title}
            - **Channel:** {channel}
            - **Processed:** {processed_date}
            - **Index:** {index_name or 'Not indexed'}
            - **Model:** {model or 'N/A'}
            """)

def main():
    st.title("Chat Interface 💬")
    
    # Initialize components
    components = init_components()
    if not components:
        st.error("Failed to initialize components. Please check the logs.")
        return
    
    db_handler, data_processor, rag_system, query_rewriter = components
    
    # Initialize session state
    init_session_state()
    
    # Get system status
    system_status = get_system_status(db_handler)
    
    # Video selection
    st.sidebar.header("Video Selection")
    
    # Get available videos with indices
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
    
    if df.empty:
        st.info("No videos available. Please process some videos in the Data Ingestion page first.")
        display_system_status(system_status)
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
        # Update system status with selected video
        system_status = get_system_status(db_handler, selected_video_id)
        display_system_status(system_status, selected_video_id)
        
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

    # Display system status
    display_system_status(system_status, selected_video_id)

if __name__ == "__main__":
    main()