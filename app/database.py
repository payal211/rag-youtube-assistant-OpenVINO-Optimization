import sqlite3
import os
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self):
        try:
            # Get database path from environment or use default
            self.db_path = os.getenv('SQLITE_DATABASE_PATH', '/app/data/sqlite.db')
            self.db_dir = os.path.dirname(self.db_path)
            logger.info(f"Using database path: {self.db_path}")

            # Ensure directory exists with proper permissions
            os.makedirs(self.db_dir, mode=0o777, exist_ok=True)
            
            # Set directory permissions
            try:
                if os.path.exists(self.db_dir):
                    os.chmod(self.db_dir, 0o777)
            except Exception as e:
                logger.warning(f"Could not set directory permissions: {str(e)}")

            # Initialize database
            self._initialize_database()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self._log_permissions()
            raise

    def _initialize_database(self):
        """Initialize database with proper settings"""
        try:
            # If database doesn't exist, create it with proper permissions
            if not os.path.exists(self.db_path):
                Path(self.db_path).touch(mode=0o666)

            # Try to set permissions on existing database
            try:
                os.chmod(self.db_path, 0o666)
            except Exception as e:
                logger.warning(f"Could not set database file permissions: {str(e)}")

            # Connect to database
            self.conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False
            )
            
            # Enable optimizations
            try:
                self.conn.execute('PRAGMA journal_mode=WAL')
                self.conn.execute('PRAGMA synchronous=NORMAL')
                self.conn.execute('PRAGMA temp_store=MEMORY')
                self.conn.execute('PRAGMA mmap_size=30000000000')
                self.conn.execute('PRAGMA page_size=4096')
            except Exception as e:
                logger.warning(f"Could not set PRAGMA settings: {str(e)}")

            # Initialize tables
            self.create_tables()
            self.update_schema()
            self.migrate_database()
            
            # Fix WAL file permissions
            self._fix_wal_permissions()
            
            logger.info("Database initialized successfully")
            
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite operational error: {str(e)}")
            self._log_permissions()
            raise
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            self._log_permissions()
            raise

    def _fix_wal_permissions(self):
        """Fix permissions for WAL and SHM files"""
        try:
            for ext in ['-wal', '-shm']:
                wal_path = f"{self.db_path}{ext}"
                if os.path.exists(wal_path):
                    try:
                        os.chmod(wal_path, 0o666)
                    except Exception as e:
                        logger.warning(f"Could not set permissions on {wal_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to fix WAL file permissions: {str(e)}")

    def _log_permissions(self):
        """Log current permissions for debugging"""
        try:
            logger.info(f"Database directory permissions: {oct(os.stat(self.db_dir).st_mode)}")
            if os.path.exists(self.db_path):
                logger.info(f"Database file permissions: {oct(os.stat(self.db_path).st_mode)}")
            logger.info(f"Current user/group: {os.getuid()}:{os.getgid()}")
        except Exception as e:
            logger.warning(f"Could not log permissions: {str(e)}")

    def create_tables(self):
        """Create database tables"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    youtube_id TEXT UNIQUE,
                    title TEXT,
                    channel_name TEXT,
                    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    upload_date TEXT,
                    view_count INTEGER,
                    like_count INTEGER,
                    comment_count INTEGER,
                    video_duration TEXT,
                    transcript_content TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    chat_id INTEGER,
                    query TEXT,
                    response TEXT,
                    feedback INTEGER CHECK (feedback IN (-1, 1)),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id),
                    FOREIGN KEY (chat_id) REFERENCES chat_history (id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embedding_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE,
                    description TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS elasticsearch_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    index_name TEXT,
                    embedding_model_id INTEGER,
                    FOREIGN KEY (video_id) REFERENCES videos (id),
                    FOREIGN KEY (embedding_model_id) REFERENCES embedding_models (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ground_truth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    question TEXT,
                    generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, question),
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    hit_rate REAL,
                    mrr REAL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    parameter_name TEXT,
                    parameter_value REAL,
                    score REAL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    question TEXT,
                    answer TEXT,
                    relevance TEXT,
                    explanation TEXT,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            ''')

            # Create indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_id ON videos(youtube_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_video ON user_feedback(video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_video ON chat_history(video_id)')
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise

    def update_schema(self):
        """Update schema with proper error handling"""
        try:
            cursor = self.conn.cursor()
            
            # Check and update videos table
            cursor.execute("PRAGMA table_info(videos)")
            columns = [column[1] for column in cursor.fetchall()]
            
            new_columns = [
                ("upload_date", "TEXT"),
                ("view_count", "INTEGER"),
                ("like_count", "INTEGER"),
                ("comment_count", "INTEGER"),
                ("video_duration", "TEXT"),
                ("transcript_content", "TEXT")
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}")
                    
        except Exception as e:
            logger.error(f"Error updating schema: {str(e)}")
            raise

    def migrate_database(self):
        """Migrate database with proper error handling"""
        try:
            cursor = self.conn.cursor()
            
            # Check if chat_id column exists in user_feedback
            cursor.execute("PRAGMA table_info(user_feedback)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'chat_id' not in columns:
                logger.info("Migrating user_feedback table")
                
                cursor.execute('''
                    CREATE TABLE user_feedback_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT,
                        query TEXT,
                        response TEXT,
                        feedback INTEGER CHECK (feedback IN (-1, 1)),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        chat_id INTEGER,
                        FOREIGN KEY (video_id) REFERENCES videos (youtube_id),
                        FOREIGN KEY (chat_id) REFERENCES chat_history (id)
                    )
                ''')
                
                cursor.execute('''
                    INSERT INTO user_feedback_new (video_id, query, response, feedback, timestamp)
                    SELECT video_id, query, response, feedback, timestamp
                    FROM user_feedback
                ''')
                
                cursor.execute('DROP TABLE user_feedback')
                cursor.execute('ALTER TABLE user_feedback_new RENAME TO user_feedback')
                
                logger.info("Migration completed successfully")
                
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            raise

    def add_video(self, video_data):
        """Add a video to the database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO videos 
                (youtube_id, title, channel_name, upload_date, view_count, like_count, 
                 comment_count, video_duration, transcript_content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_data['video_id'],
                video_data['title'],
                video_data['author'],
                video_data['upload_date'],
                video_data['view_count'],
                video_data['like_count'],
                video_data['comment_count'],
                video_data['video_duration'],
                video_data['transcript_content']
            ))
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding video: {str(e)}")
            raise

    def get_video_by_youtube_id(self, youtube_id):
        """Get video by YouTube ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM videos WHERE youtube_id = ?', (youtube_id,))
        return cursor.fetchone()

    def get_all_videos(self):
        """Get all videos"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT youtube_id, title, channel_name, upload_date
            FROM videos
            ORDER BY upload_date DESC
        ''')
        return cursor.fetchall()

    def add_chat_message(self, video_id, user_message, assistant_message):
        """Add a chat message"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (video_id, user_message, assistant_message)
            VALUES (?, ?, ?)
        ''', (video_id, user_message, assistant_message))
        return cursor.lastrowid

    def get_chat_history(self, video_id):
        """Get chat history for a video"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, user_message, assistant_message, timestamp
            FROM chat_history
            WHERE video_id = ?
            ORDER BY timestamp ASC
        ''', (video_id,))
        return cursor.fetchall()

    def add_user_feedback(self, video_id, chat_id, query, response, feedback):
        """Add user feedback"""
        try:
            cursor = self.conn.cursor()
            
            # Verify video exists
            cursor.execute('SELECT id FROM videos WHERE youtube_id = ?', (video_id,))
            if not cursor.fetchone():
                logger.error(f"Video {video_id} not found")
                raise ValueError(f"Video {video_id} not found")

            # Verify chat message exists if chat_id provided
            if chat_id:
                cursor.execute('SELECT id FROM chat_history WHERE id = ?', (chat_id,))
                if not cursor.fetchone():
                    logger.error(f"Chat message {chat_id} not found")
                    raise ValueError(f"Chat message {chat_id} not found")

            # Insert feedback
            cursor.execute('''
                INSERT INTO user_feedback 
                (video_id, chat_id, query, response, feedback)
                VALUES (?, ?, ?, ?, ?)
            ''', (video_id, chat_id, query, response, feedback))
            logger.info(f"Added feedback for video {video_id}, chat {chat_id}")
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            raise

    def get_user_feedback_stats(self, video_id):
        """Get feedback statistics for a video"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(CASE WHEN feedback = 1 THEN 1 END) as positive_feedback,
                    COUNT(CASE WHEN feedback = -1 THEN 1 END) as negative_feedback
                FROM user_feedback
                WHERE video_id = ?
            ''', (video_id,))
            return cursor.fetchone() or (0, 0)
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}")
            return (0, 0)

    def add_embedding_model(self, model_name, description):
        """Add embedding model"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO embedding_models (model_name, description)
            VALUES (?, ?)
        ''', (model_name, description))
        return cursor.lastrowid

    def add_elasticsearch_index(self, video_id, index_name, embedding_model_id):
        """Add Elasticsearch index"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO elasticsearch_indices (video_id, index_name, embedding_model_id)
            VALUES (?, ?, ?)
        ''', (video_id, index_name, embedding_model_id))

    def get_elasticsearch_index(self, video_id, embedding_model):
        """Get Elasticsearch index"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT ei.index_name 
            FROM elasticsearch_indices ei
            JOIN embedding_models em ON ei.embedding_model_id = em.id
            JOIN videos v ON ei.video_id = v.id
            WHERE v.youtube_id = ? AND em.model_name = ?
        ''', (video_id, embedding_model))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_elasticsearch_index_by_youtube_id(self, youtube_id):
        """Get Elasticsearch index by YouTube ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT ei.index_name 
            FROM elasticsearch_indices ei
            JOIN videos v ON ei.video_id = v.id
            WHERE v.youtube_id = ?
        ''', (youtube_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def add_ground_truth_questions(self, video_id, questions):
        """Add ground truth questions"""
        try:
            cursor = self.conn.cursor()
            for question in questions:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO ground_truth (video_id, question)
                        VALUES (?, ?)
                    ''', (video_id, question))
                except sqlite3.IntegrityError:
                    logger.warning(f"Duplicate question for video {video_id}: {question}")
                    continue
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding ground truth questions: {str(e)}")
            raise

    def get_ground_truth_by_video(self, video_id):
        """Get ground truth questions for a video"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT gt.*, v.channel_name
            FROM ground_truth gt
            JOIN videos v ON gt.video_id = v.youtube_id
            WHERE gt.video_id = ?
            ORDER BY gt.generation_date DESC
        ''', (video_id,))
        return cursor.fetchall()

    def get_ground_truth_by_channel(self, channel_name):
        """Get ground truth questions for a channel"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT gt.*, v.channel_name
            FROM ground_truth gt
            JOIN videos v ON gt.video_id = v.youtube_id
            WHERE v.channel_name = ?
            ORDER BY gt.generation_date DESC
        ''', (channel_name,))
        return cursor.fetchall()

    def get_all_ground_truth(self):
        """Get all ground truth questions"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT gt.*, v.channel_name
            FROM ground_truth gt
            JOIN videos v ON gt.video_id = v.youtube_id
            ORDER BY gt.generation_date DESC
        ''')
        return cursor.fetchall()

    def save_search_performance(self, video_id, hit_rate, mrr):
        """Save search performance metrics"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO search_performance (video_id, hit_rate, mrr)
                VALUES (?, ?, ?)
            ''', (video_id, hit_rate, mrr))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving search performance: {str(e)}")
            raise

    def save_search_parameters(self, video_id, parameters, score):
        """Save search parameters"""
        try:
            cursor = self.conn.cursor()
            for param_name, param_value in parameters.items():
                cursor.execute('''
                    INSERT INTO search_parameters 
                    (video_id, parameter_name, parameter_value, score)
                    VALUES (?, ?, ?, ?)
                ''', (video_id, param_name, param_value, score))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving search parameters: {str(e)}")
            raise

    def save_rag_evaluation(self, evaluation_data):
        """Save RAG evaluation results"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO rag_evaluations 
                (video_id, question, answer, relevance, explanation)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                evaluation_data['video_id'],
                evaluation_data['question'],
                evaluation_data['answer'],
                evaluation_data['relevance'],
                evaluation_data['explanation']
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving RAG evaluation: {str(e)}")
            raise

    def get_latest_evaluation_results(self, video_id=None):
        """Get latest evaluation results"""
        cursor = self.conn.cursor()
        if video_id:
            cursor.execute('''
                SELECT * FROM rag_evaluations 
                WHERE video_id = ?
                ORDER BY evaluation_date DESC
            ''', (video_id,))
        else:
            cursor.execute('''
                SELECT * FROM rag_evaluations 
                ORDER BY evaluation_date DESC
            ''')
        return cursor.fetchall()

    def get_latest_search_performance(self, video_id=None):
        """Get latest search performance metrics"""
        cursor = self.conn.cursor()
        if video_id:
            cursor.execute('''
                SELECT * FROM search_performance 
                WHERE video_id = ?
                ORDER BY evaluation_date DESC 
                LIMIT 1
            ''', (video_id,))
        else:
            cursor.execute('''
                SELECT * FROM search_performance 
                ORDER BY evaluation_date DESC
            ''')
        return cursor.fetchall()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def close(self):
        """Close database connection"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                self.conn = None
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")

    def __del__(self):
        """Destructor"""
        self.close()
