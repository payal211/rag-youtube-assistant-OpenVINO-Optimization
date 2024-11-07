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
            logger.info(f"Using database path: {self.db_path}")
            
            # Initialize database
            self._initialize_database()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _initialize_database(self):
        """Initialize database with proper settings"""
        try:
            # Connect with WAL mode and proper timeout
            self.conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False
            )
            
            # Enable WAL mode and other optimizations
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA synchronous=NORMAL')
            self.conn.execute('PRAGMA temp_store=MEMORY')
            self.conn.execute('PRAGMA mmap_size=30000000000')
            self.conn.execute('PRAGMA page_size=4096')
            
            # Initialize tables
            self.create_tables()
            self.update_schema()
            self.migrate_database()
            
            # Ensure WAL files have proper permissions
            self._fix_wal_permissions()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _fix_wal_permissions(self):
        """Fix permissions for WAL and SHM files"""
        try:
            for ext in ['-wal', '-shm']:
                wal_path = f"{self.db_path}{ext}"
                if os.path.exists(wal_path):
                    os.chmod(wal_path, 0o664)
        except Exception as e:
            logger.warning(f"Failed to fix WAL file permissions: {str(e)}")

    def create_tables(self):
        """Create database tables"""
        try:
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist (no dropping)
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
