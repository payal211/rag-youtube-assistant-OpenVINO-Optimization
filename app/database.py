import sqlite3
import os

class DatabaseHandler:
    def __init__(self, db_path='data/sqlite.db'):
        self.db_path = db_path
        self.conn = None
        self.create_tables()
        self.update_schema()

    def create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    youtube_id TEXT UNIQUE,
                    title TEXT,
                    channel_name TEXT,
                    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    query TEXT,
                    feedback INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
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
            conn.commit()

    def update_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check if columns exist, if not, add them
            cursor.execute("PRAGMA table_info(videos)")
            columns = [column[1] for column in cursor.fetchall()]
            
            new_columns = [
                ("upload_date", "TEXT"),
                ("view_count", "INTEGER"),
                ("like_count", "INTEGER"),
                ("comment_count", "INTEGER"),
                ("video_duration", "TEXT")
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}")
            
            conn.commit()

    def add_video(self, video_data):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO videos 
                (youtube_id, title, channel_name, upload_date, view_count, like_count, comment_count, video_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_data['video_id'],
                video_data['title'],
                video_data['author'],
                video_data['upload_date'],
                video_data['view_count'],
                video_data['like_count'],
                video_data['comment_count'],
                video_data['video_duration']
            ))
            conn.commit()
            return cursor.lastrowid

    def add_user_feedback(self, video_id, query, feedback):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_feedback (video_id, query, feedback)
                VALUES (?, ?, ?)
            ''', (video_id, query, feedback))
            conn.commit()

    def add_embedding_model(self, model_name, description):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO embedding_models (model_name, description)
                VALUES (?, ?)
            ''', (model_name, description))
            conn.commit()
            return cursor.lastrowid

    def add_elasticsearch_index(self, video_id, index_name, embedding_model_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO elasticsearch_indices (video_id, index_name, embedding_model_id)
                VALUES (?, ?, ?)
            ''', (video_id, index_name, embedding_model_id))
            conn.commit()

    def get_video_by_youtube_id(self, youtube_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM videos WHERE youtube_id = ?', (youtube_id,))
            return cursor.fetchone()

    def get_elasticsearch_index(self, video_id, embedding_model):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ei.index_name 
                FROM elasticsearch_indices ei
                JOIN embedding_models em ON ei.embedding_model_id = em.id
                JOIN videos v ON ei.video_id = v.id
                WHERE v.youtube_id = ? AND em.model_name = ?
            ''', (video_id, embedding_model))
            result = cursor.fetchone()
            return result[0] if result else None
        
    def get_all_videos(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT youtube_id, title, channel_name, upload_date
                FROM videos
                ORDER BY upload_date DESC
            ''')
            return cursor.fetchall()

    def get_elasticsearch_index_by_youtube_id(self, youtube_id, embedding_model):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ei.index_name 
                FROM elasticsearch_indices ei
                JOIN embedding_models em ON ei.embedding_model_id = em.id
                JOIN videos v ON ei.video_id = v.id
                WHERE v.youtube_id = ? AND em.model_name = ?
            ''', (youtube_id, embedding_model))
            result = cursor.fetchone()
            return result[0] if result else None
        
    def get_transcript_content(self, youtube_id):
        # This method assumes you're storing the transcript content in the database
        # If you're not, you'll need to modify this to retrieve the transcript from wherever it's stored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT transcript_content
                FROM videos
                WHERE youtube_id = ?
            ''', (youtube_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    # If you're not already storing the transcript content, you'll need to add a method to do so:
    def add_transcript_content(self, youtube_id, transcript_content):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE videos
                SET transcript_content = ?
                WHERE youtube_id = ?
            ''', (transcript_content, youtube_id))
            conn.commit()
            
    def get_elasticsearch_index_by_youtube_id(self, youtube_id, embedding_model):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ei.index_name 
                FROM elasticsearch_indices ei
                JOIN embedding_models em ON ei.embedding_model_id = em.id
                JOIN videos v ON ei.video_id = v.id
                WHERE v.youtube_id = ? AND em.model_name = ?
            ''', (youtube_id, embedding_model))
            result = cursor.fetchone()
            return result[0] if result else None