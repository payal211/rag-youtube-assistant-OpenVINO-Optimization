import sqlite3
import os

class DatabaseHandler:
    def __init__(self, db_path='data/sqlite.db'):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

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

    def add_video(self, youtube_id, title, channel_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO videos (youtube_id, title, channel_name)
                VALUES (?, ?, ?)
            ''', (youtube_id, title, channel_name))
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

    def get_elasticsearch_index(self, video_id, embedding_model_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT index_name FROM elasticsearch_indices
                WHERE video_id = ? AND embedding_model_id = ?
            ''', (video_id, embedding_model_id))
            result = cursor.fetchone()
            return result[0] if result else None