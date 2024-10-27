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
            # Existing tables
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
            
            # New tables for ground truth and evaluation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ground_truth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    question TEXT,
                    generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, question)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    hit_rate REAL,
                    mrr REAL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    parameter_name TEXT,
                    parameter_value REAL,
                    score REAL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def update_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
            
            conn.commit()

    def add_video(self, video_data):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO videos 
                (youtube_id, title, channel_name, upload_date, view_count, like_count, comment_count, video_duration, transcript_content)
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

    def get_elasticsearch_index_by_youtube_id(self, youtube_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ei.index_name 
                FROM elasticsearch_indices ei
                JOIN videos v ON ei.video_id = v.id
                WHERE v.youtube_id = ?
            ''', (youtube_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        
    def get_transcript_content(self, youtube_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT transcript_content
                FROM videos
                WHERE youtube_id = ?
            ''', (youtube_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    # This method is no longer needed as transcript is added in add_video
    # def add_transcript_content(self, youtube_id, transcript_content):
    #     with sqlite3.connect(self.db_path) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute('''
    #             UPDATE videos
    #             SET transcript_content = ?
    #             WHERE youtube_id = ?
    #         ''', (transcript_content, youtube_id))
    #         conn.commit()
    
    def add_ground_truth_questions(self, video_id, questions):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for question in questions:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO ground_truth (video_id, question)
                        VALUES (?, ?)
                    ''', (video_id, question))
                except sqlite3.IntegrityError:
                    continue  # Skip duplicate questions
            conn.commit()

    def get_ground_truth_by_video(self, video_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT gt.*, v.channel_name
                FROM ground_truth gt
                JOIN videos v ON gt.video_id = v.youtube_id
                WHERE gt.video_id = ?
                ORDER BY gt.generation_date DESC
            ''', (video_id,))
            return cursor.fetchall()

    def get_ground_truth_by_channel(self, channel_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT gt.*, v.channel_name
                FROM ground_truth gt
                JOIN videos v ON gt.video_id = v.youtube_id
                WHERE v.channel_name = ?
                ORDER BY gt.generation_date DESC
            ''', (channel_name,))
            return cursor.fetchall()

    def get_all_ground_truth(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT gt.*, v.channel_name
                FROM ground_truth gt
                JOIN videos v ON gt.video_id = v.youtube_id
                ORDER BY gt.generation_date DESC
            ''')
            return cursor.fetchall()

    def save_search_performance(self, video_id, hit_rate, mrr):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_performance (video_id, hit_rate, mrr)
                VALUES (?, ?, ?)
            ''', (video_id, hit_rate, mrr))
            conn.commit()

    def save_search_parameters(self, video_id, parameters, score):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for param_name, param_value in parameters.items():
                cursor.execute('''
                    INSERT INTO search_parameters (video_id, parameter_name, parameter_value, score)
                    VALUES (?, ?, ?, ?)
                ''', (video_id, param_name, param_value, score))
            conn.commit()

    def save_rag_evaluation(self, evaluation_data):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
            conn.commit()

    def get_latest_evaluation_results(self, video_id=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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