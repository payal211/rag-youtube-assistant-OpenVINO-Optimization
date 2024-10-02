from minsearch import Index
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from elasticsearch import Elasticsearch
import os

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class DataProcessor:
    def __init__(self, text_fields=["content", "title", "description"], 
                 keyword_fields=["video_id", "start_time", "author", "upload_date"], 
                 embedding_model="all-MiniLM-L6-v2"):
        self.text_index = Index(text_fields=text_fields, keyword_fields=keyword_fields)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []
        
        # Use environment variables for Elasticsearch configuration
        elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
        elasticsearch_port = int(os.getenv('ELASTICSEARCH_PORT', 9200))
        
        # Initialize Elasticsearch client with explicit scheme
        self.es = Elasticsearch([f'http://{elasticsearch_host}:{elasticsearch_port}'])

    def process_transcript(self, video_id, transcript_data):
        metadata = transcript_data['metadata']
        transcript = transcript_data['transcript']

        for i, segment in enumerate(transcript):
            cleaned_text = clean_text(segment['text'])
            doc = {
                "video_id": video_id,
                "content": cleaned_text,
                "start_time": segment['start'],
                "duration": segment['duration'],
                "segment_id": f"{video_id}_{i}",
                "title": metadata['title'],
                "author": metadata['author'],
                "upload_date": metadata['upload_date'],
                "view_count": metadata['view_count'],
                "like_count": metadata['like_count'],
                "comment_count": metadata['comment_count'],
                "video_duration": metadata['duration']
            }
            self.documents.append(doc)
            self.embeddings.append(self.embedding_model.encode(cleaned_text + " " + metadata['title']))

    def build_index(self, index_name):
        self.text_index.fit(self.documents)
        self.embeddings = np.array(self.embeddings)
        
        # Create Elasticsearch index
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body={
                "mappings": {
                    "properties": {
                        "embedding": {"type": "dense_vector", "dims": self.embeddings.shape[1]},
                        "content": {"type": "text"},
                        "video_id": {"type": "keyword"},
                        "segment_id": {"type": "keyword"},
                        "start_time": {"type": "float"},
                        "duration": {"type": "float"},
                        "title": {"type": "text"},
                        "author": {"type": "keyword"},
                        "upload_date": {"type": "date"},
                        "view_count": {"type": "integer"},
                        "like_count": {"type": "integer"},
                        "comment_count": {"type": "integer"},
                        "video_duration": {"type": "text"}
                    }
                }
            })

        # Index documents in Elasticsearch
        for doc, embedding in zip(self.documents, self.embeddings):
            doc['embedding'] = embedding.tolist()
            self.es.index(index=index_name, body=doc, id=doc['segment_id'])

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10, method='hybrid', index_name=None):
        if method == 'text':
            return self.text_search(query, filter_dict, boost_dict, num_results)
        elif method == 'embedding':
            return self.embedding_search(query, num_results, index_name)
        else:  # hybrid search
            text_results = self.text_search(query, filter_dict, boost_dict, num_results)
            embedding_results = self.embedding_search(query, num_results, index_name)
            return self.combine_results(text_results, embedding_results, num_results)

    def text_search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        return self.text_index.search(query, filter_dict, boost_dict, num_results)

    def embedding_search(self, query, num_results=10, index_name=None):
        if index_name:
            # Use Elasticsearch for embedding search
            query_vector = self.embedding_model.encode(query).tolist()
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
            response = self.es.search(
                index=index_name,
                body={
                    "size": num_results,
                    "query": script_query,
                    "_source": {"excludes": ["embedding"]}
                }
            )
            return [hit['_source'] for hit in response['hits']['hits']]
        else:
            # Use in-memory embedding search
            query_embedding = self.embedding_model.encode(query)
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:num_results]
            return [self.documents[i] for i in top_indices]

    def combine_results(self, text_results, embedding_results, num_results):
        combined = []
        for i in range(max(len(text_results), len(embedding_results))):
            if i < len(text_results):
                combined.append(text_results[i])
            if i < len(embedding_results):
                combined.append(embedding_results[i])
        
        seen = set()
        deduped = []
        for doc in combined:
            if doc['segment_id'] not in seen:
                seen.add(doc['segment_id'])
                deduped.append(doc)
        
        return deduped[:num_results]

    def process_query(self, query):
        return clean_text(query)