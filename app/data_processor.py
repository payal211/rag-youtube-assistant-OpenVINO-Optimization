import logging
from minsearch import Index
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from elasticsearch import Elasticsearch
import os
import json
from transcript_extractor import get_transcript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    if not isinstance(text, str):
        logger.warning(f"Non-string input to clean_text: {type(text)}")
        return ""
    cleaned = re.sub(r'[^\w\s.,!?]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    logger.info(f"Cleaned text: '{cleaned[:100]}...'")
    return cleaned

class DataProcessor:
    def __init__(self, text_fields=["content", "title", "description"], 
                 keyword_fields=["video_id", "author", "upload_date"], 
                 embedding_model="all-MiniLM-L6-v2"):
        self.text_index = Index(text_fields=text_fields, keyword_fields=keyword_fields)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []
        self.index_built = False
        self.current_index_name = None
        
        elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
        elasticsearch_port = int(os.getenv('ELASTICSEARCH_PORT', 9200))
        
        self.es = Elasticsearch([f'http://{elasticsearch_host}:{elasticsearch_port}'])
        logger.info(f"DataProcessor initialized with Elasticsearch at {elasticsearch_host}:{elasticsearch_port}")

    def process_transcript(self, video_id, transcript_data):
        if not transcript_data or 'metadata' not in transcript_data or 'transcript' not in transcript_data:
            logger.error(f"Invalid transcript data for video {video_id}")
            return None

        metadata = transcript_data['metadata']
        transcript = transcript_data['transcript']

        logger.info(f"Processing transcript for video {video_id}")
        logger.info(f"Number of transcript segments: {len(transcript)}")

        full_transcript = " ".join([segment.get('text', '') for segment in transcript])
        cleaned_transcript = clean_text(full_transcript)

        if not cleaned_transcript:
            logger.warning(f"Empty cleaned transcript for video {video_id}")
            return None

        doc = {
            "video_id": video_id,
            "content": cleaned_transcript,
            "segment_id": f"{video_id}_full",
            "title": clean_text(metadata.get('title', '')),
            "author": metadata.get('author', ''),
            "upload_date": metadata.get('upload_date', ''),
            "view_count": metadata.get('view_count', 0),
            "like_count": metadata.get('like_count', 0),
            "comment_count": metadata.get('comment_count', 0),
            "video_duration": metadata.get('duration', '')
        }
        self.documents.append(doc)
        self.embeddings.append(self.embedding_model.encode(cleaned_transcript + " " + metadata.get('title', '')))

        logger.info(f"Processed transcript for video {video_id}")
        return f"video_{video_id}_{self.embedding_model.get_sentence_embedding_dimension()}"

    def build_index(self, index_name):
        if not self.documents:
            logger.error("No documents to index")
            return None

        logger.info(f"Building index with {len(self.documents)} documents")
        try:
            self.text_index.fit(self.documents)
            self.index_built = True
            logger.info("Text index built successfully")
        except Exception as e:
            logger.error(f"Error building text index: {str(e)}")
            raise

        self.embeddings = np.array(self.embeddings)
        
        try:
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body={
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": self.embeddings.shape[1]},
                            "content": {"type": "text"},
                            "video_id": {"type": "keyword"},
                            "segment_id": {"type": "keyword"},
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
                logger.info(f"Created Elasticsearch index: {index_name}")

            for doc, embedding in zip(self.documents, self.embeddings):
                doc_with_embedding = doc.copy()
                doc_with_embedding['embedding'] = embedding.tolist()
                self.es.index(index=index_name, body=doc_with_embedding, id=doc['segment_id'])
            
            logger.info(f"Successfully indexed {len(self.documents)} documents in Elasticsearch")
            self.current_index_name = index_name
            return index_name
        except Exception as e:
            logger.error(f"Error building Elasticsearch index: {str(e)}")
            raise

    def ensure_index_built(self, video_id, embedding_model):
        index_name = f"video_{video_id}_{embedding_model.replace('-', '_')}".lower()
        if not self.es.indices.exists(index=index_name):
            logger.info(f"Index {index_name} does not exist. Building now...")
            transcript_data = get_transcript(video_id)
            if transcript_data:
                self.process_transcript(video_id, transcript_data)
                return self.build_index(index_name)
            else:
                logger.error(f"Failed to retrieve transcript for video {video_id}")
                return None
        return index_name

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10, method='hybrid', index_name=None):
        if not index_name:
            logger.error("No index name provided for search.")
            raise ValueError("No index name provided for search.")
        
        if not self.es.indices.exists(index=index_name):
            logger.error(f"Index {index_name} does not exist.")
            raise ValueError(f"Index {index_name} does not exist.")
        
        logger.info(f"Performing {method} search for query: {query} in index: {index_name}")
        
        if method == 'text':
            return self.text_search(query, filter_dict, boost_dict, num_results, index_name)
        elif method == 'embedding':
            return self.embedding_search(query, num_results, index_name)
        else:  # hybrid search
            text_results = self.text_search(query, filter_dict, boost_dict, num_results, index_name)
            embedding_results = self.embedding_search(query, num_results, index_name)
            return self.combine_results(text_results, embedding_results, num_results)

    def text_search(self, query, filter_dict={}, boost_dict={}, num_results=10, index_name=None):
        if not index_name:
            logger.error("No index name provided for text search.")
            raise ValueError("No index name provided for text search.")
        
        # Perform text search using Elasticsearch
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title"]
                }
            },
            "size": num_results
        }
        response = self.es.search(index=index_name, body=search_body)
        return [hit['_source'] for hit in response['hits']['hits']]

    def embedding_search(self, query, num_results=10, index_name=None):
        if not index_name:
            logger.error("No index name provided for embedding search.")
            raise ValueError("No index name provided for embedding search.")
        
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

    def set_embedding_model(self, model_name):
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model set to: {model_name}")