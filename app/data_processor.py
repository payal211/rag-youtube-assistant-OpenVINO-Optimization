from minsearch import Index
from sentence_transformers import SentenceTransformer
import numpy as np
from elasticsearch import Elasticsearch
import os
import json
import logging
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def clean_text(text):
    if not isinstance(text, str):
        logger.warning(f"Non-string input to clean_text: {type(text)}")
        return ""
    cleaned = re.sub(r'[^\w\s.,!?]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    logger.debug(f"Original text length: {len(text)}, Cleaned text length: {len(cleaned)}")
    logger.debug(f"Cleaned text sample: '{cleaned[:100]}...'")
    return cleaned

class DataProcessor:
    def __init__(self, text_fields=["content", "title", "description"], 
                 keyword_fields=["video_id", "author", "upload_date"], 
                 embedding_model="multi-qa-MiniLM-L6-cos-v1"):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.all_fields = text_fields + keyword_fields
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
        logger.info(f"Processing transcript for video {video_id}")
        
        if not transcript_data:
            logger.error(f"Transcript data is None for video {video_id}")
            return None
        
        if 'metadata' not in transcript_data or 'transcript' not in transcript_data:
            logger.error(f"Invalid transcript data structure for video {video_id}")
            logger.debug(f"Transcript data keys: {transcript_data.keys()}")
            return None

        metadata = transcript_data['metadata']
        transcript = transcript_data['transcript']

        logger.info(f"Number of transcript segments: {len(transcript)}")

        full_transcript = " ".join([segment.get('text', '') for segment in transcript])
        logger.debug(f"Full transcript length before cleaning: {len(full_transcript)}")
        logger.debug(f"Full transcript sample before cleaning: '{full_transcript[:500]}...'")

        cleaned_transcript = clean_text(full_transcript)
        logger.debug(f"Cleaned transcript length: {len(cleaned_transcript)}")
        logger.debug(f"Cleaned transcript sample: '{cleaned_transcript[:500]}...'")

        if not cleaned_transcript:
            logger.warning(f"Empty cleaned transcript for video {video_id}")
            return None

        doc = {
            "video_id": video_id,
            "content": cleaned_transcript,
            "title": clean_text(metadata.get('title', '')),
            "description": clean_text(metadata.get('description', 'Not Available')),
            "author": metadata.get('author', ''),
            "upload_date": metadata.get('upload_date', ''),
            "segment_id": f"{video_id}_full",
            "view_count": metadata.get('view_count', 0),
            "like_count": metadata.get('like_count', 0),
            "comment_count": metadata.get('comment_count', 0),
            "video_duration": metadata.get('duration', '')
        }
        
        logger.debug(f"Document created for video {video_id}")
        for field in self.all_fields:
            logger.debug(f"Document {field} length: {len(str(doc.get(field, '')))}")
            logger.debug(f"Document {field} sample: '{str(doc.get(field, ''))[:100]}...'")

        self.documents.append(doc)
        embedding = self.embedding_model.encode(cleaned_transcript + " " + metadata.get('title', ''))
        self.embeddings.append(embedding)

        logger.info(f"Processed transcript for video {video_id}")
        
        # Return a dictionary with the processed content and other relevant information
        return {
            'content': cleaned_transcript,
            'metadata': metadata,
            'index_name': f"video_{video_id}_{self.embedding_model.get_sentence_embedding_dimension()}"
        }

    def build_index(self, index_name):
        if not self.documents:
            logger.error("No documents to index")
            return None

        logger.info(f"Building index with {len(self.documents)} documents")
        
        # Fields to include in the fit function
        index_fields = self.text_fields + self.keyword_fields
        
        # Create a list of dictionaries with only the fields we want to index
        docs_to_index = []
        for doc in self.documents:
            indexed_doc = {field: doc.get(field, '') for field in index_fields}
            if all(indexed_doc.values()):  # Check if all required fields have values
                docs_to_index.append(indexed_doc)
            else:
                missing_fields = [field for field, value in indexed_doc.items() if not value]
                logger.warning(f"Document with video_id {doc.get('video_id', 'unknown')} is missing values for fields: {missing_fields}")

        if not docs_to_index:
            logger.error("No valid documents to index")
            return None

        logger.info(f"Number of valid documents to index: {len(docs_to_index)}")

        # Log the structure of the first document to be indexed
        logger.debug("Structure of the first document to be indexed:")
        logger.debug(json.dumps(docs_to_index[0], indent=2))

        try:
            logger.info("Fitting text index")
            self.text_index.fit(docs_to_index)
            self.index_built = True
            logger.info("Text index built successfully")
        except Exception as e:
            logger.error(f"Error building text index: {str(e)}")
            raise

        try:
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body={
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": len(self.embeddings[0]), "index": True, "similarity": "cosine"},
                            "content": {"type": "text"},
                            "title": {"type": "text"},
                            "description": {"type": "text"},
                            "video_id": {"type": "keyword"},
                            "author": {"type": "keyword"},
                            "upload_date": {"type": "date"},
                            "segment_id": {"type": "keyword"},
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
    
    def compute_rrf(self, rank, k=60):
        return 1 / (k + rank)

    def hybrid_search(self, query, index_name, num_results=5):
        if not index_name:
            logger.error("No index name provided for hybrid search.")
            raise ValueError("No index name provided for hybrid search.")
        
        vector = self.embedding_model.encode(query)
        
        knn_query = {
            "field": "embedding",
            "query_vector": vector.tolist(),
            "k": 10,
            "num_candidates": 100
        }

        keyword_query = {
            "multi_match": {
                "query": query,
                "fields": self.text_fields
            }
        }

        try:
            knn_results = self.es.search(
                index=index_name, 
                body={
                    "knn": knn_query, 
                    "size": 10
                }
            )['hits']['hits']
            
            keyword_results = self.es.search(
                index=index_name, 
                body={
                    "query": keyword_query, 
                    "size": 10
                }
            )['hits']['hits']
            
            rrf_scores = {}
            for rank, hit in enumerate(knn_results):
                doc_id = hit['_id']
                rrf_scores[doc_id] = self.compute_rrf(rank + 1)

            for rank, hit in enumerate(keyword_results):
                doc_id = hit['_id']
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += self.compute_rrf(rank + 1)
                else:
                    rrf_scores[doc_id] = self.compute_rrf(rank + 1)

            reranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            final_results = []
            for doc_id, score in reranked_docs[:num_results]:
                doc = self.es.get(index=index_name, id=doc_id)
                final_results.append(doc['_source'])
            
            return final_results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10, method='hybrid', index_name=None):
        if not index_name:
            logger.error("No index name provided for search.")
            raise ValueError("No index name provided for search.")
        
        if not self.es.indices.exists(index=index_name):
            logger.error(f"Index {index_name} does not exist.")
            raise ValueError(f"Index {index_name} does not exist.")
        
        logger.info(f"Performing {method} search for query: {query} in index: {index_name}")
        
        try:
            if method == 'text':
                return self.text_search(query, filter_dict, boost_dict, num_results, index_name)
            elif method == 'embedding':
                return self.embedding_search(query, num_results, index_name)
            else:  # hybrid search
                return self.hybrid_search(query, index_name, num_results)
        except Exception as e:
            logger.error(f"Error in search method {method}: {str(e)}")
            raise

    def text_search(self, query, filter_dict={}, boost_dict={}, num_results=10, index_name=None):
        if not index_name:
            logger.error("No index name provided for text search.")
            raise ValueError("No index name provided for text search.")
        
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": self.text_fields
                    }
                },
                "size": num_results
            }
            response = self.es.search(index=index_name, body=search_body)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            raise

    def embedding_search(self, query, num_results=10, index_name=None):
        if not index_name:
            logger.error("No index name provided for embedding search.")
            raise ValueError("No index name provided for embedding search.")
        
        try:
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
        except Exception as e:
            logger.error(f"Error in embedding search: {str(e)}")
            raise
    
    def set_embedding_model(self, model_name):
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model set to: {model_name}")