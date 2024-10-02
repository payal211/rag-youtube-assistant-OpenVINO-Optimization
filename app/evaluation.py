from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EvaluationSystem:
    def __init__(self, data_processor, database_handler):
        self.data_processor = data_processor
        self.db_handler = database_handler

    def relevance_scoring(self, query, retrieved_docs, top_k=5):
        query_embedding = self.data_processor.process_query(query)
        doc_embeddings = [self.data_processor.process_query(doc) for doc in retrieved_docs]
        
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        return np.mean(sorted(similarities, reverse=True)[:top_k])

    def answer_similarity(self, generated_answer, reference_answer):
        gen_embedding = self.data_processor.process_query(generated_answer)
        ref_embedding = self.data_processor.process_query(reference_answer)
        return cosine_similarity([gen_embedding], [ref_embedding])[0][0]

    def human_evaluation(self, video_id, query):
        with self.db_handler.conn:
            cursor = self.db_handler.conn.cursor()
            cursor.execute('''
                SELECT AVG(feedback) FROM user_feedback
                WHERE video_id = ? AND query = ?
            ''', (video_id, query))
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0

    def evaluate_rag_performance(self, rag_system, test_queries, reference_answers, index_name):
        relevance_scores = []
        similarity_scores = []
        human_scores = []

        for query, reference in zip(test_queries, reference_answers):
            retrieved_docs = rag_system.es_handler.search(index_name, rag_system.data_processor.process_query(query))
            generated_answer = rag_system.query(index_name, query)

            relevance_scores.append(self.relevance_scoring(query, retrieved_docs))
            similarity_scores.append(self.answer_similarity(generated_answer, reference))
            human_scores.append(self.human_evaluation(index_name, query))  # Assuming index_name can be used as video_id

        return {
            "avg_relevance_score": np.mean(relevance_scores),
            "avg_similarity_score": np.mean(similarity_scores),
            "avg_human_score": np.mean(human_scores)
        }