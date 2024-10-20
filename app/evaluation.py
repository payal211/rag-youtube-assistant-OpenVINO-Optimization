from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
import ollama

class EvaluationSystem:
    def __init__(self, data_processor, database_handler):
        self.data_processor = data_processor
        self.db_handler = database_handler

    def relevance_scoring(self, query, retrieved_docs, top_k=5):
        query_embedding = self.data_processor.embedding_model.encode(query)
        doc_embeddings = [self.data_processor.embedding_model.encode(doc['content']) for doc in retrieved_docs]
        
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        return np.mean(sorted(similarities, reverse=True)[:top_k])

    def answer_similarity(self, generated_answer, reference_answer):
        gen_embedding = self.data_processor.embedding_model.encode(generated_answer)
        ref_embedding = self.data_processor.embedding_model.encode(reference_answer)
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
            retrieved_docs = rag_system.data_processor.search(query, num_results=5, method='hybrid', index_name=index_name)
            generated_answer, _ = rag_system.query(query, search_method='hybrid', index_name=index_name)

            relevance_scores.append(self.relevance_scoring(query, retrieved_docs))
            similarity_scores.append(self.answer_similarity(generated_answer, reference))
            human_scores.append(self.human_evaluation(index_name, query))  # Assuming index_name can be used as video_id

        return {
            "avg_relevance_score": np.mean(relevance_scores),
            "avg_similarity_score": np.mean(similarity_scores),
            "avg_human_score": np.mean(human_scores)
        }

    def llm_as_judge(self, question, generated_answer, prompt_template):
        prompt = prompt_template.format(question=question, answer_llm=generated_answer)
        
        try:
            response = ollama.chat(
                model='phi3.5',
                messages=[{"role": "user", "content": prompt}]
            )
            evaluation = json.loads(response['message']['content'])
            return evaluation
        except Exception as e:
            print(f"Error in LLM evaluation: {str(e)}")
            return None

    def evaluate_rag(self, rag_system, ground_truth_file, sample_size=200, prompt_template=None):
        try:
            ground_truth = pd.read_csv(ground_truth_file)
        except FileNotFoundError:
            print("Ground truth file not found. Please generate ground truth data first.")
            return None

        sample = ground_truth.sample(n=min(sample_size, len(ground_truth)), random_state=1)
        evaluations = []

        for _, row in sample.iterrows():
            question = row['question']
            video_id = row['video_id']
            
            index_name = self.db_handler.get_elasticsearch_index_by_youtube_id(video_id)
            
            if not index_name:
                print(f"No index found for video {video_id}. Skipping this question.")
                continue

            try:
                answer_llm, _ = rag_system.query(question, search_method='hybrid', index_name=index_name)
            except ValueError as e:
                print(f"Error querying RAG system: {str(e)}")
                continue

            if prompt_template:
                evaluation = self.llm_as_judge(question, answer_llm, prompt_template)
                if evaluation:
                    evaluations.append((
                        str(video_id),
                        str(question),
                        str(answer_llm),
                        str(evaluation.get('Relevance', 'UNKNOWN')),
                        str(evaluation.get('Explanation', 'No explanation provided'))
                    ))
            else:
                # Fallback to cosine similarity if no prompt template is provided
                similarity = self.answer_similarity(answer_llm, row.get('reference_answer', ''))
                evaluations.append((
                    str(video_id),
                    str(question),
                    str(answer_llm),
                    f"Similarity: {similarity}",
                    "Cosine similarity used for evaluation"
                ))

        return evaluations