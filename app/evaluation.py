from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
import ollama
import requests
import sqlite3
from tqdm import tqdm
import csv

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
            human_scores.append(self.human_evaluation(index_name, query))

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

    def evaluate_rag(self, rag_system, ground_truth_file, prompt_template=None):
        try:
            ground_truth = pd.read_csv(ground_truth_file)
        except FileNotFoundError:
            print("Ground truth file not found. Please generate ground truth data first.")
            return None

        evaluations = []

        for _, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
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
                    evaluations.append({
                        'video_id': str(video_id),
                        'question': str(question),
                        'answer': str(answer_llm),
                        'relevance': str(evaluation.get('Relevance', 'UNKNOWN')),
                        'explanation': str(evaluation.get('Explanation', 'No explanation provided'))
                    })
            else:
                similarity = self.answer_similarity(answer_llm, row.get('reference_answer', ''))
                evaluations.append({
                    'video_id': str(video_id),
                    'question': str(question),
                    'answer': str(answer_llm),
                    'relevance': f"Similarity: {similarity}",
                    'explanation': "Cosine similarity used for evaluation"
                })

        # Save evaluations to CSV
        csv_path = 'data/evaluation_results.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_id', 'question', 'answer', 'relevance', 'explanation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for eval_data in evaluations:
                writer.writerow(eval_data)

        print(f"Evaluation results saved to {csv_path}")

        # Save evaluations to database
        self.save_evaluations_to_db(evaluations)

        return evaluations

    def save_evaluations_to_db(self, evaluations):
        with sqlite3.connect(self.db_handler.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                question TEXT,
                answer TEXT,
                relevance TEXT,
                explanation TEXT
            )
            ''')
            for eval_data in evaluations:
                cursor.execute('''
                INSERT INTO rag_evaluations (video_id, question, answer, relevance, explanation)
                VALUES (?, ?, ?, ?, ?)
                ''', (eval_data['video_id'], eval_data['question'], eval_data['answer'], 
                      eval_data['relevance'], eval_data['explanation']))
            conn.commit()
        print("Evaluation results saved to database")

    def run_full_evaluation(self, rag_system, ground_truth_file, prompt_template=None):
        # Load ground truth
        ground_truth = pd.read_csv(ground_truth_file)

        # Evaluate RAG
        rag_evaluations = self.evaluate_rag(rag_system, ground_truth_file, prompt_template)

        # Evaluate search performance
        def search_function(query, video_id):
            index_name = self.db_handler.get_elasticsearch_index_by_youtube_id(video_id)
            if index_name:
                return rag_system.data_processor.search(query, num_results=10, method='hybrid', index_name=index_name)
            return []

        search_performance = self.evaluate_search(ground_truth, search_function)

        # Optimize search parameters
        param_ranges = {'content': (0.0, 3.0)}  # Example parameter range

        def objective_function(params):
            def parameterized_search(query, video_id):
                index_name = self.db_handler.get_elasticsearch_index_by_youtube_id(video_id)
                if index_name:
                    return rag_system.data_processor.search(query, num_results=10, method='hybrid', index_name=index_name, boost_dict=params)
                return []
            return self.evaluate_search(ground_truth, parameterized_search)['mrr']

        best_params, best_score = self.simple_optimize(param_ranges, objective_function)

        return {
            "rag_evaluations": rag_evaluations,
            "search_performance": search_performance,
            "best_params": best_params,
            "best_score": best_score
        }


    def hit_rate(self, relevance_total):
        return sum(any(line) for line in relevance_total) / len(relevance_total)

    def mrr(self, relevance_total):
        scores = []
        for line in relevance_total:
            for rank, relevant in enumerate(line, 1):
                if relevant:
                    scores.append(1 / rank)
                    break
            else:
                scores.append(0)
        return sum(scores) / len(scores)

    def simple_optimize(self, param_ranges, objective_function, n_iterations=10):
        best_params = None
        best_score = float('-inf')
        for _ in range(n_iterations):
            current_params = {param: np.random.uniform(min_val, max_val) 
                              for param, (min_val, max_val) in param_ranges.items()}
            current_score = objective_function(current_params)
            if current_score > best_score:
                best_score = current_score
                best_params = current_params
        return best_params, best_score

    def evaluate_search(self, ground_truth, search_function):
        relevance_total = []
        for _, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
            video_id = row['video_id']
            results = search_function(row['question'], video_id)
            relevance = [d['video_id'] == video_id for d in results]
            relevance_total.append(relevance)
        return {
            'hit_rate': self.hit_rate(relevance_total),
            'mrr': self.mrr(relevance_total),
        }