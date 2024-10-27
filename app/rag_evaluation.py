"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import requests
import sqlite3
from minsearch import Index

# Database connection
conn = sqlite3.connect('data/sqlite.db')
cursor = conn.cursor()

# Load ground truth data from CSV
def load_ground_truth():
    return pd.read_csv('data/ground-truth-retrieval.csv')

ground_truth = load_ground_truth()

# Load transcript data
def load_transcripts():
    cursor.execute("SELECT * FROM transcript_segments")
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=['segment_id', 'video_id', 'content', 'start_time', 'duration'])

transcripts = load_transcripts()

# Create index
index = Index(
    text_fields=['content'],
    keyword_fields=['video_id', 'segment_id']
)
index.fit(transcripts.to_dict('records'))

# RAG flow
def search(query):
    boost = {}
    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )
    return results

prompt_template = '''
You're an AI assistant for YouTube video transcripts. Answer the QUESTION based on the CONTEXT from our transcript database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
'''.strip()

def build_prompt(query, search_results):
    context = "\n\n".join([f"Segment {i+1}: {result['content']}" for i, result in enumerate(search_results)])
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'phi',
        'prompt': prompt
    })
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

# Evaluation metrics
def hit_rate(relevance_total):
    return sum(any(line) for line in relevance_total) / len(relevance_total)

def mrr(relevance_total):
    scores = []
    for line in relevance_total:
        for rank, relevant in enumerate(line, 1):
            if relevant:
                scores.append(1 / rank)
                break
        else:
            scores.append(0)
    return sum(scores) / len(scores)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for _, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
        video_id = row['video_id']
        results = search_function(row['question'])
        relevance = [d['video_id'] == video_id for d in results]
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

# Parameter optimization
param_ranges = {
    'content': (0.0, 3.0),
}

def simple_optimize(param_ranges, objective_function, n_iterations=10):
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

def objective(boost_params):
    def search_function(q):
        return search(q, boost_params)
    results = evaluate(ground_truth, search_function)
    return results['mrr']

# RAG evaluation
prompt2_template = '''
You are an expert evaluator for a Youtube transcript assistant.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
'''.strip()

def evaluate_rag(sample_size=200):
    sample = ground_truth.sample(n=sample_size, random_state=1)
    evaluations = []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        question = row['question']
        answer_llm = rag(question)
        prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
        evaluation = llm(prompt)
        evaluation = json.loads(evaluation)
        evaluations.append((row['video_id'], question, answer_llm, evaluation['Relevance'], evaluation['Explanation']))
    return evaluations

# Main execution
if __name__ == "__main__":
    print("Evaluating search performance...")
    search_performance = evaluate(ground_truth, lambda q: search(q['question']))
    print(f"Search performance: {search_performance}")

    print("\nOptimizing search parameters...")
    best_params, best_score = simple_optimize(param_ranges, objective, n_iterations=20)
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    print("\nEvaluating RAG performance...")
    rag_evaluations = evaluate_rag(sample_size=200)
    
    # Store RAG evaluations in the database
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rag_evaluations (
        video_id TEXT,
        question TEXT,
        answer TEXT,
        relevance TEXT,
        explanation TEXT
    )
    ''')
    cursor.executemany('''
    INSERT INTO rag_evaluations (video_id, question, answer, relevance, explanation)
    VALUES (?, ?, ?, ?, ?)
    ''', rag_evaluations)
    conn.commit()

    print("Evaluation complete. Results stored in the database.")

    # Close the database connection
    conn.close()
"""