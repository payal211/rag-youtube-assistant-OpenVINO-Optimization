#!/bin/bash

# Start Ollama
ollama serve &

# Wait for Ollama to start
sleep 10

# Run Phi model to ensure it's loaded
ollama run phi "hello" &

# Generate ground truth
python generate_ground_truth.py

# Run RAG evaluation
python rag_evaluation.py

# Start the Streamlit app
streamlit run main.py