import os
from dotenv import load_dotenv

load_dotenv('C:/Users/tjker/Desktop/Research/Projects/lit_review/.env', override=True)

config = {
    'general': {
        'script_name': "add_chunks.py",
        'description': "This script loads pdfs, chunks them up, and creates their embeddings.",
        'version': "1.0.0"
    },
    'database': {
        'uri': os.getenv("NEO4J_URI"),
        'username': os.getenv("NEO4J_USERNAME"),
        'password': os.getenv("NEO4J_PASSWORD"),
        'database': os.getenv("NEO4J_DATABASE")
    },
    'data': {
        'data_path': os.getenv("PROJECT_PATH") + 'data',
        'titles_path': os.getenv("PROJECT_PATH") + 'data/paper_titles.txt'
    },
    'model': {
        'model_id': "meta-llama/Llama-3.2-1B"
    },
    'embedding': {
        'size': 2048,
        'similarity': 'cosine'
    }
}