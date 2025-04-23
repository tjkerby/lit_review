import os
from dotenv import load_dotenv

load_dotenv('/home/TomKerby/Research/lit_review/.env', override=True)

config = {
    'general': {
        'script_name': "ollama_rag.ipynb",
        'description': "This script loads pdfs, chunks them up, and creates their embeddings.",
        'version': "1.0.0",
        'verbose': False
    },
    'database': {
        'uri': os.getenv("NEO4J_URI"),
        'username': os.getenv("NEO4J_USERNAME"),
        'password': os.getenv("NEO4J_PASSWORD"),
        'database': os.getenv("NEO4J_DATABASE")
    },
    'model': {
        'model_id': 'deepseek-r1:70b'
    },
    'embedding': {
        'size': 768,
        'similarity': 'cosine',
        'model_id': 'nomic-embed-text'
    },
    'rag': {
        'index_name': 'paper_chunks',
        'embedding_node_property': 'textEmbedding',
        'text_node_property': 'text'
    }
}