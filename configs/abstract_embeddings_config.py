import os
from dotenv import load_dotenv

load_dotenv('/home/TomKerby/Research/lit_review/.env', override=True)

config = {
    'general':  {
        'script_name': "add_abstract_embeddings.py",
        'description': "This script creates embeddings for abstracts.",
        'version': "1.0.0",
        'verbose': False
    },
    'database': {
        'uri': os.getenv("NEO4J_URI"),
        'username': os.getenv("NEO4J_USERNAME"),
        'password': os.getenv("NEO4J_PASSWORD"),
        'database': os.getenv("NEO4J_DATABASE")
    },
    'embedding': {
        'model_id': 'nomic-embed-text',
        'size': 768,
        'similarity': 'cosine'
    }
}