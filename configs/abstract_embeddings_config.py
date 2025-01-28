import os
from dotenv import load_dotenv

load_dotenv('C:/Users/tjker/Desktop/Research/Projects/lit_review/.env', override=True)

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
    'model': {
        'model_id': "meta-llama/Llama-3.2-1B"
    },
    'embedding': {
        'size': 2048,
        'similarity': 'cosine'
    }
}