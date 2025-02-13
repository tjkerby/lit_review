import os
from dotenv import load_dotenv

load_dotenv('/home/TomKerby/Research/lit_review/.env', override=True)

config = {
    "llm": {
        "provider": "huggingface",    # options: "ollama", "openai", "huggingface", etc.
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        "max_seq_len": 16384,
        "num_predict": 16384,
        "temperature": 0.5,
        "api_key": "YOUR_API_KEY_IF_NEEDED"  # only used for some providers
    },
    'database': {
        'uri': os.getenv("NEO4J_URI"),
        'username': os.getenv("NEO4J_USERNAME"),
        'password': os.getenv("NEO4J_PASSWORD"),
        'database': os.getenv("NEO4J_DATABASE")
    },
    "embedding": {
        "provider": "huggingface",
        "model_id": 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
        # 'size': 768,
        # 'similarity': 'cosine',
        # "query_prefix": ""
    },
    'rag': {
        'index_name': 'paper_chunks',
        'embedding_node_property': 'textEmbedding',
        'text_node_property': 'text'
    }
}