import os
from dotenv import load_dotenv

load_dotenv('/home/TomKerby/Research/lit_review/.env', override=True)

config = {
    'general':  {
        'script_name': "add_papers_from_title_list.py",
        'description': "This script creates a set of paper nodes and possibly author nodes.",
        'version': "1.0.0",
        'verbose': False
    },
    'database': {
        'uri': os.getenv("NEO4J_URI"),
        'username': os.getenv("NEO4J_USERNAME"),
        'password': os.getenv("NEO4J_PASSWORD"),
        'database': os.getenv("NEO4J_DATABASE")
    },
    'file_paths': {
        'paper_titles': os.getenv("PROJECT_PATH") + "/data/paper_titles.txt",
        'output': "/path/to/output"
    },
    'api': {
        'ss_api_key': os.getenv("SS_API_KEY"),
        'max_retries': 10,
        'base_delay': 1,
        'max_delay': 512
    },
    'graph': {
        'author': True,
        'citation': True,
        'citation_fields': ["title", "abstract", "citationCount", "publicationDate", "authors"]
    }
}