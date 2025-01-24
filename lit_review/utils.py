import yaml
from langchain_neo4j import Neo4jGraph

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_kg(config):
    kg = Neo4jGraph(
        url=config['database']['uri'], 
        username=config['database']['username'], 
        password=config['database']['password'],
        database=config['database']['database']
    )
    return kg