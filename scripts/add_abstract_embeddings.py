import yaml
import os
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm

from langchain_neo4j import Neo4jGraph

  
def load_config(config_file="C:/Users/tjker/Desktop/Research/Projects/lit_review/configs/abstract_embeddings.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_embedding(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():     
            outputs = model(**inputs) 
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()

def main(config_file="C:/Users/tjker/Desktop/Research/Projects/lit_review/configs/abstract_embeddings.yaml"):
    config = load_config(config_file)
    kg = Neo4jGraph(
        url=config['database']['uri'], 
        username=config['database']['username'], 
        password=config['database']['password']
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_id'])
    model = AutoModel.from_pretrained(config['model']['model_id'])

    result = kg.query("""
        MATCH (p:Paper) 
        RETURN elementId(p) AS node_id, p.abstract AS abstract
        """
    )
    
    for record in tqdm(result):
        node_id = record["node_id"]
        abstract = record["abstract"]
        
        if abstract:
            embedding = compute_embedding(abstract, tokenizer, model)
            kg.query("""
                MATCH (p:Paper) WHERE elementId(p) = $node_id
                SET p.abstractEmbedding = $embedding
                RETURN elementId(p) AS node_id, p.abstract AS abstract
                """, params={"node_id":node_id, "embedding":embedding}
            )
            
    kg.query("""
        CREATE VECTOR INDEX abstract_embeddings IF NOT EXISTS
        FOR (p:Paper) ON (p.abstractEmbedding) 
        OPTIONS { indexConfig: {
            `vector.dimensions`: 2048,
            `vector.similarity_function`: 'cosine'
        }}"""
    )
    
    kg.query("""
        CREATE FULLTEXT INDEX paperTitleIndex FOR (p:Paper) ON EACH [p.title]
        """
    )

if __name__ == "__main__":
    main()