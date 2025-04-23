from langchain_ollama import OllamaEmbeddings
from tqdm.auto import tqdm
import sys

sys.path.append('/home/TomKerby/Research/lit_review/lit_review')
import utils 

sys.path.append('/home/TomKerby/Research/lit_review/configs')
from abstract_embeddings_config import config

def main():
    kg = utils.load_kg(config)

    embeddings = OllamaEmbeddings(model=config['embedding']['model_id'])

    result = kg.query("""
        MATCH (p:Paper) 
        RETURN elementId(p) AS node_id, p.abstract AS abstract
        """
    )
    
    for record in tqdm(result):       
        if record["abstract"]:
            embedding = embeddings.embed_query(record['abstract'])
            kg.query("""
                MATCH (p:Paper) WHERE elementId(p) = $node_id
                SET p.abstractEmbedding = $embedding
                RETURN elementId(p) AS node_id, p.abstract AS abstract
                """, params={"node_id":record["node_id"], "embedding":embedding}
            )
            
    kg.query("""
        CREATE VECTOR INDEX abstract_embeddings IF NOT EXISTS
        FOR (p:Paper) ON (p.abstractEmbedding) 
        OPTIONS { indexConfig: {
            `vector.dimensions`: $dimension,
            `vector.similarity_function`: $similarity
        }}""", params={'dimension': config['embedding']['size'], 'similarity': config['embedding']['similarity']}
    )
    
    kg.query("""
        CREATE FULLTEXT INDEX paperTitleIndex IF NOT EXISTS
        FOR (p:Paper) ON EACH [p.title]
        """
    )

if __name__ == "__main__":
    main()