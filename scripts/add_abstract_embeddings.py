from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import sys

sys.path.append('C:/Users/tjker/Desktop/Research/Projects/lit_review/lit_review')
import utils 
import rag_utils as rag

sys.path.append('C:/Users/tjker/Desktop/Research/Projects/lit_review/configs')
from abstract_embeddings_config import config

def main():
    kg = utils.load_kg(config)

    tokenizer = AutoTokenizer.from_pretrained(config['embedding']['model_id'], model_max_length=8192)
    model = AutoModel.from_pretrained(config['embedding']['model_id'], trust_remote_code=True, rotary_scaling_factor=2)
    model.eval()

    result = kg.query("""
        MATCH (p:Paper) 
        RETURN elementId(p) AS node_id, p.abstract AS abstract
        """
    )
    
    for record in tqdm(result):       
        if record["abstract"]:
            embedding = rag.compute_embedding_nomic(record["abstract"], tokenizer, model, config)
            # embedding = rag.compute_embedding(record["abstract"], tokenizer, model)
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