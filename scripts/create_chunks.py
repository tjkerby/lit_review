import json
import sys
from tqdm.auto import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

sys.path.append('C:/Users/tjker/Desktop/Research/Projects/lit_review/lit_review')
import utils
import kg_builder as kgb

sys.path.append('C:/Users/tjker/Desktop/Research/Projects/lit_review/configs')
from create_chunks_config import config



def main():
    kg = utils.load_kg(config)  

    with open(f"{config['data']['data_path']}/{config['data']['paper_chunk_output_name']}", 'r') as file:
        updated_data = json.load(file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config['chunks']['chunk_size'],
        chunk_overlap  = config['chunks']['chunk_overlap'],
        length_function = len,
        is_separator_regex = False,
    )
    
    kgb.create_chunk_nodes(kg, updated_data, text_splitter, config)
    
    kg.query("""
    CREATE VECTOR INDEX `paper_chunks` IF NOT EXISTS
    FOR (c:Chunk) ON (c.textEmbedding) 
    OPTIONS { indexConfig: {
        `vector.dimensions`: $dimension,
        `vector.similarity_function`: $similarity    
    }}""", params={'dimension': config['embedding']['size'], 'similarity': config['embedding']['similarity']}
    )

    model = SentenceTransformer(config['embedding']['model_id'])

    all_chunk_nodes = kg.query("""
        MATCH (c:Chunk) 
        RETURN elementId(c) AS chunk_id, c.text AS text
        """
    )

    for record in tqdm(all_chunk_nodes):
        chunk_id = record["chunk_id"]
        text = record["text"]
        
        if text:
            embedding = model.encode(text)
            kg.query("""
                MATCH (c:Chunk) 
                WHERE elementId(c) = $chunk_id
                SET c.textEmbedding = $embedding
                RETURN elementId(c) AS chunk_id, c.textEmbedding AS embedding
                """, params={"chunk_id":chunk_id, "embedding":embedding}
            )

if __name__ == "__main__":
    main()