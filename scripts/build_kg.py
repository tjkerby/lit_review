import sys

sys.path.append('C:/Users/tjker/Desktop/Research/Projects/lit_review/lit_review')
import semantic_scholar_api as ss_api
import neo4j_utils as nu
import kg_builder as kgb
import utils 

def main(config_file="C:/Users/tjker/Desktop/Research/Projects/lit_review/configs/build_kg.yaml"):
    config = utils.load_config(config_file)
    kg = utils.load_kg(config)

    with open(config['file_paths']['paper_titles'], 'r') as file:
        titles = [line.strip() for line in file]
    
    paper_data = ss_api.extract_paper_data(titles)
    if config['general']['verbose']: print('Creating intial paper nodes...')
    author_info = nu.create_paper_nodes(kg, paper_data, return_authors=True)
    
    if config['graph']['author']:
        if config['general']['verbose']: print('Creating intial author nodes...')
        for paper in author_info:
            kgb.build_author_data(kg, paper['authors'], paper['paperId'])
    if config['graph']['citation']:    
        for paper in paper_data:
            citation_data = kgb.build_citation_data(kg, config, paper['paperId'])

if __name__ == "__main__":
    main()