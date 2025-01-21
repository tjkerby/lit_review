import yaml
import pandas as pd
from tqdm.auto import tqdm
from neo4j import GraphDatabase

import semantic_scholar_api as ss_api
import neo4j_utils as nu
  
def load_config(config_file="C:/Users/tjker/Desktop/Research/Projects/lit_review/configs/paper_nodes.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_file="C:/Users/tjker/Desktop/Research/Projects/lit_review/configs/paper_nodes.yaml"):
    
    config = load_config(config_file)
    driver = GraphDatabase.driver(
        config['database']['uri'], 
        auth=(config['database']['username'], config['database']['password'])
    )
    
    with open(config['file_path']['paper_titles'], 'r') as file:
        paper_titles = [line.strip() for line in file]

    data = []
    for title in tqdm(paper_titles):
        try:
            paper_data = ss_api.exponential_backoff_retry(
                ss_api.search_paper_by_title,
                title=title,
                max_retries=config['api']['max_retries'],
                base_delay=config['api']['base_delay'],
                max_delay=config['api']['max_delay']
            )
            if paper_data:
                data.append(paper_data)
        except ss_api.RateLimitExceededError:
            print("Exceeded rate limit. Please try again later.")
        except Exception as e:
            print(f"An error occurred: {e}")

    df = pd.json_normalize(data)

    with driver.session() as session:
        for i, paper in tqdm(enumerate(data)):
            if paper is None:
                continue

            paper_properties = {key: value for key, value in paper.items() if key != 'authors'}
            paper_properties['level'] = 1
            paper_node = nu.get_or_create_paper_node(session, paper_properties)
            
            if config['graph']['author']:
                for author in paper['authors']:
                    author_node = nu.get_or_create_author_node(session, author)
                    session.execute_write(
                        nu.create_authored_rel,
                        {"paperId": paper['paperId']},
                        {"authorId": author['authorId']}
                    )
        if config['graph']['citation']:
            for i in tqdm(range(len(data))):
                try:
                    citation_data = ss_api.exponential_backoff_retry(
                        ss_api.get_paper_references, 
                        paper_id = df.loc[i].paperId, 
                        fields=["title", "abstract", "citationCount", "publicationDate"],
                        max_retries=config['api']['max_retries'],
                        base_delay=config['api']['base_delay'],
                        max_delay=config['api']['max_delay']
                    )
                    for cited_paper in citation_data['data']:
                        cited_paper = cited_paper['citedPaper']
                        query = "MATCH (p:Paper {paperId: $paperId}) RETURN p"
                        result = session.run(query, paperId=cited_paper['paperId'])
                        cited_paper_node = result.single()
                        if cited_paper_node is not None:
                            session.execute_write(
                                nu.create_cites_rel,
                                {"paperId": df.loc[i].paperId},
                                {"paperId": cited_paper['paperId']}
                            )
                        else:
                            cited_paper['level'] = 2
                            cited_paper_node = nu.get_or_create_paper_node(session, cited_paper)
                            session.execute_write(
                                nu.create_cites_rel,
                                {"paperId": df.loc[i].paperId},
                                {"paperId": cited_paper['paperId']}
                            )                 
                except ss_api.RateLimitExceededError:
                    print("Exceeded rate limit. Please try again later.")
                except Exception as e:
                    print(f"An error occurred: {e}")
    driver.close()

if __name__ == "__main__":
    main()