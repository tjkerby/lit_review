import os
import requests
import json

import semantic_scholar_api as ss_api
import neo4j_utils as nu

def build_citation_data(kg, config, paperId, s2_api_key=None):
    try:
        citation_data = ss_api.exponential_backoff_retry(ss_api.get_paper_references, paperId, fields=config['graph']['citation_fields'], s2_api_key=s2_api_key)
        for cited_paper in citation_data['data']:
            cited_paper = cited_paper['citedPaper']
            if cited_paper['paperId'] is None:
                continue
            cited_paper_node = nu.search_papers_by_paperid(kg, cited_paper['paperId'])
            if len(cited_paper_node) == 0:
                cited_paper['level'] = 2
                paper_properties = {key: value for key, value in cited_paper.items() if key != 'authors'}
                if config['general']['verbose']: print('adding cited paper nodes...')
                cited_paper_node = nu.create_paper_node(kg, paper_properties)
            nu.create_cites_rel(kg, paperId, cited_paper['paperId'])     
            if config['graph']['author']:
                if config['general']['verbose']: print('adding author nodes and authored relationships...')
                build_author_data(kg, cited_paper['authors'], cited_paper['paperId'])
        return citation_data
    except ss_api.RateLimitExceededError:
        print("Exceeded rate limit. Please try again later.")
    except Exception as e:
        print(f"An error occurred: {e}")
                
def build_author_data(kg, authors, paperId):
    for author in authors:
        if author['authorId'] is None:
            continue
        nu.create_author_node(kg, author)   
        nu.create_authored_rel(kg, paperId, author['authorId']) 
        
def prepare_chunks_from_urls(config, kg):

    with open(config['data']['titles_path'], 'r') as file:
        paper_titles = [line.strip() for line in file]

    json_dicts = []
    for title in paper_titles:
        paper_results = kg.query("""
            CALL db.index.fulltext.queryNodes('paperTitleIndex', $title)     
            YIELD node, score
            RETURN node.paperId, node.title, score
            LIMIT 1
            """, params={'title': title}
        )
        if paper_results:
            paper = paper_results[0]
            json_dicts.append({
                'title': paper['node.title'], 
                'paperId': paper['node.paperId'], 
                'url_path': None,
                'pdf_path': None
            })
        else:
            print(f"No match found for title: {title}")
            json_dicts.append({
                'title': title,
                'paperId': None,
                'url_path': None,
                'pdf_path': None
            })
        
    papers_json = json.dumps(json_dicts, indent=4)
    with open(f"{config['data']['data_path']}/{config['data']['prep_paper_chunk_output_name']}", 'w') as json_file:
        json_file.write(papers_json)

def download_pdfs_from_urls(config, verbose=False):
    with open(f"{config['data']['data_path']}/{config['data']['paper_chunk_input_name']}", 'r') as file:
        data = json.load(file)

    updated_data = []
    for i, paper in enumerate(data):
        pdf_path = f"{config['data']['data_path']}/paper_pdfs/{paper['paperId']}.pdf"
        
        # Check if the PDF already exists
        if os.path.exists(pdf_path):
            if verbose: print(f"PDF already exists: {pdf_path}")
            paper["pdf_path"] = pdf_path
            updated_data.append(paper)
            continue  # Skip downloading

        # Download the PDF if it doesn't exist
        response = requests.get(paper["url_path"])
        if response.status_code == 200:
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)  # Ensure the directory exists
            with open(pdf_path, 'wb') as file:
                file.write(response.content)
            if verbose: print(f"PDF saved to: {pdf_path}")
            paper["pdf_path"] = pdf_path
            updated_data.append(paper)
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")

    # Save the updated data
    papers_json = json.dumps(updated_data, indent=4)
    with open(f"{config['data']['data_path']}/{config['data']['paper_chunk_output_name']}", 'w') as json_file:
        json_file.write(papers_json)