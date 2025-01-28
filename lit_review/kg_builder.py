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