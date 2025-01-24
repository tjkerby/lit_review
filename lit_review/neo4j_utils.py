from tqdm.auto import tqdm

def create_paper_nodes(kg, paper_data, return_authors=True):
    author_info = []
    for i, paper in tqdm(enumerate(paper_data)):
        if paper is None:
            continue
        paper_properties = {key: value for key, value in paper.items() if key != 'authors'}
        paper_properties['level'] = 1
        result = create_paper_node(kg, paper_properties)
        if return_authors:
            author_info.append({key: value for key, value in paper.items() if key in ['paperId', 'authors']})
    
    if return_authors:
        return author_info
    
def create_paper_node(kg, paper_data):
    cypher = """
    MERGE (p:Paper {paperId: $paper_data.paperId})
    SET p += $paper_data
    RETURN p
    """
    return kg.query(cypher, params={'paper_data': paper_data})
    
def create_author_nodes(kg, author_info):
    for paper_authors in tqdm(author_info):
        if paper_authors is None:
            continue
        for author in paper_authors['authors']:
            result = create_author_node(kg, author)
            
def create_author_node(kg, author_info):
    cypher = """
    MERGE (a:Author {authorId: $author.authorId, name: $author.name})
    RETURN a
    """
    return kg.query(cypher, params={'author': author_info})
            
def create_authored_rels_papers(kg, author_info):   
    for paper_authors in tqdm(author_info):
        if paper_authors is None:
            continue
        for author in paper_authors['authors']:
            result = create_authored_rel(kg, paper_authors['paperId'], author['authorId'])
            
def create_authored_rel_paper(kg, paperId, authors):
    for author in authors:
        result = create_authored_rel(kg, paperId, author['authorId'])
            
def create_authored_rel(kg, paperId, authorId):
    cypher = """
    MATCH (p:Paper {paperId: $paperId})
    MATCH (a:Author {authorId: $authorId})
    MERGE (a)-[:Authored]->(p)
    """
    return kg.query(cypher, params={'paperId': paperId, 'authorId': authorId})

def create_cites_rel(kg, p1_id, p2_id):  
    cypher = """
    MATCH (p1:Paper {paperId: $paperId1})
    MATCH (p2:Paper {paperId: $paperId2})
    MERGE (p1)-[r:Cited]->(p2)
    """    
    return kg.query(cypher, params={'paperId1': p1_id, 'paperId2': p2_id})
    
def search_papers_by_paperid(kg, paperId):
    cypher = """
    MATCH (p:Paper {paperId: $paperId})
    RETURN p
    """
    result = kg.query(cypher, params={'paperId': paperId})
    assert len(result) <= 1
    return result