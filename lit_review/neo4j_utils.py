def create_node(tx, label, properties):
    query = f"CREATE (n:{label} $properties)"
    tx.run(query, properties=properties)
    
def create_relationship(tx, label1, properties1, relationship_type, label2, properties2):
    query = (
        f"MATCH (a:{label1}),(b:{label2}) "
        f"WHERE a.title = $title1 AND b.title = $title2 "
        f"CREATE (a)-[r:{relationship_type}]->(b)"
    )
    tx.run(query, title1=properties1['title'], title2=properties2['title'])
    
def create_authored_rel(tx, properties1, properties2):
    query = (
        f"MATCH (p:{'Paper'}),(a:{'Author'}) "
        f"WHERE p.paperId = $paperId AND a.authorId = $authorId "
        f"MERGE (a)-[r:{'Authored'}]->(p)"
    )
    tx.run(query, paperId=properties1['paperId'], authorId=properties2['authorId'])
    
def create_cites_rel(tx, properties1, properties2):
    query = (
        f"MATCH (p1:{'Paper'}),(p2:{'Paper'}) "
        f"WHERE p1.paperId = $paperId1 AND p2.paperId = $paperId2 "
        f"MERGE (p1)-[r:{'Cited'}]->(p2)"
    )
    tx.run(query, paperId1=properties1['paperId'], paperId2=properties2['paperId'])

def get_or_create_node(tx, label, properties):
    query = f"MATCH (n:{label} {{title: $properties.title}}) RETURN n"
    result = tx.run(query, properties=properties)
    node = result.single()
    if node is not None:
        return node["n"]
    else:
        return create_node(tx, label, properties)
    
def get_or_create_paper_node(tx, properties):
    query = f"MATCH (p:Paper {{paperId: $properties.paperId}}) RETURN p"
    result = tx.run(query, properties=properties)
    node = result.single()
    if node is not None:
        return node["p"]
    else:
        return create_node(tx, "Paper", properties)
    
def get_or_create_author_node(tx, author):
    query = f"MATCH (a:Author {{authorId: $author.authorId}}) RETURN a"
    result = tx.run(query, author=author)
    node = result.single()
    if node is not None:
        return node["a"]
    else:
        query = "MERGE (n:Author {authorId: $authorId, name: $name}) RETURN n"
        result = tx.run(query, authorId=author['authorId'], name=author['name'])
        return result.single()["n"]

