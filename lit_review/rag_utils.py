import torch
import torch.nn.functional as F
import time
import sys
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings

def compute_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():     
        outputs = model(**inputs) 
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings from token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_embedding_nomic(text, tokenizer, model, config):
    text_with_prefix = "search_document: " + text
    encoded_input = tokenizer(text_with_prefix, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :config['embedding']['size']]
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding_list = embedding.squeeze().tolist()
    return embedding_list

def query_kg(question, llm, kg, config, custom_query, prompt_template, k=30):
    chunk_vector = Neo4jVector.from_existing_index(
        OllamaEmbeddings(model=config['embedding']['model_id']),
        graph=kg, 
        index_name=config['rag']['index_name'],
        embedding_node_property=config['rag']['embedding_node_property'],
        text_node_property=config['rag']['text_node_property'],
        retrieval_query=custom_query,
    )
    question = "search_query: " + question
    retrieved_docs = chunk_vector.similarity_search_with_score(question, k=k)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved_docs])
    prompt = prompt_template.format(context=context_text, question=question)

    buffer = ""
    flush_interval = 0.5  # seconds
    last_flush = time.time()


    for chunk in llm.stream(prompt):
        buffer += chunk
        current_time = time.time()
        # Every flush_interval seconds, output the buffer and reset it.
        if current_time - last_flush >= flush_interval:
            sys.stdout.write(buffer)
            sys.stdout.flush()
            buffer = ""
            last_flush = current_time

    if buffer:
        sys.stdout.write(buffer)
        sys.stdout.flush()