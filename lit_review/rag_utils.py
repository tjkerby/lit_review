import torch
import torch.nn.functional as F

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