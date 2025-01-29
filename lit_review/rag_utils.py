import torch

def compute_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():     
        outputs = model(**inputs) 
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()