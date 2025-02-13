import time, sys

from langchain_ollama.llms import OllamaLLM
from langchain.llms import OpenAI
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

class BaseLLM:
    def stream(self, prompt: str):
        raise NotImplementedError("stream() must be implemented by subclasses.")

class OllamaAdapter(BaseLLM):
    def __init__(self, llm_config):
        self.llm = OllamaLLM(
            model=llm_config["model_id"],
            num_ctx=llm_config.get("num_ctx", 32768),
            num_predict=llm_config.get("num_predict", 4096),
            temperature=llm_config.get("temperature", 0.5)
        )
    def stream(self, prompt: str):
        return self.llm.stream(prompt)

class OpenAIAdapter(BaseLLM):
    def __init__(self, llm_config):
        self.llm = OpenAI(
            model=llm_config["model_id"],
            temperature=llm_config.get("temperature", 0.5),
            openai_api_key=llm_config.get("api_key")
        )
    def stream(self, prompt: str):
        response = self.llm(prompt)
        yield response

import time
import torch

class HuggingFaceAdapter(BaseLLM):
    def __init__(self, llm_config):
        max_seq_len = llm_config.get("max_seq_len", 4096)
        self.max_seq_len = max_seq_len
        self.temperature = llm_config.get("temperature", 0.5)
        model_config = AutoConfig.from_pretrained(llm_config["model_id"], trust_remote_code=True)
        model_config.max_seq_len = max_seq_len
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_config["model_id"],
            config=model_config,
            trust_remote_code=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_config["model_id"], trust_remote_code=True)
        self.tokenizer.model_max_length = max_seq_len
   
    def stream(self, prompt: str):
        import torch

        self.model.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        generated_ids = input_ids

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        past = outputs.past_key_values

        while True:
            last_token_id = generated_ids[:, -1].unsqueeze(-1)
            with torch.no_grad():
                outputs = self.model(input_ids=last_token_id, past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = outputs.past_key_values

            next_token_logits = logits[:, -1, :] / self.temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            token_id = next_token.item()
            decoded = self.tokenizer.decode(next_token.squeeze(), skip_special_tokens=True)
            yield decoded

            if token_id == self.tokenizer.eos_token_id or generated_ids.shape[-1] >= self.max_seq_len:
                break     
    # def stream(self, prompt: str, max_new_tokens=100):
    #     # Encode the prompt with attention mask.
    #     encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)
    #     input_ids = encoded.input_ids.to(self.model.device)
    #     attention_mask = encoded.attention_mask.to(self.model.device)
        
    #     buffer = ""
    #     last_flush = time.time()
        
    #     # Iteratively generate one token at a time.
    #     for _ in range(max_new_tokens):
    #         outputs = self.model.generate(
    #             input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=50,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
    #         # Extract the newly generated token (the last token)
    #         new_token_id = outputs[0][-1].unsqueeze(0)
            
    #         # Update input_ids and attention_mask to include the new token
    #         input_ids = torch.cat([input_ids, new_token_id.unsqueeze(0)], dim=-1)
    #         attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.model.device)], dim=-1)
            
    #         # Decode the new token
    #         new_token = self.tokenizer.decode(new_token_id, skip_special_tokens=True)
    #         buffer += new_token
            
    #         # Flush the buffer every 0.5 seconds
    #         if time.time() - last_flush >= 0.5:
    #             yield buffer
    #             buffer = ""
    #             last_flush = time.time()
            
    #         # Stop if EOS token is generated
    #         if new_token_id.item() == self.tokenizer.eos_token_id:
    #             break
        
    #     if buffer:
    #         yield buffer


# class HuggingFaceAdapter(BaseLLM):
#     def __init__(self, llm_config):
#         max_seq_len = llm_config.get("max_seq_len", 4096)     
#         model_config = AutoConfig.from_pretrained(llm_config["model_id"], trust_remote_code=True)
#         model_config.max_seq_len = max_seq_len
#         self.model = AutoModelForCausalLM.from_pretrained(
#             llm_config["model_id"],
#             config=model_config,
#             trust_remote_code=True,
#             device_map="auto"
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(llm_config["model_id"], trust_remote_code=True)
#         self.tokenizer.model_max_length = max_seq_len
        
#         # device = 1 if torch.cuda.is_available() else -1
#         hf_pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             # device=device,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
        
#         self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
#     # def stream(self, prompt: str):
#     #     # HuggingFacePipeline doesn't provide streaming natively,
#     #     # so here we simply yield the full response as one chunk.
#     #     response = self.llm.invoke(prompt)
#     #     yield response

#     def stream(self, prompt: str):
#         input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
#         generated_ids = input_ids
#         max_new_tokens = 100  
#         for _ in range(max_new_tokens):
#             outputs = self.model.generate(generated_ids, max_new_tokens=1, do_sample=True)
#             new_token_id = outputs[0][-1].unsqueeze(0)
#             generated_ids = torch.cat([generated_ids, new_token_id.unsqueeze(0)], dim=-1)
#             new_token = self.tokenizer.decode(new_token_id, skip_special_tokens=True)
#             yield new_token
#             if new_token_id.item() == self.tokenizer.eos_token_id:
#                 break

    
def get_llm(config):
    provider = config["llm"]["provider"].lower()
    if provider == "ollama":
        return OllamaAdapter(config["llm"])
    elif provider == "openai":
        return OpenAIAdapter(config["llm"])
    elif provider == "huggingface":
        return HuggingFaceAdapter(config["llm"])
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

class BaseEmbeddings:
    def embed_query(self, text: str):
        raise NotImplementedError("embed_query() must be implemented.")
    def prepare_query(self, query: str):
        # Default is no extra processing
        return query

class NomicEmbeddingAdapter(BaseEmbeddings):
    def __init__(self, emb_config):
        self.embeddings = OllamaEmbeddings(model=emb_config["model_id"])
        self.query_prefix = emb_config.get("query_prefix", "")
    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)
    def prepare_query(self, query: str):
        return self.query_prefix + query

class OpenAIEmbeddingAdapter(BaseEmbeddings):
    def __init__(self, emb_config):
        self.embeddings = OpenAIEmbeddings(model=emb_config["model_id"],
                                           openai_api_key=emb_config.get("api_key"))
    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)
    def prepare_query(self, query: str):
        return query

class HuggingFaceEmbeddingAdapter(BaseEmbeddings):
    def __init__(self, emb_config):
        # Use the model name from the configuration and pass any desired kwargs.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=emb_config["model_id"],
            model_kwargs={'trust_remote_code': True}
        )
    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)
    def prepare_query(self, query: str):
        # No extra processing needed by default.
        return query

def get_embeddings(config):
    provider = config["embedding"]["provider"].lower()
    if provider == "nomic":
        return NomicEmbeddingAdapter(config["embedding"])
    elif provider == "openai":
        return OpenAIEmbeddingAdapter(config["embedding"])
    elif provider == "huggingface":
        return HuggingFaceEmbeddingAdapter(config["embedding"])
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings


# def query_kg(question, llm, kg, config, custom_query, prompt_template, k=30):
#     chunk_vector = Neo4jVector.from_existing_index(
#         OllamaEmbeddings(model=config['embedding']['model_id']),
#         graph=kg, 
#         index_name=config['rag']['index_name'],
#         embedding_node_property=config['rag']['embedding_node_property'],
#         text_node_property=config['rag']['text_node_property'],
#         retrieval_query=custom_query,
#     )
#     question = "search_query: " + question
#     retrieved_docs = chunk_vector.similarity_search_with_score(question, k=k)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved_docs])
#     prompt = prompt_template.format(context=context_text, question=question)

#     buffer = ""
#     flush_interval = 0.5  # seconds
#     last_flush = time.time()


#     for chunk in llm.stream(prompt):
#         buffer += chunk
#         current_time = time.time()
#         # Every flush_interval seconds, output the buffer and reset it.
#         if current_time - last_flush >= flush_interval:
#             sys.stdout.write(buffer)
#             sys.stdout.flush()
#             buffer = ""
#             last_flush = current_time

#     if buffer:
#         sys.stdout.write(buffer)
#         sys.stdout.flush()

def query_kg(question, llm_adapter, emb_adapter, kg, config, custom_query, prompt_template, k=30):
    prepared_question = emb_adapter.prepare_query(question)
    
    chunk_vector = Neo4jVector.from_existing_index(
        emb_adapter.embeddings,
        graph=kg, 
        index_name=config["rag"]["index_name"],
        embedding_node_property=config["rag"]["embedding_node_property"],
        text_node_property=config["rag"]["text_node_property"],
        retrieval_query=custom_query,
    )
    
    retrieved_docs = chunk_vector.similarity_search_with_score(prepared_question, k=k)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved_docs])
    prompt = prompt_template.format(context=context_text, question=question)
    buffer = ""
    flush_interval = 0.5  # seconds
    last_flush = time.time()
    for chunk in llm_adapter.stream(prompt):
        buffer += chunk
        if time.time() - last_flush >= flush_interval:
            sys.stdout.write(buffer)
            sys.stdout.flush()
            buffer = ""
            last_flush = time.time()
    if buffer:
        sys.stdout.write(buffer)
        sys.stdout.flush()