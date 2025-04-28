import gradio as gr
import asyncio
from langchain_core.messages import HumanMessage

import sys
sys.path.append('/home/TomKerby/Research/lit_review/configs')
from new_llm_config import config

sys.path.append('/home/TomKerby/Research/lit_review/lit_review')
import utils
from agent_tools import SearchNeo4jVectorTool
import rag_utils as rag

from langchain_neo4j import Neo4jVector

kg = utils.load_kg(config)
# llm_adapter = rag.get_llm(config)
emb_adapter = rag.get_embeddings(config)

custom_query = """
MATCH (c:Chunk)
WITH DISTINCT c, vector.similarity.cosine(c.textEmbedding, $embedding) AS score
ORDER BY score DESC LIMIT $k
RETURN c.text AS text, score, {source: c.source, chunkId: c.chunkId} AS metadata
"""

chunk_vector = Neo4jVector.from_existing_index(
    emb_adapter.embeddings,
    graph=kg, 
    index_name=config["rag"]["index_name"],
    embedding_node_property=config["rag"]["embedding_node_property"],
    text_node_property=config["rag"]["text_node_property"],
    retrieval_query=custom_query,
)

tool = SearchNeo4jVectorTool(vector_db=chunk_vector)

from langchain_ollama import ChatOllama

llm_config = config['llm']

model = ChatOllama(
    model=llm_config["model_id"],
    num_ctx=llm_config.get("num_ctx", 32768),
    num_predict=llm_config.get("num_predict", 4096),
    temperature=llm_config.get("temperature", 0.5),
    disable_streaming="tool_calling",
)

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

from lit_review import agent

async def main():
    db_path = "./langgraph_agent.db"
    memory = agent.load_memory(db_path, async_run=True)
    abot = agent.Agent(model, [tool], system=prompt, checkpointer=memory)

    async def interact_with_agent(message, chat_history):
        thread = {"configurable": {"thread_id": "1"}}
        full_response = ""

        # Append the user message at the start!
        chat_history.append((message, ""))  # User message, empty assistant response
        yield chat_history

        llm_stream = abot.graph.astream(
            {"messages": [HumanMessage(content=message)], "buffer": ""},
            thread,
            stream_mode=["custom", "updates"],
        )
        async for mode, payload in llm_stream:
            if mode == "custom":
                for chunk in payload["messages"]:
                    if getattr(chunk, "content", None):
                        full_response += chunk.content
                        # Update the last assistant message
                        user_msg, _ = chat_history[-1]
                        chat_history[-1] = (user_msg, full_response)
                        yield chat_history
            elif mode == "updates":
                llm_info = payload.get("llm")
                if llm_info:
                    msgs = llm_info.get("messages", [])
                    if msgs and hasattr(msgs[0], "tool_calls"):
                        for tc in msgs[0].tool_calls:
                            name = tc.get("name")
                            args = tc.get("arguments") or tc.get("args") or {}
                            k = args.get("k", 10)
                            q = args.get("question", args.get("query", ""))
                            chat_history.append((
                                "", 
                                f"🔍 Planning to call `{name}` with k={k}, query={q!r}"
                            ))
                            yield chat_history



    with gr.Blocks(css="""
.system-msg { 
    color: #666;
    border-left: 3px solid #ccc;
    padding-left: 10px;
    margin: 10px 0;
}
""") as demo:
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(label="Input", placeholder="Type your research question...")
        clear = gr.Button("Clear")

        @clear.click
        def clear_chat():
            return [], ""
        
        async def respond(message, chat_history):
            chat_history = chat_history or []
            async for updated in interact_with_agent(message, chat_history):
                # second element clears the textbox
                yield updated, ""


        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
            concurrency_limit=1,
        )
        demo.queue()   
        demo.launch(server_port=7681)

if __name__=="__main__":
    asyncio.run(main())
