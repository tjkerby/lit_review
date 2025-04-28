# import asyncio
# import json
# from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
# from langgraph.graph import StateGraph, END, START
# from typing import TypedDict, Annotated
# import operator

# async def run_agent(abot, messages, thread):
#     buffer = ""
#     async for event in abot.graph.astream({"messages": messages, "buffer": ""}, thread, stream_mode="updates"):
#         if 'buffer' in event:
#             new_content = event['buffer']
#             buffer += new_content
#             print(new_content, end='', flush=True)

# def load_memory(db_path, async_run=True):
#     if async_run:
#         import aiosqlite
#         from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
#         conn = aiosqlite.connect(db_path)
#         memory = AsyncSqliteSaver(conn)
#     else:
#         import sqlite3
#         from langgraph.checkpoint.sqlite import SqliteSaver
#         conn = sqlite3.connect(db_path, check_same_thread=False)
#         memory = SqliteSaver(conn)
#     return memory

# class AgentState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]
#     buffer: str

# class Agent:
#     def __init__(self, model, tools, checkpointer, system=""):
#         self.system = system
#         graph = StateGraph(AgentState)
#         graph.add_node("llm", self.call_model)
#         graph.add_edge(START, "llm")
#         graph.add_node("action", self.take_action)
#         graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
#         graph.add_edge("action", "llm")
#         graph.set_entry_point("llm")
#         graph.add_node("stream_buffer", self.handle_partial_outputs)
#         graph.add_edge("llm", "stream_buffer")
#         graph.add_edge("stream_buffer", END)
#         self.graph = graph.compile(checkpointer=checkpointer)
#         self.tools = {t.name: t for t in tools}
#         self.model = model.bind_tools(tools)

#     async def call_model(self, state: AgentState):
#         # print(">>> FULL STATE MESSAGES for this run:", state["messages"])
#         messages = state['messages']
#         if self.system:
#             messages = [SystemMessage(content=self.system)] + messages
#         async for chunk in self.model.astream(messages):
#             # print(f"Received chunk: {chunk}")
#             yield {"messages": [chunk], "buffer": chunk.content if chunk.content else ""}
#             if chunk.tool_calls:
#                 break

#     def handle_partial_outputs(self, state: AgentState):
#         buffer = state.get('buffer', '')
#         return {'buffer': buffer}

#     def exists_action(self, state: AgentState):
#         result = state['messages'][-1]
#         return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

#     async def take_action(self, state: AgentState):
#         tool_calls = state['messages'][-1].tool_calls
#         tasks = [asyncio.create_task(self._execute_tool(t)) 
#                 for t in tool_calls]
        
#         results = []
#         while tasks:
#             done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
#             for task in done:
#                 results.append(await task)
        
#         return {'messages': results}

#     async def _execute_tool(self, tool_call):
#         print(f"Calling: {tool_call}")
#         tool_name = tool_call.name if hasattr(tool_call, 'name') else tool_call.get('name')
        
#         # Handle different possible structures of tool_call
#         if hasattr(tool_call, 'arguments'):
#             tool_args = tool_call.arguments
#         elif 'arguments' in tool_call:
#             tool_args = tool_call['arguments']
#         elif 'args' in tool_call:
#             tool_args = tool_call['args']
#         else:
#             tool_args = {}

#         if isinstance(tool_args, str):
#             try:
#                 tool_args = json.loads(tool_args)
#             except json.JSONDecodeError:
#                 tool_args = {"question": tool_args}

#         if not isinstance(tool_args, dict):
#             tool_args = {"question": str(tool_args)}

#         if 'k' not in tool_args:
#             tool_args['k'] = 10 

#         result = await self.tools[tool_name].ainvoke(input=tool_args)
#         return ToolMessage(
#             tool_call_id=tool_call.id if hasattr(tool_call, 'id') else tool_call.get('id', ''),
#             name=tool_name,
#             content=str(result)
#         )

import asyncio
import json
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
import operator

def load_memory(db_path, async_run=True):
    if async_run:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = aiosqlite.connect(db_path)
        memory = AsyncSqliteSaver(conn)
    else:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn)
    return memory

# Define the state without a separate 'buffer' field
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        # Build a simple graph: llm -> [action?] -> END
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_edge(START, "llm")
        graph.add_node("action", self.take_action)
        # If the LLM produced tool calls, go to `action`; otherwise, end
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        # After tooling, loop back to the model
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        # Compile without a stream_buffer node
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    from langgraph.types import StreamWriter
    async def call_model(self, state: AgentState, writer: StreamWriter):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        collected = []
        async for chunk in self.model.astream(messages):

            collected.append(chunk)
            # import asyncio
            # await asyncio.sleep(0.5)
            writer({"messages": [chunk]})
            if getattr(chunk, "tool_calls", None):
                break
        return {"messages": collected}

    def exists_action(self, state: AgentState) -> bool:
        # Check if the most recent message triggered tool calls
        last = state['messages'][-1]
        return hasattr(last, 'tool_calls') and bool(last.tool_calls)

    async def take_action(self, state: AgentState):
        # Execute any pending tool calls
        tool_calls = state['messages'][-1].tool_calls
        tasks = [asyncio.create_task(self._execute_tool(tc)) for tc in tool_calls]

        results = []
        while tasks:
            done, tasks = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                results.append(await task)

        # Return all tool outputs as messages
        return {"messages": results}

    async def _execute_tool(self, tool_call):
        # print(f"Calling: {tool_call}")
        tool_name = getattr(tool_call, 'name', tool_call.get('name'))
        args = getattr(tool_call, 'arguments', None) or tool_call.get('arguments') or tool_call.get('args') or {}

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"question": args}
        if not isinstance(args, dict):
            args = {"question": str(args)}

        # Default 'k' if not provided
        args.setdefault('k', 10)

        # Invoke and wrap as ToolMessage
        result = await self.tools[tool_name].ainvoke(input=args)
        return ToolMessage(
            tool_call_id=getattr(tool_call, 'id', tool_call.get('id', '')),
            name=tool_name,
            content=str(result)
        )
