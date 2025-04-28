from typing import Optional, Type
from langchain_neo4j import Neo4jVector
# from langchain_core.vectorstores import VectorStore

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

class SearchChunksInput(BaseModel):
    question: str = Field(description="input for searching vector database")
    k: int = Field(description="number of chunks to retrieve")
    
    class Config:
        arbitrary_types_allowed = True


class SearchNeo4jVectorTool(BaseTool):
    name: str = "search_chunks"
    description: str = "Searches chunks for relevant context relative to the input question."
    args_schema: Type[BaseModel] = SearchChunksInput
    return_direct: bool = True
    _vector_db: Neo4jVector = PrivateAttr()
    
    def __init__(self, vector_db: Neo4jVector):
        super().__init__()
        self._vector_db  = vector_db

    def _run(
        self, question: str, k: int=10, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        retrieved_docs = self._vector_db.similarity_search_with_score(question, k=k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved_docs])
        return context_text

    async def _arun(
        self,
        question: str,
        k: int=10, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(question, k, run_manager=run_manager.get_sync())