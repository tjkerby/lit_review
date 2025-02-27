from lit_review.agent.AgentComponents import (
    AgentNodes,
    AgentState,
    AgentEdges,
)
from langgraph.graph import StateGraph, END

from typing import Any, List
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
import uuid
import logging

logging.basicConfig(
    format="%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("Agent")
logger.setLevel(logging.INFO)


class ResearchAgent:
    """
    A class representing a research agent that manages the workflow of research tasks.

    This agent utilizes a state graph to navigate through various stages of research,
    including planning, writing, reviewing, and revising reports.

    Attributes:
        nodes (AgentNodes): An instance of AgentNodes containing the nodes for the state graph.
        agent (StateGraph): The state graph that manages the research workflow.
    """

    def __init__(self, model: Any, searcher: Any) -> None:
        """
        Initialize the ResearchAgent with a model and a searcher.

        Args:
            model (Any): The model used for generating research content.
            searcher (Any): The searcher used for retrieving relevant information.
        """
        self.nodes = AgentNodes(model, searcher)
        self.edges = AgentEdges()
        logger.info("Setting up agent graph")
        self.agent = self._setup()

    def _setup(self) -> StateGraph:
        """
        Set up the state graph for the research agent.

        This method initializes the state graph, adds nodes, and defines the edges
        that represent the workflow of the research process.

        Returns:
            StateGraph: The configured state graph for the research agent.
        """
        agent = StateGraph(AgentState)

        ## Nodes
        agent.add_node("initial_plan", self.nodes.plan_node)
        agent.add_node("write", self.nodes.generation_node)
        agent.add_node("review", self.nodes.review_node)
        agent.add_node("do_research", self.nodes.research_plan_node)
        agent.add_node("research_revise", self.nodes.research_response_node)
        agent.add_node("reject", self.nodes.reject_node)
        agent.add_node("accept", self.nodes.accept_node)
        agent.add_node("editor", self.nodes.editor_node)

        ## Edges
        agent.set_entry_point("initial_plan")
        agent.add_edge("initial_plan", "do_research")
        agent.add_edge("do_research", "write")
        agent.add_edge("write", "editor")

        ## Conditional edges
        agent.add_conditional_edges(
            "editor",
            self.edges.should_continue,
            {"accepted": "accept", "to_review": "review", "rejected": "reject"},
        )
        agent.add_edge("review", "research_revise")
        agent.add_edge("research_revise", "write")
        agent.add_edge("reject", END)
        agent.add_edge("accept", END)

        return agent

    def display_components(self, stage, verbose=True):

        logger.info("#" * 20)
        level_1_keys = list(stage.keys())
        for k1 in level_1_keys:
            level_2_keys = list(stage[k1].keys())
            logger.info(f"Node : {k1}")
            for k2 in level_2_keys:
                logger.info(f"Task : {k2}")
                if verbose:
                    logger.info(stage[k1][k2])
        logger.info("#" * 20)

    def run_task(self, task_description: str, max_revisions: int = 1) -> List[Any]:
        """
        Execute a research task based on the provided description.

        This method compiles the state graph and streams the task through the agent,
        collecting results along the way.

        Args:
            task_description (str): A description of the task to be executed.
            max_revisions (int): The maximum number of revisions allowed. Defaults to 1.

        Returns:
            List[Any]: A list of results generated during the task execution.
        """

        checkpointer = MemorySaver()
        self.in_memory_store = InMemoryStore()
        graph = self.agent.compile(
            checkpointer=checkpointer, store=self.in_memory_store
        )
        results = []
        # Invoke the graph
        user_id = "1"
        config = {"configurable": {"thread_id": "1", "user_id": user_id}}
        namespace = (user_id, "memories")

        for i, update in enumerate(
            graph.stream(
                {
                    "task": task_description,
                    "max_revisions": max_revisions,
                    "revision_number": 0,
                },
                config,
                stream_mode="updates",
            )
        ):
            self.display_components(update)
            memory_id = str(uuid.uuid4())
            self.in_memory_store.put(namespace, memory_id, {"memory": update})
            results.append(update)
            logger.info(f"Agent at stage {i + 1}")

        return results