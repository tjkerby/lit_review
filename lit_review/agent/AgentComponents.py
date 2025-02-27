from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from lit_review.agent.prompts import (
    ResearchPlanPrompt,
    ResearchEditorPrompt,
    ResearchResponsePrompt,
    ResearchReviewPrompt,
    ResearchWritePrompt,
    ResearchQueryPrompt,
)

from typing import List, Dict, Any
from pydantic import BaseModel
from typing_extensions import TypedDict
import logging

logging.basicConfig(
    format="%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("AgentNodes")
logger.setLevel(logging.INFO)


class AgentState(TypedDict):
    """
    A dictionary representing the state of the research agent.

    Attributes:
        task (str): The description of the task to be performed.
        plan (str): The research plan generated for the task.
        draft (str): The current draft of the research report.
        critique (str): The critique received for the draft.
        content (List[str]): A list of content gathered during research.
        revision_number (int): The current revision number of the draft.
        max_revisions (int): The maximum number of revisions allowed.
        finalized_state (bool): Indicates whether the report is finalized.
    """

    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    editor_comment: str
    revision_number: int
    max_revisions: int
    finalized_state: bool


class Queries(BaseModel):
    """
    A model representing a list of search queries.

    Attributes:
        queries (List[str]): A list of search queries to be executed.
    """

    queries: List[str]
    
    
class FinalizedState(BaseModel):
    """
    A model representing the finalized state of a report.

    Attributes:
        state (bool): Indicates whether the report is finalized.
    """

    state: bool
    reasoning: str
    

class AgentNodes:
    """
    A class that encapsulates the nodes of the research agent's state graph.

    This class contains methods for handling various stages of the research process,
    including planning, writing, reviewing, and revising reports.

    Attributes:
        model (Any): The language model used for generating content.
        searcher (Any): The searcher used for retrieving relevant information.
    """

    def __init__(self, llm: Any, searcher: Any) -> None:
        """
        Initialize the AgentNodes with a language model and a searcher.

        Args:
            llm (Any): The language model used for generating research content.
            searcher (Any): The searcher used for retrieving relevant information.
        """
        self.model = llm
        self.searcher = searcher

    def plan_node(self, state: AgentState) -> Dict[str, str]:
        """
        Generate a research plan based on the current state.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, str]: A dictionary containing the generated research plan.
        """
        messages = [
            SystemMessage(content=ResearchPlanPrompt.system_template),
            HumanMessage(content=state["task"]),
        ]
        response = self.model.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState) -> Dict[str, List[str]]:
        """
        Generate search queries and perform research based on the current state.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, List[str]]: A dictionary containing the gathered content.
        """
        queries = self.model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=ResearchQueryPrompt.system_template),
                HumanMessage(content=state["task"]),
            ]
        )
        content = state.get("content", [])

        for q in queries.queries:
            response = self.searcher.invoke({'question':q, 'k':5})
            response = response.split("search_document:")[1:]
            for r in response:
                content.append("search_document:" + r)
        return {"content": content}

    def generation_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate a draft based on the current state and the research plan.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, Any]: A dictionary containing the generated draft and updated revision number.
        """
        content = "\n\n".join(state.get("content", []))
        reviewer_critique = state.get(
            "critique",
            "The article has not been reviewed yet, so there are no reviewer comments available",
        )
        previous_draft = state.get("draft", "There is no previous draft available")
        editor_comment = state.get(
            "editor_comment", "There is no previous editor comment available"
        )
        messages = [
            SystemMessage(
                content=ResearchWritePrompt.system_template.format(content=content)
            ),
            HumanMessage(content="The original task: {}".format(state["task"])),
            HumanMessage(content="The original plan: {}".format(state["plan"])),
            HumanMessage(content="The previous draft: {}".format(previous_draft)),
            HumanMessage(
                content="The reviewer comments about the previous draft: {}".format(
                    reviewer_critique
                )
            ),
            HumanMessage(
                content="The editor comments about the previous draft and review: {}".format(
                    editor_comment
                )
            ),
        ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", -1) + 1,
        }

    def review_node(self, state: AgentState) -> Dict[str, str]:
        """
        Review the current draft and generate critique.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, str]: A dictionary containing the generated critique.
        """
        messages = [
            SystemMessage(content=ResearchReviewPrompt.system_template),
            HumanMessage(content=state["draft"]),
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content}

    def research_response_node(self, state: AgentState) -> Dict[str, List[str]]:
        """
        Generate search queries based on the critique and perform research.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, List[str]]: A dictionary containing the gathered content.
        """
        queries = self.model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=ResearchResponsePrompt.system_template),
                HumanMessage(content=state["critique"]),
            ]
        )
        content = state["content"] or []
        for q in queries.queries:
            response = self.searcher.invoke({'question':q, 'k':5})
            response = response.split("search_document:")[1:]
            for r in response:
                content.append("search_document:" + r)
        return {"content": content}

    def editor_node(self, state: AgentState):

        # always send to review if we don't have a review yet
        current_reviewer_comments = state.get("critique", [])
        if not current_reviewer_comments:
            current_reviewer_comments = "The article has not been reviewed yer"

        continue_state = self.model.with_structured_output(FinalizedState).invoke(
            [
                SystemMessage(content=ResearchEditorPrompt.system_template),
                HumanMessage(
                    content="The previous critique: {}".format(
                        current_reviewer_comments
                    )
                ),
                HumanMessage(
                    content="The article has been through {} of revision".format(
                        state["revision_number"]
                    )
                ),
                HumanMessage(content="The original task {}".format(state["task"])),
                HumanMessage(content="The current essay: {}".format(state["draft"])),
            ]
        )
        editor_accepts = continue_state.state
        editor_reasoning = continue_state.reasoning

        return {"finalized_state": editor_accepts, "editor_comment": editor_reasoning}

    def reject_node(self, state: AgentState) -> Dict[str, bool]:
        """
        Indicate that the report has been rejected.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, bool]: A dictionary indicating the finalized state as False.
        """
        return {"finalized_state": False}

    def accept_node(self, state: AgentState) -> Dict[str, bool]:
        """
        Indicate that the report has been accepted.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            Dict[str, bool]: A dictionary indicating the finalized state as True.
        """
        return {"finalized_state": True}


class AgentEdges:

    @staticmethod
    def should_continue(state: AgentState) -> str:
        """
        Determine whether the research process should continue based on the current state.

        Args:
            state (AgentState): The current state of the research agent.

        Returns:
            str: The next state to transition to ("to_review", "accepted", or "rejected").
        """
        # always send to review if editor hasn't made comments yet
        current_editor_comments = state.get("editor_comment", [])
        if not current_editor_comments:
            return "to_review"

        final_state = state.get("finalized_state", False)
        if final_state:
            return "accepted"
        elif state["revision_number"] > state["max_revisions"]:
            logger.info("Revision number > max allowed revisions")
            return "rejected"
        else:
            return "to_review"