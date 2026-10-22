from typing import TypedDict, Annotated, Sequence, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

class LeaderInputState(TypedDict):
    """
    Input state for the Leader Agent.

    Contains additional information which is needed at the beginning of the graph invokation,
    and not during its inner workings.
    """

    regions: List[str]

    supply_surplus: int = 0
    upkeep_phase: bool = False

    player_id: str
    recent_events: List[str]
    leader_messages: Annotated[Sequence[BaseMessage], add_messages]
    order_list: List[Order] = []
    recruitment_choice: List[RecruitmentChoice] = []
    generals_units: dict[str, list[str]]
    generals_messages: dict[str, list[BaseMessage]]


class LeaderState(TypedDict):
    """
    State for the Leader Agent.

    Maintains most control of the MAS over the player actions,
    however to perform the military actions (like give order to a unit)
    it invokes sub-agents.
    """

    regions: List[str]
    player_id: str
    recent_events: List[str]
    leader_messages: Annotated[Sequence[BaseMessage], add_messages]
    order_list: List[Order]

    generals_units: dict[str, list[str]]
    generals_messages: dict[str, list[BaseMessage]]

    recruitment_choice: List[RecruitmentChoice]

class LeaderOutputState(TypedDict):
    """
    Output state for the Leader Agent.

    Contains only information necessary for executing player's decisions,
    and maintaining the state of the Leader agent.
    """

    leader_messages: Annotated[Sequence[BaseMessage], add_messages]

    generals_units: dict[str, list[str]]
    generals_messages: dict[str, list[BaseMessage]]

    order_list: List[Order]
    recruitment_choice: List[RecruitmentChoice]

class GeneralState(TypedDict):
    """
    State for the General Sub-Agent.

    Commands only part of the player's units, and performes only when activated by the Leader Agent (as an sub-agent).
    """

    general_id: str
    general_messages: Annotated[Sequence[BaseMessage], add_messages]
    received_orders: str
    controlled_units: dict[str, Optional[Order]]

class GeneralOutputState(TypedDict):
    """
    Output state for the General Sub-Agent.

    Contains only the data necessary to maintain the state of the sub-agent, 
    and explain to the Leader Agent taken actions.
    """

    general_messages: Annotated[Sequence[BaseMessage], add_messages]
    controlled_units: dict[str, Optional[Order]]
    report: str