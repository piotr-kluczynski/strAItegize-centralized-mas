from typing import TypedDict, Annotated, Sequence, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

class CentralizedState(TypedDict):
    units: List[str]
    regions: List[str]

    supply_surplus: int = 0
    upkeep_phase: bool = False

    player_id: str
    units_per_general: int
    general_id_counter: int = 0
    
    blocked_tools: list[str]    
    
    recent_events: List[str]
    leader_messages: Annotated[Sequence[BaseMessage], add_messages]

    order_list: List[Order] = []
    recruitment_choice: List[RecruitmentChoice] = []
    
    active_general: str
    generals_units: dict[str, list[str]]
    generals_messages: dict[str, list[BaseMessage]]

    # Shared channels with GeneralState
    general_id: Optional[str]
    general_messages: Optional[Annotated[Sequence[BaseMessage], add_messages]]
    received_orders: Optional[str]
    controlled_units: Optional[dict[str, Optional[Order]]]

class GeneralState(TypedDict):
    """
    State for the General Sub-Agent.

    Commands only part of the player's units, and performes only when activated by the Leader Agent (as an sub-agent).
    """

    general_id: str
    general_messages: Annotated[Sequence[BaseMessage], add_messages]
    received_orders: str
    controlled_units: dict[str, Optional[Order]]
