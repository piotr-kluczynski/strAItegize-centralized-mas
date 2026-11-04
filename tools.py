from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.tools import InjectedState
from typing import Annotated

from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.communication.message_types import MessageType

from strAItegize_common.others.order import describe_order
from strAItegize_common.client.tools import get_generic_tools

# General-Agent Tools
def get_general_tools(communication: CommunicationContext) -> list:

    @tool
    class AssignOrder(BaseModel):
        """Tool for assigning an order to the owned unit."""
        unit_id: str = Field(
            description="Identifier of the unit to assign order to."
        )

        unit_order: Order = Field(
            description="The order to assign to the unit."
        )

    @tool
    class RemoveOrder(BaseModel):
        """Tool for removing already assigned orders."""
        unit_id: str = Field(
            description="Identifier of the unit to remove order from."
        )

    return [AssignOrder, RemoveOrder] + get_generic_tools(communication)

# Leader-Agent Tools
def get_leader_tools(communication: CommunicationContext) -> list:
    @tool
    class RecruitUnit(BaseModel):
        """Tool for adding a recruitment choice to the recruitment list."""
        recruitment_choice: RecruitmentChoice = Field(
            description="The choice of where to recruit which type of unit."
        )

    @tool
    class DelegateGeneral(BaseModel):
        """Tool for delegating the military action to the chosen general agent."""
        general_id: str = Field(
            description="Identifier of the general agent."
        )

        command: str = Field(
            description="Orders for the general agent to follow."
        )

    @tool 
    def get_players() -> list[str]:
        """Get the list of the remaining players.
        
        Returns:
            List of remaining player's identifiers.
        """

        # Retrieving the player lit from the simulation.
        data = communication.request_data(
            request_type = MessageType.GetPlayers,
            request_args = None
        )

        if isinstance(data, dict):
            if data.get("error"):
                return f"An error has occurred during tool execution.\r\nDescription: {data["desc"]}\r\nDetails: {data['details']}"

        # Formatting the tool output to be more readable for the LLM
        if len(data.players_list) > 0:
            observation = f"Player's participating within the simulation:"
            for player in data.players_list:
                observation += f"{player}, "
            observation = observation[:-2]
            observation += "."
        else:
            observation = "No players still participate within the simulation. You shouldn't be here."

        return observation

    @tool
    def get_order_list(state: Annotated[dict, InjectedState]) -> str:
        """Read the current list of orders given to the units.

        Returns:
            The list of orders assigned this round.
        """
        order_list = state["order_list"]

        # Formatting the tool output to be more readable for the LLM
        if len(order_list) > 0:
            observation = f"Currently assigned orders:\r\n"
            for order in order_list:
                observation += f"- {describe_order(order)};"
            observation += "."
        else:
            observation = "There are no orders yet, assigned to your units."

        return observation

    @tool
    def get_recent_events(state: Annotated[dict, InjectedState]) -> str:
        """Read the list of all events that took place this round.

        Returns:
            The list of events from this round.    
        """
        events = state["recent_events"]

        # Formatting the tool output to be more readable for the LLM
        if len(events) > 0:
            observation = "Events that had occurred on this round:\r\n"
            for event in events:
                observation += f"- {event},"
            observation += "."
        else:
            observation = "There've been no events since the last round."

        return observation

    return [RecruitUnit, DelegateGeneral, get_players, get_order_list, get_recent_events] + get_generic_tools(communication)