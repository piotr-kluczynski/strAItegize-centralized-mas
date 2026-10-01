from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolArg
from langchain.tools import InjectedState
from langchain_core.runnables import RunnableConfig
from typing import Annotated, Literal

from shared.order import Order
from shared.recruitment_choice import RecruitmentChoice
from communication.communication import Communication

from utils import describe_order
from game_rules_store import rules_store

def get_generic_tools(communication: Communication) -> list:
    @tool
    def think_tool(reflection: str) -> str:
        """Tool for strategic reflection on the simulation state and decision-making.

        Use this tool after each interaction with environment, other players or agents within your system and plan next steps systematically.
        This creates a deliberate pause in the planning process for quality decision-making.

        When to use:
        - After receiving the information from the environment: How does new information influences my plans?
        - Before making the final decision: Can I make well informed decision?
        
        Reflection should address:
        1. Analysis of the currently gathered information - What do I know about the problem at hand?
        2. Missing information assessment - What more information can I gather about the problem?
        3. Predicted consequences - What will be the result of considered possiblities?
        4. Strategic decision - Should I continue observing the environment or make the decision now?

        Args:
            reflection: Your detailed reflection on the problem at hand, findings, and next steps

        Returns:
            Confirmation that reflection was recorded for decision-making
        """

        return f"Reflection recorded: {reflection}"

    @tool
    def observe_unit(unit_id: str) -> str:
        """Observe the surroundings around owned unit with given id.

        Args:
            unit_id: Identifier of the unit to observe.

        Returns:
            Textual description of the surroundings within distance of 2 tiles from the unit.
        """

        # Retrieving the description of unit surroundings from the simulation.
        data = communication.request_data(
            request_type = "OBSERVE_UNIT",
            request_args = {"unit_id": unit_id}
        )

        if data is None:
            return "Error: Could not retrieve data from the simulation server."

        return str(data)

    @tool
    def observe_region(region_id: str) -> str:
        """Observe the terrain of the owned region with given id.

        Args:
            unit_id: Identifier of the region to observe.

        Returns:
            Textual description of the entire region.
        """

        # Retrieving the description of region terrain from the simulation.
        data = communication.request_data(
            request_type = "OBSERVE_REGION",
            request_args = {"region_id": region_id}
        )

        if data is None:
            return "Error: Could not retrieve data from the simulation server."

        return str(data)

    @tool
    def get_units(state: Annotated[dict, InjectedState], sender_agent: Annotated[Literal["leader", "general"], InjectedToolArg] = "leader") -> list[str]:
        """Reads the list of all unit controlled by this agent.
        Args:
            sender_agent: The identity of the agent calling the tool.

        Returns:
            List containing the identifiers of all controlled units.    
        """

        # Retrieving the uint list from the simulation.
        data = communication.request_data(
            request_type = "GET_UNITS",
            request_args = None
        )

        if data is None:
            return "Error: Could not retrieve data from the simulation server."

        # If the agent requesting the tool is General Agent, its units list is limited
        if sender_agent == "general":
            filtered_data = [u for u in data.units_list if u in state["controlled_units"]]
            return str(filtered_data)
        
        return str(data)

    @tool
    def get_regions() -> list[str]:
        """Reads the list of all regions controlled by the player.

        Returns:
            List containing the identifiers of all controlled regions.    
        """

        # Retrieving the uint list from the simulation.
        data = communication.request_data(
            request_type = "GET_REGIONS",
            request_args = None
        )

        if data is None:
            return "Error: Could not retrieve data from the simulation server."

        return str(data)

    @tool
    def get_unit_details(unit_id: str) -> str:
        """Read the parameters, such as strength and speed, of any unit with given id.
        
        Args:
            unit_id: Identifier of the unit to analyze.

        Returns:
            Textual description of the unit's parameters.
        """

        # Retrieving the unit description from the simulation.
        data = communication.request_data(
            request_type = "GET_UNIT_DETAILS",
            request_args = {"unit_id": unit_id}
        )

        if data is None:
            return "Error: Could not retrieve data from the simulation server."

        return str(data)

    @tool
    def get_game_rules(category: Literal['Game map', 'Units', 'Player Actions', 'Combat system', 'Round details', 'Economy and Upkeep', 'Player elimination', 'Simulation end conditions'], question: str) -> str:
        """Read the rules of the simulation about the chosen subject.
        
        Args:
            category: The category of the query regarding the game rules. Existing categories: Game map, Units, Player Actions, Combat system, Round details, Economy and Upkeep, Player elimination, Simulation end conditions.
            question: The question about the simulation rules.
        
        Returns:
            Game rules explaining the given topic.
        """

        query_results = rules_store.similarity_search(
            question,
            k = 3,
            filter={"category": category}
        )

        response = """The game facts retrieved from the game rules vector store:"""
        for result in query_results:
            response += f"- {result}\r\n"

        return response

    return [think_tool, observe_unit, observe_region, get_units, get_regions, get_unit_details, get_game_rules]

# General-Agent Tools
def get_general_tools(communication: Communication) -> list:

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
        order_id: str = Field(
            description="Identifier of the order to remove."
        )

    return [AssignOrder, RemoveOrder] + get_generic_tools(communication)

# Leader-Agent Tools
def get_leader_tools(communication: Communication) -> list:
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
            request_type = "GET_PLAYERS",
            request_args = None
        )

        if data is None:
            return "Error: Could not retrieve data from the simulation server."

        return str(data)

    @tool
    class MessagePlayer(BaseModel):
        """Tool for sending a text message to another player."""
        player_id: str = Field(
            description="Identifier of the player."
        )
        message: str = Field(
            description="Content of the message."
        )

    @tool
    def get_player_conversation(player_id: str) -> str:
        """Read the conversation history with another player with given id.
        
        Args:
            player_id: Identifier of the player

        Returns:
            The conversation history between you and the player.
        """

        # Retrieve the conversation history from the simulation

        return "To be implemented"

    @tool
    def get_order_list(state: Annotated[dict, InjectedState]) -> str:
        """Read the current list of orders given to the units.

        Returns:
            The list of orders assigned this round.
        """
        data = []

        order_list = state["order_list"]
        for order in order_list:
            data.append(describe_order(order))

        return data

    @tool
    def get_recent_events(state: Annotated[dict, InjectedState]) -> str:
        """Read the list of all events that took place this round.

        Returns:
            The list of events from this round.    
        """
        return state["recent_events"]

    # Add MessagePlayer and get_player_conversation later
    return [RecruitUnit, DelegateGeneral, get_players, get_order_list, get_recent_events] + get_generic_tools(communication)