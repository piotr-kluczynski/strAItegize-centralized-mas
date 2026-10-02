from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from strAItegize_common.others.human import human

from general_agent import get_general_agent
from prompts import leader_agent_prompt
from states import LeaderState, LeaderInputState, LeaderOutputState, GeneralState
from tools import get_leader_tools, get_general_tools

# Temporary variable set up by graph initializer
HUMAN_MODE = False

# Dynamic list of tools initialization
leader_tools = []
general_tools = []

available_leader_tools = []
tools_by_name = {}

# Initializing llm
model = init_chat_model(
    model="openai:gpt-4o"
)

# Agent nodes
def input_node(state: LeaderInputState) -> LeaderState:
    """Prepares the data and tool selection before the agent loop begins.
    
    Args:
        state: Leader Input State containing additional information necessary to execute the upkeep phase.

    Returns:
        Leader state, without additional information about the round
    """
    global leader_tools, available_leader_tools, tools_by_name
    leader_messages = []

    # Dynamically reducing tools if invokation happened as result of the unit recruitment process
    if state["upkeep_phase"]:
        available_leader_tools = [tool for tool in leader_tools if tool.name not in ["DelegateGeneral", "get_order_list", "get_recent_events"]]

    tools_by_name = {tool.name: tool for tool in available_leader_tools}

    return {
        "leader_messages": leader_messages,
        "player_id": state["player_id"],
        "recent_events": state["recent_events"],
        "order_list": state["order_list"],
        "regions": state["regions"],
        "generals_units": state["generals_units"],
        "generals_messages": state["generals_messages"],
        "recruitment_choice": state["recruitment_choice"]
    }

def leader(state: LeaderState) -> LeaderState:
    """Makes the macro-decisions within the simulation, delegating micro-decisions to their generals.
    
        Analyzes the current state of the simulation and tries to gain an advantage by:
        - observing the simulation environment
        - using diplomacy with other players
        - analyzing different possibilities
        - delegating tasks to the sub-agents
        - recruiting new units
    
        Args:
            state: Current Leader State with the list of messages, controlled regions, currently assigned orders and list of sub-agents.

        Returns:
            New, updated Leader State.
        
    """

    return {
        "leader_messages": [
            model.bind_tools(available_leader_tools).invoke(
                [SystemMessage(content=leader_agent_prompt(state["player_id"], state["generals_units"], state["regions"]))] + state["leader_messages"],
                config={""}
            )
        ]
    }

def human_leader(state: LeaderState) -> LeaderState:
    return {
        "leader_messages":
            human(
                [SystemMessage(content=leader_agent_prompt(state["player_id"], state["generals_units"], state["regions"]))] + state["leader_messages"],
                agent_name="leader",
                tools=available_leader_tools
            )
        
    }

def tool_node(state: LeaderState) -> LeaderState:
    """Execute Leader Agent tool calls.

    Handles:
    - Executing think_tool for strategic reflection.
    - Observing simulation environment.
    - Communicating with other players.
    - Aggregating learned information.
    - Assign tasks to the sub-agents.
    
    Args:
        state: Current Leader State with the list of messages, controlled regions, currently assigned orders and list of sub-agents.

    Returns:
        New, updated state with tool messages.
    """
    tool_calls = state["leader_messages"][-1].tool_calls
    observations = []

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]

        # If cases for specific tools
        if tool_call["name"] == "DelegateGeneral":
            general_id = tool_call["args"]["general_id"]
            command = tool_call["args"]["command"]

            general_input_state = GeneralState(
                general_messages=state["generals_messages"][general_id],
                received_orders=command,
                controlled_units=state["generals_units"][general_id]
            )

            general_output_state = get_general_agent(general_tools, human_mode=HUMAN_MODE).invoke(general_input_state)

            # Updating general's message log
            state["generals_messages"][general_id] = general_output_state["general_messages"]

            # Updating orders list
            for order in general_output_state["controlled_units"].values():
                state["order_list"].append(order)

            observations.append(general_output_state["report"])
        elif tool_call["name"] == "RecruitUnit":
            recruitment_choice = tool_call["args"]["recruitment_choice"]

            state["recruitment_choice"].append(recruitment_choice)

            observations.append(f"Unit recruitment choice: {recruitment_choice.region}(region): {recruitment_choice.unit_type}(unit type) was successfuly added to the list.")
        
        elif tool_call["name"] == "MessagePlayer":
            # Use outer functions to send message to another player within simultion
            pass
        elif tool_call["name"] == "get_units":
            args = {
                **tool_call["args"],
                "sender_agent": "leader"
            }
            observations.append(tool.invoke(args, config={"configurable": {"state": state}}))
        else:
            observations.append(tool.invoke(tool_call["args"], config={"configurable": {"state": state}}))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"leader_messages": state["leader_messages"] + tool_outputs}

def output_node(state: LeaderState) -> LeaderOutputState:
    """Node transforming the output of the architecture to the correct object type.

        Args:
            state: Internal LeaderState object, which will be transformed into LeaderOutputState

        Returns:
            Output state object containing only necessary information about the architecture.
    """

    return LeaderOutputState(
        leader_messages=state["leader_messages"],
        order_list=state["order_list"],
        generals_units=state["generals_units"],
        generals_messages=state["generals_messages"],
        recruitment_choice=state["recruitment_choice"]
    )

def upkeep_condition(state: LeaderState) -> Literal["leader", "output_node"]:
    """Decides whether to continue the graph loop, depending on whether the graph call was 
       a result of a recruitment procedure or not.
    
        Args:
            state: Current Leader State with the list of messages, controlled regions, currently assigned orders and list of sub-agents.
        
        Returns:
            The text description of the next node for navigation.
    """

    if state["recruitment_response"]:
        return "output_node"
    return "leader"

def tool_condition(state: LeaderState) -> Literal["tool_node", "output_node"]:
    """Decides whether to continue the graph loop, based on or on the lack of tool calls.
    
        Args:
            state: Current Leader State with the list of messages, controlled regions, currently assigned orders and list of sub-agents.
        
        Returns:
            The text description of the next node for navigation.
    """

    leader_messages = state["leader_messages"]
    last_message = leader_messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return "output_node"

# Function for creating new leader agents
def get_leader_agent(communication, human_mode=False):
    global HUMAN_MODE, leader_tools, general_tools

    # Setting up the tool set
    leader_tools = get_leader_tools(communication)
    general_tools = get_general_tools(communication)

    # Graph construction
    leader_agent_builder = StateGraph(LeaderState, input_schema=LeaderInputState)

    leader_agent_builder.add_node("input_node", input_node)
    leader_agent_builder.add_node("tool_node", tool_node)
    leader_agent_builder.add_node("output_node", output_node)

    if human_mode:
        leader_agent_builder.add_node("leader", human_leader)
        HUMAN_MODE = True
    else:
        leader_agent_builder.add_node("leader", leader)

    leader_agent_builder.add_edge(START, "input_node")
    leader_agent_builder.add_edge("input_node", "leader")
    leader_agent_builder.add_edge("output_node", END)

    leader_agent_builder.add_conditional_edges(
        "leader",
        tool_condition,
        {
            "tool_node": "tool_node",
            "output_node": "output_node"
        }
    )

    return leader_agent_builder.compile()