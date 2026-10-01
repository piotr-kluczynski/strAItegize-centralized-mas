from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from shared.human import human

from states import GeneralState, GeneralOutputState
from prompts import general_agent_prompt

# Setup general agent tools
available_tools = []
tools_by_name = {tool.name: tool for tool in available_tools}

# Initializing llm
model = init_chat_model(
    model="openai:gpt-4o"
)

# Agent nodes
def general(state: GeneralState) -> GeneralState:
    """Makes the micro-decisions within the simulation, to fulfill received orders.

    Analyzes the orders received from the Leader Agent and decides how to carry out them by:
    - observing the simulation environment
    - analyzing different possibilities
    - giving orders to the controlled units

    Args:
        state: Current General State with the list of messages, command from Leader agent, and dictionary with units and orders assigned to them.
    
    Returns:
        New, updated General State.
    
    """

    return {
        "general_messages": [
            model.bind_tools(available_tools).invoke(
                [SystemMessage(content=general_agent_prompt())] + state["general_messages"]
            )
        ]
    }

def human_general(state: GeneralState) -> GeneralState:
    return {
        "general_messages": 
            human(
                [SystemMessage(content=general_agent_prompt())] + state["general_messages"],
                agent_name=state["general_id"],
                tools=available_tools
            )
    }


def tool_node(state: GeneralState) -> GeneralState:
    """Execute General Agent tool calls.

    Handles:
    - Executing think_tool for strategic reflection.
    - Observing simulation environment.
    - Aggregating learned information.
    - Give orders to the controlled units.

    Args:
        state: Current General State with the list of messages, command from Leader agent, and dictionary with units and orders assigned to them.

    Returns:
        New, updated state with tool messages.
    """

    tool_calls = state["general_messages"][-1].tool_calls
    observations = []

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]

        if tool["name"] == "AssignOrder":
            # Add new order to the list of orders
            unit_id = tool_call["args"]["unit_id"]
            unit_order = tool_call["args"]["unit_order"]

            # Assigning order to the unit
            state["controlled_units"][unit_id] = unit_order

            # Generating tool response
            observations.append(f"The order was assigned to the unit {unit_id}.")
        elif tool["name"] == "RemoveOrder":
            # Remove an order from the list of orders
            unit_id = tool["args"]["unit_id"]

            # Removing an order from the unit
            state["controlled_units"][unit_id] = None

            # Generating tool response
            observations.append(f"The order was removed from the unit {unit_id}.")
        elif tool["name"] == "get_units":
            args = {
                **tool_call["args"],
                "sender_agent": "general"
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


    return {"general_messages": state["general_messages"] + tool_outputs}

def should_continue(state: GeneralState) -> Literal["tool_node", "output_node"]:
    """Decides whether to continue the graph loop, based on or on the lack of tool calls.

    Args:
        state: Current General State with the list of messages, command from Leader agent, and dictionary with units and orders assigned to them.     
    
    Returns:
        The text description of the next node for navigation.
    """

    general_messages = state["general_messages"]
    last_messages = general_messages[-1]

    if last_messages.tool_calls:
        return "tool_node"
    return "output_node"

def output_node(state: GeneralState) -> GeneralOutputState:
    """Generates the textual description of the actions taken by the General Agent
    and preserves only absolutely needed data about the agent's state.

    Args:
        state: Current General State with the list of messages, command from Leader agent, and dictionary with units and orders assigned to them.     

    Returns:
        Output state for the Leader Agent to use.
   
    """

    report = f"The General Agent with general_id: {state["general_id"]}, gave the following orders to the units under their command:"
    for order in state["controlled_units"].values():
        target_txt = f"{order.target_id}" if order.target_id != None else "None"
        vec_txt = f"[{order.move_q}, {order.move_r}, {order.move_s}]" if order.move_q != None else "None"

        report += f"{order.order_type.capitalize()} order to the unit {order.unit_id} (movement vector: {vec_txt}, targeted unit: {target_txt})\r\n"

    return GeneralOutputState(
        general_messages=state["general_messages"],
        controlled_units=state["controlled_units"],
        report=report
    )

# Function for creating new general agents
def get_general_agent(tools, human_mode=False):
    global available_tools

    # Setting up the tools
    available_tools = tools

    # Graph construction
    general_agent_builder = StateGraph(GeneralState, output_schema=GeneralOutputState)

    general_agent_builder.add_node("tool_node", tool_node)
    general_agent_builder.add_node("output_node", output_node)

    if human_mode:
        general_agent_builder.add_node("general", human_general)
    else:
        general_agent_builder.add_node("general", general)

    general_agent_builder.add_edge(START, "general")
    general_agent_builder.add_edge("output_node", END)

    general_agent_builder.add_conditional_edges(
        "general",
        should_continue,
        {
            "tool_node": "tool_node",
            "__end__": "output_node"
        }
    )

    return general_agent_builder.compile()