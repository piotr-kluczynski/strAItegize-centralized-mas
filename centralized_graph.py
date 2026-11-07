from typing import Literal
from langchain.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

from leader_agent import get_leader_agent 
from general_agent import get_general_agent
from states import CentralizedState, GeneralState

from tools import assign_order, remove_order, recruit_unit, delegate_general
from strAItegize_common.client.tools import get_generic_tools
from strAItegize_common.client.communication.communication_context import CommunicationContext

def phase_node(state: CentralizedState) -> CentralizedState:
    if state["upkeep_phase"]:
        blocked_tools = ["delegate_general"]
    else:
        blocked_tools = ["recruit_unit"]
        
    return {"blocked_tools": blocked_tools}

def general_manage_node(state: CentralizedState) -> CentralizedState:
    # Find all new units
    new_units = state["units"]
    for general, old_units in state["generals_units"].items():
        new_units = list(set(new_units) - set(old_units))
    
    # Adding new units
    for new_unit in new_units:
        unit_added = False
    
        for general, old_units in state["generals_units"].items():
            if len(old_units) < state["units_per_general"]:
                old_units.append(new_unit)
                unit_added = True
                break
    
        if not unit_added:
            # Creating new general
            new_general_id = f"general_{state["general_id_counter"]}"
    
            state['generals_units'][new_general_id] = [new_unit]
            state['generals_messages'][new_general_id] = []
    
            state['general_id_counter'] += 1
    
    # Removing missing units
    for general, old_units in state['generals_units'].items():
        state['generals_units'][general] = list(set(old_units) & set(state['units']))
    
        # Removing generals without units
        if len(state['generals_units'][general]) == 0:
            del state['generals_units'][general]
            del state['generals_messages'][general]

def general_output_node(state: CentralizedState) -> CentralizedState:
    # State reassignment GeneralState -> CentralizedState
    new_orders = [order for order in state["controlled_units"].values() if order is not None]
    
    # Building ToolMessage
    general_id = state["active_general"]
    if not general_id:
        return {}
    
    last_msg = state["leader_messages"][-1]
    
    # If the last message is the ToolMessage from delegate_general, read its tool_call_id directly
    if hasattr(last_msg, "tool_call_id"):
        tool_call_id = last_msg.tool_call_id
    else:
        # Searching for the AIMessage that triggered the delegation
        ai_msg = next((m for m in reversed(state["leader_messages"]) if getattr(m, "tool_calls", None)), None)
        delegation_call = next((c for c in ai_msg.tool_calls if c["name"] == "delegate_general"), None) if ai_msg else None
        tool_call_id = delegation_call["id"] if delegation_call else "manual_delegation"
    
    report = f"Report from General {general_id}:\n"
    if new_orders:
        for order in new_orders:
            target_txt = f"target: {order.target_id}" if order.target_id else ""
            vec_txt = f"vector: [{order.move_q}, {order.move_r}, {order.move_s}]" if order.move_q is not None else ""
            report += f"- Unit {order.unit_id}: {order.order_type.upper()} {vec_txt} {target_txt}\n"
    else:
        report += "No new orders issued this turn.\n"
        
    tool_msg = ToolMessage(
        content=report,
        name="delegate_general",
        tool_call_id=tool_call_id,
        id=f"tool_{tool_call_id}"
    )
    
    # Cleaning the shared channels
    return {
        "order_list": state["order_list"] + new_orders,
        "leader_messages": [tool_msg],
        "generals_messages": {
            **state["generals_messages"],
            general_id: state["general_messages"]
        },
        "active_general": None,
        "general_messages": None,
        "controlled_units": None,
        "general_id": None,
    }

def leader_tool_router(state: CentralizedState) -> Literal["leader_tools", "end"]:
    last_msg = state["leader_messages"][-1]
    if last_msg.tool_calls:
        return "leader_tools"
    
    return "end"

def general_tool_router(state: GeneralState) -> Literal["tools", "end"]:
    last_msg = state["general_messages"][-1]
    if last_msg.tool_calls:
        return "general_tools"
    
    return "end"

def delegation_router(state: CentralizedState) -> Literal["general_subgraph", "leader_agent"]:
    if state.get("active_general") is not None:
        return "general_subgraph"
    
    return "leader_agent"

def get_centralized_mas(
        communication: CommunicationContext, 
        leader_model = init_chat_model("openai:gpt-4o"),
        general_model = init_chat_model("openai:gpt-4o")
    ):
    # Assigning tools
    general_tools = [assign_order, remove_order] + get_generic_tools(communication=communication)
    leader_tools = [delegate_general, recruit_unit] + get_generic_tools(communication=communication)
    
    # Building General sub-graph
    general_builder = StateGraph(GeneralState)

    general_builder.add_node("general_agent", get_general_agent(tools=general_tools, model=general_model))
    general_builder.add_node("tools", ToolNode(general_tools, messages_key="general_messages"))

    general_builder.add_edge(START, "general_agent")
    general_builder.add_conditional_edges("general_agent", general_tool_router, {
        "general_tools": "tools",
        "end": END
    })
    general_builder.add_edge("tools", "general_agent")

    compiled_general = general_builder.compile()

    # Building Centralized Multi-Agent System graph
    mas_builder = StateGraph(CentralizedState)
    
    mas_builder.add_node("leader_agent", get_leader_agent(tools=leader_tools, model=leader_model))
    mas_builder.add_node("tools", ToolNode(leader_tools, messages_key="leader_messages"))
    mas_builder.add_node("general_subgraph", compiled_general)
    mas_builder.add_node("general_manage_node", general_manage_node)
    mas_builder.add_node("general_output_node", general_output_node)
    mas_builder.add_node("phase_node", phase_node)
    
    mas_builder.add_edge(START, "phase_node")
    mas_builder.add_edge("phase_node", "general_manage_node")
    mas_builder.add_edge("general_manage_node", "leader_agent")
    mas_builder.add_edge("general_subgraph", "general_output_node")
    mas_builder.add_edge("general_output_node", "leader_agent")
    
    mas_builder.add_conditional_edges("leader_agent", leader_tool_router, {
        "leader_tools": "tools",
        "end": END
    })
    
    mas_builder.add_conditional_edges("tools", delegation_router, {
        "general_subgraph": "general_subgraph",
        "leader_agent": "leader_agent"
    })
    
    checkpointer = MemorySaver()
    return mas_builder.compile(checkpointer=checkpointer)