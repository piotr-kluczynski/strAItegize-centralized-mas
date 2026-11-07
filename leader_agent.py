from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage
from strAItegize_common.others.human import human

from prompts import leader_agent_prompt
from states import CentralizedState

def get_leader_agent(tools: list, model):
    def leader(state: CentralizedState, config: RunnableConfig) -> CentralizedState:
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

        configurable = config.get("configurable", {})
        is_human = configurable.get("human_mode", False)
        
        active_tools = [tool for tool in tools if tool.name not in state["blocked_tools"]]
        
        model_with_tools = model.bind_tools(active_tools)

        message_list = [
            SystemMessage(content=leader_agent_prompt(
                state["player_id"], 
                state["generals_units"], 
                state["regions"]
            ))
        ] + state["leader_messages"]

        if is_human:
            return {
                "leader_messages": human(
                    message_list,
                    agent_name="leader",
                    tools=active_tools
                )
            }
        else:
            return {
                "leader_messages": [
                    model_with_tools.invoke(message_list)
                ]
            }

    return leader