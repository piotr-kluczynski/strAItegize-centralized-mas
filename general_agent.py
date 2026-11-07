from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage

from strAItegize_common.others.human import human

from states import GeneralState
from prompts import general_agent_prompt

def get_general_agent(tools: list, model):
    def general(state: GeneralState, config: RunnableConfig) -> GeneralState:
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

        configurable = config.get("configurable", {})
        is_human = configurable.get("human_mode", False)

        message_list = [
            SystemMessage(content=general_agent_prompt(
                state["received_orders"], 
                state["controlled_units"]
            ))
        ] + state["general_messages"]

        if is_human:
            return {
                "general_messages": human(
                    message_list,
                    agent_name=state["general_id"],
                    tools=tools
                )
            }
        else:
            return {
                "general_messages": [
                    model.bind_tools(tools).invoke(message_list)
                ]
            }

    return general
