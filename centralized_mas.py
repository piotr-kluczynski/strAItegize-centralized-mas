from strAItegize_common.client.prompts import round_start_prompt, simulation_start_prompt, units_recruitment_prompt
from states import LeaderInputState
from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice

from leader_agent import get_leader_agent

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.client.multiagent_system import MultiAgentSystem

from langchain_core.messages import BaseMessage, SystemMessage

class CentralizedMAS(MultiAgentSystem):
    def __init__(self, communication: CommunicationContext, human_mode: bool=False, **kwargs):
        self.communication: CommunicationContext = communication
        self.human_mode: bool = human_mode
        self.player_id: str = None

        # Graph
        self.thread_id: int = 1
        self.centralized_mas = get_leader_agent(communication, human_mode=human_mode)

        # Leader Agents
        self.leader_messages: list[BaseMessage] = []

        # General Agents
        self.general_id_counter: int = 1
        self.units_per_general: int = kwargs.get("units_per_general", 3)

        self.generals_messages: dict[str, list[BaseMessage]] = {} # Name: List[BaseMessage]
        self.generals_units: dict[str, list[str]] = {} # Name: List[str]

    def initialize_graph(self,
        player_id: str, players: list[str],
        regions: list[str], neighbouring_regions: dict[str, list[str]],
        final_round: int
    ):
        self.player_id = player_id
        
        self.leader_messages.append(
            SystemMessage(content=simulation_start_prompt(
                player_id=player_id,
                players=players,
                regions=regions,
                neighbouring_regions=neighbouring_regions,
                final_round=final_round
            ))
        )

    def invoke_orders(self, 
            current_round: int, final_round: int,
            players: list[str], leaderboard: dict[str, int], 
            units: list[str], regions: list[str],
            tresspasers: dict[str, list[str]], events: list[str]
        ) -> list[Order]:
        self._manage_subagents(units=units)

        # Adding the round start prompt to the leader messages
        self.leader_messages.append(SystemMessage(content=round_start_prompt(
            current_round=current_round,
            final_round=final_round,
            players=players,
            leaderboard_dict=leaderboard,
            controlled_units=units,
            controlled_regions=regions,
            enemy_units_per_region=tresspasers,
            events=events
        )))

        # Creating new state for the planning phase
        input_state = LeaderInputState(
            regions=regions,
            supply_surplus=0,
            upkeep_phase=False,
            player_id=self.player_id,
            recent_events=events,
            leader_messages=self.leader_messages,
            order_list=[],
            recruitment_choice=[],
            generals_units=self.generals_units,
            generals_messages=self.generals_messages
        )

        # Invoking graph
        output_state = self.centralized_mas.invoke(input_state, {"configurable": {"thread_id": f"{self.thread_id}"}})    

        # Updating the MAS state
        self.leader_messages = output_state["leader_messages"]

        # Retrieving the recruitment choices from the output state
        order_list = output_state["order_list"]
        orders_to_send = [order.model_dump() for order in order_list]

        return orders_to_send

    def invoke_upkeep(self, supply_surplus: int, regions: list[str]) -> list[RecruitmentChoice]:
        # Adding the upkeep prompt to the leader messages
        self.leader_messages.append(SystemMessage(content=units_recruitment_prompt(
            supply_surplus=supply_surplus,
            regions=regions
        )))

        # Creating new state for the upkeep phase
        input_state = LeaderInputState(
            regions=regions,
            supply_surplus=supply_surplus,
            upkeep_phase=True,
            player_id=self.player_id,
            recent_events={},
            leader_messages=self.leader_messages,
            order_list=[],
            recruitment_choice=[],
            generals_units=self.generals_units,
            generals_messages=self.generals_messages
        )

        # Invoking graph
        output_state = self.centralized_mas.invoke(input_state, {"configurable": {"thread_id": f"{self.thread_id}"}})

        # Updating the MAS state
        self.leader_messages = output_state["leader_messages"]

        # Retrieving the recruitment choices from the output state
        recruitment_choice = output_state["recruitment_choice"]
        recruitment_to_send = [recruitment.model_dump() for recruitment in recruitment_choice]

        return recruitment_to_send

    def _manage_subagents(self, units: list[str]):
        # Calculate new units
        new_units = units
        for general, old_units in self.generals_units.items():
            new_units = list(set(new_units) - set(old_units))
        
        # Adding new units
        for new_unit in new_units:
            unit_added = False
        
            for general, old_units in self.generals_units.items():
                if len(old_units) < self.units_per_general:
                    old_units.append(new_unit)
                    unit_added = True
                    continue
        
            if not unit_added:
                # Creating new general
                new_general_id_counter = f"general_{self.general_id_counter}"
        
                self.generals_units[new_general_id_counter] = [new_unit]
                self.generals_messages[new_general_id_counter] = []
        
                self.general_id_counter += 1
        
        # Removing missing units
        for general, old_units in zip(list(self.generals_units.keys()), list(self.generals_units.values())):
            self.generals_units[general] = list(set(old_units) & set(units))
        
            # Removing generals without units
            if len(self.generals_units[general]) == 0:
                del self.generals_units[general]
                del self.generals_messages[general]