from strAItegize_common.client.prompts import round_start_prompt, simulation_start_prompt, units_recruitment_prompt
from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice

from centralized_graph import get_centralized_mas

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.client.multiagent_system import MultiAgentSystem

from langchain_core.messages import SystemMessage

class CentralizedMAS(MultiAgentSystem):
    def __init__(self, communication: CommunicationContext, human_mode: bool=False, **kwargs):
        self.communication: CommunicationContext = communication
        self.human_mode: bool = human_mode
        self.player_id: str = None
        self.thread_id = "session"
        self.units_per_general = kwargs.get("units_per_general", 3)

        # Graph
        self.graph = get_centralized_mas(communication=communication, leader_model=kwargs.get('leader_model'), general_model=kwargs.get('general_model'))

    def initialize_graph(self,
        player_id: str, players: list[str],
        regions: list[str], neighbouring_regions: dict[str, list[str]],
        final_round: int
    ):
        self.player_id = player_id
        
        init_msg = SystemMessage(content=simulation_start_prompt(
                player_id=player_id,
                players=players,
                regions=regions,
                neighbouring_regions=neighbouring_regions,
                final_round=final_round
        ))
        
        self.graph.update_state(
            values={
                "player_id": player_id,
                "regions": regions,
                "units": [],
                "units_per_general": self.units_per_general,
                "general_id_counter": 1,
                "generals_units": {},
                "generals_messages": {},
                "generals_orders": {},
                "leader_messages": [init_msg],
                "order_list": [],
                "recruitment_choice": [],
                "recent_events": [],
                "upkeep_phase": False,
                "supply_surplus": 0
            },
            config={"configurable": {"thread_id": self.thread_id}}
        )

    def invoke_orders(self, 
            current_round: int, final_round: int,
            players: list[str], leaderboard: dict[str, int], 
            units: list[str], regions: list[str],
            tresspasers: dict[str, list[str]], events: list[str]
        ) -> list[Order]:

        round_msg = SystemMessage(content=round_start_prompt(
            current_round=current_round,
            final_round=final_round,
            players=players,
            leaderboard_dict=leaderboard,
            controlled_units=units,
            controlled_regions=regions,
            enemy_units_per_region=tresspasers,
            events=events
        ))

        # Invoking graph
        output_state = self.graph.invoke(
            {
                "leader_messages": [round_msg],
                "units": units,
                "regions": regions,
                "recent_events": events,
                "order_list": [],
                "upkeep_phase": False   
            }, 
            {
                "configurable": {
                    "thread_id": f"{self.thread_id}",
                    "human_mode": self.human_mode
                }
            }
        )    

        # Retrieving the recruitment choices from the output state
        order_list = output_state["order_list"]
        return [order.model_dump() for order in order_list]

    def invoke_upkeep(self, supply_surplus: int, regions: list[str]) -> list[RecruitmentChoice]:
        upkeep_msg = SystemMessage(content=units_recruitment_prompt(
            supply_surplus=supply_surplus,
            regions=regions
        ))

        # Invoking graph
        output_state = self.graph.invoke(
            {
                "leader_messages": [upkeep_msg],
                "supply_surplus": supply_surplus,
                "regions": regions,
                "recruitment_choice": [],
                "upkeep_phase": True      
            }, 
            {
                "configurable": {
                    "thread_id": f"{self.thread_id}",
                    "human_mode": self.human_mode
                }
            }
        )

        # Retrieving the recruitment choices from the output state
        recruitment_choice = output_state["recruitment_choice"]
        return [recruitment.model_dump() for recruitment in recruitment_choice]
