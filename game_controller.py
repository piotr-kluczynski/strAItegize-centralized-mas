from langchain_core.messages import SystemMessage

from graph import get_centralized_mas
from prompts import simulation_start_prompt, round_start_prompt, units_recruitment_prompt
from states import LeaderInputState
from communication.communication_context import CommunicationContext

class GameController:
    def __init__(self, simulation_port, units_per_general=3, human_mode=False):
        self.communication = CommunicationContext()
        self.communication.start_connection(simulation_port=int(simulation_port))
        self.is_next_round_upkeep = False

        self.units_per_general = units_per_general

        self.player_id = -1
        self.general_id = 1

        # Graphs
        self.centralized_mas = get_centralized_mas(self.communication, human_mode=human_mode)
        self.thread_id = 1

        self.leader_messages = []

        self.generals_messages = {} # Name: List[BaseMessage]
        self.generals_units = {} # Name: List[str]

    def initialize_simulation(self):
        # Waiting for the simulation to start 
        session_data = self.communication.start_simulation()

        # Generating simulation start prompt message
        self.your_id = session_data["your_id"]

        # Checking if next round is an upkeep one
        self.is_next_round_upkeep = session_data["is_start_upkeep"]
        
        self.leader_messages.append(SystemMessage(content=simulation_start_prompt(
            your_id=session_data["your_id"],
            players=session_data["players"],
            regions=session_data["regions"],
            neighbouring_regions=session_data["neighbouring_regions"],
            final_round=session_data["final_round"]
        )))

    def manage_subagents(self, round_data):
        # Calculate new units
        new_units = round_data["units"]
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
                new_general_id = f"general_{self.general_id}"
        
                self.generals_units[new_general_id] = [new_unit]
                self.generals_messages[new_general_id] = []
        
                self.general_id += 1
        
        # Removing missing units
        for general, old_units in zip(list(self.generals_units.keys()), list(self.generals_units.values())):
            self.generals_units[general] = list(set(old_units) & set(round_data["units"]))
        
            # Removing generals without units
            if len(self.generals_units[general]) == 0:
                del self.generals_units[general]
                del self.generals_messages[general]

    def start_simulation(self):
        while True:
            # Perform upkeep phase if needed
            if self.is_next_round_upkeep:
                upkeep_data = self.communication.await_upkeep()

                self.leader_messages.append(SystemMessage(content=units_recruitment_prompt(
                    supply_surplus=upkeep_data["supply_surplus"],
                    regions=upkeep_data["regions"]
                )))

                # Creating the new state for the upkeep phase
                input_state = LeaderInputState(
                    regions=upkeep_data["regions"],
                    supply_surplus=upkeep_data["supply_surplus"],
                    upkeep_phase=True,
                    player_id=self.your_id,
                    recent_events={},
                    leader_messages=self.leader_messages,
                    order_list=[],
                    recruitment_choice=[],
                    generals_units=self.generals_units,
                    generals_messages=self.generals_messages
                )

                # MAS processing current round
                output_state = self.centralized_mas.invoke(input_state, {"configurable": {"thread_id": f"{self.thread_id}"}})

                # Retrieving information from the graph
                self.leader_messages = output_state["leader_messages"]
                recruitment_choice = output_state["recruitment_choice"]
                recruitment_to_send = [recruitment.model_dump() for recruitment in recruitment_choice]

                # Sending the recruitment choices to the simulation
                self.communication.submit_recruitment(recruitment=recruitment_to_send)

            # Await for the new round
            round_data = self.communication.await_round_start()

            # Adjusting the number of generals
            self.manage_subagents(round_data)
            
            # Generating round start prompt message
            self.leader_messages.append(SystemMessage(content=round_start_prompt(
                current_round=round_data["current_round"],
                final_round=round_data["final_round"],
                players=round_data["players"],
                leaderboard_dict=round_data["leaderboard"],
                controlled_units=round_data["units"],
                controlled_regions=round_data["regions"],
                enemy_units_per_region=round_data["tresspasers"],
                events=round_data["events"]
            )))

            # Creating the new state for this round
            input_state = LeaderInputState(
                regions=round_data["regions"],
                supply_surplus=0,
                upkeep_phase=False,
                player_id=self.your_id,
                recent_events=round_data["events"],
                leader_messages=self.leader_messages,
                order_list=[],
                recruitment_choice=[],
                generals_units=self.generals_units,
                generals_messages=self.generals_messages
            )
            
            # MAS processing current round
            output_state = self.centralized_mas.invoke(input_state, {"configurable": {"thread_id": f"{self.thread_id}"}})
            
            # Retrieving information from the graph
            self.leader_messages = output_state["leader_messages"]
            order_list = output_state["order_list"]
            
            orders_to_send = [order.model_dump() for order in order_list]

            # Sending the orders to the simulation
            self.communication.submit_orders(orders=orders_to_send)

            # Awaiting next round
            self.is_next_round_upkeep = self.communication.end_round()

            print(f"Does simulation ends? {self.is_next_round_upkeep}")

            # Checking if the server signalled the simulation end
            if self.is_next_round_upkeep is None:
                break