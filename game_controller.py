from langchain_core.messages import SystemMessage

from graph import get_centralized_mas
from prompts import simulation_start_prompt, round_start_prompt, units_recruitment_prompt
from states import LeaderInputState
from communication.communication import Communication

class GameController:
    def __init__(self, simulation_port, units_per_general=3, human_mode=False):
        self.communication = Communication()
        self.communication.start_connection(port=int(simulation_port))

        self.simulation_port = simulation_port
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
        session_data = self.communication.await_simulation_start()
        self.communication.simulation_socket.settimeout(None)   

        # Generating simulation start prompt message
        self.your_id = session_data["your_id"]
        
        self.leader_messages.append(SystemMessage(content=simulation_start_prompt(
            your_id=session_data["your_id"],
            players=session_data["players"],
            regions=session_data["regions"],
            neighbouring_regions=session_data["neighbouring_regions"],
            final_round=session_data["max_round"]
        )))

    def manage_subagents(self, round_data):
        # Calculate new units
        new_units = round_data["controlled_units"]
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
            self.generals_units[general] = list(set(old_units) & set(round_data["controlled_units"]))
        
            # Removing generals without units
            if len(self.generals_units[general]) == 0:
                del self.generals_units[general]
                del self.generals_messages[general]

    def start_simulation(self):
        while True:
            # Await new round to begin
            round_data = self.communication.await_round_start()
            self.communication.simulation_socket.settimeout(None)

            # Executing the upkeep phase if signaled
            if round_data.get("upkeep_phase"):
                self.leader_messages.append(SystemMessage(content=units_recruitment_prompt(
                    supply_surplus=round_data["supply_surplus"],
                    regions=round_data["regions"]
                )))

                # Creating the new state for the upkeep phase
                input_state = LeaderInputState(
                    regions=round_data["regions"],
                    supply_surplus=round_data["supply_surplus"],
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
                messages = output_state["leader_messages"]
                recruitment_choice = output_state["recruitment_choice"]
                recruitment_to_send = [recruitment.model_dump() for recruitment in recruitment_choice]
                
                # Sending the orders to the simulation
                self.communication.execute_upkeep(recruitment_to_send)

                # Skipping the remaining part of the round loop
                continue

            # Checking if the simulation ended
            if round_data.get("end_session"):
                break
        
            # Adjusting the number of generals
            self.manage_subagents(round_data)
        
            # Generating round start prompt message
            messages.append(SystemMessage(content=round_start_prompt(
                current_round=round_data["current_round"],
                final_round=round_data["final_round"],
                players=round_data["players"],
                leaderboard_dict=round_data["leaderboard_dict"],
                controlled_units=round_data["controlled_units"],
                controlled_regions=round_data["controlled_regions"],
                enemy_units_per_region=round_data["enemy_units_per_region"],
                events=round_data["event_dict"]
            )))
        
            # Creating the new state for this round
            input_state = LeaderInputState(
                regions=round_data["controlled_regions"],
                supply_surplus=0,
                upkeep_phase=False,
                player_id=self.your_id,
                recent_events=round_data["event_dict"],
                leader_messages=messages,
                order_list=[],
                generals_units=self.generals_units,
                generals_messages=self.generals_messages
            )
        
            # MAS processing current round
            output_state = self.centralized_mas.invoke(input_state, {"configurable": {"thread_id": f"{self.thread_id}"}})
        
            # Retrieving information from the graph
            messages = output_state["leader_messages"]
            order_list = output_state["order_list"]
            orders_to_send = [order.model_dump() for order in order_list]
        
            # Sending the orders to the simulation
            self.communication.end_round(orders_to_send)
            self.communication.simulation_socket.settimeout(None)

    def end_simulation(self):
        pass