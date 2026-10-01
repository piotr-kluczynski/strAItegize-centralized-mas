from dotenv import load_dotenv
load_dotenv()

from game_controller import GameController

HUMAN_MODE = True

if __name__ == '__main__':
    print("Welcome to the Centralized Multi-Agent System Architecture!")
    simulation_port = input("Enter the port of the empty player slot from the simulation app:\r\n")

    game_controler = GameController(simulation_port, human_mode=HUMAN_MODE)

    game_controler.initialize_simulation()

    game_controler.start_simulation()
    game_controler.end_simulation()