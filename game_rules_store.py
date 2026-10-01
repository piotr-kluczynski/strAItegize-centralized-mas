from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

documents = []
rules_store = Chroma(
    collection_name="simulation_rules",
    embedding_function=OpenAIEmbeddings()
)

# Game map
documents.append(Document(page_content="The map consists of hexagonal tiles.", 
    metadata={"category": "Game Map"}
))

documents.append(Document(page_content="Each tile can be occupied by only one unit at a time.", 
    metadata={"category": "Game Map"}
))

documents.append(Document(page_content="The map is divided into distinct regions.", 
    metadata={"category": "Game Map"}
))

documents.append(Document(page_content="Regions consist of many connected tiles.", 
    metadata={"category": "Game Map"}
)) 

documents.append(Document(page_content="Each region contains exactly one special tile called a Command Center.", 
    metadata={"category": "Game Map"}
)) 

documents.append(Document(page_content="A player controls a region if they control its Command Center.", 
    metadata={"category": "Game Map"}
))

documents.append(Document(page_content="A player gains control of a Command Center when one of their units moves onto its tile.", 
    metadata={"category": "Game Map"}
))
documents.append(Document(page_content="Player keeps the control over the Command Center, until another player’s unit moves on it.", 
    metadata={"category": "Game Map"}
)) 

documents.append(Document(page_content="Information outside a player's vision range is not directly observable.", 
    metadata={"category": "Game Map"}
))

# Units
documents.append(Document(page_content="Each unit is characterized by three attributes: movement speed, upkeep cost, and strength.", 
    metadata={"category": "Units"}
))

documents.append(Document(page_content="There are three types of units: light infantry, heavy infantry, and cavalry.", 
    metadata={"category": "Units"}
))

documents.append(Document(page_content="Light infantry is the cheapest and most flexible type of unit. Movement speed: 2. Upkeep cost: 1. Strength: 6.", 
    metadata={"category": "Units"}
))

documents.append(Document(page_content="Heavy infantry is the strongest and least mobile type of unit. Movement speed: 1. Upkeep cost: 2. Strength: 8.", 
    metadata={"category": "Units"}
))

documents.append(Document(page_content="Cavalry is the most mobile and most expensive type of unit. Movement speed: 3. Upkeep cost: 3. Strength: 6.", 
    metadata={"category": "Units"}
))

documents.append(Document(page_content="A unit can observe map tiles within its vision range.", 
    metadata={"category": "Units"}
))

# Player Actions
documents.append(Document(page_content="There are four types of actions that can be performed by a unit: hold, move, attack, and support.", 
    metadata={"category": "Player Actions"}
))  

documents.append(Document(page_content="Hold action makes the unit remain inactive for the round. ", 
    metadata={"category": "Player Actions"}
))   

documents.append(Document(page_content="Move action allows a unit to move to any unoccupied tile within a distance equal to or less than its movement speed.", 
    metadata={"category": "Player Actions"}
))  

documents.append(Document(page_content="Attack action makes a unit attempt to engage another unit in combat within a distance of 1 tile.", 
    metadata={"category": "Player Actions"}
))    

documents.append(Document(page_content="Support action makes a unit attempt to support another unit within a distance of 1 tile.", 
    metadata={"category": "Player Actions"}
))    

documents.append(Document(page_content="When a support action is successfully performed, the target unit adds the supporting unit to its supporting_unit list, allowing the supporting unit to assist it in combat during the current round.", 
    metadata={"category": "Player Actions"}
))    

documents.append(Document(page_content="An action may become impossible to perform due to changes in the game state, such as the target unit moving out of range. In such cases, the action is cancelled, and the unit remains inactive for the round.", 
    metadata={"category": "Player Actions"}
))    

documents.append(Document(page_content="Each unit can perform at most one action per round.", 
    metadata={"category": "Player Actions"}
))    

documents.append(Document(page_content="A unit performs an action only if an appropriate order was assigned to it during the planning phase.", 
    metadata={"category": "Player Actions"}
))  

# Combat system

documents.append(Document(page_content="Combat between two units occurs when one unit successfully performs an attack action against another unit.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="Combat can involve only two units directly: one attacker and one defender. Other units can influence the outcome by supporting either the attacker or the defender.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="Combat strength is calculated separately for the attacker and defender using the following formula:  combat strength = random.randint(1, unit_strength)) + 2 * len(supporting_units)).", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="After the combat strength of both units is calculated, the values are compared and one of three outcomes occurs: defenders' victory, attackers' victory, or decisive attackers' victory.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="Defenders' victory occurs when the attacker's combat strength is less than or equal to the defender's combat strength.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="As an outcome of defenders' victory, both units remain in their original positions.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="Attackers' victory occurs when the attacker's combat strength is greater than the defender's combat strength but less than or equal to twice the defender's combat strength.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="As an outcome of attackers' victory, the defending unit is pushed one tile in the opposite direction from the attack, and the attacking unit moves into the defender's original position.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="Decisive attackers' victory occurs when the attacker's combat strength is greater than twice the defender's combat strength.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="As an outcome of decisive attackers' victory, the defending unit is destroyed, and the attacking unit moves into the defender's original position.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="When a unit is pushed, it is forced to move one tile in a specified direction.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="If a pushed unit cannot move because its destination tile is blocked, one of three outcomes occurs depending on the obstacle.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="If a pushed unit is blocked by a unit belonging to another player, the pushed unit is destroyed.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="If a pushed unit is blocked by the edge of the map, the pushed unit is destroyed.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="If a pushed unit is blocked by a unit belonging to the same player, the blocking unit is chain pushed.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="A chain-pushed unit attempts to move in the same direction as the originally pushed unit.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="If chain pushing would result in the chain-pushed unit being destroyed, the originally pushed unit is destroyed instead.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="A unit on a neighboring tile can support an attacking or defending unit by successfully performing a support action on that unit beforehand.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="Each supporting unit provides a +2 bonus to the combat strength of the supported unit.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="The support bonus lasts for one round only. A support action must be performed again in a subsequent round to provide the bonus again.", 
    metadata={"category": "Combat system"}
))

documents.append(Document(page_content="A unit can be attacked any number of times during a single round.", 
    metadata={"category": "Combat system"}
))

# Round details
documents.append(Document(page_content="Each simulation session consists of multiple rounds. The total number of rounds is determined by the session setup.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="Each round consists of two phases: 1. Planning phase. 2. Execution phase.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="During the Planning phase, each player can observe the current state of the simulation, communicate with other players, and create a list of orders for their units.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="Orders created during the Planning phase are not executed immediately. Instead, they are stored and executed during the Execution phase.", 
    metadata={"category": "Round details"}
))  

documents.append(Document(page_content="During the Execution phase, all orders from all players are grouped by action type and executed in the following order: 1. Hold. 2. Move. 3. Support. 4. Attack.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="Within each action type, orders are executed according to the player's order.", 
    metadata={"category": "Round details"}
))  

documents.append(Document(page_content="The player order is randomized at the beginning of each round.", 
    metadata={"category": "Round details"}
))  

documents.append(Document(page_content="If an order becomes invalid before it is executed, the order is ignored. For example, an attack order is ignored if the attacking unit is destroyed before the attack is executed.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="During the Planning phase, players can communicate with each other using text messages.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="Once the Execution phase is completed, the next round begins.", 
    metadata={"category": "Round details"}
))  

documents.append(Document(page_content="After a specified number of rounds, an Upkeep phase takes place before the Planning phase.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="The number of rounds between the Upkeep phases is determined separately at the beginning of each session.", 
    metadata={"category": "Round details"}
))  

documents.append(Document(page_content="During the Upkeep phase, each player's supply balance is calculated to determine whether units must be disbanded or whether new units can be recruited.", 
    metadata={"category": "Round details"}
))

documents.append(Document(page_content="After all players have completed their recruitment or disbanding, the Upkeep phase ends and the Planning phase begin.", 
    metadata={"category": "Round details"}
))

# Economy and Upkeep
documents.append(Document(page_content="Regions grant supply income at the start of each upkeep round.", 
    metadata={"category": "Economy and Upkeep"}
))  

documents.append(Document(page_content="A region's supply income is determined by its size and remains constant throughout the session.", 
    metadata={"category": "Economy and Upkeep"}
))

documents.append(Document(page_content="A player's supply balance is calculated by subtracting the total upkeep cost of their controlled units from the total supply income of their controlled regions.", 
    metadata={"category": "Economy and Upkeep"}
))

documents.append(Document(page_content="If a player's supply balance is positive, the player can recruit new units.", 
    metadata={"category": "Economy and Upkeep"}
)) 

documents.append(Document(page_content="If a player's supply balance is negative, the player must disband units.", 
    metadata={"category": "Economy and Upkeep"}
))  

documents.append(Document(page_content="If a player's supply balance is zero, no recruitment or disbanding is required.", 
    metadata={"category": "Economy and Upkeep"}
))

documents.append(Document(page_content="When a player has a negative supply balance, units are disbanded one at a time. After each unit is disbanded, the supply balance is recalculated. The process continues until the supply balance reaches zero or becomes positive.", 
    metadata={"category": "Economy and Upkeep"}
))

documents.append(Document(page_content="During the disbanding process, units are destroyed in order from the oldest to the newest.", 
    metadata={"category": "Economy and Upkeep"}
))  

documents.append(Document(page_content="A player can recruit at most one unit in each controlled region, provided that the total upkeep cost of all recruited units does not exceed the player's positive supply balance.", 
    metadata={"category": "Economy and Upkeep"}
))

# Player elimination
documents.append(Document(page_content="The number of players in a simulation session cannot increase, but it can decrease under certain circumstances.", 
    metadata={"category": "Player elimination"}
))

documents.append(Document(page_content="A player is eliminated when they have lost control of all their regions and have no units remaining.", 
    metadata={"category": "Player elimination"}
))

documents.append(Document(page_content="A player who has lost control of all their regions but still has units remaining is not eliminated.", 
    metadata={"category": "Player elimination"}
))

documents.append(Document(page_content="A player who has lost all their units but still controls at least one region is not eliminated.", 
    metadata={"category": "Player elimination"}
))

# Simulation end conditions
documents.append(Document(page_content="The simulation ends when either of the following conditions is met: 1. Only one player remains in the simulation. 2. The final round of the simulation has been completed.", 
    metadata={"category": "Simulation end conditions"}
))

documents.append(Document(page_content="The final round of the simulation is determined at the beginning of each session.", 
    metadata={"category": "Simulation end conditions"}
))