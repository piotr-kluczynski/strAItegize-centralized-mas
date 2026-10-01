from leader_agent import get_leader_agent

# Creating the overall Multi-Agent System architecture

# In this case, the architecture itself is encased fully within the Leader Agent, 
# due to its simple Sub-Agent architecture
def get_centralized_mas(communication, human_mode=False):
    return get_leader_agent(communication, human_mode)