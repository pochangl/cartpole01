from reinforcement.agent import Agent
from reinforcement.history import ReinforcementHistory
from reinforcement.backends.environment import GymEnvironment
from .graph import V1Graph


class CartPoleV0Agent1(Agent):
    environment_class = GymEnvironment
    graph_class = V1Graph
    history_class = ReinforcementHistory
