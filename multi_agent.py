import torch
from model import Actor, Critic
from agent import Agent
from ReplayBuffer import ReplayBuffer
import numpy as np
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512 # minibatch size
GAMMA = 0.99  # discount factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent():
    """Contains all the agents"""

    def __init__(self, n_agents, state_size, action_size, seed=0):
        """
        Params
        ======
            n_agents (int): number of distinct agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.seed = random.seed(1)
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agents = []

        # create agents
        #self.agent_0 = Agent(0, self.n_agents, self.state_size, self.action_size, seed)
        #self.agent_1 = Agent(1, self.n_agents, self.state_size, self.action_size, seed)
        self.agents = [Agent(i, n_agents, state_size, action_size, random_seed=1) for i in range(n_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Each agent stores its own experiences
        # for id, agent in enumerate(self.agents):
        #     agent.step(states[id], actions[id], rewards[id], next_states[id], dones[id])        

        # Save shared experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):        
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # if len(self.memory) > BATCH_SIZE * self.n_agents:
        if len(self.memory) > BATCH_SIZE:            

            experiences = self.memory.sample()

            for id, agent in enumerate(self.agents):
                agent.learn(id, experiences, GAMMA)
                

    def act(self, states, add_noise=True):

        return [agent.act(states[id], add_noise) for id, agent in enumerate(self.agents)]
            

    def reset(self):
        for agent in self.agents:
            agent.reset()            