import torch
from model import Actor, Critic
from agent import Agent
from ReplayBuffer import ReplayBuffer
import numpy as np
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 2  # minibatch size
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
        self.agent_0 = Agent(0, self.state_size, self.action_size, seed)
        self.agent_1 = Agent(1, self.state_size, self.action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()

            states, actions, rewards, next_states, dones = experiences

            #actions_next_0 = self.agent_0.actor_target(next_states[:, 0, :])
            #print("actions_next_0 ", actions_next_0)
            #actions_next_1 = self.agent_1.actor_target(next_states[:, 1, :])
            #actions_next = torch.tensor([actions_next_0, actions_next_1])
            #actions_next = [actions_next_0, actions_next_1]

            #actions_pred_0 = self.agent_0.actor_local(states[:, 0, :])
            #actions_pred_1 = self.agent_1.actor_local(states[:, 1, :])
            #actions_pred = [actions_pred_0, actions_pred_1]

            self.agent_0.learn(states, actions, rewards, next_states, dones, GAMMA)
            self.agent_1.learn(states, actions, rewards, next_states, dones, GAMMA)

    def act(self, states, add_noise=True):

        #print('Act States: ', states)

        action_0 = self.agent_0.act(states[0], add_noise)
        action_1 = self.agent_1.act(states[1], add_noise)

        return [action_0, action_1]

