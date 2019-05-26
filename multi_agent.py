
import torch
from actor import Actor
from critic import Critic
from agent import Agent
from ReplayBuffer import ReplayBuffer
import numpy as np
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
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
        self.critic_input_size = (state_size + action_size) * self.n_agents
        self.agents = []

        # create two agents, each with their own actor and critic
        self.actor_local_left = Actor(state_size, action_size, seed).to(device)
        self.actor_target_left = Actor(state_size, action_size, seed).to(device)
        self.critic_local_left = Critic(self.critic_input_size, seed).to(device)
        self.critic_target_left = Critic(self.critic_input_size, seed).to(device)
        self.agent_left = Agent("left", self.action_size, self.actor_local_left, self.actor_target_left,
                                self.critic_local_left, self.critic_target_left, seed)

        self.actor_local_rigth = Actor(state_size, action_size, seed).to(device)
        self.actor_target_rigth = Actor(state_size, action_size, seed).to(device)
        self.critic_local_rigth = Critic(self.critic_input_size, seed).to(device)
        self.critic_target_right = Critic(self.critic_input_size, seed).to(device)
        self.agent_right = Agent("right", self.action_size, self.actor_local_rigth, self.actor_target_rigth,
                            self.critic_local_rigth, self.critic_target_right, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add(states[0], states[1], actions[0], actions[1], rewards[0], rewards[1],
                        next_states[0], next_states[1], dones[0], dones[1])

        #print('elementos en memoria: {}'.format(len(self.memory)))
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            # print('Experiences = ', experiences)
            self.learn(experiences, GAMMA)


    def learn(self, experiences, GAMMA):

        states_left, states_right, actions_left, actions_right, rewards_left, rewards_right, \
            next_states_left, next_states_right, dones_left, dones_right = experiences

        actions_next_left = self.agent_left.actor_target(next_states_left)
        actions_next_right = self.agent_right.actor_target(next_states_right)
        actions_next = [actions_next_left, actions_next_right]

        actions_pred_left = self.agent_left.actor_local(states_left)
        actions_pred_right = self.agent_right.actor_local(states_right)
        actions_pred = [actions_pred_left, actions_pred_right]

        self.agent_left.learn(experiences, actions_next, actions_pred, GAMMA)
        self.agent_right.learn(experiences, actions_next, actions_pred, GAMMA)

    def act(self, states, add_noise=True):

        action_left = self.agent_left.act(states[0], add_noise)
        action_right = self.agent_right.act(states[1], add_noise)

        return [action_left, action_right]

        # actions = np.random.randn(2, 2)
        # actions = np.clip(actions, -1, 1)
        # return actions
