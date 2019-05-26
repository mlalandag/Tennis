import numpy as np
import random
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, id, action_size, actor_local, actor_target, critic_local, critic_target, random_seed):
        """Initialize an Agent object.

        Params
        ======
            actor_local (Actor):  local Actor
            actor_target (Actor): target
            critic_local (Critic):  local Critic
            critic_target (Critic): target Critic
            random_seed (int): random seed
        """
        self.id = id
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = actor_local
        self.actor_target = actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_local
        self.critic_target = critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    # def step(self, states, actions, rewards, next_states, dones, actions_next, actions_pred):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     # Save experience / reward
    #     # print('Step state = ', state.size())
    #     # print('Step action = ', action.size())
    #     # print('Step reward = ', reward.size())
    #     # print('Step next_state = ', next_state.size())
    #     # print('Step done = ', done.size())
    #
    #     self.memory.add(states[0], states[1], actions[0], actions[1], rewards[0], rewards[1],
    #                     next_states[0], next_states[1], dones[0], dones[1], actions_next[0], actions_next[1],
    #                     actions_pred[0], actions_pred[1])
    #     #print('elementos en memoria: {}'.format(len(self.memory)))
    #
    #     # Learn, if enough samples are available in memory
    #     if len(self.memory) > BATCH_SIZE:
    #         experiences = self.memory.sample()
    #         # print('Experiences = ', experiences)
    #         self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, actions_next, actions_pred, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_left, states_right, actions_left, actions_right, rewards_left, rewards_right, \
            next_states_left, next_states_right, dones_left, dones_right = experiences

        actions_next_left = actions_next[0]
        actions_next_right = actions_next[1]
        actions_pred_left = actions_pred[0]
        actions_pred_right = actions_pred[1]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # if self.id == "left":
        #     actions_next_left = self.actor_target(next_states_left)
        # else:
        #     actions_next_right = self.actor_target(next_states_right)
        # print('actions_next_left = ', actions_next_left.size())
        # print('actions_next_right = ', actions_next_right.size())
        Q_targets_next = self.critic_target(next_states_left, next_states_right, actions_next_left, actions_next_right)
        # Compute Q targets for current states (y_i)
        if self.id == "left":
            Q_targets = rewards_left + (gamma * Q_targets_next * (1 - dones_left))
        else:
            Q_targets = rewards_right + (gamma * Q_targets_next * (1 - dones_right))

        # Compute critic loss
        Q_expected = self.critic_local(states_left, states_right, actions_left, actions_right)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # actions_pred_left = self.actor_local(states_left)
        # actions_pred_right = self.actor_local(states_right)
        actor_loss = -self.critic_local(states_left, states_right, actions_pred_left, actions_pred_right).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)