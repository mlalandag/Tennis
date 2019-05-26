from collections import deque, namedtuple
import torch
import time
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_left", "state_right", "action_left",
                                                                "action_right", "reward_left", "reward_right",
                                                                "next_state_left", "next_state_right", "done_left",
                                                                "done_right"])
        self.seed = random.seed(seed)

    def add(self, state_left, state_right, action_left, action_right, reward_left, reward_right,
                next_state_left, next_state_right, done_left, done_right):
        """Add a new experience to memory."""
        e = self.experience(state_left, state_right, action_left, action_right, reward_left, reward_right,
                            next_state_left, next_state_right, done_left, done_right)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states_left = torch.from_numpy(np.vstack([e.state_left for e in experiences if e is not None])).float().to(device)
        states_right = torch.from_numpy(np.vstack([e.state_right for e in experiences if e is not None])).float().to(device)
        actions_left = torch.from_numpy(np.vstack([e.action_left for e in experiences if e is not None])).float().to(device)
        actions_right = torch.from_numpy(np.vstack([e.action_right for e in experiences if e is not None])).float().to(device)
        rewards_left = torch.from_numpy(np.vstack([e.reward_left for e in experiences if e is not None])).float().to(device)
        rewards_right = torch.from_numpy(np.vstack([e.reward_right for e in experiences if e is not None])).float().to(device)
        next_states_left = torch.from_numpy(np.vstack([e.next_state_left for e in experiences if e is not None])).float().to(
            device)
        next_states_right = torch.from_numpy(
            np.vstack([e.next_state_right for e in experiences if e is not None])).float().to(
            device)
        dones_left = torch.from_numpy(np.vstack([e.done_left for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        dones_right = torch.from_numpy(np.vstack([e.done_right for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states_left, states_right, actions_left, actions_right, rewards_left, rewards_right,
                next_states_left, next_states_right, dones_left, dones_right)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)