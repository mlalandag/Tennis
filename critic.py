import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        # print('Critic input_size {}'.format(input_size))

        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states_left, states_right, actions_left, actions_right):

        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # print('Critic states_left dim', states_left.size())
        # print('Critic states_right dim', states_right.size())
        # print('Critic actions_left dim ', actions_left.size())
        # print('Critic actions_right dim ', actions_right.size())

        x = torch.cat((states_left, states_right, actions_left, actions_right), dim=1)

        # print('x in forward size {}'.format(x.size))

        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)

        # print('x shape {}'.format(x.size()))

        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        return self.fc3(x)