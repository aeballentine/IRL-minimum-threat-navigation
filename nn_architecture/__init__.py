import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset
from torch import nn


def to_2d(loc, dims):
    """
    :param loc: coordinate in the graph, measured from the lower left-hand corner (left to right then down to up)
    :param dims: dimensions of the graph
    :return: the [i, j] index
    """
    # loc is a one-dimensional index
    # NOTE: this indexes from the lower left corner
    i, j = loc // dims[1], loc % dims[1]
    return [i, j]


def to_1d(array, dims):
    """
    :param array: [i, j] index, from the lower left hand corner
    :param dims: dimensions of the graph
    :return: single index
    """
    loc = array[0] * dims[1] + array[1]
    return loc


def neighbors_of_four(dims, target):
    """
    This function gives the four-point adjacency neighbors for a single threat field
    :param dims: dimensions of the graph
    :param target: goal location
    :return: dataframe with relevant parameters: left, right, up, and down neighbors, and distances
    """
    size = dims[0] * dims[1]
    my_points = np.arange(0, size, 1)
    coords = to_2d(my_points, dims)
    target = to_2d(target, dims)

    # Euclidean distance to the target location (based on the index only - not scaled)
    x_distance = target[1] - coords[1]
    y_distance = target[0] - coords[0]
    dist = (x_distance**2 + y_distance**2) ** 0.5
    dist = minmax_scale(dist, feature_range=(0, 10))
    x_dist = minmax_scale(x_distance, feature_range=(0, 5))
    y_dist = minmax_scale(y_distance, feature_range=(0, 5))
    coords = np.concatenate(
        (np.reshape(coords[0], (size, 1)), np.reshape(coords[1], (size, 1))), axis=1
    )
    movements = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # left, right, down, up

    # find all four neighbors
    left_neighbors = [
        [coords[i][0] + movements[0][0], coords[i][1] + movements[0][1]]
        for i in range(len(coords))
    ]
    right_neighbors = [
        [coords[i][0] + movements[1][0], coords[i][1] + movements[1][1]]
        for i in range(len(coords))
    ]
    down_neighbors = [
        [coords[i][0] + movements[2][0], coords[i][1] + movements[2][1]]
        for i in range(len(coords))
    ]
    up_neighbors = [
        [coords[i][0] + movements[3][0], coords[i][1] + movements[3][1]]
        for i in range(len(coords))
    ]

    # find if anything is out of bounds
    for i in range(len(left_neighbors)):
        if (0 <= left_neighbors[i][0] < dims[0]) & (
            0 <= left_neighbors[i][1] < dims[1]
        ):
            left_neighbors[i] = to_1d(left_neighbors[i], dims)
        else:
            left_neighbors[i] = max(my_points) + 1

        if (0 <= right_neighbors[i][0] < dims[0]) & (
            0 <= right_neighbors[i][1] < dims[1]
        ):
            right_neighbors[i] = to_1d(right_neighbors[i], dims)
        else:
            right_neighbors[i] = max(my_points) + 1

        if (0 <= down_neighbors[i][0] < dims[0]) & (
            0 <= down_neighbors[i][1] < dims[1]
        ):
            down_neighbors[i] = to_1d(down_neighbors[i], dims)
        else:
            down_neighbors[i] = max(my_points) + 1

        if (0 <= up_neighbors[i][0] < dims[0]) & (0 <= up_neighbors[i][1] < dims[1]):
            up_neighbors[i] = to_1d(up_neighbors[i], dims)
        else:
            up_neighbors[i] = max(my_points) + 1

    my_neighbors = pd.DataFrame(
        {
            "points": my_points,
            "left": left_neighbors,
            "right": right_neighbors,
            "up": up_neighbors,
            "down": down_neighbors,
            "dist": dist,
            "x_dist": x_dist,
            "y_dist": y_dist,
        }
    )
    return my_neighbors


class CustomRewardDataset(Dataset):
    """
    Dataset for the reward function:
    Two returns: feature map and the associated expert expectation
    """

    def __init__(self, feature_map, expert_expectation):
        self.feature_map = feature_map
        self.expert_expectation = expert_expectation

    def __len__(self):
        return len(self.expert_expectation)

    def __getitem__(self, item):
        feature_map = self.feature_map[item]
        expert_expectation = self.expert_expectation[item]

        return feature_map.float(), expert_expectation.float()


class RewardFunction(nn.Module):
    """
    Set up a reward function network for the projection method
    Guess is equal weights for each entry, i.e. (1, 1)
    """

    def __init__(self, feature_dimensions, my_device, reward_lr):
        super(RewardFunction, self).__init__()
        # initialize the weights
        self.weights = torch.ones(feature_dimensions).float().to(my_device)

        # normalize the starting weights
        self.weights = self.weights / torch.sqrt(torch.sum(self.weights**2))
        self.device = my_device

        # learning rate
        self.learning_rate = reward_lr

    def forward(self, features):
        # return the anticipated reward
        return -25 * torch.matmul(features, self.weights)

    def set_weights(self, new_weights):
        # mu should be a torch tensor - these are the new reward function weights
        new_weights = new_weights / torch.sqrt(
            torch.sum(new_weights**2)
        )  # normalize mu
        self.weights = (
            new_weights.to(self.device) * self.learning_rate
            + (1 - self.learning_rate) * self.weights
        )
        self.weights = self.weights / torch.sqrt(
            torch.sum(self.weights**2)
        )  # normalize the weights


class DQN(nn.Module):
    """
    Deep Q-Learning network: multi-layer perceptron
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        return self.layer4(x)
