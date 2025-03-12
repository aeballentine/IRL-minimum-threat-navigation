"""
This is the module used for the main training loop
Goal: train an agent to move according to a known reward function
Possible movements: left, right, up, and down

IDEAS: try using 2 q-learning networks, & weighting the results OR try using multiple "steps" per iteration
"""

import math
import random
import torch
from torch import optim
import copy

from nn_architecture import DQN


class DeepQLearning:
    def __init__(
        self,
        reward_dims,
        n_observations,
        n_actions,
        device,
        LR,
        neighbors,
        gamma,
        target_loc,
        tau,
        criterion,
        path_length,
        starting_coords,
        failed_loc
    ):
        """
        Deep Q Learning module, adapted from Mnih et. al. (2013)
        :param reward_dims: number of features used for the reward function
        :param n_observations: number of features to account for each movement
        :param n_actions: number of possible actions
        :param device: device (cpu, cuda, or mps)
        :param LR: learning rate for the policy network
        :param neighbors: neighbors tensor (used to determine adjacency)
        :param gamma: discount rate
        :param target_loc: goal location index
        :param tau: learning rate for the target network
        :param criterion: criterion to determine the loss function
        :param path_length: maximum path length to calculate
        :param starting_coords: starting coordinates for sample paths
        """
        # basic parameters
        self.reward_dims = reward_dims
        self.n_observations = (
            n_observations  # number of characteristics to determine the state
        )
        self.n_actions = n_actions  # number of possible actions
        self.LR = LR  # learning rate
        self.tau = tau  # parameter to update the target network
        self.target_loc = target_loc  # target location
        self.path_length = path_length  # maximum path length to calculate (this goes into calculating the loss)
        # self.expert_paths = expert_paths[0]      # expert paths to use during training
        self.starting_coords = (
            starting_coords  # starting coords (goes into the loss function)
        )

        # for the neural network
        self.device = device  # device to save variables to
        self.criterion = criterion  # criterion to update the policy network
        self.loss = None

        # create the two neural networks: target and policy
        self.policy_net = DQN(
            n_observations=self.n_observations, n_actions=n_actions
        ).to(self.device)
        self.target_net = DQN(
            n_observations=self.n_observations, n_actions=n_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  #
        # self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.LR, weight_decay=1e-5
        )

        # counters and things
        self.step = 0  # how many optimization steps have been taken?
        self.EPS_START = 0.85  # starting value
        self.EPS_END = 0.05  # lowest possible value
        self.EPS_DECAY = 750  # decay rate

        # movement tracking
        self.neighbors = neighbors

        # for rewards
        self.gamma = gamma
        self.reward = None

        # relevant to account for time-varying when needed
        self.moving_indices = None
        self.moving_locations = None
        self.relevant_features = None
        self.all_features = None
        self.not_moving_indices = None
        self.unique_locations = None

        self.fail_loc = failed_loc

    def select_action(self):
        # we have two cases: take a random sample of action or sample according to the current policy

        # determine if we are sampling randomly or not
        sample = random.random()  # generate a random number
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1 * self.step / self.EPS_DECAY
        )
        if sample > eps_threshold:  # decide actions according to the policy
            with torch.no_grad():
                actions = (
                    self.policy_net(self.all_features.to(self.device)).max(1).indices
                )
        else:  # generate random actions - one for each point in the threat field (not finishes/failures)
            actions = torch.randint(0, 4, (len(self.all_features),)).to(self.device)
        self.step += 1  # incremement step by 1
        return actions

    def find_next_state(self, actions):
        original_actions = actions[self.moving_locations]   # set of actions if we don't care about finish/fail locs
        # neighbors should have rows corresponding to locations and columns corresponding to locations
        # find the new locations according to the moving indices & corresponding actions
        next_locs = self.neighbors[self.moving_indices, original_actions]
        # find the new features according to the next locs
        new_features = self.all_features[next_locs].to(self.device)
        with torch.no_grad():
            rewards = self.reward(new_features[:, :self.reward_dims])   # calculate the reward

        # now we want to filter actions, rewards, etc to only start from non-finished/failed locations
        moving_locs = torch.isin(next_locs, self.moving_indices).bool()     # moving locs only if next locs is in moving_indices
        finished = next_locs == self.target_loc     # find finishes
        failed = next_locs == self.fail_loc     # find failures

        next_locs = next_locs[moving_locs]  # filter the new locations to only those which are not finishes/failures
        new_actions = actions[next_locs]    # find the new actions from the original action vector & new locations

        return original_actions, rewards, new_features, moving_locs, new_actions, failed, finished

    def optimize_model(
        self,
        actions,
        new_features,
        rewards,
        not_finished,
        new_actions,
        failed,
        finished,
    ):
        state_action_values = self.policy_net(self.relevant_features).gather(1, actions)    # find the output of Q
        next_state_values = torch.zeros(len(self.relevant_features), 1).to(self.device)     # vector for Q'
        with torch.no_grad():
            next_state_values[not_finished] = self.target_net(
                new_features[not_finished]
            ).gather(1, new_actions)    # if the next state is not a finish/failure, find Q'
            next_state_values[failed] = state_action_values[failed]     # if next state is a failure, set Q'=Q
            next_state_values[finished] = 100     # if next state is a finish, set Q'=100

        expected_state_action_values = (
            next_state_values * self.gamma
        ) + rewards.unsqueeze(1)    # find r + gamma * Q' for all states

        loss = self.criterion(state_action_values, expected_state_action_values)    # calculate the loss: Q - (r + gamma * Q')

        # take one gradient descent step
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 50)
        self.optimizer.step()

        return loss.item()

    def run_q_learning(self, feature_vector, pretraining=False):
        # do one training step

        # each time: calculate the relevant threat field features
        self.all_features = copy.deepcopy(
            feature_vector[0]
        ).detach()
        self.all_features = self.all_features.to(self.device)   # this is the list of all possible features

        # generate the vector for not-finishes, not-failures (locations)
        self.moving_locations = torch.ones(len(self.all_features)).bool().to(self.device)
        self.moving_locations[-1] = False   # this is the failure location
        finish_locations = torch.arange(
            self.target_loc, len(self.all_features), self.target_loc + 1
        )   # this marks where all the finish locations are (if time-varying)
        self.moving_locations[finish_locations] = False
        self.relevant_features = copy.deepcopy(self.all_features[self.moving_locations]).to(self.device)    # features for moving locations
        not_moving_indices = ~self.moving_locations  # for failures and finishes

        self.unique_locations = torch.arange(0, len(self.all_features), 1).to(
            self.device
        )
        self.not_moving_indices = self.unique_locations[not_moving_indices]   # indices for finishes/failures
        self.moving_indices = self.unique_locations[self.moving_locations]      # indices if not finished/failed

        action = self.select_action()
        actions, rewards, new_features, not_finished, new_actions, failed, finished = (
            self.find_next_state(actions=action)
        )

        loss = self.optimize_model(
            actions=actions.view(-1, 1),
            new_features=new_features.to(self.device),
            rewards=rewards,
            not_finished=not_finished,
            new_actions=new_actions.view(-1, 1),
            failed=failed,
            finished=finished,
        )

        # update the target network with a soft update: θ′ ← τ θ + (1 - τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        self.loss = loss

        if (
            self.starting_coords is not None and pretraining is False
        ):  # if we gave the class a set of coordinates to test from & want a sample of the results
            sums, failures, finishes = self.find_feature_expectation(
                feature_function=feature_vector
            )
            return sums, loss, failures, finishes

        else:
            return None, loss, None, None

    def find_feature_expectation(self, feature_function):
        # map the feature function to one long vector (nx15 instead of nx626x15)
        feature_function = feature_function.view(-1, self.n_observations).to(
            self.device
        )

        # tile the starting coordinates
        coords = self.starting_coords.to(self.device)

        # feature function that we can use to return
        my_features = torch.zeros(
            len(self.starting_coords), self.path_length, self.n_observations
        ).to(self.device)
        # starting features
        my_features[:, 0, :] = feature_function[coords]  # + coords_conv]
        # features used to calculate the desired action
        new_features = copy.deepcopy(my_features[:, 0, :]).to(self.device)

        # mask for any finished paths (or terminated paths)
        mask = torch.ones(coords.shape).bool().to(self.device)
        finished_mask = torch.ones(coords.shape).bool().to(self.device)

        for step in range(
            self.path_length - 1
        ):  # path length - 1 because the first coordinate counts too

            with torch.no_grad():
                resultant = self.policy_net(new_features)
                action = resultant.max(1).indices[
                    mask
                ]  # determine the action according to the policy network

            coords[mask] = self.neighbors[coords[mask], action]
            # coords[mask] = torch.tensor(
            #     map(
            #         lambda index: self.neighbors[
            #             index[1], action[index[0]]
            #         ],
            #         enumerate(coords[mask]),
            #     )
            # )  # find the next coordinate according to the initial location and action

            ind = coords == self.target_loc
            mask[ind] = False
            finished_mask[ind] = False
            failures = coords == (self.target_loc + 1)
            mask[failures] = False

            new_features = (
                feature_function[coords]
                .view(-1, self.n_observations)
                .to(self.device)
            )  # find the features at the new location
            # now add the features: should be gamma^t * new_features for t in [0, T]
            my_features[finished_mask, step + 1, :] += (
                self.gamma ** (step + 1) * new_features[finished_mask]
            )

        total_paths = len(coords)
        not_finishes_failures = sum(mask)
        not_finishes = sum(finished_mask)
        finishes = total_paths - not_finishes
        failures = total_paths - not_finishes_failures - finishes

        return (
            my_features,
            failures,
            finishes,
        )
