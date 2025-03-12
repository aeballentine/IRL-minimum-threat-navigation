"""
Inverse reinforcement learning: learn the reward function from expert demonstrations
Possible cases:
- static threat field
- dynamic threat field
- alterations to the expert cost function (only applicable for static threat fields)
- training from multiple threat fields (only applicable for static threat fields)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle
import glob

from nn_architecture import CustomRewardDataset, RewardFunction
import generate_expert_demonstrations
from generate_expert_demonstrations import time_varying
from nn_architecture import neighbors_of_four
from evaluation import dijkstra_evaluation
from nn_architecture.deep_Q_learning import DeepQLearning

torch.set_printoptions(linewidth=800)
print('Started Training. The current time is ', datetime.datetime.now())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USER-SPECIFIED PARAMETERS
dynamic_threat = True  # False if static, True if time-varying
number_fields = (
    1  # if not time-varying, set how many threat fields to use during training
)
num = 300  # number of expert paths to calculate (per threat field)

# for the static case only
new_field = False  # do evaluation on a new threat field?
single_distance = (
    True  # True if using {distance} in reward system, False if using {distance_x, distance_y}
)
lambda_val = 0  # lambda adds a weight to the total length of the path
heuristic = 0  # heuristics adds a weight to the y-distance from the finish (single distance must be False)
heuristic_index = 2
# note: both lambda and heuristic are useful for policy discrimination
threat_field_override = False   # if using a specific threat field
threat_field_name = "../01_data/custom_fields/custom_threat.csv"    # name of desired threat field

# values set only for time-varying fields
transition_steps = (
    50  # how many steps to fully transition from one threat field to another
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA PARAMETERS
# threat field - basic
target_loc = 624  # final location in the threat field
gamma = 1  # discount factor
path_length = 50  # maximum number of points to calculate along a path
dims = (25, 25)  # dimensions of the threat field
max_len = 625  # for our eval at the end of this loop, this is basically target_loc + 1
if dynamic_threat:
    max_len = -1    # this gives the total number of entries in the threat vector

# MACHINE LEARNING PARAMETERS
# reward function
if single_distance is True:
    feature_dims = (
        2  # number of features to take into account (for the reward function)
    )
else:
    feature_dims = (
        3  # number of features to take into account (for the reward function)
    )
if dynamic_threat is True:
    feature_dims = 2    # not accounting for x and y distance for time-varying fields
batch_size = 1  # number of samples to take per batch
learning_rate = 0.05  # learning rate
epochs = 300  # number of epochs for the main training loop
criterion = nn.MSELoss()
reward_min_accuracy = 10  # value to terminate the IRL loop
reward_clip_value = 100

# value function
q_tau = 0.0001  # rate at which to update the target_net variable inside the Q-learning module
q_lr = 0.0001  # learning rate for Q-learning
q_criterion = (
    nn.HuberLoss()
)  # criterion to determine the loss during training
q_features = 5 * feature_dims  # number of features to take into consideration
q_pretraining = 50  # times to run q_learning inside each reward function loop

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# select the device to use: cpu, mps, or gpu (mps/cuda is faster)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print('The device used for training is \t', device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEIGHBORS OF FOUR AND DISTANCES
distance = None
distance_x = None
distance_y = None
if dynamic_threat:
    neighbors_tensor, distance = time_varying.time_dependent_neighbors(
        dims=dims, steps=transition_steps, target=target_loc
    )
else:
    neighbors = neighbors_of_four(dims=dims, target=target_loc)
    neighbors_tensor = torch.from_numpy(
        neighbors[["left", "right", "up", "down"]].values
    ).to(device)
    if single_distance is False:
        # x-distance to the finish
        distance_x = torch.from_numpy(neighbors[["x_dist"]].values).float().to(device)
        max_distance_x = torch.tensor([[torch.max(distance_x)]]).to(device)
        distance_x = torch.concat((distance_x, max_distance_x))
        # y-distance to the finish
        distance_y = torch.from_numpy(neighbors[["y_dist"]].values).float().to(device)
        max_distance_y = torch.tensor([[torch.max(distance_y)]]).to(device)
        distance_y = torch.concat((distance_y, max_distance_y))
    else:
        # single distance to the finish
        distance = torch.from_numpy(neighbors[["dist"]].values).float().to(device)
        max_distance = torch.tensor([[torch.max(distance)]]).to(device)
        distance = torch.concat((distance, max_distance))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOAD THE DATA
# run get_expert_demos()
# first, select which threat field to use
if threat_field_override is False:
    if dynamic_threat:
        data_path = "../01_data/cost_function/"
        file_list = glob.glob(data_path + "*")
        # choose two threat fields
        random_seed_a = np.random.randint(0, 999)
        random_seed_b = np.random.randint(0, 999)
        file_list = [file_list[i] for i in [random_seed_a, random_seed_b]]
    else:
        data_path = "../01_data/cost_function/"
        all_files = glob.glob(data_path + "*")
        random_seed = np.random.randint(0, 999, number_fields)
        file_list = []
        for seed in random_seed:
            file_list.append(all_files[seed])
else:
    file_list = [threat_field_name]

# run get_expert_demos() and return the dictionary of states
data = generate_expert_demonstrations.get_expert_demos(
    num_paths=num,
    neighbors_tensor=neighbors_tensor,
    device=device,
    distance_x=distance_x,
    distance_y=distance_y,
    distance=distance,
    dynamic=dynamic_threat,
    constant=lambda_val,
    heuristic=heuristic,
    heuristic_index=heuristic_index,
    file_list=file_list,
    end_index=target_loc,
    transition_steps=transition_steps
)

feature_averages = data["expert_feat"]
feature_function = data["feature_map"]
threat_fields = data["threat_field"]
test_points = data["test_points"][0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# constants for the network & initialize the reward model
rewards = RewardFunction(
    feature_dimensions=feature_dims, my_device=device, reward_lr=learning_rate
)
mu_bar_prior = rewards.weights  # set this for the first iteration of IRL

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create the dataloader and the testloader
dataset = CustomRewardDataset(
    feature_map=feature_function, expert_expectation=feature_averages
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set up the deep Q network
q_learning = DeepQLearning(
    n_observations=q_features,
    n_actions=4,
    device=device,
    LR=q_lr,
    neighbors=neighbors_tensor.to(device),
    gamma=gamma,
    target_loc=target_loc,
    tau=q_tau,
    criterion=q_criterion,
    path_length=path_length,
    starting_coords=torch.tensor(test_points),
    reward_dims=feature_dims,
    failed_loc=max_len
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train the model

# save variable for plotting and analysis
total_losses = torch.tensor([]).to(device)
finish_counter = torch.tensor([]).to(device)
q_learning_losses = []
failure_counter = []
q_learning.loss = 50

for epoch in range(epochs):
    for batch_num, input_data in enumerate(dataloader):
        x, y = (
            input_data  # x is the feature function field and y is the expert average feature expectation
        )

        # reset the target network
        if q_learning.loss < 50:
            q_learning.target_net.load_state_dict(q_learning.policy_net.state_dict())
        # run pre-training module
        q_learning.reward = rewards
        for counter in torch.linspace(1, 11, q_pretraining).to(device):
            output, q_learning_loss, q_learning_failures, q_learning_finishes = (
                q_learning.run_q_learning(feature_vector=x, pretraining=True)
            )

        # recalculate the expectation
        output, q_learning_loss, q_learning_failures, q_learning_finishes = (
            q_learning.run_q_learning(feature_vector=x)
        )

        # find mu
        mu = torch.sum(output.view(-1, q_features)[:, :feature_dims], dim=0)
        mu = mu / num  # divide by the number of paths

        # find mu_expert
        mu_expert = torch.sum(y[0].view(-1, q_features)[:, :feature_dims], dim=0)
        mu_expert = mu_expert / num  # divide by the number of paths

        # calculate the loss using the "Projection Method" from (Abbeel and Ng)
        mu_line = mu - mu_bar_prior
        mu_bar = mu_bar_prior + torch.dot(
            mu_line, (mu_expert - mu_bar_prior)
        ) * mu_line / torch.dot(mu_line, mu_line)

        # clip the loss function
        mu_bar = mu_bar.clip(min=-reward_clip_value, max=reward_clip_value)
        mu_bar_prior = mu_bar

        # update the reward weights
        rewards.set_weights(new_weights=mu_bar)  # update the reward function weights

        # this is for the termination condition
        loss = criterion(mu, mu_expert)

        total_losses = torch.concat((total_losses, torch.tensor([loss]).to(device))).to(
            device
        )
        finish_counter = torch.concat(
            (finish_counter, torch.tensor([q_learning_finishes]).to(device))
        ).to(device)
        q_learning_losses.append(q_learning_loss)
        failure_counter.append(q_learning_failures.item())

    if (
        (loss.item() < reward_min_accuracy)
        & (q_learning_finishes > 0.9 * num)
    ):
        print(epoch)
        print(loss.item())
        print(q_learning_finishes)
        epochs = epoch + 1
        break
print(loss.item())
print(q_learning_finishes)
print('Finished Training! The current time is ', datetime.datetime.now())
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# evaluate against Dijkstra's
(
    nn_length,
    dijkstra_length,
    nn_average,
    dijkstra_average,
    n_failures,
    n_departures,
    nn_cost,
    dijkstra_cost,
    nn_paths,
) = dijkstra_evaluation(
    policy_net=q_learning.policy_net,
    device=device,
    feature_function_=feature_function[0],
    neighbors_tensor=neighbors_tensor,
    n_observations=q_features,
    max_len=max_len,
    target=target_loc,
    constant=lambda_val,
    time_varying=dynamic_threat
)

# make plots
# reward loss
reward_learning = go.Figure(
    data=[go.Scatter(x=np.arange(0, epochs * number_fields, 1), y=total_losses.cpu())]
)
reward_learning.update_xaxes(title_text="Epoch")
reward_learning.update_yaxes(title_text="Reward Loss")
reward_learning.show()

# q learning loss
q_losses_training = go.Figure(
    data=[go.Scatter(x=np.arange(0, epochs * number_fields, 1), y=q_learning_losses)]
)
q_losses_training.update_xaxes(title_text="Epoch")
q_losses_training.update_yaxes(title_text="Q-learning Losses")
q_losses_training.show()

# number of finishes
failures = go.Figure(
    data=[go.Scatter(x=np.arange(0, epochs * number_fields, 1), y=failure_counter)]
)
failures.update_xaxes(title_text="Epoch")
failures.update_yaxes(title_text="Failures")
failures.show()

# number of failures
finishes = go.Figure(
    data=[go.Scatter(x=np.arange(0, epochs * number_fields, 1), y=finish_counter.cpu())]
)
finishes.update_xaxes(title_text="Epoch")
finishes.update_yaxes(title_text="Finishes")
finishes.show()

# eval
results = pd.DataFrame(
    {
        "Dijkstra Path Length": dijkstra_length,
        "Error [%]": 100
        * (np.array(nn_average) - np.array(dijkstra_average))
        / np.array(dijkstra_average),
    }
)
final_results = go.Figure(
    data=[
        go.Scatter(
            x=results["Dijkstra Path Length"], y=results["Error [%]"], mode="markers"
        )
    ]
)
final_results.update_xaxes(title_text="Dijkstra Path Length")
final_results.update_yaxes(title_text="Error [%]")
final_results.show()

print('Number of Failures: \t', n_failures)
print('Number of Departures: \t', n_departures)

percent_error = results["Error [%]"]
average_error = torch.mean(torch.tensor(percent_error).float()).round().item()
print('Average Error (for all successful paths): \t', average_error)

print('Finished Analysis! The current time is ', datetime.datetime.now())

# make a dictionary of our data (makes it easier to do analysis later)
time = datetime.datetime.now()
file_name = (
    "../01_data/results/"
    + str(time.month)
    + "_"
    + str(time.day)
    + "_"
    + str(time.hour)
    + ":"
    + str(time.minute)
    + "_performance_"
    + str(average_error)[:-2]
    + "_finishes_"
    + str(624 - n_departures - n_failures)
    + ".pkl"
)
my_dict = {
    "dims": target_loc,
    "number_of_training_paths": num,
    "reward_features": feature_dims,
    "reward_min_accuracy": reward_min_accuracy,
    "reward_LR": learning_rate,
    "reward_clip_value": reward_clip_value,
    "q_tau": q_tau,
    "q_lr": q_lr,
    "q_features": q_features,
    "reward_loss": total_losses,
    "q_loss_training": q_learning_losses,
    "failures": failures,
    "finishes": finishes,
    "dijkstra_length": dijkstra_length,
    "nn_length": nn_length,
    "percent_error": 100
    * (np.array(nn_average) - np.array(dijkstra_average))
    / np.array(dijkstra_average),
    "dijkstra_cost": dijkstra_average,
    "nn_cost": nn_average,
    "final_failures": n_failures,
    "final_departures": n_departures,
    "nn_pointwise_cost": nn_cost,
    "dijkstra_pointwise_cost": dijkstra_cost,
    "threat_field": threat_fields,
    "feature_function": feature_function[0],
    "nn_paths": nn_paths,
}
# Note that in the nn_cost vector, if the values are zero, it means that the algorithm fully left the graph
# Conversely, if the values in the nn_cost and dijkstra_cost are -1, it means that the algorithm failed to find the goal

f = open(file_name, "wb")  # make our file to save the data
pickle.dump(my_dict, f)  # save the dictionary to the file
f.close()
