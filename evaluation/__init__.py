"""
Evaluation file...this is supposed to be used after fully training the neural network
Required arguments: the feature function, the policy network, neighbors tensor, and dimensions and related things
Note: if trying to evaluate an already trained network, the best way to do that is likely to have the Q-learning network
class in another file, not going to add that here.
"""

import torch
import numpy as np
from dijkstras_algorithm import dijkstra_bwd, dijkstra_fwd


def find_nn_path(
    feature_function,
    starting_coord,
    policy_net,
    neighbors,
    device,
    n_observations,
    max_len,
    target,
):
    """
    :param feature_function: feature function for the threat field
    :param starting_coord: starting coordinate
    :param policy_net: policy network (from Q-learning)
    :param neighbors: neighbors tensor
    :param device: device (cpu, mps, or cuda)
    :param n_observations: number of inputs
    :param max_len: maximum length (fail loc)
    :param target: target location/index (index in neighbors tensor)
    :return: path, if reached the finish, if left the graph, final location
    """
    feature_function = feature_function.view(-1, n_observations)
    new_coord = starting_coord
    new_feat = feature_function[new_coord].to(device)
    path = [new_coord]

    success = True
    left = False
    for step in range(100):
        with torch.no_grad():
            action = (
                policy_net(new_feat).max(0).indices
            )  # this should be max(1) for multi-threat

        new_coord = neighbors[new_coord, action]
        path.append(int(new_coord))

        if new_coord == max_len:
            success = False
            left = True
            break

        elif new_coord == target:
            break

        new_feat = feature_function[new_coord].to(device)

        if step == 99:
            success = False

    return path, success, left, new_coord


def dijkstra_evaluation(
    policy_net,
    device,
    feature_function_,
    neighbors_tensor,
    n_observations,
    time_varying,
    max_len=625,
    target=624,
    constant=0,
    heuristic=0,
    heuristic_index=2
):
    """
    :param policy_net: policy network
    :param device: device (cpu, mps, or cuda)
    :param feature_function_: feature function for the threat field
    :param neighbors_tensor: neighbors tensor
    :param n_observations: number of inputs to the policy network
    :param time_varying: if time varying or not
    :param max_len: fail loc (referenced in the neighbors tensor)
    :param target: target loc (referenced in the neighbors tensor)
    :param constant: for Dijkstra's algorithm
    :param heuristic: for Dijkstra's algorithm
    :param heuristic_index: for Dijkstra's algorithm
    :return: evaluation metrics
    """
    if time_varying is True:
        return eval_dynamic(
            policy_net=policy_net,
            device=device,
            feature_function_=feature_function_,
            neighbors_tensor=neighbors_tensor,
            n_observations=n_observations,
            target=target,
            max_len=max_len
        )
    else:
        return eval_static(
            policy_net=policy_net,
            device=device,
            feature_function_=feature_function_,
            neighbors_tensor=neighbors_tensor,
            n_observations=n_observations,
            max_len=max_len,
            target=target,
            constant=constant,
            heuristic=heuristic,
            heuristic_index=heuristic_index
        )


def eval_dynamic(
    policy_net,
    device,
    feature_function_,
    neighbors_tensor,
    n_observations,
    max_len,
    target,
):
    # for Dijkstra's algorithm
    vertices = np.arange(0, len(feature_function_), 1)

    # chose starting coordinates
    # starting_coords = torch.arange(0, max_len - 1, 1).to(device)
    starting_coords = np.arange(0, 624, 1)

    # vectors to hold results
    average_discrepancy = []
    dijkstra_average = []
    nn_average = []

    # hold the results for the threat field from loc 1 to 624
    nn_cost_vector = []
    dijkstra_cost_vector = []

    # path length
    dijkstra_length = []
    nn_length = []

    # number of failures of the neural network
    n_failures = 0
    n_departures = 0

    failed_loc = []

    nn_paths = []

    for coord in starting_coords:
        print(coord)
        # call dijsktra's and the nn
        nn_path, info, outside, fail_loc = find_nn_path(
            feature_function=feature_function_.float(),
            starting_coord=int(coord),
            policy_net=policy_net,
            neighbors=neighbors_tensor,
            device=device,
            n_observations=n_observations,
            max_len=max_len,
            target=target,
        )
        nn_paths.append(nn_path)

        # if the network fails, terminate the loop
        if info is False:
            print('Failed!')
            if outside is True:
                n_departures += 1
                dijkstra_cost_vector.append(0)
                nn_cost_vector.append(0)
            else:
                n_failures += 1
                failed_loc.append(fail_loc)
                dijkstra_cost_vector.append(-1)
                nn_cost_vector.append(-1)
            continue

        dijkstra_info = dijkstra_fwd(
            feature_function=feature_function_,
            vertices=vertices,
            source=int(coord),
            neighbors=neighbors_tensor,
            max_len=max_len,
            node_f=target,
        )

        # recover the Dijkstra algorithm path
        node = target
        dijkstra_path = [node]
        while node != coord:
            previous_node = dijkstra_info[node].parent
            dijkstra_path.append(previous_node)
            node = previous_node
        dijkstra_path = dijkstra_path[::-1]

        # determine the cost of the neural network path and of Dijkstra's algorithm
        dijkstra_cost = 0
        for node in dijkstra_path[1:]:
            dijkstra_cost += feature_function_[node][0]

        nn_cost = 0
        for node in nn_path[1:]:
            nn_cost += feature_function_[node][0]

        # add our values to keep track of them
        nn_length.append(len(nn_path))
        dijkstra_length.append(len(dijkstra_path))
        average_discrepancy.append(float(dijkstra_cost - nn_cost))
        dijkstra_average.append(float(dijkstra_cost))
        nn_average.append(float(nn_cost))
        dijkstra_cost_vector.append(dijkstra_cost)
        nn_cost_vector.append(nn_cost)

    return (
        nn_length,
        dijkstra_length,
        nn_average,
        dijkstra_average,
        n_failures,
        n_departures,
        nn_cost_vector,
        dijkstra_cost_vector,
        nn_paths,
    )


def eval_static(
    policy_net,
    device,
    feature_function_,
    neighbors_tensor,
    n_observations,
    max_len=625,
    target=624,
    constant=0,
    heuristic=0,
    heuristic_index=2,
):
    # for Dijkstra's algorithm
    vertices = np.arange(0, max_len, 1)
    dijkstra_info = dijkstra_bwd(
        feature_function=feature_function_,
        vertices=vertices,
        source=int(target),
        neighbors=neighbors_tensor,
        max_len=max_len,
        constant=constant,
        heuristic=heuristic,
        heuristic_index=heuristic_index
    )

    # chose starting coordinates
    # starting_coords = torch.arange(0, max_len - 1, 1).to(device)
    starting_coords = np.arange(0, 624, 1)

    # vectors to hold results
    average_discrepancy = []
    dijkstra_average = []
    nn_average = []

    # hold the results for the threat field from loc 1 to 624
    nn_cost_vector = []
    dijkstra_cost_vector = []

    # path length
    dijkstra_length = []
    nn_length = []

    # number of failures of the neural network
    n_failures = 0
    n_departures = 0

    failed_loc = []

    nn_paths = []

    for coord in starting_coords:
        # call dijsktra's and the nn
        nn_path, info, outside, fail_loc = find_nn_path(
            feature_function=feature_function_.float(),
            starting_coord=int(coord),
            policy_net=policy_net,
            neighbors=neighbors_tensor,
            device=device,
            n_observations=n_observations,
            max_len=max_len,
            target=target,
        )
        nn_paths.append(nn_path)

        # if the network fails, terminate the loop
        if info is False:
            if outside is True:
                n_departures += 1
                dijkstra_cost_vector.append(0)
                nn_cost_vector.append(0)
            else:
                n_failures += 1
                failed_loc.append(fail_loc)
                dijkstra_cost_vector.append(-1)
                nn_cost_vector.append(-1)
            continue

        # recover the Dijkstra algorithm path
        node = coord
        dijkstra_path = [coord]
        while node != target:
            previous_node = dijkstra_info[node].parent
            dijkstra_path.append(previous_node)
            node = previous_node

        # determine the cost of the neural network path and of Dijkstra's algorithm
        dijkstra_cost = 0
        for node in dijkstra_path[1:]:
            dijkstra_cost += (
                constant + feature_function_[node][0] + heuristic * feature_function_[node][heuristic_index]
            )

        nn_cost = 0
        for node in nn_path[1:]:
            nn_cost += (
                constant + feature_function_[node][0] + heuristic * feature_function_[node][heuristic_index]
            )

        # add our values to keep track of them
        nn_length.append(len(nn_path))
        dijkstra_length.append(len(dijkstra_path))
        average_discrepancy.append(float(dijkstra_cost - nn_cost))
        dijkstra_average.append(float(dijkstra_cost))
        nn_average.append(float(nn_cost))
        dijkstra_cost_vector.append(dijkstra_cost)
        nn_cost_vector.append(nn_cost)

    return (
        nn_length,
        dijkstra_length,
        nn_average,
        dijkstra_average,
        n_failures,
        n_departures,
        nn_cost_vector,
        dijkstra_cost_vector,
        nn_paths,
    )
