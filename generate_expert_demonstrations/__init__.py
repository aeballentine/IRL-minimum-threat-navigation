"""
Generate the required expert demonstrations. This set of functions uses Dijkstra's algorithm to find the least-cost
path through static and dynamic threat fields, and returns the feature function for each threat field as well as metrics
regards the expert paths
"""

import torch
import numpy as np
import pandas as pd
import random
import generate_expert_demonstrations.time_varying
import dijkstras_algorithm


def make_time_dependent(field_a, field_b, steps):
    """
    :param field_a: one threat field (representative of the starting time)
    :param field_b: another threat field (representative of the final time
    :param steps: number of timesteps for the interpolation
    :return: a one-dimensional vector, size: (steps x threat_field_len + 1)
    """
    # vector dimensions: steps by field dimensions: steps by 625 -> will need to map to the feature function as well
    # inputs should be torch tensors
    # note that we have steps available. if we need a further timestep, we need to round down
    # DO NOT input fields with "outside points" already appended
    field_a = field_a.repeat(steps, 1)
    field_b = field_b.repeat(steps, 1)
    multiplier_a = torch.linspace(0, 1, steps).view(-1, 1)
    multiplier_b = torch.linspace(1, 0, steps).view(-1, 1)
    evolving_threat = multiplier_a * field_a + multiplier_b * field_b

    # formatting
    max_value = torch.max(evolving_threat).view(1, 1)
    evolving_threat = evolving_threat.view(-1, 1)
    evolving_threat = torch.cat((evolving_threat, max_value))

    return evolving_threat.view(-1)


def create_feature_map(my_field, my_neighbors, relevant_features, device):
    """
    :param my_field: threat field of interest
    :param my_neighbors: neighbors tensor
    :param relevant_features: other features to include, namely distance or {distance_x, distance_y}
    :return: feature function for the threat field
    """
    for index in range(5):
        if index == 0:
            # find the values at the original locations
            feature_func = my_field[:-1].view(-1, 1)
            outside_values = torch.max(my_field).view(
                -1,
            )
            for vals in relevant_features:
                feature_func = torch.concat(
                    (feature_func, vals[:-1].view(-1, 1)), dim=1
                )
                outside_values = torch.concat(
                    (
                        outside_values,
                        torch.max(vals).view(
                            -1,
                        ),
                    ),
                    dim=0,
                )
        else:
            # find values at neighboring cells...indexing through the neighbors vector in order
            feature_func = torch.concat(
                (feature_func, my_field[my_neighbors[:, index - 1]].view(-1, 1)), dim=1
            )
            outside_values = torch.concat(
                (
                    outside_values,
                    torch.max(my_field).view(
                        -1,
                    ),
                ),
                dim=0,
            )
            for vals in relevant_features:
                feature_func = torch.concat(
                    (feature_func, vals[my_neighbors[:, index - 1]].view(-1, 1)), dim=1
                )
                outside_values = torch.concat(
                    (
                        outside_values,
                        torch.max(vals).view(
                            -1,
                        ),
                    ),
                    dim=0,
                )
    feature_func = torch.concat((feature_func, outside_values.view(1, -1)), dim=0)

    return feature_func.float().to(device)


def find_feature_expectation(coords, feature_function):
    """
    :param coords: coordinates along the path
    :param feature_function: feature function associated with the threat field
    :return: the list of feature vectors at each point along the path
    """
    relevant_features = feature_function[coords]

    return relevant_features


def get_expert_demos(
    num_paths,
    file_list,
    device,
    end_index,
    neighbors_tensor,
    distance_x=None,
    distance_y=None,
    distance=None,
    dynamic=False,
    transition_steps=0,
    heuristic=0,
    heuristic_index=2,
    constant=0,
):
    """
    :param num_paths: number of paths to calculate
    :param file_list: files from which to extract threat fields
    :param device: device (cpu, mps, or cuda)
    :param end_index: the goal location
    :param neighbors_tensor: tensor with the relevant neighbors
    :param distance_x: x distance to the finish (if desired)
    :param distance_y: y distance to the finish (if desired)
    :param distance: euclidean distance to the finish (if desired)
    :param dynamic: if a dynamic threat field
    :param transition_steps: number of timesteps for a dynamic field
    :param heuristic: heuristic for dijkstra's algorithm to weight one entry of the feature function
    :param heuristic_index: index of the entry in the feature function to weight
    :param constant: a constant to penalize path length (for dijkstra's algorithm)
    :return: a dictionary with the expert features, the feature map, the original threat field, and the starting coords
    """

    relevant_features = []
    if distance_x is not None:
        relevant_features.append(distance_x)
    if distance_y is not None:
        relevant_features.append(distance_y)
    if distance is not None:
        relevant_features.append(distance)
    if (distance_x is None) and (distance_y is None) and (distance is None):
        raise Exception("Need to specify at least one distance metric")

    if dynamic is False:
        return expert_static(
            num_paths=num_paths,
            file_list=file_list,
            device=device,
            end_index=end_index,
            neighbors_tensor=neighbors_tensor,
            relevant_features=relevant_features,
            heuristic=heuristic,
            heuristic_index=heuristic_index,
            constant=constant,
        )

    else:
        return expert_dynamic(
            num_paths=num_paths,
            file_list=file_list,
            device=device,
            end_index=end_index,
            neighbors_tensor=neighbors_tensor,
            relevant_features=relevant_features,
            transition_steps=transition_steps,
        )


def expert_static(
    num_paths,
    file_list,
    device,
    end_index,
    neighbors_tensor,
    relevant_features,
    heuristic=0,
    heuristic_index=2,
    constant=0,
):

    starting_coords = random.sample(range(0, end_index - 1), num_paths)     # starting coordinates for the expert paths

    # place to store the relevant characteristics
    features = []
    feature_map = []
    threat_map = []

    vertices = np.arange(0, 625, 1)     # vertices for Dijkstra's algorithm
    for file in file_list:
        threat = pd.read_csv(file)  # read the file
        max_threat = max(threat["Threat Intensity"].to_numpy())     # find the place of maximum threat
        threat_field = np.append(threat["Threat Intensity"].to_numpy(), 10 * max_threat)    # formatting for the pathfinder
        threat_field = torch.from_numpy(threat_field).float().to(device)    # move the field to torch
        threat_map.append(threat_field[:-1])    # save the threat field

        # create the feature map for this threat field
        my_feature_map = create_feature_map(
            my_field=threat_field,
            my_neighbors=neighbors_tensor,
            relevant_features=relevant_features,
            device=device
        )
        feature_map.append(my_feature_map)
        my_features = torch.tensor([]).to(device)

        dijkstra_info = dijkstras_algorithm.dijkstra_bwd(
            feature_function=my_feature_map,
            vertices=vertices,
            source=624,
            neighbors=neighbors_tensor,
            constant=constant,
            heuristic=heuristic,
            heuristic_index=heuristic_index,
        )  # solve the threat field using Dijkstra's algorithm

        for loc in starting_coords:
            node = loc      # extract the path
            dijkstra_path = [node]
            while node != end_index:
                previous_node = dijkstra_info[node].parent
                dijkstra_path.append(previous_node)
                node = previous_node

            # find the feature expectation of the path
            my_features = torch.concat(
                (
                    my_features,
                    find_feature_expectation(
                        coords=dijkstra_path,
                        feature_function=my_feature_map,
                    ),
                )
            )

        features.append(my_features)

    return {
        "expert_feat": features,
        "feature_map": feature_map,
        "threat_field": threat_map,
        "test_points": [starting_coords],
    }


def expert_dynamic(
    num_paths,
    file_list,
    device,
    end_index,
    neighbors_tensor,
    relevant_features,
    transition_steps=0,
):
    starting_coords = random.sample(range(0, end_index - 1), num_paths)     # starting coords for the expert

    # store relevant characteristics
    feature_map = []
    threat_map = []
    features = []

    # read the csv files
    field_a = pd.read_csv(file_list[0])
    field_b = pd.read_csv(file_list[1])
    # extract and format the data for the threat fields
    threat_a = torch.from_numpy(field_a["Threat Intensity"].to_numpy())
    threat_b = torch.from_numpy(field_b["Threat Intensity"].to_numpy())

    evolving_threat = make_time_dependent(
        field_a=threat_a, field_b=threat_b, steps=transition_steps
    )
    threat_map.append(evolving_threat)

    # create the feature map for this threat field
    my_feature_map = create_feature_map(
        my_field=evolving_threat,
        my_neighbors=neighbors_tensor,
        relevant_features=relevant_features,
        device=device
    )
    feature_map.append(my_feature_map)
    my_features = torch.tensor([]).to(device)

    # set up to run Dijkstra's algorithm
    vertices = np.arange(0, len(evolving_threat), 1)

    for loc in starting_coords:
        dijkstra_info = dijkstras_algorithm.dijkstra_fwd(
            feature_function=my_feature_map,
            vertices=vertices,
            source=int(loc),
            node_f=end_index,
            neighbors=neighbors_tensor,
            max_len=-1,
        )  # find the optimal path using the threat field and value function (single path only)
        # note: max_len = -1 for Dijkstra's algorithm because that is the "out of bounds" coordinate

        node = end_index
        dijkstra_path = [node]
        while node != loc:
            previous_node = dijkstra_info[node].parent
            dijkstra_path.append(previous_node)
            node = previous_node
        path = dijkstra_path[::-1]

        # find the feature expectation of the path
        path_data = find_feature_expectation(
            coords=path, feature_function=my_feature_map
        )
        my_features = torch.cat((my_features, path_data))

        features.append(my_features)

        return {
            "expert_feat": features,
            "feature_map": feature_map,
            "threat_field": threat_map,
            "test_points": [starting_coords],
        }
