import torch
from sklearn.preprocessing import minmax_scale

from nn_architecture import neighbors_of_four


def time_dependent_neighbors(dims, steps, target):
    """
    :param dims: dimensions of the threat field
    :param steps: number of tiemsteps
    :param target: goal location
    :return: neighbors tensor and distance tensor
    """
    # neighbors vector
    length = dims[0] * dims[1]  # length of the threat field...this is max_index + 1 -> corresponds to out of bounds
    neighbors = neighbors_of_four(dims=dims, target=target)     # get regular neighbors
    neighbors_vector = torch.from_numpy(
        neighbors[["left", "right", "up", "down"]].values
    ).repeat(steps, 1)  # repeat neighbors vector by the number of steps
    neighbors_vector[neighbors_vector == length] = -1   # where we have the max_len, mark -1 to show outside bounds
    add_vector = length * torch.repeat_interleave(
        torch.linspace(1, steps, steps).long(), length
    ).view(-1, 1)   # this vector accounts for the time increment per movement, so the neighbor of {1} is the neighbors + 625 to get to the next timestep
    add_vector[-length:] = add_vector[-(length + 1)]    # modify the last timestep so that its neighbors are itself
    # at the final timestep, don't advance in time, just keep moving in that same threat field
    new_neighbors = neighbors_vector + add_vector   # add the vector to the neighbors vector
    new_neighbors[neighbors_vector == -1] = -1      # anywhere that we moved out of bounds, account for that

    neighbors_calc = new_neighbors
    neighbors_calc[neighbors_vector == target] = target     # set any target destination to have the right index (624) to trigger break statements

    # distance vector
    distance = minmax_scale(neighbors[["dist"]].values, feature_range=(0, 10))
    distance = torch.from_numpy(distance).repeat(steps, 1)
    max_value = torch.max(distance).view(-1, 1)
    distance = torch.cat((distance, max_value))
    return neighbors_calc, distance
