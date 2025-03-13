"""
Make a standardized heatmap plot
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_dataframe(values, dims):
    # values should be a one-dimensional vector
    # dims is a tuple: number of x points by number of y points

    # reshape the input vector. The input is listed from the lower left corner, left to right, and up
    # reshape the input to the specified dimensions, and then flip the vector
    values = np.flip(np.reshape(values, dims), axis=0)

    # create the names of the columns and the rows. The columns will become the x-axis coordinates and the rows will
    # become the y-axis
    columns = np.round(np.linspace(-1, 1, dims[0]), 2)
    indices = np.round(np.linspace(1, -1, dims[1]), 2)
    ind = []
    for x in indices:
        if np.where(indices == x)[0][0] % 2 == 0:
            ind.append(str(x))
        else:
            ind.append(" ")

    cols = []
    for x in columns:
        if np.where(columns == x)[0][0] % 2 == 0:
            cols.append(str(x))
        else:
            cols.append(" ")
    # indices = [str(x) for x in indices]

    # create a data frame with the x coordinates as the columns, the y coordinates as the indices, and the values as
    # the input vector
    value_map = pd.DataFrame(values, columns=cols, index=ind)

    return value_map, values

def make_heatmap(values, vmin=None, vmax=None, title=None, color=None):
    threat_field, values = create_dataframe(values, (25, 25))

    if vmin is not None:
        sns.heatmap(threat_field, cmap=color, cbar=True, annot=False, fmt='g', vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(threat_field, cmap=color, cbar=True, annot=False, fmt='g')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # plot time evolution of a threat field
    data_1 = pd.read_csv('../01_data/cost_function/001.csv')
    data_2 = pd.read_csv('../01_data/cost_function/100.csv')

    threat_field_1, _ = create_dataframe(data_1['Threat Intensity'].to_numpy(), (25, 25))
    threat_field_2, _ = create_dataframe(data_2['Threat Intensity'].to_numpy(), (25, 25))
    threat_intermediate = 0.5 * data_1['Threat Intensity'].to_numpy() + 0.5 * data_2['Threat Intensity'].to_numpy()
    threat_intermediate, _ = create_dataframe(threat_intermediate, (25, 25))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(threat_field_1, ax=ax1, cmap='RdYlBu_r', cbar=True, fmt='g', vmin=0, vmax=5)
    ax1.set_title('t=0')
    sns.heatmap(threat_intermediate, ax=ax2, cmap='RdYlBu_r', cbar=True, fmt='g', vmin=0, vmax=5)
    ax2.set_title('t=25')
    sns.heatmap(threat_field_2, ax=ax3, cmap='RdYlBu_r', cbar=True, fmt='g', vmin=0, vmax=5)
    ax3.set_title('t=50')
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()

    data_2 = pd.read_csv('../01_data/cost_function/430.csv')
    threat_field, _ = create_dataframe(data_1['Threat Intensity'].to_numpy(), (25, 25))
    make_heatmap(threat_field, color='RdYlBu_r')

    # plot a single field: Dijkstra's, NN, and error
    single = pd.read_pickle('../01_data/paper_data/static_results/1_15_random_169_performance_2_finishes_624.pkl')
    print(single.keys())

    nn_cost = np.array(single['nn_cost'])
    nn_cost = np.append(nn_cost, 0)
    max_cost = max(nn_cost)
    make_heatmap(nn_cost, vmin=0, vmax=max_cost, color='viridis')

    dijsktra_cost = np.array(single['dijkstra_cost'])
    dijkstra_cost = np.append(dijsktra_cost, 0)
    make_heatmap(dijkstra_cost, vmin=0, vmax=max_cost, color='viridis')

    percent_error = np.array(single['percent_error'])
    percent_error = np.append(percent_error, 0)
    make_heatmap(percent_error, color='Greys_r')

    # plot the difference between a threat field used during training and the performance on a new threat field
    multi_data = pd.read_pickle('../01_data/paper_data/eval_on_new_threat/1_22_random_909_performance_2_finishes_624.pkl')
    print(multi_data.keys())

    threat_test = multi_data['threat_field_test_test'][0]
    vmax_threat = max(threat_test)
    make_heatmap(threat_test, vmin=0, vmax=vmax_threat, color='RdYlBu_r')

    error_test = np.array(multi_data['percent_error_test'])
    error_test = np.append(error_test, 0)
    vmax_error = max(error_test)
    make_heatmap(error_test, vmin=0, vmax=vmax_error, color='Greys_r')

    threat_training = multi_data['threat_field'][0]
    make_heatmap(threat_training, vmin=0, vmax=vmax_threat, color='RdYlBu_r')

    error_training = np.array(multi_data['percent_error'])
    error_training = np.append(error_training, 0)
    make_heatmap(error_training, vmin=0, vmax=vmax_error, color='Greys_r')
