import pandas as pd
import glob
import torch

from heatmap_plotting_standard import make_heatmap


# load the data
data_path = "../01_data/paper_data/time_varying/"
file_list = glob.glob(data_path + "*")
error = torch.tensor([])
maxi = 0
for file in file_list:
    print(file)
    obj = pd.read_pickle(file)
    err = torch.tensor(obj['percent_error'])
    error = torch.concat((error, err))
    maximum = torch.max(error)
    if maximum > maxi:
        maxi = maximum
        print(maxi)

error = error.view(-1, 624)

# find the statistics
mean = torch.mean(error, dim=0)
std = torch.std(error, dim=0)
mean = torch.concat((mean, torch.tensor([0])))
# mean = mean.view(25, 25)

std = torch.concat((std, torch.tensor([0])))
# std = std.view(25, 25)

make_heatmap(mean.numpy(), color='Greys_r', vmin=0, vmax=30)
make_heatmap(std.numpy(), color='bone', vmin=0, vmax=65)
