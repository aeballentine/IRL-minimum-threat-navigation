import numpy as np
import pandas as pd

from nn_architecture import to_1d

# define the center of the threat:
[i_threat, j_threat] = [12, 12]
# define the gradient in the x and y directions
dx = -0.005
dy = -0.005
dydx = -0.001

# define the maximum intensity of the threat
peak = 5

# solve for all the relevant locations
threat_values = np.zeros((625, ))
grad_x = np.zeros((625, ))
grad_y = np.zeros((625, ))
for i in range(25):
    for j in range(25):
        single_index = to_1d([i, j], (25, 25))
        threat_values[single_index] = peak * (1 - np.sqrt((i - i_threat)**2 + (j - j_threat)**2) / 24.5)
        grad_x[single_index] = peak * (1 - np.sqrt((i + 1 - i_threat)**2 + (j - j_threat)**2) / 24.5) - peak * (1 - np.sqrt((i - 1 - i_threat)**2 + (j - j_threat)**2) / 24.5)
        grad_y[single_index] = peak * (1 - np.sqrt((i - i_threat)**2 + (j + 1 - j_threat)**2) / 24.5) - peak * (1 - np.sqrt((i - i_threat)**2 + (j - 1 - j_threat)**2) / 24.5)

# save the threat field
threat_field = pd.DataFrame({'Threat Intensity': threat_values, 'Threat Gradient x_1': grad_x, 'Threat Gradient x_2': grad_y})
threat_field.to_csv('../01_data/custom_fields/custom_threat.csv')
