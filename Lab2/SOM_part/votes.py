import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from Utils.SOM import SOM

# Import data

with open('././data_lab2/votes.dat', "r") as f:
    line = f.read().strip()

data = np.array(line.split(","), dtype=float)
X = data.reshape(349, 31)
# print(X)



# Import the model
epochs = 20
grid_dimension = 2
input_dimension = 31
units_number = 100

model = SOM(grid_dimension, input_dimension, units_number, sigma0=25, epochs=epochs)
# Train the model
bmus = model.train(X.T)[epochs]


