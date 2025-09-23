import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.SOM import SOM
import numpy as np




# Import data
with open("./data_lab2/animals.dat", "r") as f:
    line = f.read().strip()

vals = np.array(line.split(","), dtype=int)
X = vals.reshape(32, 84)
# Import animals names

with open("./data_lab2/animalnames.txt", "r") as f:
    animals = [line.strip() for line in f]


# Initialize the model
epochs = 20
grid_dimension = 1
input_dimension = 84
units_number = 100

model = SOM(grid_dimension, input_dimension, units_number, sigma0=10)
bmus=model.train(X.T)

sorted_pairs = sorted(zip(bmus[epochs], animals))
bmus_sorted, animals_sorted = zip(*sorted_pairs)
print(animals_sorted)


