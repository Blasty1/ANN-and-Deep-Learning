import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
sys.path.append(os.path.abspath("Lab1b/"))

from BackpropCode import NeuralNetwork

# --- Data Preparation ---
def generate_data():
    # Define input vectors
    x = np.arange(-5, 5.5, 0.5).reshape(-1, 1)  # (21, 1)
    y = np.arange(-5, 5.5, 0.5).reshape(-1, 1)  # (21, 1)

    # Compute target surface z
    z = np.exp(-0.1 * x**2) @ np.exp(-0.1 * y**2).T - 0.5  # (21, 21)

    # Create meshgrid for plotting
    xx, yy = np.meshgrid(np.arange(-5, 5.5, 0.5), np.arange(-5, 5.5, 0.5))

    # Prepare patterns (inputs) and targets (outputs) for the network
    patterns = np.vstack([xx.reshape(1, -1), yy.reshape(1, -1)])  # (2, 441)
    targets = z.reshape(1, -1)                                    # (1, 441)
    return patterns, targets, z, xx, yy

def split_data(patterns, targets, fraction):
    np.random.seed(42)
    nsamples = int(patterns.shape[1] * fraction)
    indices = np.random.permutation(patterns.shape[1])[:nsamples]
    train_X, train_y = patterns[:, indices], targets[:, indices]
    return train_X, train_y

def plot_function_estimation(z, xx, yy, z_prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, z, color='blue', label='Target')
    ax.plot_wireframe(xx, yy, z_prediction, color='red', label='Prediction')
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
    ax.legend(custom_lines, ['Target', 'Prediction'])
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from BackpropCode import NeuralNetwork

# --- Data Preparation ---
def generate_data():
    # Define input vectors
    x = np.arange(-5, 5.5, 0.5).reshape(-1, 1)  # (21, 1)
    y = np.arange(-5, 5.5, 0.5).reshape(-1, 1)  # (21, 1)

    # Compute target surface z
    z = np.exp(-0.1 * x**2) @ np.exp(-0.1 * y**2).T - 0.5  # (21, 21)

    # Create meshgrid for plotting
    xx, yy = np.meshgrid(np.arange(-5, 5.5, 0.5), np.arange(-5, 5.5, 0.5))

    # Prepare patterns (inputs) and targets (outputs) for the network
    patterns = np.vstack([xx.reshape(1, -1), yy.reshape(1, -1)])  # (2, 441)
    targets = z.reshape(1, -1)                                    # (1, 441)
    return patterns, targets, z, xx, yy

def split_data(patterns, targets, fraction):
    np.random.seed(42)
    nsamples = int(patterns.shape[1] * fraction)
    indices = np.random.permutation(patterns.shape[1])[:nsamples]
    train_X, train_y = patterns[:, indices], targets[:, indices]
    return train_X, train_y

def plot_function_estimation(z, xx, yy, z_prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, z, color='blue', label='Target')
    ax.plot_wireframe(xx, yy, z_prediction, color='red', label='Prediction')
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
    ax.legend(custom_lines, ['Target', 'Prediction'])
    plt.show()

def plot_mse_over_epochs(train_MSEs, valid_MSEs,epochs, nnodes):
    plt.figure()
    nepochs = np.array(range(1, epochs + 1))
    plt.plot(nepochs, train_MSEs, marker='+', label='Train MSE', color='blue')
    plt.plot(nepochs, valid_MSEs, marker='o', label='Valid MSE', color='red')
    plt.title(f"MSE over Epochs - Hidden Nodes: {nnodes}")
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.legend()
    plt.show()


def train_and_evaluate(layer_sizes, train_X, train_y, patterns, targets, epochs):
    train_MSEs, valid_MSEs = [], []
    model = NeuralNetwork(layer_sizes=layer_sizes)
    train_MSEs, predictions = model.train_evaluate(train_X, train_y, patterns, epochs=epochs)
    valid_MSEs = [np.mean((prediction - targets) ** 2) for prediction in predictions]
    return train_MSEs, valid_MSEs, predictions


def nodes_mse(train_X, train_y, patterns, targets,epochs):
    v_mses, t_mses = [], []
    for i in range(1, 26):
        layer_sizes = [2, i, 1]  # [input, hidden, output]
        model = NeuralNetwork(layer_sizes=layer_sizes)
        train_MSEs, predictions = model.train_evaluate(train_X, train_y, patterns, epochs)
        valid_MSEs = np.mean((np.array(predictions[-1]) - targets) ** 2) 
        v_mses.append(valid_MSEs)
        t_mses.append(train_MSEs[-1])
    return t_mses, v_mses

def plot_nodes_vs_mse(t_mses, v_mses):
    plt.figure()
    plt.plot(range(1, 26), v_mses, marker='o', label='Validation MSE', color='red')
    plt.plot(range(1, 26), t_mses, marker='+', label='Training MSE', color='blue')
    plt.title("Number of Hidden Nodes vs. Final Validation MSE")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Final Validation MSE")
    plt.grid(True)
    plt.legend()
    plt.show()




# --- Model Setup and Training ---
nnodes = 25 # Number of nodes in the hidden layer
epochs = 100
fraction = 0.5

layer_sizes = [2, nnodes, 1]  # [input, hidden, output]
patterns, targets, z, xx, yy = generate_data()
train_X, train_y = split_data(patterns, targets, fraction)


train_MSEs, valid_MSEs, predictions = train_and_evaluate(layer_sizes, train_X, train_y, patterns, targets, epochs)
# plot_mse_over_epochs(train_MSEs, valid_MSEs, epochs, nnodes)
z_predictions = [prediction.reshape(z.shape[0], z.shape[0]) for prediction in predictions]

t_mses, v_mses = nodes_mse(train_X, train_y, patterns, targets, epochs)
print(f"Train MSE: {t_mses[-1]}, Valid MSE: {v_mses[-1]}, generalization gap: {v_mses[-1]-t_mses[-1]}")
# plot_nodes_vs_mse(t_mses,v_mses)
