import matplotlib.pyplot as plt
import data_generation as data
import part2 as p
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

# Part I

def plot_decision_boundary(network, X_A, X_B, resolution=100):
    """
    Plots the decision boundary of the neural network
    
    Args:
        network: the trained neural network
        X_A, X_B: data from both classes (format: features x samples)
        resolution: grid resolution for the boundary
    """
    
    # Concatenate all data to find plot limits
    all_data = np.hstack([X_A, X_B])
    
    # Find data limits with some margin
    x_min, x_max = all_data[0, :].min() - 0.5, all_data[0, :].max() + 0.5
    y_min, y_max = all_data[1, :].min() - 0.5, all_data[1, :].max() + 0.5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Flatten the grid to pass it to the network
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Shape: (2, resolution²)
    
    # Calculate predictions for all grid points
    cache = network.forward(grid_points)
    L = len(network.layer_sizes) - 1
    predictions = cache[f"A{L}"]
    
    # Reshape predictions to match the grid shape
    Z = predictions.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the decision boundary as contour
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Network Output')
    
    # Add the decision boundary line (output = 0)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    # Plot the original data
    plt.scatter(X_A[0, :], X_A[1, :], c='red', marker='o', s=50, 
                edgecolors='black', label='Class A (target: -1)', alpha=0.8)
    plt.scatter(X_B[0, :], X_B[1, :], c='blue', marker='s', s=50, 
                edgecolors='black', label='Class B (target: +1)', alpha=0.8)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    return plt

# Alternative simpler version that shows only classification regions
def plot_decision_regions(network, X_A, X_B, resolution=200):
    """
    Plots the decision regions (simpler visualization)
    
    Args:
        network: the trained neural network
        X_A, X_B: data from both classes (format: features x samples)
        resolution: grid resolution for the boundary
    """
    
    # Concatenate all data to find plot limits
    all_data = np.hstack([X_A, X_B])
    
    # Find data limits with some margin
    x_min, x_max = all_data[0, :].min() - 0.5, all_data[0, :].max() + 0.5
    y_min, y_max = all_data[1, :].min() - 0.5, all_data[1, :].max() + 0.5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Flatten the grid to pass it to the network
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    
    # Calculate predictions
    cache = network.forward(grid_points)
    L = len(network.layer_sizes) - 1
    predictions = cache[f"A{L}"]
    
    # Classify points (>0 -> class B, <0 -> class A)
    classifications = (predictions > 0).astype(int)
    Z = classifications.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot regions with different colors
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightcoral', 'lightblue'], alpha=0.7)
    
    # Plot the original data
    plt.scatter(X_A[0, :], X_A[1, :], c='red', marker='o', s=60, 
                edgecolors='black', label='Class A (target: -1)', alpha=0.9)
    plt.scatter(X_B[0, :], X_B[1, :], c='blue', marker='s', s=60, 
                edgecolors='black', label='Class B (target: +1)', alpha=0.9)
    
    # Add the decision boundary line
    cache = network.forward(grid_points)
    predictions = cache[f"A{L}"]
    Z_continuous = predictions.reshape(xx.shape)
    plt.contour(xx, yy, Z_continuous, levels=[0], colors='black', linewidths=3)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt


def plot_learning_curve(MSEs,filename=None):
    """
    Ploting error convergence.
    
    Inputs:
    MSEs : array - contains mean squere error from each epoch

    """
    fig = plt.figure()
    ax = plt.axes()
    x_values = range(len(MSEs))
    ax.plot(x_values, MSEs, linestyle='-')
    plt.title('Error convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    if(filename != None):
        os.makedirs('Lab1\\3_1_3_plots\\', exist_ok=True)
        plt.savefig(f'Lab1\\3_1_3_plots\\{filename}.png')
    plt.show()

# Part II
# Plot mackey-glass time series data
def plot_mackey_glass_data():
    X, y = data.generate_time_series_data()
    _, train_y, _, valid_y, _, test_y=data.split_data_for_train_valid_test(X,y) 
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(range(300,1100)), train_y, label='Train Data', color='blue')
    plt.plot(np.array(range(1100,1300)), valid_y, label='Valid Data', color='green')
    plt.plot(np.array(range(1300,1500)), test_y, label='Test Data', color='red')
    plt.legend()
    plt.title('Mackey-Glass Time Series')
    plt.xlabel('Time')
    plt.ylabel('Mackey-Glass Value')
    plt.grid(True)
    plt.show()



# Plot showing training and validation MSE for different alpha values
def plot_alpha_choices():
    plt.figure(figsize=(10, 6))
    train_mse = np.array([p.train_dict[alpha] for alpha in p.alphas])
    valid_mse = np.array([p.valid_dict[alpha] for alpha in p.alphas])
    plt.plot(np.array(p.alphas), train_mse, marker='o', color = 'blue',label=f'Train MSE ')
    plt.plot(np.array(p.alphas), valid_mse, marker='x',color='green', label=f'Validation MSE')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training and Validation MSE for Different Alpha Values')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_mackey_glass_data()
    plot_alpha_choices()





