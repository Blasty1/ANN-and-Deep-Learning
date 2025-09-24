import matplotlib.pyplot as plt
import numpy as np
import os

def plot_dataset(x_train, y_train, x_test, y_test, fun_name = 'sin(2x)', noise = ""):
    """
    Plots sin(2x) or square(2x) dataset.
    To make noise use noise = 'noisy'; it will be added for file name and label name in legend.

    """
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=10, label=f'Training data {noise}')
    plt.plot(x_test, y_test, color='red', label=f'Real function {fun_name}')
    
    plt.xlabel('X', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    save_path = os.path.join('Lab2', 'plots', f'Dataset_{noise}{fun_name}.png')
    plt.savefig(save_path)
    plt.show()

def plot_dataset_centers(x_train, y_train, x_test, y_test, centers, fun_name = 'sin(2x)', noise = ""):
    """
    Plots sin(2x) or square(2x) dataset with chosen centers as yellow triangles.
    To make noise use noise = 'noisy' it will be added for file name and label name in legend.

    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=10, label=f'Training data {noise}')
    plt.plot(x_test, y_test, color='red', label=f'Real function {fun_name}')
    plt.plot(centers, np.zeros_like(centers), 'y^', markersize=7, label='RBF Centers')

    plt.xlabel('X', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    save_path = os.path.join('Lab2', 'plots', f'Dataset_{noise}{fun_name}_centers.png')
    plt.savefig(save_path)
    plt.show()

def plot_prediction(x_train, y_train, x_test, y_pred, fun_name = 'sin(2x)', add_info = ""):
    """
    Plots sin(2x) or square(2x) prediction.

    """
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=10, label=f'Training data')
    plt.plot(x_test, y_pred, color='red', label=f'Prediction of {fun_name}')
    
    plt.xlabel('X', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    save_path = os.path.join('Lab2', 'plots', f'Prediction_{fun_name}{add_info}.png')
    plt.savefig(save_path)
    plt.show()


def plot_RBF_kernel_comprision(data, name):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for nodes, values in data.items():
        ax.plot(values['sigma'], values['valid_mse'], 'o-', label=f'{nodes} nodes')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Variance σ', fontsize=14)
    ax.set_ylabel('Validation MSE', fontsize=14)
    ax.legend(title='Number of RBF nodes', fontsize=10)
    ax.grid(True)
    plt.savefig(os.path.join('Lab2', 'plots', f'RBF_kernel_comprision{name}.png'))
    plt.show()