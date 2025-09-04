import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def plot_convergence(error):
    """
    Ploting error convergence.
    
    Inputs:
    error : array - contains mean squere error from each epoch
    
    """

    fig = plt.figure()
    ax = plt.axes()
    x_values = range(len(error))
    ax.plot(x_values, error, linestyle='-')
    plt.title('Error convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def plot_accuracy(accuracy):
    """
    Ploting error convergence.
    
    Inputs:
    accuaracy : array - contains accuracy value form each epoch
    
    """
    fig = plt.figure()
    ax = plt.axes()
    x = range(len(accuracy))
    ax.plot(x, accuracy, linestyle='-')
    plt.title('Accuracy for epoch')
    plt.grid(True)
    plt.show()

def animate(i, X, all_weights, targets, line):
    """
    Ploting decision boundary line for single step of animation.
    
    Inputs:
    i : int - number of animation step
    X : array(n, 2) - datapoints
    weights : array(3) - weights [x1, x2, bias]
    targets : array(n) - targets for X inputs, values 1 or -1
    line : plot function format to decision boundary
    
    Output
    line : plot function to decision boundary
    """

    weights = all_weights[i]
    
    # Przeliczanie pozycji granicy decyzyjnej
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 100)
    
    if weights[2] != 0:
        y_values = -(weights[1] / weights[2]) * x_values - (weights[0] / weights[2])
    elif weights[1] != 0:
        y_values = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
        x_values = np.full_like(y_values, -weights[0] / weights[1])
    else:
        # Obsługa przypadku, gdy wagi są zerowe
        y_values = []
        x_values = []

    # Aktualizacja danych linii
    line.set_data(x_values, y_values)
    return line,

def create_animation(X, all_weights, targets, filename, title):
    """
    Create animation of decision boundary during learning
    
    Inputs:
    X : array(2, n) -- array of ale datapoints
    all_weights : array[number_of_epoch, 3] - vector containing weights vestor [w0, w1, w2], w0 is bias
    targets: array(n) -- array of lebales of X
    filename: string -- name of file to save (without .gif)
    title: string -- title for plot

    """

    fig, ax = plt.subplots(figsize=(8,8))
    
    # Rysowanie punktów
    colors = np.where(targets == 1, 'blue', 'red')
    print(colors)
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=50)

    # Inicjalizacja linii granicy decyzyjnej
    line, = ax.plot([], [], color='b', linestyle='-')
    
    # Ustawianie granic wykresu
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.title('Decision boundary for batch delta rule')
    plt.grid(True)
    anim = FuncAnimation(fig, animate, frames=len(all_weights), fargs=(X, all_weights, targets, line), interval=50, blit=True, repeat=False)
    os.makedirs("Lab1\\animations\\", exist_ok=True)
    anim.save(f'Lab1\\animations\\{filename}.gif', writer='pillow')
    plt.show()