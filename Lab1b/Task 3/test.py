import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("Lab1b/"))

# Custom activation (same as your code: scaled tanh)
class CustomTanh(nn.Module):
    def forward(self, x):
        return 2 / (1 + torch.exp(-x)) - 1

# Derivative is handled automatically by autograd, so no need to implement af_derivative

class NeuralNetTorch(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetTorch, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=True))
            # Apply activation to all but the last layer
            if i < len(layer_sizes) - 2:
                layers.append(CustomTanh())
            else:
                layers.append(CustomTanh())  # you also use tanh at output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Setup
# ---------------------------

def train_torch(X, targets, layer_sizes, epochs=100, lr=0.01, momentum=0.9):
    # Convert to torch tensors
    X_tensor = torch.tensor(X.T, dtype=torch.float32)     # shape (N, features)
    y_tensor = torch.tensor(targets.T, dtype=torch.float32)  # shape (N, 1)

    # Model
    model = NeuralNetTorch(layer_sizes)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    MSEs = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        MSEs.append(loss.item())
    
    return model, MSEs

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Same data as your main
    import data_generation

    X_A = data_generation.generate_splited_data(50,0,0)[0].T # class A -> -1
    X_B = data_generation.generate_splited_data(50,0,0)[2].T # class B -> 1
    X = np.hstack([X_A, X_B])
    targets = np.zeros((1, X.shape[1]))
    targets[0, :50] = -1
    targets[0, 50:] = 1

    layer_sizes = [X.shape[0], 20, 20, 20, 20, 1]

    model, MSEs = train_torch(X, targets, layer_sizes, epochs=100, lr=0.01, momentum=0.9)

    # Plot learning curve
    import matplotlib.pyplot as plt
    plt.plot(MSEs)
    plt.title("Torch Training Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
