import matplotlib.pyplot as plt
import data_generation as data
import part2 as p
import numpy as np


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

plot_mackey_glass_data(    )

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

plot_alpha_choices()





