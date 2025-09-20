import numpy as np
import matplotlib.pyplot as plt
from Plot_functions import plot_dataset


def generate_data_sin_2x():
    x_train_sin = np.arange(0, 2 * np.pi, 0.1).reshape(-1, 1)
    y_train_sin = np.sin(2 * x_train_sin)

    x_test_sin = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1, 1)
    y_test_sin = np.sin(2 * x_test_sin)

    print("\n--- Data for sin(2x) ---")
    print(f"Training set dimensions: X={x_train_sin.shape}, Y={y_train_sin.shape}")
    print(f"Testing set dimensions: X={x_test_sin.shape}, Y={y_test_sin.shape}")
    plot_dataset(x_train_sin, y_train_sin, x_test_sin, y_test_sin, fun_name='sin(2x)')
    return x_train_sin, y_train_sin, x_test_sin, y_test_sin

def generate_data_square_2x():
    x_train_square = np.arange(0, 2 * np.pi, 0.1).reshape(-1, 1)
    y_train_square = np.where(np.sin(2 * x_train_square) >= 0, 1, -1)
    x_test_square = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1, 1)
    y_test_square = np.where(np.sin(2 * x_test_square) >= 0, 1, -1)

    print("\n--- Data for square(2x) ---")
    print(f"Training set dimensions: X={x_train_square.shape}, Y={y_train_square.shape}")
    print(f"Testing set dimensions: X={x_test_square.shape}, Y={y_test_square.shape}")
    plot_dataset(x_train_square, y_train_square, x_test_square, y_test_square, fun_name='square(2x)')
    return x_train_square, y_train_square, x_test_square, y_test_square

def noise_generate_data_sin_2x():
    x_train_sin = np.arange(0, 2 * np.pi, 0.1).reshape(-1, 1)
    y_train_sin = np.sin(2 * x_train_sin)
    
    noise_train = np.random.normal(0, np.sqrt(0.1), x_train_sin.shape)
    y_train_sin_noisy = y_train_sin + noise_train
    
    x_test_sin = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1, 1)
    y_test_sin = np.sin(2 * x_test_sin)

    noise_test = np.random.normal(0, np.sqrt(0.1), x_test_sin.shape)
    y_test_sin_noisy = y_test_sin + noise_test

    print("\n--- Noisy data for sin(2x) ---")
    print(f"Training set dimensions: X={x_train_sin.shape}, Y={y_train_sin_noisy.shape}")
    print(f"Testing set dimensions: X={x_test_sin.shape}, Y={y_test_sin_noisy.shape}")
    plot_dataset(x_train_sin, y_train_sin_noisy, x_test_sin, y_test_sin_noisy, fun_name='sin(2x)', noise='noisy')
    return x_train_sin, y_train_sin_noisy, x_test_sin, y_test_sin_noisy

def noise_generate_data_square_2x():
    x_train_square = np.arange(0, 2 * np.pi, 0.05).reshape(-1, 1)
    y_train_square = np.where(np.sin(2 * x_train_square) >= 0, 1, -1)

    noise_train = np.random.normal(0, np.sqrt(0.1), x_train_square.shape)
    y_train_square_noisy = y_train_square + noise_train

    x_test_square = np.arange(0.05, 2 * np.pi, 0.01).reshape(-1, 1)
    y_test_square = np.where(np.sin(2 * x_test_square) >= 0, 1, -1)

    noise_test = np.random.normal(0, np.sqrt(0.1), x_test_square.shape)
    y_test_square_noisy = y_test_square + noise_test

    print("\n--- Noisy data for square(2x) ---")
    print(f"Training set dimensions: X={x_train_square.shape}, Y={y_train_square_noisy.shape}")
    print(f"Testing set dimensions: X={x_test_square.shape}, Y={y_test_square_noisy.shape}")
    plot_dataset(x_train_square, y_train_square_noisy, x_test_square, y_test_square_noisy, fun_name='square(2x)', noise="noisy")
    return x_train_square, y_train_square_noisy, x_test_square, y_test_square_noisy

# Example usage for testing
#generate_data_sin_2x()
# generate_data_square_2x()
# noise_generate_data_sin_2x()
# noise_generate_data_square_2x()