import numpy as np
import matplotlib.pyplot as plt
from Utils.Plot_functions import plot_dataset


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

def noise_generate_data_sin_2x(variance = 0.1):
    x_train_sin = np.arange(0, 2 * np.pi, 0.1).reshape(-1, 1)
    y_train_sin = np.sin(2 * x_train_sin)
    np.random.seed(42)

    noise_train = np.random.normal(0, np.sqrt(variance), x_train_sin.shape)
    y_train_sin_noisy = y_train_sin + noise_train
    
    x_test_sin = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1, 1)
    y_test_sin = np.sin(2 * x_test_sin)

    noise_test = np.random.normal(0, np.sqrt(variance), x_test_sin.shape)
    y_test_sin_noisy = y_test_sin + noise_test

    print("\n--- Noisy data for sin(2x) ---")
    print(f"Training set dimensions: X={x_train_sin.shape}, Y={y_train_sin_noisy.shape}")
    print(f"Testing set dimensions: X={x_test_sin.shape}, Y={y_test_sin_noisy.shape}")
    plot_dataset(x_train_sin, y_train_sin_noisy, x_test_sin, y_test_sin_noisy, fun_name='sin(2x)', noise='noisy')
    return x_train_sin, y_train_sin_noisy, x_test_sin, y_test_sin_noisy

def noise_generate_data_square_2x(variance = 0.1):
    x_train_square = np.arange(0, 2 * np.pi, 0.1).reshape(-1, 1)
    y_train_square = np.where(np.sin(2 * x_train_square) >= 0, 1, -1)
    np.random.seed(42)

    noise_train = np.random.normal(0, np.sqrt(variance), x_train_square.shape)
    y_train_square_noisy = y_train_square + noise_train

    x_test_square = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1, 1)
    y_test_square = np.where(np.sin(2 * x_test_square) >= 0, 1, -1)

    noise_test = np.random.normal(0, np.sqrt(variance), x_test_square.shape)
    y_test_square_noisy = y_test_square + noise_test

    print("\n--- Noisy data for square(2x) ---")
    print(f"Training set dimensions: X={x_train_square.shape}, Y={y_train_square_noisy.shape}")
    print(f"Testing set dimensions: X={x_test_square.shape}, Y={y_test_square_noisy.shape}")
    plot_dataset(x_train_square, y_train_square_noisy, x_test_square, y_test_square_noisy, fun_name='square(2x)', noise="noisy")
    return x_train_square, y_train_square_noisy, x_test_square, y_test_square_noisy

def create_data_with_dead_units(n_train_samples=100, n_sparse_samples=5, variance=0.1):
    """
    Generates data designed to create "dead units" in competitive learning.
    It creates a main cluster of data and a small, isolated cluster.
    
    Returns: x_train, y_train, x_test, y_test
    """
    
    # 1. Main, dense data cluster (sinusoidal function)
    x_train_main = np.arange(0, 2 * np.pi, 2 * np.pi / (n_train_samples - n_sparse_samples)).reshape(-1, 1)
    y_train_main = np.sin(2 * x_train_main)
    noise_main = np.random.normal(0, np.sqrt(variance), x_train_main.shape)
    y_train_main_noisy = y_train_main + noise_main
    
    # 2. Small, isolated data cluster (a few samples far from the main cluster)
    x_train_sparse = np.random.uniform(low=2.5 * np.pi, high=3 * np.pi, size=(n_sparse_samples, 1))
    y_train_sparse = np.sin(2 * x_train_sparse)
    noise_sparse = np.random.normal(0, np.sqrt(variance), x_train_sparse.shape)
    y_train_sparse_noisy = y_train_sparse + noise_sparse
    
    # Concatenate the two clusters
    x_train = np.vstack((x_train_main, x_train_sparse))
    y_train = np.vstack((y_train_main_noisy, y_train_sparse_noisy))
    
    # Test data (can be clean for evaluation)
    x_test = np.arange(0, 3 * np.pi, 0.1).reshape(-1, 1)
    y_test = np.sin(2 * x_test)
    
    print("\n--- Data with isolated cluster ---")
    print(f"Training set dimensions: X={x_train.shape}, Y={y_train.shape}")
    print(f"Testing set dimensions: X={x_test.shape}, Y={y_test.shape}")
    plot_dataset(x_train, y_train, x_test, y_test, fun_name='sin(2x)', noise='bad')

    return x_train, y_train, x_test, y_test

def create_clustered_data_for_dead_units(n_clusters=3, samples_per_cluster=20, variance=0.1):
    """
    Generates data with distinct, separated clusters to demonstrate "dead units".
    
    Args:
        n_clusters (int): Number of separate data clusters to create.
        samples_per_cluster (int): Number of data points in each cluster.
        variance (float): The variance of the added Gaussian noise.
        
    Returns:
        tuple: (x_train, y_train_noisy, x_test, y_test_clean)
    """
    
    x_train_list = []
    y_train_list = []
    
    # Generate data for each cluster
    for i in range(n_clusters):
        # Create x values in distinct, separated ranges
        x_cluster = np.linspace(i * 3, i * 3 + 1, samples_per_cluster).reshape(-1, 1)
        y_cluster = np.sin(2 * x_cluster)
        
        # Add noise to each cluster
        noise_cluster = np.random.normal(0, np.sqrt(variance), x_cluster.shape)
        y_cluster_noisy = y_cluster + noise_cluster
        
        x_train_list.append(x_cluster)
        y_train_list.append(y_cluster_noisy)
    
    # Combine all clusters into a single training set
    x_train = np.vstack(x_train_list)
    y_train_noisy = np.vstack(y_train_list)
    
    # Generate a smooth test set for plotting
    x_test_clean = np.arange(0, n_clusters * 3 + 1, 0.1).reshape(-1, 1)
    y_test_clean = np.sin(2 * x_test_clean)
    
    print("\n--- Clustered data to create dead units ---")
    print(f"Number of clusters: {n_clusters}")
    print(f"Training set dimensions: X={x_train.shape}, Y={y_train_noisy.shape}")
    
    # A simple plot function to visualize the data distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train_noisy, label='Dane treningowe (klastry)', alpha=0.7)
    plt.plot(x_test_clean, y_test_clean, 'r-', label='Prawdziwa funkcja')
    plt.title('Dane treningowe w odseparowanych klastrach')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    return x_train, y_train_noisy, x_test_clean, y_test_clean


# Example usage for testing
#generate_data_sin_2x()
# generate_data_square_2x()
# noise_generate_data_sin_2x()
# noise_generate_data_square_2x()