# Przykład użycia
from Utils.Plot_functions import *
from Utils.RBF import RBF_NN as rbf
from Utils.Data_generation import *

# Choose hyperparamiters
n_hidden_neurons = 50 # Max 63
sigma_value = 0.9

#Choose dataset 
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


# sin(2x) data generation
X_train, Y_train, X_validation, Y_validation = create_clustered_data_for_dead_units()

# square(2x) data generation
# X_train, Y_train, X_validation, Y_validation = generate_data_square_2x()

# noisy sin(2x) data generation
# X_train, Y_train, X_validation, Y_validation = noise_generate_data_sin_2x()

# noisy square(2x) data generation
# X_train, Y_train, X_validation, Y_validation = noise_generate_data_square_2x()


print(f"Trening X set shape: {X_train.shape}")
print(f"Trening Y set shape: {Y_train.shape}")
print(f"Testing X set shape: {X_validation.shape}")
print(f"Testing Y set shape: {Y_validation.shape}")

#Inicialise network
rbf_net = rbf(n_hidden=n_hidden_neurons, sigma=sigma_value)

# Find centers
_, dead_units = rbf_net.choose_centers(X_train)
plot_dataset_centers(X_train, Y_train, X_validation, Y_validation, rbf_net.centers, fun_name='sin(2x)')

print(f'Center vestor shape: {rbf_net.centers.shape}, Nr of dead units {len(dead_units)}')

# Choose output layer trening method
# train_mse, validation_mse = rbf_net.fit_least_square(X_train, Y_train, X_validation, Y_validation)
# print(f"Train MSE: {train_mse}")
# print(f"Valid MSE: {validation_mse}")

train_mse, validation_mse = rbf_net.fit_delta_rule(X_train, Y_train, X_validation, Y_validation)
print(f"Train MSE: {train_mse[-1]:.5f}")
print(f"Valid MSE: {validation_mse[-1]:.5f}")


# Plotting prediction 
predict_y = rbf_net.predict(X_validation)

plot_prediction(X_train, Y_train, X_validation, predict_y, 'sin(2x)', '_delta_rule') # '_least_square')
