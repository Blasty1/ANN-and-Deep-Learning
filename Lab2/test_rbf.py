# Przykład użycia
from Utils.Plot_functions import *
from Utils.RBF import RBF_NN as rbf
from Utils.Data_generation import *

# Choose hyperparamiters
n_hidden_neurons = 20 # Max 63
sigma_value = 0.2

#Choose dataset 

# sin(2x) data generation
X_train, Y_train, X_validation, Y_validation = generate_data_sin_2x()

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
rbf_net.choose_centers(X_train)
plot_dataset_centers(X_train, Y_train, X_validation, Y_validation, rbf_net.centers, fun_name='sin(2x)')

print(f'Center vestor shape: {rbf_net.centers.shape}')

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
