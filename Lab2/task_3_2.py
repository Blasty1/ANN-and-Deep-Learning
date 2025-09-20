from Utils.Plot_functions import *
from Utils.RBF import RBF_NN as rbf
from Utils.Data_generation import *

print("\n==== FIND BEST KERNEL FOR NOISY DATA ====")

# noisy sin(2x) data generation 0.1
X_train_sin, Y_train_sin, X_test_sin, Y_test_sin = noise_generate_data_sin_2x()

# noisy square(2x) data generation variance 0.1
X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr = noise_generate_data_square_2x()

X_train_sin_length = max(X_train_sin) - min(X_train_sin)
X_train_sqr_length = max(X_train_sqr) - min(X_train_sqr)

n_hidden_vec = [5, 10, 15, 20, 50]
sigma_vec = [0.01, 0.1, 0.2, 0.3, 0.5]

print('\n==== Search best hiperparamiters for sin(2x) ====\n')

data_sin_kernels = {n: {} for n in n_hidden_vec}
for n_hidden in n_hidden_vec:
    centers = []
    centers_gaps = (X_train_sin_length/n_hidden)*(1+1/n_hidden)
    for n in range(n_hidden):
        centers.append(min(X_train_sin) + n*centers_gaps)
    # plot_dataset_centers(X_train_sin, Y_train_sin, X_test_sin, Y_test_sin, centers)
    train_mse_vec = []
    valid_mse_vec = []
    data_sin_kernels[n_hidden]['sigma'] = sigma_vec
    for sigma in sigma_vec:
        rbf_net = rbf(n_hidden=n_hidden, sigma=sigma)
        rbf_net.set_centers(centers)
        train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_sin, Y_train_sin, X_test_sin, Y_test_sin)
        train_mse_vec.append(train_mse[-1])
        valid_mse_vec.append(valid_mse[-1])
        print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}")
    data_sin_kernels[n_hidden]['train_mse'] = train_mse_vec
    data_sin_kernels[n_hidden]['valid_mse'] = valid_mse_vec
plot_RBF_kernel_comprision(data_sin_kernels)
#TODO find best combination for sin

print('\n==== Search best hiperparamiters for square(2x) ====\n')

data_sqr_kernels = {n: {} for n in n_hidden_vec}
for n_hidden in n_hidden_vec:
    centers = []
    centers_gaps = (X_train_sqr_length/n_hidden)*(1+1/n_hidden)
    for n in range(n_hidden):
        centers.append(min(X_train_sqr) + n*centers_gaps)
    # plot_dataset_centers(X_train_sin, Y_train_sin, X_test_sin, Y_test_sin, centers)
    train_mse_vec = []
    valid_mse_vec = []
    data_sqr_kernels[n_hidden]['sigma'] = sigma_vec
    for sigma in sigma_vec:
        rbf_net = rbf(n_hidden=n_hidden, sigma=sigma)
        rbf_net.set_centers(centers)
        train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr)
        train_mse_vec.append(train_mse[-1])
        valid_mse_vec.append(valid_mse[-1])
        print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}")
    data_sqr_kernels[n_hidden]['train_mse'] = train_mse_vec
    data_sqr_kernels[n_hidden]['valid_mse'] = valid_mse_vec
plot_RBF_kernel_comprision(data_sqr_kernels)
#TODO find best combination for square