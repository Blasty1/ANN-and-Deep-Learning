# File: run_square_2x.py

from Utils.Plot_functions import *
from Utils.RBF import RBF_NN as rbf
from Utils.Data_generation import *
import numpy as np
import time
from Utils.BackpropCode import NeuralNetwork as mlp


print("\n==== FIND BEST KERNEL FOR NOISY SQUARE(2x) DATA ====")

# noisy square(2x) data generation with variance 0.1
X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr = noise_generate_data_square_2x()

X_train_sqr_length = max(X_train_sqr) - min(X_train_sqr)

n_hidden_vec = [5, 15, 20, 30, 50]
sigma_vec = [0.01, 0.1, 0.2, 0.3, 0.5]

print('\n==== Search best hiperparamiters for square(2x) ====\n')

data_sqr_kernels = {n: {} for n in n_hidden_vec}
for n_hidden in n_hidden_vec:
    centers = []
    centers_gaps = (X_train_sqr_length/n_hidden)*(1+1/n_hidden)
    for n in range(n_hidden):
        centers.append(min(X_train_sqr) + n*centers_gaps)
    train_mse_vec = []
    valid_mse_vec = []
    gen_gap_vec = []
    model_vec = []
    time_vec = []
    data_sqr_kernels[n_hidden]['sigma'] = sigma_vec
    for sigma in sigma_vec:
        rbf_net = rbf(n_hidden=n_hidden, sigma=sigma)
        rbf_net.set_centers(centers)
        start_time = time.time()
        train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr, epochs = 300)
        elapsed_time = time.time() - start_time         
        train_mse_vec.append(train_mse[-1])
        valid_mse_vec.append(valid_mse[-1])
        model_vec.append(rbf_net)
        time_vec.append(elapsed_time)
        gen_gap_vec.append((train_mse[-1] - valid_mse[-1]))
        print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}")
    data_sqr_kernels[n_hidden]['train_mse'] = train_mse_vec
    data_sqr_kernels[n_hidden]['valid_mse'] = valid_mse_vec
    data_sqr_kernels[n_hidden]['genralisation_gap'] = gen_gap_vec
    data_sqr_kernels[n_hidden]['model'] = model_vec
    data_sqr_kernels[n_hidden]['time'] = time_vec

plot_RBF_kernel_comprision(data_sqr_kernels)
# TODO find best combination for square

best_combo = None
best_score = float('inf')

all_results = []

for n_nodes, n_data in data_sqr_kernels.items():
    for i in range(len(n_data['sigma'])):
        sigma = n_data['sigma'][i]
        model = n_data['model'][i]
        train_mse = n_data['train_mse'][i]
        valid_mse = n_data['valid_mse'][i]
        generalization_gap = n_data['genralisation_gap'][i]
        time_s = n_data['time'][i]
        score = valid_mse + abs(generalization_gap)*2 
        
        all_results.append({
            'nodes': n_nodes,
            'sigma': sigma,
            'model': model,
            'train_mse': train_mse,
            'valid_mse': valid_mse,
            'time': time_s,
            'generalization_gap': generalization_gap,
            'score': score
        })

        if score < best_score:
            best_score = score
            best_combo = {
                'nodes': n_nodes,
                'sigma': sigma,
                'model': model,
                'train_mse': train_mse,
                'valid_mse': valid_mse,
                'time': time_s,
                'generalization_gap': generalization_gap
            }

# Sortowanie wyników
sorted_results = sorted(all_results, key=lambda x: x['score'])

# Wyświetlenie 5 najlepszych konfiguracji
print("Five best combination based on valid MSE and generalisation gap:")
for result in sorted_results[:5]:
    # y_pred = result['model'].predict(X_test_sqr)
    print(f"Nodes: {result['nodes']}, Sigma: {result['sigma']:.2f}, Valid MSE: {result['valid_mse']:.4f}, Gap: {result['generalization_gap']:.4f}, Score: {result['score']:.4f}")
    # plot_prediction(X_train_sqr, Y_train_sqr, X_test_sqr, y_pred )

# Wyświetlenie najlepszej kombinacji
print("\nBest combination:")
print(f"Nodes: {best_combo['nodes']}, Sigma: {best_combo['sigma']:.2f}, Valid MSE: {best_combo['valid_mse']:.4f}, Gap: {best_combo['generalization_gap']:.4f}")

y_best_prediction = best_combo['model'].predict(X_test_sqr)


plot_prediction(X_train_sqr, Y_train_sqr, X_test_sqr, y_best_prediction )


#! NEWSECTION ======================================================================


print("\n" + "="*50 + "\n")

print('\n==== 1. Compare least square batch least square and online delta rule ====\n')

# TODO Train best for square(2x) for least square and delta rule
centers = []
centers_gaps = (X_train_sqr_length/best_combo['nodes'])*(1+1/best_combo['nodes'])
for n in range(best_combo['nodes']):
    centers.append(min(X_train_sqr) + n*centers_gaps)

data_least_quare = {}
rbf_net = rbf(n_hidden=best_combo['nodes'], sigma=best_combo['sigma'])
rbf_net.set_centers(centers)
train_mse, valid_mse = rbf_net.fit_least_square(X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr)
gen_gap = train_mse - valid_mse
data_least_quare['train_mse'] = train_mse
data_least_quare['valid_mse'] = valid_mse
data_least_quare['genralisation_gap'] = gen_gap
data_least_quare['model'] = rbf_net

print(f"Delta rule --> Train MSE: {best_combo['train_mse']:.4f}, Valid MSE: {best_combo['valid_mse']:.4f}")
print(f"Lest square --> Train MSE: {data_least_quare['train_mse']:.4f}, Valid MSE: {data_least_quare['valid_mse']:.4f}")

y_predict_least_square = data_least_quare['model'].predict(X_test_sqr)

# TODO compare with plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_train_sqr, Y_train_sqr, s=10, label=f'Training data')
plt.plot(X_test_sqr, y_predict_least_square, color='red', label=f'Least square model')
plt.plot(X_test_sqr, y_best_prediction, color='green', label=f'Delta rule model')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
# save_path = os.path.join('Lab2', 'plots', f'Prediction_{fun_name}{add_info}.png')
# plt.savefig(save_path)
plt.show()


#! NEWSECTION ======================================================================
print("\n" + "="*50 + "\n")

print('\n==== 2. Test convergence for different learning rate/eta on online delta rule ====\n')

# TODO test eta for square
eta_vec = [0.1, 0.05, 0.01, 0.005, 0.001]
centers = []
centers_gaps = (X_train_sqr_length/best_combo['nodes'])*(1+1/best_combo['nodes'])
for n in range(best_combo['nodes']):
    centers.append(min(X_train_sqr) + n*centers_gaps)
data_choose_eta = {eta: {} for eta in eta_vec}
for eta in eta_vec:
        rbf_net = rbf(n_hidden=best_combo['nodes'], sigma=best_combo['sigma'])
        rbf_net.set_centers(centers)
        train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr, learning_rate=eta)
        gen_gap = train_mse[-1] - valid_mse[-1]
        print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}, gap: {(train_mse[-1] - valid_mse[-1]):.5f}")
        data_choose_eta[eta]['train_mse'] = train_mse
        data_choose_eta[eta]['valid_mse'] = valid_mse
        data_choose_eta[eta]['genralisation_gap'] = gen_gap
        data_choose_eta[eta]['model'] = rbf_net


fig, ax = plt.subplots(figsize=(10, 6))
epochs = np.array(range(1, 101))
for eta, eta_data in data_choose_eta.items():
        ax.plot(epochs, eta_data['valid_mse'], label=f'eta {eta}')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_xlabel('Epochs', fontsize=14)
ax.set_ylabel('Validation MSE', fontsize=14)
ax.legend(title='Lerinig rate value', fontsize=10)
ax.grid(True)
plt.show()


#! NEWSECTION ======================================================================


print("\n" + "="*50 + "\n")

print('\n==== 3. Compare random centers localisation with my strategy ====\n')

# TODO Train on random centers for square

centers = []
centers_gaps = (X_train_sqr_length/best_combo['nodes'])*(1+1/best_combo['nodes'])
for n in range(best_combo['nodes']):
    centers.append(min(X_train_sqr) + n*centers_gaps)

random_centers = np.random.uniform(
    low=min(X_train_sqr),
    high=max(X_train_sqr),
    size=best_combo['nodes'])
random_centers = np.array(random_centers).reshape(-1, 1)

data_random_centers = {}
rbf_net = rbf(n_hidden=best_combo['nodes'], sigma=best_combo['sigma'])
rbf_net.set_centers(random_centers)
train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_sqr, Y_train_sqr, X_test_sqr, Y_test_sqr)
gen_gap = train_mse[-1] - valid_mse[-1]
print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}, gap: {(train_mse[-1] - valid_mse[-1]):.5f}")
data_random_centers['train_mse'] = train_mse
data_random_centers['valid_mse'] = valid_mse
data_random_centers['genralisation_gap'] = gen_gap
data_random_centers['model'] = rbf_net

y_predict_random_centers = data_random_centers['model'].predict(X_test_sqr)
y_predict_my_centers =  data_choose_eta[0.01]['model'].predict(X_test_sqr)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_sqr, Y_train_sqr, s=10, label=f'Training data')
plt.plot(X_test_sqr, y_predict_random_centers, color='red', label=f'Random center prediction')
plt.plot(X_test_sqr, y_predict_my_centers, color='green', label=f'Our method center prediction')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
# save_path = os.path.join('Lab2', 'plots', f'Prediction_{fun_name}{add_info}.png')
# plt.savefig(save_path)
plt.show()


#! NEWSECTION ======================================================================


print("\n" + "="*50 + "\n")

print('\n==== 4. Test trained model on clean dataset ====\n')

# TODO test square on clean data

_, _, X_test_clean, Y_test_clean = generate_data_square_2x()

# TODO test sqr on clean data
y_best_prediction_clean = best_combo['model'].predict(X_test_clean)

print(f"Test MSE on clean data: {np.mean((Y_test_clean-y_best_prediction_clean)**2)}")

plt.figure(figsize=(10, 6))
plt.plot(X_test_clean, Y_test_clean, color='red', label=f'Clean data')
plt.plot(X_test_clean, y_best_prediction_clean, color='green', label=f'Prediction')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
# save_path = os.path.join('Lab2', 'plots', f'Prediction_{fun_name}{add_info}.png')
# plt.savefig(save_path)
plt.show()



#! NEWSECTION ======================================================================


print("\n" + "="*50 + "\n")

print('\n==== 5. Compare RBF with perceptron with one hidden layer form Lab1b ====\n')

# TODO Train perceptron on noised data square(2x)
mlp_nn = mlp([X_train_sqr.shape[1]] + [best_combo['nodes']] + [1] )

start_time = time.time()
MLP_train_mse, MLP_valid_mse = mlp_nn.train_with_validation_set(X_train_sqr.T, Y_train_sqr.T, X_test_sqr.T, Y_test_sqr.T, 3000)
elapsed_time = time.time() - start_time

print(f"MLP --> Train MSE: {MLP_train_mse[-1]:.5f}, Valid MSE: {MLP_valid_mse[-1]:.5f}, Generalisation gap: {(MLP_train_mse[-1]-MLP_valid_mse[-1]):.5f}, Time: {elapsed_time:.3f} s")
print(f"RBF --> Train MSE: {best_combo['train_mse']:.5f}, Valid MSE: {best_combo['valid_mse']:.5f}, Generalisation gap: {best_combo['generalization_gap']:.5f}, Time: {best_combo['time']:.3f} s")

# TODO Compare errors and plots
y_predict_mlp = mlp_nn.predict(X_test_sqr.T)
if y_best_prediction.shape != y_predict_mlp.shape:
     y_predict_mlp = y_predict_mlp.T

plt.figure(figsize=(10, 6))
plt.scatter(X_train_sqr, Y_train_sqr, s=10, label=f'Training data')
plt.plot(X_test_sqr, y_predict_mlp, color='red', label=f'MLP prediction')
plt.plot(X_test_clean, y_best_prediction, color='green', label=f'RBF prediction')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
# save_path = os.path.join('Lab2', 'plots', f'Prediction_{fun_name}{add_info}.png')
# plt.savefig(save_path)
plt.show()
