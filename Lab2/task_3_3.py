from Utils.Plot_functions import *
from Utils.RBF import RBF_NN as rbf
from Utils.Data_generation import *
import time

#TODO best noisy model

X_train_noisy, Y_train_noisy, X_test_noisy, Y_test_noisy = noise_generate_data_sin_2x()
X_train_clean, Y_train_clean, X_test_clean, Y_test_clean = generate_data_sin_2x()

X_train_noisy_length = max(X_train_noisy) - min(X_train_noisy)

n_hidden_vec = [5, 10, 15, 20, 50]
sigma_vec = [0.01, 0.1, 0.2, 0.3, 0.5]


print('\n==== Search best hiperparamiters for sin(2x) ====\n')

data_noisy_kernels = {n: {} for n in n_hidden_vec}
for n_hidden in n_hidden_vec:
    centers = []
    centers_gaps = (X_train_noisy_length/n_hidden)*(1+1/n_hidden)
    for n in range(n_hidden):
        centers.append(min(X_train_noisy) + n*centers_gaps)
    train_mse_vec = []
    valid_mse_vec = []
    gen_gap_vec = []
    model_vec = []
    time_vec = []
    data_noisy_kernels[n_hidden]['sigma'] = sigma_vec
    for sigma in sigma_vec:
        rbf_net = rbf(n_hidden=n_hidden, sigma=sigma)
        rbf_net.set_centers(centers)
        start_time = time.time()
        train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_noisy, Y_train_noisy, X_test_noisy, Y_test_noisy)
        elapsed_time = time.time() - start_time
        train_mse_vec.append(train_mse[-1])
        valid_mse_vec.append(valid_mse[-1])
        model_vec.append(rbf_net)
        time_vec.append(elapsed_time)
        gen_gap_vec.append((train_mse[-1] - valid_mse[-1]))
        print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}, gap: {(train_mse[-1] - valid_mse[-1]):.5f}")
    data_noisy_kernels[n_hidden]['train_mse'] = train_mse_vec
    data_noisy_kernels[n_hidden]['valid_mse'] = valid_mse_vec
    data_noisy_kernels[n_hidden]['genralisation_gap'] = gen_gap_vec
    data_noisy_kernels[n_hidden]['model'] = model_vec
    data_noisy_kernels[n_hidden]['time'] = time_vec

# plot_RBF_kernel_comprision(data_noisy_kernels)
# TODO find best combination for sin

best_combo_noisy = None
best_score = float('inf')

all_results = []

for n_nodes, n_data in data_noisy_kernels.items():
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
            best_combo_noisy = {
                'nodes': n_nodes,
                'sigma': sigma,
                'model': model,
                'train_mse': train_mse,
                'valid_mse': valid_mse,
                'time': time_s,
                'generalization_gap': generalization_gap
            }

# Sortowanie wyników
sorted_results_noisy = sorted(all_results, key=lambda x: x['score'])

# Wyświetlenie 5 najlepszych konfiguracji
print("Five best combination for noisy based on valid MSE and generalisation gap:")
for result in sorted_results_noisy[:5]:
    print(f"Nodes: {result['nodes']}, Sigma: {result['sigma']:.2f}, Valid MSE: {result['valid_mse']:.4f}, Gap: {result['generalization_gap']:.4f}, Score: {result['score']:.4f}")

# Wyświetlenie najlepszej kombinacji
print("\nBest combination for noisy:")
print(f"Nodes: {best_combo_noisy['nodes']}, Sigma: {best_combo_noisy['sigma']:.2f}, Valid MSE: {best_combo_noisy['valid_mse']:.4f}, Gap: {best_combo_noisy['generalization_gap']:.4f}")

y_noisy_prediction = best_combo_noisy['model'].predict(X_test_noisy)


# plot_prediction(X_train_noisy, Y_train_noisy, X_test_noisy, y_noisy_prediction)


#!======================================================================

#TODO Bets clean model

X_train_clean, Y_train_clean, X_test_clean, Y_test_clean = generate_data_sin_2x()

X_train_clean_length = max(X_train_clean) - min(X_train_clean)

n_hidden_vec = [5, 10, 20, 30, 50]
sigma_vec = [0.01, 0.1, 0.2, 0.3, 0.5]

print('\n==== Search best hiperparamiters for sin(2x) ====\n')

data_clean_kernels = {n: {} for n in n_hidden_vec}
for n_hidden in n_hidden_vec:
    centers = []
    centers_gaps = (X_train_clean_length/n_hidden)*(1+1/n_hidden)
    for n in range(n_hidden):
        centers.append(min(X_train_clean) + n*centers_gaps)
    train_mse_vec = []
    valid_mse_vec = []
    gen_gap_vec = []
    model_vec = []
    time_vec = []
    data_clean_kernels[n_hidden]['sigma'] = sigma_vec
    for sigma in sigma_vec:
        rbf_net = rbf(n_hidden=n_hidden, sigma=sigma)
        rbf_net.set_centers(centers)
        start_time = time.time()
        train_mse, valid_mse = rbf_net.fit_delta_rule(X_train_clean, Y_train_clean, X_test_clean, Y_test_clean)
        elapsed_time = time.time() - start_time
        train_mse_vec.append(train_mse[-1])
        valid_mse_vec.append(valid_mse[-1])
        model_vec.append(rbf_net)
        time_vec.append(elapsed_time)
        gen_gap_vec.append((train_mse[-1] - valid_mse[-1]))
        print(f"{n_hidden} hidden nodes, sigma {sigma}, train MSE: {train_mse[-1]:.5f}, valid MSE: {valid_mse[-1]:.5f}, gap: {(train_mse[-1] - valid_mse[-1]):.5f}")
    data_clean_kernels[n_hidden]['train_mse'] = train_mse_vec
    data_clean_kernels[n_hidden]['valid_mse'] = valid_mse_vec
    data_clean_kernels[n_hidden]['genralisation_gap'] = gen_gap_vec
    data_clean_kernels[n_hidden]['model'] = model_vec
    data_clean_kernels[n_hidden]['time'] = time_vec

# plot_RBF_kernel_comprision(data_clean_kernels)
# TODO find best combination for sin

best_combo_clean = None
best_score = float('inf')

all_results = []

for n_nodes, n_data in data_clean_kernels.items():
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
            best_combo_clean = {
                'nodes': n_nodes,
                'sigma': sigma,
                'model': model,
                'train_mse': train_mse,
                'valid_mse': valid_mse,
                'time': time_s,
                'generalization_gap': generalization_gap
            }

sorted_results_clean = sorted(all_results, key=lambda x: x['score'])

print("Five best combination based on valid MSE and generalisation gap:")
for result in sorted_results_clean[:5]:
    print(f"Nodes: {result['nodes']}, Sigma: {result['sigma']:.2f}, Valid MSE: {result['valid_mse']:.4f}, Gap: {result['generalization_gap']:.4f}, Score: {result['score']:.4f}")

print("\nBest combination:")
print(f"Nodes: {best_combo_clean['nodes']}, Sigma: {best_combo_clean['sigma']:.2f}, Valid MSE: {best_combo_clean['valid_mse']:.4f}, Gap: {best_combo_clean['generalization_gap']:.4f}")

y_clean_prediction = best_combo_clean['model'].predict(X_test_clean)


# plot_prediction(X_train_clean, Y_train_clean, X_test_clean, y_clean_prediction )
#
#!======================================================================

print("\n" + "="*50 + "\n")

print('\n==== 1. Compare noisy and clean data with CL method ====\n')

#TODO CLEAN DATA WITH CL
rbf_clean = rbf(n_hidden=best_combo_clean['nodes'], sigma=best_combo_clean['sigma'])
_, dead_units_clean = rbf_clean.choose_centers(X_train_clean, 200, 0.1)
train_mse_clean, valid_mse_clean = rbf_clean.fit_delta_rule(X_train_clean, Y_train_clean, X_test_clean, Y_test_clean)

#TODO NOISY DATA WITH CL
rbf_noisy = rbf(n_hidden=best_combo_noisy['nodes'], sigma=best_combo_noisy['sigma'])
_, dead_units_noisy = rbf_noisy.choose_centers(X_train_noisy, 200, 0.1)
train_mse_noisy, valid_mse_noisy = rbf_noisy.fit_delta_rule(X_train_noisy, Y_train_noisy, X_test_noisy, Y_test_noisy)

y_clean_prediction = rbf_clean.predict(X_test_clean)
y_noisy_prediction = rbf_noisy.predict(X_test_noisy)

print(f"Clean data --> Train MSE: {train_mse_clean[-1]:.5f}, Valid MSE: {valid_mse_clean[-1]:.5f}, Gap: {(train_mse_clean[-1]-valid_mse_clean[-1]):.5f}, MAE: {np.mean(np.abs(Y_test_clean - y_clean_prediction))} Nr of dead units: {len(dead_units_clean)}")
print(f"Noisy data --> Train MSE: {train_mse_noisy[-1]:.5f}, Valid MSE: {valid_mse_noisy[-1]:.5f}, Gap: {(train_mse_noisy[-1]-valid_mse_noisy[-1]):.5f}, MAE: {np.mean(np.abs(Y_test_clean - y_noisy_prediction))} Nr of dead units: {len(dead_units_noisy)}")




plt.figure(figsize=(10, 6))
plt.plot(X_test_clean, Y_test_clean, color='blue', linestyle = 'dashed', label=f'Clean data')
plt.plot(X_test_clean, y_clean_prediction, color='red', label=f'Clean prediction')
plt.plot(X_test_noisy, y_noisy_prediction, color='green', label=f'Noisy prediction')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
save_path = os.path.join('Lab2', 'plots', f'Compere_prediction_CL_noisy_clean_sin2x.png')
plt.savefig(save_path)
plt.show()

 
#!======================================================================

print("\n" + "="*50 + "\n")

print('\n==== 2. Dead units prevention ====\n')

# X_train_du, Y_train_du, X_test_du, Y_test_du = create_clustered_data_for_dead_units()

rbf_soch = rbf(n_hidden=35, sigma=0.5)
_, dead_units_soch = rbf_soch.choose_centers_sochastic(X_train_noisy, 200, 0.1)
train_mse_soch, valid_mse_soch = rbf_soch.fit_delta_rule(X_train_noisy, Y_train_noisy, X_test_noisy, Y_test_noisy)

rbf_wta = rbf(n_hidden=35, sigma=0.5)
_, dead_units_wta = rbf_wta.choose_centers(X_train_noisy, 200, 0.1)
train_mse_wta, valid_mse_wta = rbf_wta.fit_delta_rule(X_train_noisy, Y_train_noisy, X_test_noisy, Y_test_noisy)

print(f"Sochastic centers noisy--> Train MSE: {train_mse_soch[-1]:.5f}, Valid MSE: {valid_mse_soch[-1]:.5f}, Generalisation gap: {(train_mse_soch[-1]-valid_mse_soch[-1]):.5f}, Nr of dead units: {len(dead_units_soch)}")
print(f"WTA centers noisy--> Train MSE: {train_mse_wta[-1]:.5f}, Valid MSE: {valid_mse_wta[-1]:.5f}, Generalisation gap: {(train_mse_wta[-1]-valid_mse_wta[-1]):.5f}, Nr of dead units: {len(dead_units_wta)}")

y_pred_soch = rbf_soch.predict(X_test_noisy)

y_pred_wta = rbf_wta.predict(X_test_noisy)

plt.figure(figsize=(10, 6))
plt.plot(X_test_clean, Y_test_clean, color='blue', linestyle = 'dashed', label=f'Clean data')
plt.plot(X_test_noisy, y_pred_wta, color='red', label=f'WTA prediction')
plt.plot(X_test_noisy, y_pred_soch, color='green', label=f'Sochastic prediction')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
save_path = os.path.join('Lab2', 'plots', f'Prediction_sochastic_normal_CL.png')
plt.savefig(save_path)
plt.show()

#!==============================================

rbf_soch = rbf(n_hidden=50, sigma=0.3)
_, dead_units_soch = rbf_soch.choose_centers_sochastic(X_train_clean, 200, 0.1)
train_mse_soch, valid_mse_soch = rbf_soch.fit_delta_rule(X_train_clean, Y_train_clean, X_test_clean, Y_test_clean)

rbf_wta = rbf(n_hidden=50, sigma=0.3)
_, dead_units_wta = rbf_wta.choose_centers(X_train_clean, 200, 0.1)
train_mse_wta, valid_mse_wta = rbf_wta.fit_delta_rule(X_train_clean, Y_train_clean, X_test_clean, Y_test_clean)

print(f"Sochastic centers clean--> Train MSE: {train_mse_soch[-1]:.5f}, Valid MSE: {valid_mse_soch[-1]:.5f}, Generalisation gap: {(train_mse_soch[-1]-valid_mse_soch[-1]):.5f}, Nr of dead units: {len(dead_units_soch)}")
print(f"WTA centers clean --> Train MSE: {train_mse_wta[-1]:.5f}, Valid MSE: {valid_mse_wta[-1]:.5f}, Generalisation gap: {(train_mse_wta[-1]-valid_mse_wta[-1]):.5f}, Nr of dead units: {len(dead_units_wta)}")

y_pred_soch = rbf_soch.predict(X_test_clean)

y_pred_wta = rbf_wta.predict(X_test_clean)

plt.figure(figsize=(10, 6))
plt.plot(X_test_clean, Y_test_clean, color='blue', linestyle = 'dashed', label=f'Clean data')
plt.plot(X_test_clean, y_pred_wta, color='red', label=f'WTA prediction')
plt.plot(X_test_clean, y_pred_soch, color='green', label=f'Sochastic prediction')

plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
save_path = os.path.join('Lab2', 'plots', f'Prediction_sochastic_normal_CL_clean.png')
plt.savefig(save_path)
plt.show()

