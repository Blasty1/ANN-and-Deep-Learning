import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from part2 import *
from data_generation import *

# PART II
print("\n", "="*50, "\n")

# // TODO Generate data and split for train valid and test set 
print("1. Generating data spliting for train and test sets")
X, y = generate_time_series_data()
#train_X, train_y, valid_X, valid_y, test_X, test_y = split_data_for_train_valid_test(X, y)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=float(200/1200), shuffle=False, random_state=42)
print("Train X shape: ", train_X.shape, " Test X shape: ", test_X.shape)
print("Train Y shape: ", train_y.shape, " Test Y shape: ", test_y.shape)
print("\n", "="*50, "\n")

# // TODO Testing a few configurations of a three-layer perceptron with different # combinations of the number of nodes in the hidden layer5 (nh1 = 3, 4, 5, nh2 = 2, 4, 6) 
# // TODO Choose best be validation error and use early stopping
print("2. Testing different configuration of hidden layers")
nh1 = [3, 4, 5] 
nh2 = [2, 4, 6] 
print("Hidden layer 1 nodes: ", nh1, "\nHidden layer 2 nodes: ", nh2)
results = {}

for i in nh1:
    for j in nh2:
        _, _, mlp_reg = train_mlp(train_X, 
                            train_y, 
                            test_X, 
                            test_y, 
                            alpha = 0.01, 
                            learning_rate=0.1, 
                            hidden_layers=(i, j), 
                            early_stopping=True, 
                            max_iter = 10000)

        results[(i, j)] = {'mlp_reg': mlp_reg}

print("Trening results:")
for (i, j), res in results.items():
    print(f"Hidden Layers: ({i}, {j}) - Train MSE: {res['mlp_reg'].loss_:.4f}, Valid MSE: {(1-res['mlp_reg'].validation_scores_[-1]):.4f}")
best_config = min(results, key=lambda x: 1-results[x]['mlp_reg'].validation_scores_[-1])
print(best_config)
worst_config = max(results, key=lambda x: 1-results[x]['mlp_reg'].validation_scores_[-1])
best_model = results[best_config]['mlp_reg']
worst_model = results[worst_config]['mlp_reg']
print(f"Best Configuration: {best_config} with Valid MSE: {(1-results[best_config]['mlp_reg'].validation_scores_[-1]):.4f}")
print(f"Worst Configuration: {worst_config} with Valid MSE: {(1-results[worst_config]['mlp_reg'].validation_scores_[-1]):.4f}")
#print(f"Best model scores: {best_model.loss_curve_}, valid score: {best_model.validation_scores_}")

# // TODO Random weights initialization
print("\n", "="*50, "\n")
print("Random weights initialization test for best and worst configuration")
random_weights_seed = [12, 24, 38, 42, 53, 67]
seed_test_results = {'best': {}, 'worst': {}}

for seed in random_weights_seed:
    _, _, best_mlp_reg = train_mlp(train_X, 
                            train_y, 
                            test_X, 
                            test_y, 
                            alpha = 0.01, 
                            learning_rate=0.1, 
                            hidden_layers=best_config, 
                            early_stopping=True, 
                            max_iter = 10000,
                            random_state=seed)
    _, _, worst_mlp_reg = train_mlp(train_X, 
                            train_y, 
                            test_X, 
                            test_y, 
                            alpha = 0.01, 
                            learning_rate=0.1, 
                            hidden_layers=worst_config, 
                            early_stopping=True, 
                            max_iter = 10000,
                            random_state=seed)

    seed_test_results['best'][seed] = {best_mlp_reg}
    seed_test_results['worst'][seed] = {worst_mlp_reg}

print(seed_test_results)
print("Random seed test results:")
for model_type, models in seed_test_results.items():
    print(f"\nModel type: {model_type}")
    for seed, model_set in models.items():
        for model in model_set:
            print(f"Seed: {seed} - Train MSE: {model.loss_:.4f}, Valid MSE: {(1-model.validation_scores_[-1]):.4f}")

# // TODO Report the test performance for the two architectures. This is the conclu-sive estimate of the generalisation error on the unseen data subset. Please plot also these test predictions along with the known target values.
print("\n", "="*50, "\n")
print("3. Test te two models on unseen data")

best_pred = best_model.predict(test_X)
worst_pred = worst_model.predict(test_X)
print(test_y[:10])
print(worst_pred[:10])

best_error = mean_squared_error(test_y, best_pred)
worst_error = mean_squared_error(test_y, worst_pred)

print(f"Best model Test MSE: {best_error:.4f}")
print(f"Worst model Test MSE: {worst_error:.4f}")   

# TODO visualisation of predictions and true values
plt.figure(figsize=(10, 10))
plt.plot(test_y, label='True Values', color='black')
plt.plot(best_pred, label='Best Model Predictions', color='green')
plt.plot(worst_pred, label='Worst Model Predictions', color='red')
plt.title('Test Set Predictions vs True Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('Lab1b\\test_prediction_4.3.png')
plt.show()

# TODO Adding a guassian noise experiment 
print("\n", "="*50, "\n")
print("4. Adding zero-mean Guasian noise to input data")
results_noisy = {variance: {} for variance in [0.05, 0.15]}
variance_levels = [0.05, 0.15]
nh2_noisy = [3, 6, 9]
alpha_vec = [0.001, 0.01, 0.05, 0.1]
nh1_noisy = best_config[0]
for variance in variance_levels:
    train_X_noisy = add_noise(train_X, variance=variance)
    for nh2_n in nh2_noisy:
        for alpha in alpha_vec:
            _, test_mse, noisy_mlp_reg = train_mlp(train_X_noisy, 
                                train_y, 
                                test_X, 
                                test_y, 
                                alpha = alpha, 
                                learning_rate=0.1, 
                                hidden_layers=(nh1_noisy, nh2_n), 
                                early_stopping=True, 
                                max_iter = 10000,
                                random_state=32)
            results_noisy[variance][(nh1_noisy, nh2_n, alpha)] = noisy_mlp_reg
            print(f"Var: {variance}, H-Layers: ({nh1_noisy}, {nh2_n}), A: {alpha} - T MSE: {noisy_mlp_reg.loss_:.4f}, V MSE: {(1-noisy_mlp_reg.validation_scores_[-1]):.4f}, Tst MSE: {test_mse:.4f}")

print("\n", "="*50, "\n")

# TODO Non mandatory compere of models with noise 
print("5. Comparing models with noise")
print("Fonding best hiden layers number and alpha for noisy data - variance 0.09")
results_comparing = {'noisy_model': {}}
variance_lvl = 0.09
nh2_noisy = [3, 6, 9]
alpha_vector = [0.001, 0.005, 0.01, 0.05, 0.1]
nh1_noisy = best_config[0]
train_X_noisy = add_noise(train_X, variance=variance_lvl)
for nh2_n in nh2_noisy:
    for alpha in alpha_vector:
        _, test_mse, noisy_mlp_reg = train_mlp(train_X_noisy, 
                            train_y, 
                            test_X, 
                            test_y, 
                            alpha = alpha, 
                            learning_rate=0.1, 
                            hidden_layers=(nh1_noisy, nh2_n), 
                            early_stopping=True, 
                            max_iter = 10000,
                            random_state=32)
        results_comparing["noisy_model"][(nh1_noisy, nh2_n, alpha)] = noisy_mlp_reg
        print(f"Var: {variance}, H-Layers: ({nh1_noisy}, {nh2_n}), A: {alpha} - T MSE: {noisy_mlp_reg.loss_:.4f}, V MSE: {(1-noisy_mlp_reg.validation_scores_[-1]):.4f}, Tst MSE: {test_mse:.4f}")
best_noisy_config = min(results_comparing["noisy_model"], key=lambda x: 1-results_comparing["noisy_model"][x].validation_scores_[-1])
best_noisy_model = results_comparing["noisy_model"][best_noisy_config]
print(f"Best Noisy Configuration: {best_noisy_config} with Valid MSE: {(1-results_comparing['noisy_model'][best_noisy_config].validation_scores_[-1]):.4f}")
best_noisy_pred = best_noisy_model.predict(test_X)
best_noisy_error = mean_squared_error(test_y, best_noisy_pred)
print(f"Best Noisy model Test MSE: {best_noisy_error:.4f}")

print( "="*50, "\n")
# TODO Trainh best and worst model from point 2 on nosy data 
print("Train best and worst model form point 2 on nosy data - variance 0.09")
_, _, best_2_noisy = train_mlp(train_X_noisy, 
                            train_y, 
                            test_X, 
                            test_y, 
                            alpha = 0.01, 
                            learning_rate=0.1, 
                            hidden_layers=(best_config), 
                            early_stopping=True, 
                            max_iter = 10000,
                            random_state=32)
_, _, worst_2_noisy = train_mlp(train_X_noisy, 
                            train_y, 
                            test_X, 
                            test_y, 
                            alpha = 0.01, 
                            learning_rate=0.1, 
                            hidden_layers=(worst_config), 
                            early_stopping=True, 
                            max_iter = 10000,
                            random_state=32)
best_2_noisy_pred = best_2_noisy.predict(test_X)
worst_2_noisy_pred = worst_2_noisy.predict(test_X)
best_2_noisy_error = mean_squared_error(test_y, best_2_noisy_pred)
worst_2_noisy_error = mean_squared_error(test_y, worst_2_noisy_pred)
print(f"Best 2 model Test MSE on noisy data: {best_2_noisy_error:.4f}")
print(f"Worst 2 model Test MSE on noisy data: {worst_2_noisy_error:.4f}")

# TODO visualisation and comparision of predictions and true values
test_y_noisy = add_noise(test_y, variance=variance_lvl)
plt.figure(figsize=(10, 10))
plt.plot(test_y, label='True Values', color='black')
plt.plot(test_y_noisy, label='Noisy Values', color='yellow')
plt.plot(best_2_noisy_pred, label='Best Non-Noisy Model', color='green')
plt.plot(worst_2_noisy_pred, label='Worst Non-Noisy Model', color='red')
plt.plot(best_noisy_pred, label='Best Noisy Model', color='blue')
plt.title('Comper of nosy models')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('Lab1b\\compere_noisy_prediction_4.3.png')
plt.show()
