import numpy as np
import matplotlib.pyplot as plt
import generate_non_linear_data as gnd
import PerceptronLearningRule as plr
from matplotlib.animation import FuncAnimation
from plot_utils import *

# * Perceptron learnig rule
def weight_sum(weights, x):
    return np.dot(weights[1:], x) + weights[0]

def activation_function(weights, X):
    return 1 if weight_sum(weights, X) > 0.0 else -1

def perceptron_learning_rule_online(X, targets, weights, eta=0.01, max_epochs=50):
    traning_data = []#{epoch: {'weights': [], 'errors': [], 'epoch_error' : float, 'accuracy': float} for epoch in range(max_epochs)}
    
    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')
        correct_predictions = 0
        weights_results = []
        error_sum = 0
        for x, t in zip(X.T, targets):
            
            y_pred = activation_function(weights, x)
            if y_pred == t:
                correct_predictions += 1
            error = t - y_pred
            update = eta * error
            weights[1:] += update * x
            weights[0] += update
            weights_results.append(weights.copy())
            error_sum += error**2
        epoch_results = {
            'weights': weights_results.copy(),
            'error': error_sum / X.shape[1],
            'accuracy': correct_predictions / X.shape[1]
        }
        print(f'Error: {error_sum}, Accuracy: {epoch_results['accuracy']}')
        traning_data.append(epoch_results)
        # if error_sum == 0:
        #     break
    return traning_data


# * Delta rule online -- ADELINE

def delta_activation_function(weights, X):
    return weight_sum(weights, X)

def delta_predict(weights, x):
    return 1 if weight_sum(weights, x) > 0.0 else -1

def delta_rule_online(X, targets, weights, eta=0.01, max_epochs=50):
    traning_data = []

    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')
        correct_predictions = 0
        weights_results = []
        error_sum = 0
        for x, t in zip(X.T, targets):
            
            y_pred = delta_activation_function(weights, x)
            if delta_predict(weights, x) == t:
                correct_predictions +=1
            error = t - y_pred
            serror = error*error
            delta_w = eta * error
            weights[1:] += delta_w * x
            weights[0] += delta_w
            weights_results.append(weights.copy())
            error_sum += serror
        epoch_results = {
            'weights': weights_results.copy(),
            'error': error_sum/X.shape[1],
            'accuracy': correct_predictions / X.shape[1]
        }
        print(f'MSE: {serror}, Error: {error_sum}, Accuracy: {epoch_results['accuracy']}')

        traning_data.append(epoch_results)
    return traning_data


# * Delta rule batch -- ADELINE

def delta_activation_function_batch(weights, X_batch):
    return np.dot(X_batch, weights[1:]) + weights[0]

def delta_rule_batch(X, targets, weights, eta=0.01, max_epochs=50, batch_size = 20):
    traning_data = []

    for epoch in range(max_epochs):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        targets_shuffled = targets[indices]
        print(f'Epoch {epoch}')
        correct_predictions = 0
        weights_results = []
        error_sum = 0
        for i in range(0, len(targets), batch_size):
            xi = X_shuffled[i:i+batch_size]
            print(xi)
            
            ti = targets_shuffled[i:i+batch_size]
            print(ti)
            y_pred = delta_activation_function_batch(weights, xi)
            print(y_pred)
            predictions = np.where(y_pred > 0, 1, -1)
            correct_in_batch = np.sum(predictions == ti)
            correct_predictions += correct_in_batch
            errors = ti - y_pred
            batch_error = np.mean(errors)
            print(errors)
            delta_w = np.dot(xi.T, errors) / xi.shape[0]
            weights[1:] += eta * delta_w 
            weights[0] += eta * np.mean(errors)
            weights_results.append(weights.copy())
            error_sum += np.sum(errors**2) / xi.shape[0]

            
        epoch_results = {
            'weights': weights_results.copy(),
            'error': error_sum / X.shape[0],
            'accuracy': correct_predictions / X.shape[0]
        }
        print(f'Error: {error_sum}, Accuracy: {epoch_results['accuracy']}')
        traning_data.append(epoch_results)
    return traning_data

def run_delta_buch_for_split_dataset(n, eta, num_epoch, percentA, percentB, task_d = False):
    classA, classB = gnd.generate_splited_data(n, percentA, percentB, task_d)
    X, targets = gnd.suffle_and_create_labels(classA, classB)
    np.random.seed(52)
    weights = np.random.randn(3) *0.2 

    traning_data = delta_rule_batch(X.T, targets, weights, eta=eta, max_epochs=num_epoch)
    errors = [epoch['error'] for epoch in traning_data]    
    accuracies = [epoch['accuracy'] for epoch in traning_data]    
    weights = [epoch['weights'][-1] for epoch in traning_data]
    if task_d:
        create_animation(X.T, weights, targets, f'delta_rule_batch_taskD', f'Delta rule batch for delete 20%A1 and 80%A2 data')
    else:  
        create_animation(X.T, weights, targets, f'delta_rule_batch_{int(percentA*100)}_{int(percentB*100)}', f'Delta rule batch for delete {int(percentA*100)}%A and {int(percentB*100)}%B data')
    return errors, accuracies, weights

if __name__ == "__main__":
    n = 100
    eta = 0.01
    num_epochs = 150
    errors = []
    accuracies = []
    weights_vec = []

    deleteA = [0.0, 0.25, 0.5, 0.0]
    deleteB = [0.0, 0.25, 0.0, 0.5]
    for A, B in zip(deleteA, deleteB):
        err, acc, w =  run_delta_buch_for_split_dataset(n, eta, num_epochs, A, B)
        errors.append(err)
        accuracies.append(acc)
        weights_vec.append(w)
    err, acc, w =  run_delta_buch_for_split_dataset(n, eta, num_epochs, 0.0, 0.0, True)
    errors.append(err)
    accuracies.append(acc)
    weights_vec.append(w)
    labels_vec = ['All data', 'Without 25% each class', 'Without 50% A', 'Without 50% B', 'Without 20% A1 & 80% A2']
    compere_accuracy(accuracies, labels_vec, 'accuracy_delete_samples_comper')
    compere_convergence(errors, labels_vec, 'error_delete_samples_comper')
    # classA, classB = gnd.generate_overlaping_data(n)
    # #classA, classB = gnd.generate_splited_data(n, 0.0, 0.0, True)
    # X, targets = gnd.suffle_and_create_labels(classA, classB)

    # bias = [1] * X.shape[1]
    # np.random.seed(42)
    # weights = np.random.randn(3) *0.5 # [bias, x1, x2]

    # pl_traning_data = perceptron_learning_rule_online(X, targets, weights, eta=eta, max_epochs=num_epochs)
    # pl_errors = [epoch['error'] for epoch in pl_traning_data]    
    # pl_accuracies = [epoch['accuracy'] for epoch in pl_traning_data]    
    # # plot_convergence(pl_errors, 'erros_perceptron_rule_overlaping')
    # plot_accuracy(pl_accuracies, 'accuracy_perceptron_rule_overlaping')
    # pl_weights = [epoch['weights'][-1] for epoch in pl_traning_data]
    # create_animation(X.T, pl_weights, targets, 'Perceptron_rule_overlaping', 'Perceptron leraning rule for overlaping data')

    
    # dr_traning_data = delta_rule_online(X, targets, weights, eta=eta, max_epochs=num_epochs)
    # dr_errors = [epoch['error'] for epoch in dr_traning_data]    
    # dr_accuracies = [epoch['accuracy'] for epoch in dr_traning_data]    
    # # plot_convergence(dr_errors, 'error_delta_rule_online_overlaping')
    # plot_accuracy(dr_accuracies, 'accuracy_delta_rule_online_overlaping')
    # dr_weights = [epoch['weights'][-1] for epoch in dr_traning_data]
    # create_animation(X.T, dr_weights, targets, 'Delta_rule_online_overlaping', 'Delta rule online for overlaping data')

    # classA, classB = gnd.generate_splited_data(n, 0.0, 0.0, True)
    # X, targets = gnd.suffle_and_create_labels(classA, classB)
    # bias = [1] * X.shape[1]
    # np.random.seed(42)
    # weights = np.random.randn(3) *0.5 # [bias, x1, x2]


    # drb_traning_data = delta_rule_batch(X.T, targets, weights, eta=0.005, max_epochs=num_epochs)
    # drb_errors = [epoch['error'] for epoch in drb_traning_data]    
    # drb_accuracies = [epoch['accuracy'] for epoch in drb_traning_data]    
    # # plot_convergence(drb_errors, 'error_delta_rule_batch_overlaping')
    # # plot_accuracy(drb_accuracies, 'accuracy_delta_rule_batch_overlaping')
    # drb_weights = [epoch['weights'][-1] for epoch in drb_traning_data]# for weight in epoch['weights'][-1]]
   
    # create_animation(X.T, drb_weights, targets, 'delta_rule_batch_overlaping', 'Delta rule batch leraning for overlaping data')

    # compere_convergence(pl_errors, drb_errors, dr_errors, 'error_compere_overlaping')
    
    # compere_accuracy(pl_accuracies, drb_accuracies, dr_accuracies, 'accuracy_compere_overlaping')