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
            weights_results.append(weights.copy)
            error_sum += error
        epoch_results = {
            'weights': weights_results.copy(),
            'error': error_sum,
            'accuracy': correct_predictions / X.shape[1]
        }
        print(f'Error: {error_sum}, Accuracy: {epoch_results['accuracy']}')
        traning_data.append(epoch_results)
        if error_sum == 0:
            break
    return traning_data


# * Delta rule online -- ADELINE

def delta_activation_function(weights, X):
    return weight_sum(weights, X)

def delta_predict(weights, x):
    return 1 if weight_sum(weights, x) > 0.0 else 0

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
            serror = error**2
            delta_w = eta * serror
            weights[1:] += delta_w * x
            weights[0] += delta_w
            weights_results.append(weights.copy())
            error_sum += error
        epoch_results = {
            'weights': weights_results.copy(),
            'error': error_sum/X.shape[1],
            'accuracy': correct_predictions / X.shape[1]
        }
        print(f'Error: {error_sum}, Accuracy: {epoch_results['accuracy']}')

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


if __name__ == "__main__":
    n = 100
    num_epochs = 150
    classA, classB = gnd.generate_overlaping_data(n)
    #classA, classB = gnd.generate_splited_data(n, 0.0, 0.0, True)
    X, targets = gnd.suffle_and_create_labels(classA, classB)

    bias = [1] * X.shape[1]
    np.random.seed(42)
    weights = np.random.randn(3) *0.5 # [bias, x1, x2]

    # pl_traning_data = perceptron_learning_rule_online(X, targets, weights, eta=0.01, max_epochs=num_epochs)
    # pl_errors = [epoch['error'] for epoch in pl_traning_data]    
    # pl_accuracies = [epoch['accuracy'] for epoch in pl_traning_data]    
    # plot_convergence(pl_errors, num_epochs)
    # plot_accuracy(pl_accuracies, num_epochs)

    # dr_traning_data = delta_rule_online(X, targets, weights, eta=0.01, max_epochs=num_epochs)
    # dr_errors = [epoch['error'] for epoch in dr_traning_data]    
    # dr_accuracies = [epoch['accuracy'] for epoch in dr_traning_data]    
    # plot_convergence(dr_errors, num_epochs)
    # plot_accuracy(dr_accuracies, num_epochs)

    drb_traning_data = delta_rule_batch(X.T, targets, weights, eta=0.005, max_epochs=num_epochs)
    drb_errors = [epoch['error'] for epoch in drb_traning_data]    
    drb_accuracies = [epoch['accuracy'] for epoch in drb_traning_data]    
    # plot_convergence(drb_errors, num_epochs)
    # plot_accuracy(drb_accuracies, num_epochs)
    drb_weights = [epoch['weights'][-1] for epoch in drb_traning_data]# for weight in epoch['weights'][-1]]
    print(drb_weights)
    
    create_animation(X.T, drb_weights, targets, 'delta_rule_batch_overlaping', 'Delta rule batch leraning for overlaping data')

    