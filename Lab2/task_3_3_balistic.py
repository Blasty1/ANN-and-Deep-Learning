import numpy as np
from Utils.Plot_functions import *
from Utils.RBF import RBF_NN as rbf
from Utils.Data_generation import *
import os
from matplotlib.patches import Circle


def load_data(file_path):
    """
    Loads data from a .dat file and splits it into input and output matrices.
    
    """
    try:
        data = np.loadtxt(file_path)
        input_data = data[:, :2]
        output_data = data[:, 2:]
        
        print(f"Data successfully loaded from {file_path}")
        print(f"Input data shape: {input_data.shape}")
        print(f"Output data shape: {output_data.shape}")
        
        return input_data, output_data
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None


def plot_data(X, Y, title="Ballistic Data"):
    """
    Visualizes the input and output data on two separate scatter plots.
    
    Args:
        X (np.ndarray): The input features, shape (n_samples, 2).
        Y (np.ndarray): The output targets, shape (n_samples, 2).
        title (str): The main title for the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot input data (angle vs velocity)
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
    plt.title(f"{title} - Input Data")
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.grid(True)
    
    # Plot output data (distance vs height)
    plt.subplot(1, 2, 2)
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.7, color='orange')
    plt.title(f"{title} - Output Data")
    plt.xlabel("Distance")
    plt.ylabel("Height")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    
    train_file = 'S:\\KTH\ANNDL\\ANN-and-Deep-Learning\\Lab2\\data\\ballist.dat'
    test_file = 'S:\\KTH\ANNDL\\ANN-and-Deep-Learning\\Lab2\\data\\balltest.dat'
    
    X_train, Y_train = load_data(train_file)
    X_test, Y_test = load_data(test_file)
    if X_train is not None and X_test is not None:
        print("\nData loaded and ready for RBF network training and testing.")
    plot_data(X_train, Y_train, title="Training Data (ballist.dat)")

    n_hidden_vec = [20, 25, 30, 40, 45, 50]
    sigma_vec= [0.05, 0.1, 0.15, 0.17]
    epochs = 100
    # n_hidden = 50
    # sigma = 0.1
    # epochs = 200
    results = {n_hidden : {} for n_hidden in n_hidden_vec}
    for n_hidden in n_hidden_vec:
        results[n_hidden] = {sigma : {} for sigma in sigma_vec}
        for sigma in sigma_vec:

            rbf_balistic = rbf(n_hidden, sigma)
            centers, dead_units = rbf_balistic.choose_centers_sochastic(X_train, 300, 0.2)
            train_mse, valid_mse = rbf_balistic.fit_delta_rule(X_train, Y_train, X_test, Y_test, epochs=epochs, learning_rate=0.05)
            y_pred = rbf_balistic.predict(X_test)

            results[n_hidden][sigma]['centers'] = centers
            results[n_hidden][sigma]['dead_units'] = dead_units
            results[n_hidden][sigma]['train_mse'] = train_mse
            results[n_hidden][sigma]['valid_mse'] = valid_mse
            results[n_hidden][sigma]['model'] = rbf_balistic
            results[n_hidden][sigma]['prediction'] = y_pred
                   
            print(f"Nodes {n_hidden}, Sigma {sigma}, Train MSE: {train_mse[-1]:.5f}, Valid MSE: {valid_mse[-1]:.5f}, Gap: {(valid_mse[-1]-train_mse[-1]):.5f}, Nr of dead units: {len(dead_units)}")
    best_combo = None
    best_score = float('inf')
    all_results = []

    for n_nodes, n_data_dict in results.items():
        for sigma, data_dict in n_data_dict.items():
            model = data_dict['model']
            prediction = data_dict['prediction']
            centers = data_dict['centers']
            dead_units = data_dict['dead_units']
            train_mse = data_dict['train_mse']
            valid_mse = data_dict['valid_mse']
            generalization_gap = valid_mse[-1] - train_mse[-1]
            score = valid_mse[-1] + abs(generalization_gap)*2 + len(dead_units)
            
            all_results.append({
                'nodes': n_nodes,
                'sigma': sigma,
                'model': model,
                'prediction': prediction,
                'centers': centers,
                'dead_units': dead_units,
                'train_mse': train_mse,
                'valid_mse': valid_mse,
                'generalization_gap': generalization_gap,
                'score': score
            })

            if score < best_score:
                best_score = score
                best_combo = {
                    'nodes': n_nodes,
                    'sigma': sigma,
                    'model': model,
                    'prediction': prediction,
                    'centers': centers,
                    'dead_units': dead_units,
                    'train_mse': train_mse,
                    'valid_mse': valid_mse,
                    'generalization_gap': generalization_gap
                }

    sorted_results_noisy = sorted(all_results, key=lambda x: x['score'])

    print("Five best combination for noisy based on valid MSE and generalisation gap:")
    for result in sorted_results_noisy[:5]:
        print(f"Nodes: {result['nodes']}, Sigma: {result['sigma']:.2f}, Valid MSE: {result['valid_mse'][-1]:.4f}, Gap: {result['generalization_gap']:.4f}, Score: {result['score']:.4f}")

    print("\nBest combination for noisy:")
    print(f"Nodes: {best_combo['nodes']}, Sigma: {best_combo['sigma']:.2f}, Valid MSE: {best_combo['valid_mse'][-1]:.4f}, Gap: {best_combo['generalization_gap']:.4f}")


    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, epochs+1)), best_combo['train_mse'], color = 'green', label = 'Traning MSE')
    plt.plot(list(range(1, epochs+1)), best_combo['valid_mse'], color = 'red', label = 'Valid MSE')
    plt.xlabel("Epochs")
    plt.ylabel("MSE value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(12, 6))

    ax = plt.subplot(1, 2, 1)
    for center in best_combo['centers']:
        circle = Circle(center, radius=best_combo['sigma'], color='lime', alpha=0.2, fill=True)
        ax.add_patch(circle)
    plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.7)
    plt.scatter(best_combo['centers'][:, 0], best_combo['centers'][:, 1], c='lime', marker='^', s=10, edgecolors='black', label='Centra RBF')
    
    plt.title(f"Testing - Input Data")
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.grid(True)
    
    # Plot output data (distance vs height)
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test[:, 0], Y_test[:, 1], alpha=0.7, color='orange')
    plt.scatter(best_combo['prediction'][:, 0], best_combo['prediction'][:, 1], alpha=0.7, color='blue')
    plt.title(f"Prediction - Output Data")
    plt.xlabel("Distance")
    plt.ylabel("Height")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
