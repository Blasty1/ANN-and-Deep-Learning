import numpy as np
from Utils.RBF import RBF_NN as rbf
from Utils.Plot_functions import plot_dataset, plot_curve

def square(x):
    return np.sign(np.sin(2*x))

points = np.arange(0,2*np.pi,0.1)
sin_points = np.sin(2*points)
square_points = square(points)

testing_points = np.arange(0.05,2*np.pi,0.1)
testing_sin_points = np.sin(2*testing_points)
testing_square_points = square(testing_points)

## we see the points to choose the centers
plot_dataset(points, sin_points, testing_points, testing_sin_points, fun_name='sin(2x)')
plot_dataset(points, square_points, testing_points, testing_square_points, fun_name='square(2x)')

set_of_number_of_nodes = range(0,100)
absolute_residual_error_sin = []
absolute_residual_error_square = []
absolute_residual_error_square_sig = []
notifications_sin=0
notifications_square=0
lower_value = 1
best_n_hidden_nodes = 0
for n_hidden_nodes in set_of_number_of_nodes:
    centers = np.linspace(0, 2*np.pi, n_hidden_nodes) # choose centers evenly spaced
    rbf_sin = rbf(centers, 0.5,centers=centers)
    rbf_square = rbf(centers, 0.5,centers=centers)

    rbf_sin.fit_least_square(points, sin_points,testing_points, testing_sin_points)
    rbf_square.fit_least_square(points, square_points, testing_points, testing_square_points)
    
    rbf_sin_predictions = rbf_sin.predict(testing_points)
    rbf_square_predictions = rbf_square.predict(testing_points)
    
    transformed_predictions = np.sign(rbf_square_predictions) # 3.1.2
    sin_error = np.mean(np.abs(testing_sin_points - rbf_sin_predictions))
    square_error = np.mean(np.abs(testing_square_points - rbf_square_predictions))
    square_sig_error = np.mean(np.abs(testing_square_points - transformed_predictions))
    absolute_residual_error_sin.append(sin_error)
    absolute_residual_error_square.append(square_error)
    absolute_residual_error_square_sig.append(np.mean(np.abs(testing_square_points - transformed_predictions)))
    if(sin_error < 0.1 and notifications_sin==0):
        print(f"Number of hidden nodes needed for sin(2x) with error < 0.1: {n_hidden_nodes}")
        notifications_sin+=1
    if(sin_error < 0.01 and notifications_sin==1):
        print(f"Number of hidden nodes needed for sin(2x) with error < 0.01: {n_hidden_nodes}")
        notifications_sin+=1
    if(sin_error < 0.001 and notifications_sin==2):
        print(f"Number of hidden nodes needed for sin(2x) with error < 0.001: {n_hidden_nodes}")
        notifications_sin+=1
    
    if(square_sig_error < 0.1 and notifications_square==0):
        print(f"Number of hidden nodes needed for square(2x) with error < 0.1: {n_hidden_nodes}")
        notifications_square+=1
    if(square_sig_error < 0.01 and notifications_square==1):
        print(f"Number of hidden nodes needed for square(2x) with error < 0.01: {n_hidden_nodes}")
        notifications_square+=1
    if(square_sig_error < 0.001 and notifications_square==2):
        print(f"Number of hidden nodes needed for square(2x) with error < 0.001: {n_hidden_nodes}")
        notifications_square+=1
    
    if( lower_value > square_error):
        lower_value = square_error
        best_n_hidden_nodes = n_hidden_nodes

print(f"Best number of hidden nodes for square(2x) is {best_n_hidden_nodes} with error {lower_value}")
    
    
plot_curve(set_of_number_of_nodes, absolute_residual_error_sin, 'Number of hidden nodes', 'Absolute Residual Error', 'Absolute Residual Error vs Number of hidden nodes for sin(2x)', 'sin_absolute_residual_error_vs_hidden_nodes')
plot_curve(set_of_number_of_nodes, absolute_residual_error_square, 'Number of hidden nodes', 'Absolute Residual Error', 'Absolute Residual Error vs Number of hidden nodes for square(2x)', 'square_absolute_residual_error_vs_hidden_nodes')    
plot_curve(set_of_number_of_nodes, absolute_residual_error_square_sig, 'Number of hidden nodes', 'Absolute Residual Error', 'Absolute Residual Error vs Number of hidden nodes for square(2x) using signal transformation', 'square_sign_absolute_residual_error_vs_hidden_nodes')


    
    
    