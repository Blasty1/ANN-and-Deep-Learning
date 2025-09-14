import sys
import os
import numpy as np
sys.path.append(os.path.abspath("Lab1b/"))

import data_generation
import BackpropCode
import plots
import matplotlib.pyplot as plt
from plots import plot_decision_boundary, plot_decision_regions
########### 1 part
X_A = data_generation.generate_splited_data(50,0,0)[0].T # class A -> -1
X_B = data_generation.generate_splited_data(50,0,0)[2].T # class B -> 1
#data_generation.plot_data(X_A,X_B)

X = np.hstack([X_A,X_B])
targets = np.zeros((1,X.shape[1]))
targets[0, 0:50]= -1
targets[0, 50:]= 1

NUMBER_OF_INPUTS = [X.shape[0]]
NUMBER_OF_OUTPUTS = [1]
HIDDEN_LAYERS = [15,15] # number of nodes per hidden layer

network = BackpropCode.NeuralNetwork(NUMBER_OF_INPUTS + HIDDEN_LAYERS + NUMBER_OF_OUTPUTS )

[MSEs,ratios] = network.train(X,targets, 100)
#plotting MSE over epoch
#plots.plot_curve_over_epoch(MSEs, title="MSE_over_epoch", filename="MSE_over_epoch")

#plotting mislcassification ratio over epoch
#plots.plot_curve_over_epoch(ratios, title="Ratio of misclassifications over epoch",filename="Ratio_of_misclassifications_over_epoch")

#plots.plot_decision_regions(network,X_A,X_B)
#plt.show()

########### 2 part
cache = {}

#First case: 25% of class A and 25% of class B
trainA1, validA1, trainB1, validB1 = data_generation.generate_splited_data(
    n=100,  
    percent_of_A=0.25,
    percent_of_B=0.25,
    task_d=False
)
cache['trainA1'] = trainA1.T
cache['trainB1'] = trainB1.T
cache['validA1'] = validA1.T if validA1 is not None else None
cache['validB1'] = validB1.T if validB1 is not None else None

##############################################################################
#Second case: 50% of class A 
trainA2, validA2, trainB2, validB2 = data_generation.generate_splited_data(
    n=100,  
    percent_of_A=0.5,
    percent_of_B=0.0,
    task_d=False
)
cache['trainA2'] = trainA2.T
cache['trainB2'] = trainB2.T
cache['validA2'] = validA2.T if validA2 is not None else None
cache['validB2'] = validB2.T if validB2 is not None else None


##############################################################################
#Third case: 20% of class A for which A(1,:) <0 and 80% of class A for which A(1,:) >0
trainA3, validA3, trainB3, validB3 = data_generation.generate_splited_data(
    n=100,  
    percent_of_A=0.0,
    percent_of_B=0.0,
    task_d=True
)
cache['trainA3'] = trainA3.T
cache['trainB3'] = trainB3.T
cache['validA3'] = validA3.T if validA3 is not None else None
cache['validB3'] = validB3.T if validB3 is not None else None


#3 cases ( from 1 to 3)
for i in range(1,4):
    #training data set
    X_train, targets_train, X_validation , targets_validation  = data_generation.prepare_train_valid_dataset(cache[f"trainA{i}"],cache[f"trainB{i}"],cache[f"validA{i}"],cache[f"validB{i}"])
    NUMBER_OF_INPUTS = [X_train.shape[0]]
    NUMBER_OF_OUTPUTS = [1]
    HIDDEN_LAYERS = [2] # number of nodes per hidden layer
    
    network = BackpropCode.NeuralNetwork(NUMBER_OF_INPUTS + HIDDEN_LAYERS + NUMBER_OF_OUTPUTS )
    MSEs_training, MSEs_validation = network.train_with_validation_set(X_train,targets_train,X_validation,targets_validation, 100)
    
    #### plot the MSEs for training and validation set
    epoch_step=10
    title=f"MSEs for training and validation set - case {i} - {HIDDEN_LAYERS[0]} hidden units" 
    filename=f"MSEs_for_training_and_validation_set_case_{i}_{HIDDEN_LAYERS[0]}_hidden_units"  
    
    fig = plt.figure()
    ax = plt.axes()
    x_values = range(0,len(MSEs_training)*epoch_step, epoch_step)
    ax.plot(x_values, MSEs_training, 'b-', label='Training Error', linewidth=2)
    ax.plot(x_values, MSEs_validation, 'r-', label='Validation Error', linewidth=2)
    
    plt.title(f'Learning Curve - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    
    if(filename != None):
        save_path = os.path.join('Lab1b', 'Task_3', 'Plots', f'{filename}.png')
        plt.savefig(save_path)
    plt.show()
    
    
    ####### Difference betwen batch and sequential learning approach in terms of validation performance
    
    MSEs_training, MSEs_validation = network.train_online_with_validation_set(X_train,targets_train,X_validation,targets_validation, 100)

    #### plot the MSEs for training and validation set
    epoch_step=10
    title=f"MSEs for training and validation set - case {i} - {HIDDEN_LAYERS[0]} hidden units online_approach" 
    filename=f"MSEs_for_training_and_validation_set_case_{i}_{HIDDEN_LAYERS[0]}_hidden_units_online_approach"  
    
    fig = plt.figure()
    ax = plt.axes()
    x_values = range(0,len(MSEs_training)*epoch_step, epoch_step)
    ax.plot(x_values, MSEs_training, 'b-', label='Training Error', linewidth=2)
    ax.plot(x_values, MSEs_validation, 'r-', label='Validation Error', linewidth=2)
    
    plt.title(f'Learning Curve - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    
    if(filename != None):
        save_path = os.path.join('Lab1b', 'Task_3', 'Plots', f'{filename}.png')
        plt.savefig(save_path)
    plt.show()