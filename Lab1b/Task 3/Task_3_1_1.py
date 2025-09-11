import sys
import os
import numpy as np
sys.path.append(os.path.abspath("Lab1b/"))

import data_generation
import BackpropCode
import plots

X_A = data_generation.generate_splited_data(50,0,0)[0].T # class A -> -1
X_B = data_generation.generate_splited_data(50,0,0)[2].T # class B -> 1

# data_generation.plot_data(X_A,X_B)

X = np.hstack([X_A,X_B])
targets = np.zeros((1,X.shape[1]))
targets[0, 0:50]= -1
targets[0, 50:]= 1

NUMBER_OF_INPUTS = [X.shape[0]]
NUMBER_OF_OUTPUTS = [1]
HIDDEN_LAYERS = [20,20,20,20] # number of nodes per hidden layer

network = BackpropCode.NeuralNetwork(NUMBER_OF_INPUTS + HIDDEN_LAYERS + NUMBER_OF_OUTPUTS )

MSEs = network.train(X,targets, 100)
plots.plot_learning_curve(MSEs)