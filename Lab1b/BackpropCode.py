
import numpy as np

## Activation function used in the code
def activation_function(x):
    return 2/(1+np.exp(-x)) - 1

def af_derivative(x):
    return  0.5*(1 + activation_function(x))*(1 - activation_function(x))


#############

#### HOW IT WORKS?
# three phases:
#       1. Forward pass -> compute the nodes output
#       2. Backward pass -> compute the error signal sigma ( starting from th output layer ) going back
#       3. Update of weights
#  X -> input matrice ( one value per column and features on the row )
#  W -> vectors of matrices of weights W = [ V , W , S ] ( a matrice for each layer )
#  HIDDEN_LAYERS -> each number is nodes per hidden layer 

class NeuralNetwork:
    def __init__(self, layer_sizes,coefficient_of_the_momentum=0.9,learning_rate=0.01):
        self.layer_sizes = layer_sizes # it contains the size of each layer ( including the input and output layers )
        self.learning_rate = learning_rate
        self.weights = self._init_weights()
        self.momentums = self._init_momentum()
        self.coefficient_of_the_momentum = coefficient_of_the_momentum
    
    def _init_weights(self):
        weights = {}
        for i in range(len(self.layer_sizes) - 1):
            sigma = 0.5
            # +1 on the right dimension is for adding the bias in the weights
            weights[f"W{i+1}"] = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]+1) * sigma
        return weights
    
    def _init_momentum(self):
        teta = {}
        # this for is equivalent to the one seen in _init_weights
        for i in range(1,len(self.layer_sizes)):
            teta[f"teta{i}"] = np.zeros_like(self.weights[f"W{i}"])
            
        return teta

    def forward(self,X):
        # we are interested the bias , so we are increasing the rows of the input matrice
        cache = {"A0" : np.vstack([X, np.ones((1,X.shape[1]))])} # we save some important values for other computations
        
        # we start from 1 because the input layer is not considered
        for i in range(1, len(self.layer_sizes)):
            A_i_pre_activation_function = self.weights[f"W{i}"] @ cache[f"A{i-1}"]
            A_i = activation_function(A_i_pre_activation_function)
            
            if i < len(self.layer_sizes) - 1:  # hidden layer
                A_i = np.vstack([A_i, np.ones((1, A_i.shape[1]))]) # Add bias for the next layer

            cache[f"A{i}"] = A_i
            cache[f"A{i}_pre_activation_function"] = A_i_pre_activation_function
        return cache


    def backward(self,cache,targets):
        gradients = {} # delta * input
        deltas = {}
        L = len(self.layer_sizes) - 1 # number of layers with weights
        
        # Output layer signal error 
        deltas[f"delta_{L}"] = (cache[f"A{L}"] - targets) * af_derivative(cache[f"A{L}_pre_activation_function"])
        gradients[f"dW{L}"] =  deltas[f"delta_{L}"]  @ cache[f"A{L-1}"].T
        


        # we go through the hidden layer for computing the error signals
        for i in reversed(range(1, L)):
            # V^T @ signal error matrice of the last iteration  * derivative
            weight_multiplied_by_signal_errors = (self.weights[f"W{i+1}"].T @ deltas[f"delta_{i+1}"]) 
            
            # we remove the extra row that we previously added to the forward pass to take care of the bias term
            weight_multiplied_by_signal_errors  = weight_multiplied_by_signal_errors[:-1,:]
            
            new_delta = weight_multiplied_by_signal_errors * af_derivative(cache[f"A{i}_pre_activation_function"])
            
    
            # we update the delta
            deltas[f"delta_{i}"] = new_delta
            
            gradients[f"dW{i}"] = new_delta @ cache[f"A{i-1}"].T

        return gradients
    
    def update_weights(self, gradients):

        for i in range(1,len(self.layer_sizes)):

            ### Momentum term -> alpha * teta - (1-alpha) * gradient
            teta = self.coefficient_of_the_momentum*self.momentums[f"teta{i}"] - (1-self.coefficient_of_the_momentum)*gradients[f"dW{i}"]
            deltaW = self.learning_rate*teta
            
            self.weights[f"W{i}"] += deltaW

            # we update the momentum
            self.momentums[f"teta{i}"] = teta
    
    # it returns the list of MSE in order to be able to study the learning phase
    def train(self,X, targets, epochs):
        # it contains the MSE for each epoch
        MSEs = []
        L=len(self.layer_sizes)-1
        for epoch in range(epochs):
            cache = self.forward(X)
            gradients = self.backward(cache, targets)
            self.update_weights(gradients)
            
            MSEs.append( np.mean((cache[f"A{L}"] - targets)**2) )
        
        return MSEs

