
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
        ratioOfMisclassifications = []
        L=len(self.layer_sizes)-1
        for epoch in range(epochs):
            cache = self.forward(X)
            gradients = self.backward(cache, targets)
            self.update_weights(gradients)
            
            MSEs.append( np.mean((cache[f"A{L}"] - targets)**2) )
            ratioOfMisclassifications.append(np.mean( ( (cache[f"A{L}"] > 0).astype(int) != (targets > 0).astype(int) ).astype(int) ))
        
        return (MSEs, ratioOfMisclassifications)
    
    def train_with_validation_set(self,X_train, Y_train,X_validation,Y_validation, epochs, number_of_epochs_for_validation=10):
        """ it trains the network using a validation set to monitor the overfitting, returns two MSEs, one for the training set and one for the validation set
        Args:
            X_train (_type_): t
            Y_train (_type_): _description_
            X_validation (_type_): _description_
            Y_validation (_type_): _description_
            epochs (_type_): _description_
            number_of_epochs_for_validation (int, optional): after each epoches we compiute the MSEs for training and validation dataset. Defaults to 10.

        Returns:
            : _description_
        """
        
        # it contains the MSE for each epoch
        MSEs_train_error = []
        MSEs_validation_error = []
        L=len(self.layer_sizes)-1
        for epoch in range(epochs):
            cache = self.forward(X_train)
            gradients = self.backward(cache, Y_train)
            self.update_weights(gradients)
                    
            
            if epoch % number_of_epochs_for_validation == 0:
                cache_validation = self.forward(X_validation)
                MSEs_validation_error.append( np.mean((cache_validation[f"A{L}"] - Y_validation)**2) ) 
                MSEs_train_error.append( np.mean((cache[f"A{L}"] - Y_train)**2) )
       
        return MSEs_train_error, MSEs_validation_error
    
    # it evaluate the validation dataset after each epoch during training
    def train_evaluate(self,X, Y, patterns, epochs):
        # it contains the MSE for each epoch
        train_MSEs = []
        predictions = []
        L=len(self.layer_sizes)-1
        for epoch in range(epochs):
            cache = self.forward(X)
            gradients = self.backward(cache, Y)
            self.update_weights(gradients)
            cache_prediction = self.forward(patterns) # test on the whole dataset
            prediction = cache_prediction[f"A{L}"]         # select the output layer
            predictions.append(prediction)
            train_MSEs.append( np.mean((cache[f"A{L}"] - Y)**2) )
        
        return train_MSEs, predictions
    
    
    ######### Online Training
        
        
    def train_online_with_validation_set(self,X_train, Y_train,X_validation,Y_validation, epochs, number_of_epochs_for_validation=10):
        """ it trains the network using a validation set to monitor the overfitting and an online approach, returns two MSEs, one for the training set and one for the validation set
        Args:
            X_train (_type_): t
            Y_train (_type_): _description_
            X_validation (_type_): _description_
            Y_validation (_type_): _description_
            epochs (_type_): _description_
            number_of_epochs_for_validation (int, optional): after each epoches we compiute the MSEs for training and validation dataset. Defaults to 10.

        Returns:
            : _description_
        """
        
        # it contains the MSE for each epoch
        MSEs_train_error = []
        MSEs_validation_error = []
        L=len(self.layer_sizes)-1
        for epoch in range(epochs):
            
            # Shuffle the order of samples for each epoch
            indices = np.random.permutation(X_train.shape[1])
            epoch_mse = 0
            epoch_misclass = 0
            
            for index in indices:
                X_train_sample = X_train[:, index:index+1]  # Select a single sample
                Y_train_sample = Y_train[:, index:index+1]  # Select the corresponding target
                
                # Forward pass for single sample
                cache = self.forward(X_train_sample)
                
                # Backward pass for single sample
                gradients = self.backward(cache, Y_train_sample)
                
                # Update weights immediately after each sample
                self.update_weights(gradients)
                
                # Accumulate error for epoch statistics
                prediction = cache[f"A{L}"]
                epoch_mse += np.mean((prediction - Y_train_sample)**2)
                epoch_misclass += np.mean(((prediction > 0).astype(int) != (Y_train_sample > 0).astype(int)).astype(int))

            if epoch % number_of_epochs_for_validation == 0:
                cache_validation = self.forward(X_validation)
                MSEs_validation_error.append( np.mean((cache_validation[f"A{L}"] - Y_validation)**2) ) 
                MSEs_train_error.append(epoch_mse / X_train.shape[1])       
       
        return MSEs_train_error, MSEs_validation_error

    def predict(self, X):

        cache = self.forward(X)
        L = len(self.layer_sizes) - 1
        return cache[f"A{L}"]
