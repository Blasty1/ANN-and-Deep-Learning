import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class RBF_NN:
    """
    Radial Basis Function Network
    Training is divided in 2 steps:
    1) Unsupervised Learning -> Choose the centers
    2) Supervised Learning -> Train the output weights ( choose one of the two methods below, fit_*)
    Then we have Prediction
    
    
    """
    
    def __init__(self, n_hidden,sigma,centers=None):
        self.n_hidden = n_hidden
        self.sigma = sigma
        self.centers = centers
    
    def _gaussian_rbf(self, X, centers):
        """
        Compute Radial-basis function networks
        """
        distances = cdist(X, centers, metric='euclidean')

        # Applica la funzione gaussiana
        rbf_values = np.exp(-(distances**2) / (2 * self.sigma**2))
        
        return rbf_values
    
    # First Stage.( Unsupervised Learning) -> Centers are chosen with unsupervised methods
    def choose_centers(self, X):
        pass
    
    # Second Stage.( Supervised Learning) -> Output weights are trained using supervised methods
    def fit_least_square(self, X, Y,X_validation, Y_validation):
        """
        Train the RBF network with batch and least squares approach
        """
        if(self.centers == None):
            # Choose the centers
            self.centers = self.choose_centers(X)
        
        #compute Phi -> n x N
        Phi = self._gaussian_rbf(X, self.centers)
        
        # Add the bias term
        Phi = np.vstack([Phi, np.ones((1,Phi.shape[1]))])
        
        W = np.linalg.pinv(Phi) @ Y.T
        
        self.weights = W
    
        # Compute the training error
        y_pred = self.predict(X)
        train_mse = np.mean((Y-y_pred)**2)
        
        y_pred_validation = self.predict(X_validation)
        validation_mse = np.mean((Y_validation-y_pred_validation)**2)
        
        return (train_mse, validation_mse)
    
    # For fit_delta_rule, ignore it
    def initialize_weights_random_uniform(self, n_outputs, range_val=0.5):
        """
        2. RANDOM UNIFORM [-range, +range]
        Simple but it can cause vanishing/exploding of the gradient
        """
        n_weights = self.n_hidden + 1 ## +1 for the bias
        weights = np.random.uniform(-range_val, range_val, (n_weights, n_outputs))
        return weights
    
    # Second Stage.( Supervised Learning) -> Output weights are trained using supervised methods
    def fit_delta_rule(self, X, Y,X_validation,Y_validation,weights=None,epochs=100, learning_rate=0.01):
        """
        Train the RBF network with sequential/online and gradient descent approach
        """
        train_mse = []
        validation_mse = []
        n_samples = X.shape[1]
        self.weights = self.initialize_weights_random_uniform(Y.shape[0]) if weights is None else weights
        
        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[: , i:i+1]
                y_i = Y[:, i:i+1]
                
                #compute Phi -> n x N
                Phi_i = self._gaussian_rbf(x_i, self.centers)
                
                # Add the bias term
                Phi_i = np.vstack([Phi_i, [[1]]])
                
                # Compute the output
                y_pred_i = self.weights.T @ Phi_i              # (n_outputs, 1)
                
                # compute the error
                e = y_i - y_pred_i
                
                deltaW = learning_rate * Phi_i @ e.T
                
                self.weights += deltaW

            # Compute the training error
            y_pred = self.predict(X)
            train_mse.append(np.mean((Y-y_pred)**2))
            
            y_pred_validation = self.predict(X_validation)
            validation_mse.append(np.mean((Y_validation-y_pred_validation)**2))
        
        return (train_mse, validation_mse)

    def predict(self,X):
        """
        Predict the output of the RBF network
        """
        Phi = self._gaussian_rbf(X, self.centers)  # (n_hidden, n_samples)
        Phi = np.vstack([Phi, np.ones((1, Phi.shape[1]))])  # + bias
        
        return self.weights.T @ Phi   # (n_outputs, n_samples)