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
        self.weights = None #! lost weights variable 

    def set_centers(self, centers_vec):
        self.centers = centers_vec
    
    def _gaussian_rbf(self, X, centers):
        """
        Compute Radial-basis function networks
        """
        distances = cdist(X, centers, metric='euclidean')

        # Applica la funzione gaussiana
        rbf_values = np.exp(-(distances**2) / (2 * self.sigma**2))
        
        return rbf_values

    
    # First Stage.( Unsupervised Learning  ) -> Centers are chosen with unsupervised methods
    def choose_centers(self, X, n_iterations = 1000, eta = 0.1):
        """
        Unsupervised choosing of centers. 
        Returns only centers coordinates
        """
        # np.random.seed(42)
        # initaial_coordinates = np.random.choice(X.shape[0], self.n_hidden, replace=False)
        centers = np.random.uniform(low=np.min(X), high=np.max(X), size=(self.n_hidden, X.shape[1]))
        # centers = X[initaial_coordinates].copy()
        win_counts = np.zeros(self.n_hidden, dtype=int)

        for it in range(n_iterations):
            shuffled_indices = np.random.permutation(X.shape[0])
            for i in shuffled_indices:
                x_i = X[i:i+1]
                dist_vec = np.sum((centers - x_i)**2, axis=1)
                
                win_id = np.argmin(dist_vec)
                win_counts[win_id] += 1
                centers[win_id] += eta * (x_i.squeeze() - centers[win_id].squeeze())
            eta *=0.99
        self.centers = centers
        dead_units_indices = np.where(win_counts == 0)[0]
        return centers, dead_units_indices
    
    def choose_centers_sochastic(self, X, n_iterations=1000, eta=0.1):
        """
        Unsupervised choosing of centers with a probabilistic update rule
        to prevent dead units.
        Returns centers coordinates and a list of dead unit indices.
        """
        np.random.seed(42)
        centers = np.random.uniform(low=np.min(X), high=np.max(X), size=(self.n_hidden, X.shape[1]))
        win_counts = np.zeros(self.n_hidden, dtype=int)

        prob_closest = 0.80
        
        for it in range(n_iterations):
            shuffled_indices = np.random.permutation(X.shape[0])
            for i in shuffled_indices:
                x_i = X[i:i+1]
                dist_vec = np.sum((centers - x_i)**2, axis=1)

                closest_nodes_indices = np.argsort(dist_vec)[:2]

                if np.random.rand() < prob_closest:
                    win_id = closest_nodes_indices[0]  
                else:
                    if len(closest_nodes_indices) > 1:
                        win_id = closest_nodes_indices[1]
                    else:
                        win_id = closest_nodes_indices[0]

                win_counts[win_id] += 1
                centers[win_id] += eta * (x_i.squeeze() - centers[win_id].squeeze())

            eta *= 0.99
        
        self.centers = centers
        dead_units_indices = np.where(win_counts == 0)[0]

        return centers, dead_units_indices
        
    # Second Stage.( Supervised Learning) -> Output weights are trained using supervised methods
    def fit_least_square(self, X, Y,X_validation, Y_validation):
        """
        Train the RBF network with batch and least squares approach
        """
        if self.centers is None:    #! change "== None" for "is None" better for arrays 
            # Choose the centers
            self.centers = self.choose_centers(X)
        
        #compute Phi -> n x N
        Phi = self._gaussian_rbf(X, self.centers)
        
        # Add the bias term
        Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))]) #! change dimmensions
        # Phi = np.vstack([Phi, np.ones((1,Phi.shape[1]))]) <-- OLD
       
        W = np.linalg.pinv(Phi) @ Y
        
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
        np.random.seed(42)
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
        n_samples = X.shape[0]
        self.weights = self.initialize_weights_random_uniform(Y.shape[1]) if weights is None else weights
        np.random.seed(42)

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(n_samples) #! added shuffle, is required in 3.2
            X_shuffled = X[shuffled_indices]
            Y_shuffled = Y[shuffled_indices]
            for i in range(n_samples):
                x_i = X_shuffled[i:i+1, :] #! change dimensions
                y_i = Y_shuffled[i:i+1, :]
                
                #compute Phi -> n x N
                Phi_i = self._gaussian_rbf(x_i, self.centers)
                
                # Add the bias term
                Phi_i_bias = np.hstack([Phi_i, np.ones((Phi_i.shape[0], 1))]) #! change dimensions
                #Phi_i = np.vstack([Phi_i, [[1]]])<-- OLD
                
                # Compute the output
                y_pred_i = Phi_i_bias @ self.weights #! change order of multiplication
                #y_pred_i = self.weights.T @ Phi_i <-- OLD             # (n_outputs, 1)
                
                # compute the error
                e = y_i - y_pred_i
                
                deltaW = learning_rate * Phi_i_bias.T @ e #! change dimensions
                
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
        Phi = self._gaussian_rbf(X, self.centers)  # (N_samples, n_hidden)
        Phi_with_bias = np.hstack([Phi, np.ones((Phi.shape[0], 1))]) #! change dimmensions
        #Phi = np.vstack([Phi, np.ones((1, Phi.shape[1]))])  # + bias <-- OLD
        y_pred = Phi_with_bias @ self.weights
        return y_pred #self.weights.T @ Phi   # (n_outputs, n_samples)