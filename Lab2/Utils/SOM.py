import numpy as np

class SOM:
    def __init__(self, grid_dimension, input_dimension, units_number, sigma0=1, epochs=20, lr=0.2, circular=False):
        self.grid_dimension = grid_dimension
        self.input_dimension = input_dimension
        self.units_number = units_number       # number of used units
        self.cols = int(np.sqrt(self.units_number))
        self.rows = int(self.units_number / self.cols)
        self.lr = lr
        self.sigma0 = sigma0      # the initial radius for sigma used in neighborhood function
        self.epochs = epochs
        self.W = self._init_weights()
        self.circular = circular
        self.t = 0
        self.total_steps = None    # will be set in the train method

    def _init_weights(self):
        sigma = 0.5
        np.random.seed(42)
        weights = np.random.randn(self.input_dimension, self.units_number) * sigma
        return weights

    def select_BMU(self, X):
        # Selects the best matching unit
        bmu = np.argmin([np.sum((X - self.W[:, i])**2) for i in range(self.units_number)])
        return bmu
 

    def neighborhood_function(self, distance):
        tau = max(1, self.total_steps)
        sigma = self.sigma0 * np.exp(-self.t / tau)
        return np.exp(-(distance**2) / (2 * sigma**2 ))
    
    def grid_distance(self, i, j):
        ri, ci = divmod(i, self.cols)  # To find the coordiantes of a certain unit in the grid
        rj, cj = divmod(j, self.cols)
        return abs(ri-rj) + abs(ci-cj)
    

    def update_weights(self, X, bmu):
        for i in range(self.units_number):
            if self.grid_dimension == 1:
                if self.circular:
                    distance = min(abs(bmu - i), self.units_number-abs(bmu-i))
                else:
                    distance = abs(bmu - i)
            else:
                distance = self.grid_distance(i, bmu)
            h = self.neighborhood_function(distance)
            self.W[:, i] += self.lr * h * (X - self.W[:, i])

    def train(self, X):
        M = X.shape[1]
        self.total_steps = self.epochs * M
        for epoch in range(self.epochs):
            for i in range(M):
                bmu = self.select_BMU(X[:, i])
                self.update_weights(X[:, i], bmu)
                self.t += 1
