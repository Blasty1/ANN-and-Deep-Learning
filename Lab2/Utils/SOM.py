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
        weights = np.random.rand(self.input_dimension, self.units_number) * sigma
        return weights

    def select_BMU(self, X):
        # Selects the best matching unit
        bmu = np.argmin([np.sum((X - self.W[:, i])**2) for i in range(self.units_number)])
        return bmu
 

    def neighborhood_function(self, distance):
        tau = max(1, self.total_steps / np.log((self.sigma0+1e-9) / 0.5))  
        sigma = max(1e-3, self.sigma0 * np.exp(-self.t / tau))
        return np.exp(-(distance**2) / (2 * sigma**2))
    
    def grid_distance(self, i, j):
        if self.grid_dimension == 1:
            if self.circular:
                # For circular topology: distance between nodes i and j
                direct_distance = abs(i - j)
                wraparound_distance = self.units_number - direct_distance
                return min(direct_distance, wraparound_distance)
            else:
                return abs(i - j)
        else:
            # For 2D grid
            ri, ci = divmod(i, self.cols)
            rj, cj = divmod(j, self.cols)
            return abs(ri-rj) + abs(ci-cj)

    def update_weights(self, X, bmu):
        for i in range(self.units_number):
            distance = self.grid_distance(i, bmu)  # Use unified grid_distance method
            h = self.neighborhood_function(distance)
            self.W[:, i] += self.lr * h * (X - self.W[:, i])

    def train(self, X):
        M = X.shape[1]
        self.total_steps = self.epochs * M
        bmu_epoch ={}
        for epoch in range(self.epochs):
            bmus = []
            for i in range(M):
                bmu = self.select_BMU(X[:, i])
                self.update_weights(X[:, i], bmu)
                bmus.append(bmu)
                self.t += 1
            bmu_epoch[epoch+1] = bmus
        return bmu_epoch
