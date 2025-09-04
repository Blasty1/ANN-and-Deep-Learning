import numpy as np
import matplotlib.pyplot as plt
from DeltaRulePer import *


###### Parameters
n = 100  # number of points
# Class A (mean and standard deviation)
mA = np.array([ 2.0, 1.0]) #(x,y)
sigmaA = 0.3 #how data are sparsed around the mean

# Class B (mean and standard deviation)
mB = np.array([0.0, 0.0]) #(x,y) centring the class B around (0, 0) to show the effect of bias=0
sigmaB = 0.3 #how data are sparsed around the mean

np.random.seed(42) # to have same values over different iterations

#Return n samples in the form (x,y) from the “standard normal” distribution ( gaussian distribution with mean=0 and deviation=1).
classA = np.random.randn(2, n) * sigmaA + mA.reshape(2,1)

np.random.seed(42) # to have same values over different iterations
classB = np.random.randn(2, n) * sigmaB + mB.reshape(2,1)

# data creation
X = np.hstack([classA, classB])
labelsDL = np.hstack([np.ones(n),-np.ones(n)])
labelsCL = np.hstack([np.ones(n),np.zeros(n)])
np.random.seed(42) # to have same values over different iterations

# Shuffling dei dati (equivalente a randperm in MATLAB)
indices = np.random.permutation(2*n)
X = X[:, indices]  # shuffling lungo le colonne
labelsDL = labelsDL[indices]
labelsCL = labelsCL[indices]

# Data Visualization For DL
plt.figure(figsize=(10, 8))
plt.scatter(X[0, labelsDL == 1], X[1, labelsDL == 1], c='red', alpha=0.7, label='Classe A', s=50)
plt.scatter(X[0, labelsDL == -1], X[1, labelsDL == -1], c='blue', alpha=0.7, label='Classe B', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly-Separable Data for Binary Classification Delta Rule')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

##### Applying Learning Rules

np.random.seed(42) # to have same values over different iterations

# W dimensions 
# Before the learning phase can be executed, the weights must be initialised (have initial values assigned).
# The normal procedure is to start with small random numbers drawn from the normal distribution with zero mean
sigmaN = 0.5 
W = np.random.randn(1,2) * sigmaN
print(W)


# Plot for Delta rule on batch mode without bias
epochs = 50
eta = 0.0001
mse = delta_batch(X, labelsDL, epochs, W, eta )

nepochs = np.arange(1, epochs + 1)
plt.figure()
plt.plot(nepochs, mse, label=f"Batch (η={eta})")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title(f"Delta Rule: Batch mode without bias and mB= [0.0, 0.0]")
plt.legend()
plt.grid(True, linestyle=":")
plt.show()

