import numpy as np
import matplotlib.pyplot as plt
from CLRulePer import *
from DeltaRulePer import * 

###### Parameters
n = 100  # number of points
# Class A (mean and standard deviation)
mA = np.array([2.0, 1.0]) #(x,y)
sigmaA = 2 #how data are sparsed around the mean

# Class B (mean and standard deviation)
mB = np.array([1.0, 0.0]) #(x,y)
sigmaB = 3 #how data are sparsed around the mean

np.random.seed(42) # to have same values over different iterations

#Return n samples in the form (x,y) from the “standard normal” distribution ( gaussian distribution with mean=0 and deviation=1).
classA = np.random.randn(2, n) * sigmaA + mA.reshape(2,1)

np.random.seed(42) # to have same values over different iterations
classB = np.random.randn(2, n) * sigmaB + mB.reshape(2,1)

# data creation
X = np.hstack([classA, classB])
labels = np.hstack([np.ones(n), np.zeros(n)])

np.random.seed(42) # to have same values over different iterations

# Shuffling dei dati (equivalente a randperm in MATLAB)
indices = np.random.permutation(2*n)
X = X[:, indices]  # shuffling lungo le colonne
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


# Data Visualization For CL
plt.figure(figsize=(10, 8))
plt.scatter(X[0, labelsCL == 1], X[1, labelsCL == 1], c='red', alpha=0.7, label='Classe A', s=50)
plt.scatter(X[0, labelsCL == 0], X[1, labelsCL == 0], c='blue', alpha=0.7, label='Classe B', s=50)
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

#adding new weight w0 for the bias
np.random.seed(42)
bias = np.random.randn(1,1)   
Wb = np.hstack([bias, W])     
print(Wb)

#adding a new row of ones for the bias
Xb = np.vstack([np.ones((1, X.shape[1])), X]) 

#Adjust the learning rate and study the convergence of the Delta rule learning in online mode
epochs = 50
etas = [0.001, 0.01]    # values of learning rate for stability in sequential delta

mseDl = delta_online_etas(Xb, labelsDL, epochs, Wb, etas)
mseCl = cl_online(Xb, labelsCL, etas, epochs, Wb)
draw_mse_dl(epochs,etas, mseDl)
draw_mse_cl(epochs, etas, mseCl)

# COMPARING DL BATCH MODE AND ONLINE MODE WITH TWO VALUES OF INIT WEIGHTS
eta = 0.0001
# Multiply initial weights by O.01 to have a small initial values
W_low = Wb*0.01

dl_mse_batch_low = delta_batch(Xb, labelsDL, epochs, W_low, eta)
dl_mse_onl_low = delta_online(Xb, labelsDL, epochs, W_low, eta)

nepochs = np.arange(1, epochs + 1)
# Plot with low initial value
plt.figure()
plt.plot(nepochs, dl_mse_onl_low,   label=f"Online (η={eta})")
plt.plot(nepochs, dl_mse_batch_low, label=f"Batch (η={eta})", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Delta Rule: Batch vs Online (Fixed η and low init weights)")
plt.legend()
plt.grid(True, linestyle=":")
plt.show()

dl_mse_batch_high = delta_batch(Xb, labelsDL, epochs, Wb, eta)
dl_mse_onl_high= delta_online(Xb, labelsDL, epochs, Wb, eta)
plt.figure()
plt.plot(nepochs, dl_mse_onl_high,   label=f"Sequential (η={eta})")
plt.plot(nepochs, dl_mse_batch_high, label=f"Batch (η={eta})", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Delta Rule: Batch vs Online (Fixed η and high init weights)")
plt.legend()
plt.grid(True, linestyle=":")
plt.show()


