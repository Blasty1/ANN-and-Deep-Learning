from PerceptronLearningRule import *
from DeltaRule import *
import numpy as np
import matplotlib.pyplot as plt

###### Parameters
n = 100  # number of points
# Class A (mean and standard deviation)
mA = np.array([4.0, 1]) #(x,y)
sigmaA = 1 #how data are sparsed around the mean

# Class B (mean and standard deviation)
mB = np.array([-4.0, -1]) #(x,y)
sigmaB = 1 #how data are sparsed around the mean

np.random.seed(42) # to have same values over different iterations

#Return n samples in the form (x,y) from the “standard normal” distribution ( gaussian distribution with mean=0 and deviation=1).
classA = np.random.randn(2, n) * sigmaA + mA.reshape(2,1)

np.random.seed(42) # to have same values over different iterations
classB = np.random.randn(2, n) * sigmaB + mB.reshape(2,1)

# data creation
X = np.hstack([classA, classB])
labels = np.hstack([np.ones(n),np.zeros(n)])  # 1 for classA and 0 for classB

np.random.seed(42) # to have same values over different iterations

# Shuffling dei dati (equivalente a randperm in MATLAB)
indices = np.random.permutation(2*n)
X = X[:, indices]  # shuffling lungo le colonne
labels = labels[indices]

# Data Visualization
plt.figure(figsize=(10, 8))
plt.scatter(X[0, labels == 1], X[1, labels == 1], c='red', alpha=0.7, label='Classe A', s=50)
plt.scatter(X[0, labels == 0], X[1, labels == 0], c='blue', alpha=0.7, label='Classe B', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly-Separable Data for Binary Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()


##### Applying all the 4 Learning Rules

np.random.seed(45) # to have same values over different iterations

# W dimensions 
# Before the learning phase can be executed, the weights must be initialised (have initial values assigned).
# The normal procedure is to start with small random numbers drawn from the normal distribution with zero mean
sigmaN = 0.5 
W = np.random.randn(1,2) * sigmaN
bias = np.random.randn() * sigmaN
eta=0.01
classicalLearningRuleBatch(X,labels,W,bias,500,eta)

np.random.seed(45) # to have same values over different iterations

# W dimensions 
# Before the learning phase can be executed, the weights must be initialised (have initial values assigned).
# The normal procedure is to start with small random numbers drawn from the normal distribution with zero mean
sigmaN = 0.5 
W = np.random.randn(1,2) * sigmaN*500
bias = np.random.randn() * sigmaN
eta=0.01
classicalLearningRuleOnline(X,labels,W,bias,500,eta)
    


np.random.seed(45) # to have same values over different iterations

sigmaN = 0.5 
W = np.random.randn(1,2) * sigmaN
bias = np.random.randn() * sigmaN
labels = np.hstack([np.ones(n),-np.ones(n)])  # 1 for classA and -1 for classB
labels = labels[indices]
eta=0.01
deltaRuleBatch(X,labels,W,bias,20,eta)


np.random.seed(45) # to have same values over different iterations

sigmaN = 0.5 
W = np.random.randn(1,2) * sigmaN
bias = np.random.randn() * sigmaN
labels = np.hstack([np.ones(n),-np.ones(n)])  # 1 for classA and -1 for classB
labels = labels[indices]
eta=0.01
deltaRuleOnline(X,labels,W,bias,20,eta)