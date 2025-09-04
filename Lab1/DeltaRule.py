import numpy as np
import matplotlib.pyplot as plt

def print_decision_boundary(X,targets,weights,bias,epoch_number,name):
    plt.figure(figsize=(10, 8))

    plt.scatter(X[0, targets == 1], X[1, targets == 1], c='red',label='Class A')
    plt.scatter(X[0, targets == -1], X[1, targets == -1], c='blue', label='Class B')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # nice axis limits
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    ymin = X[1,:].min()
    ymax = X[1,:].max()
    pad = 0.1 * max(xmax - xmin, ymax - ymin)
    xlim = (xmin - pad, xmax + pad)
    ylim = (ymin - pad, ymax + pad)

    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    """
        Draws the decision boundary defined by w[0]*x + w[1]*y + b = 0
    """
    if abs(weights[0,1]) > 1e-12:  # non-vertical line
        xs = np.array(xlim)
        ys = -(weights[0,0] * xs + bias) / weights[0,1]
        ax.plot(xs, ys, label="decision boundary")
    else:  # vertical line case
        x0 = - bias/ weights[0,0]
        ax.plot([x0, x0], ylim, label="decision boundary")

    if(epoch_number == 0):
        plt.title(f"{name} - Initial conditions")
    else:
        plt.title(f"{name} - Decision boundary after epoch {epoch_number}")
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
def deltaRuleBatch(X, targets,weights,bias,max_epochs,eta):
    mse_hist = []
    eta = 0.05 #hyperparameter
    print_decision_boundary(X,targets,weights,bias,0,"Delta Rule Batch")

    #iterate through the wholde dataset
    for epoch in range(max_epochs):
        weightsAccum = np.zeros(weights.shape)
        biasAccum = 0
        sqerr=0
        #iterate through each sample
        for i in range(X.shape[1]):
            x_i = X[:,i] #point Xi
            target_i = targets[i]

            y_prethreshold = weights @  x_i  + bias

            error = target_i - y_prethreshold 
            
            updated_weights=eta*error*x_i.reshape(1,2)
            sqerr += error**2
            
            #we want to check if all the elements of the weights has not changed
            weightsAccum = weightsAccum + updated_weights
            biasAccum = biasAccum + eta*error
 
        mse_hist.append(sqerr / X.shape[1])
    
        weights = weights + weightsAccum
        bias = bias + biasAccum
    
        print_decision_boundary(X,targets,weights,bias,epoch+1,"Delta Rule Batch")
    return mse_hist

def deltaRuleOnline(X, targets,weights,bias,max_epochs,eta):
    eta = 0.05 #hyperparameter
    mse_hist = []
    print_decision_boundary(X,targets,weights,bias,0,"Delta Rule Online")
    
    #iterate through the wholde dataset
    for epoch in range(max_epochs):
        sqerr=0
        
        #iterate through each sample
        for i in range(X.shape[1]):
            x_i = X[:,i] #point Xi
            target_i = targets[i]

            y_prethreshold = weights @  x_i  + bias
            
            error = target_i - y_prethreshold 
            sqerr += error**2
            updated_weights=eta*error*x_i.reshape(1,2)
            
            weights = weights + updated_weights
            bias = bias + eta*error

        mse_hist.append(sqerr/X.shape[1])
        print_decision_boundary(X,targets,weights,bias,epoch+1, "Delta Rule Online")
    return mse_hist