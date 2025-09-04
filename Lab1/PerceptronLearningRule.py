import numpy as np
import matplotlib.pyplot as plt

def activation_function(y_pre):
    y_post = y_pre.copy()
    y_post[y_post > 0] = 1 
    y_post[y_post <= 0] = 0
    
    return y_post

def print_decision_boundary(X,targets,weights,bias,epoch_number):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[0, targets == 1], X[1, targets == 1], c='red',label='Class A')
    plt.scatter(X[0, targets == 0], X[1, targets == 0], c='blue', label='Class B')
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
        plt.title(f"Initial conditions")
    else:
        plt.title(f"Decision boundary after epoch {epoch_number}")
    
    plt.legend()
    plt.grid(True)
    plt.show()


def classicalLearningRuleBatch(X, targets,weights,bias,max_epochs=100):
    eta = 0.05 #hyperparameter
    
    print_decision_boundary(X,targets,weights,bias,0)

    #iterate through the wholde dataset
    for epoch in range(max_epochs):
        changed=0
        weightsAccum = np.zeros(weights.shape)
        
        #iterate through each sample
        for i in range(X.shape[1]):
            x_i = X[:,i] #point Xi
            target_i = targets[i]

            y_prethreshold = weights @  x_i  + bias
            y_afterthreshold = activation_function(y_prethreshold)

            error = target_i - y_afterthreshold 
            
            updated_weights=eta*error*x_i.reshape(1,2)
            
            #we want to check if all the elements of the weights has not changed
            if(not (updated_weights == 0).all()):
                weightsAccum = weightsAccum + updated_weights
                changed=1
        weights = weights + weightsAccum
        
        # nothing has changed from the last epoch, the convergence has been reached
        if changed == 0:
            break
    
        print_decision_boundary(X,targets,weights,bias,epoch)





def classicalLearningRuleOnline(X, targets,weights,bias,max_epochs=100):
    eta = 0.05 #hyperparameter
    print_decision_boundary(X,targets,weights,bias,0)

    #iterate through the wholde dataset
    for epoch in range(max_epochs):
        changed=0
        #iterate through each sample
        for i in range(X.shape[1]):
            x_i = X[:,i] #point Xi
            target_i = targets[i]

            y_prethreshold = weights @  x_i  + bias
            y_afterthreshold = activation_function(y_prethreshold)

            error = target_i - y_afterthreshold 
            
            updated_weights=eta*error*x_i.reshape(1,2)
            
            #we want to check if all the elements of the weights has not changed
            if(not (updated_weights == 0).all()):
                weights = weights + updated_weights
                changed=1
        
        # nothing has changed from the last epoch, the convergence has been reached
        if changed == 0:
            break
    
        print_decision_boundary(X,targets,weights,bias,epoch+1)


