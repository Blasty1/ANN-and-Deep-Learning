import numpy as np

def activation_function(y_pre):
    y_post = y_pre.copy()
    y_post[y_post > 0] = 1 
    y_post[y_post <= 0] = 0

def classicalLearningRuleOnline(X, targets,weights,max_epochs=50):
    eta = 3 #hyperparameter

    #iterate through the wholde dataset
    for epoch in range(max_epochs):
        changed=0
        #iterate through each sample
        for i in range(X.shape[1]):
            x_i = X[:,i] #point Xi
            target_i = targets[i]
            bias=weights[0].reshape(-1, 1)
            y_prethreshold = np.traspose(weights) * x_i  + bias
            y_afterthreshold = activation_function(y_prethreshold)

            error = y_afterthreshold - target_i 
            updated_weights=eta*error*x_i
            if(updated_weights != 0):
                weights = weights + eta*error*x_i
                changed=1
        
        # nothing has changed from the last epoch, the convergence has been reached
        if changed == 0:
            break
    