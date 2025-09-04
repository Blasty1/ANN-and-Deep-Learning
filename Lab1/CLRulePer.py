import numpy as np
import matplotlib.pyplot as plt

def active_function(z):
    return (z>=0).astype(int)

def cl_online(data, labels, etas, epochs, weights):
    mse_curves = {} # k= eta, v= mean square error
    #iterate through each learning rate value
    for eta in etas:
        w = weights.copy()
        mse_hist = [] #to store the values of mis & mse
        for _ in range(epochs):
            sqerr=0.0
            mis = 0
            for j in range(data.shape[1]):
                y = active_function(np.dot(w, data[:,j]))
                e = y - labels[j]
                sqerr+=e**2
                if e<0:
                    w+=eta*data[:,j] #the case where the target is positive but the result is negative
                elif e>0:
                    w-=eta*data[:,j] #the case where the target is negative but the result is positive
            mse_hist.append(sqerr / data.shape[1])
        mse_curves[eta] = np.array(mse_hist)
    return mse_curves

def draw_mse_cl(epochs, etas, mse):
    plt.figure()
    nepochs = np.arange(1, epochs + 1)
    for eta in etas:
        plt.plot(nepochs, np.array(mse[eta]), label=f"η={eta}")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Classical Rule (Online) – MSE vs Epochs")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

