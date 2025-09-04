import numpy as np
import matplotlib.pyplot as plt


#DELTA RULE ONLINE
def delta_online(data, labels, epochs, weights, eta):
        w = weights.copy() 
        mse_hist = []
        for _ in range(epochs):
            sqerr=0.0
            for j in range(data.shape[1]):
                y = np.dot(w, data[:, j])  
                err = labels[j]-y
                sqerr+=err**2
                w += eta*err*data[:,j]
            mse_hist.append(sqerr / data.shape[1])
        return mse_hist

#DELTA RULE ONLINE MODE WITH DIIFERENT ETA VALUES
def delta_online_etas(data, labels, epochs, weights, etas):
    mse_dict = {} # k= eta, v= mean square error
    for eta in etas:
        mse_dict[eta] = delta_online(data, labels, epochs, weights, eta)
    return mse_dict



#DELTA RULE BATCH MODE 
def delta_batch(data, labels, epochs, weights,eta): 
    mse_hist = []
    w = weights.copy()
    for _ in range(epochs):
        w_accum = np.zeros(w.shape[1])
        sqerr=0.0
        for j in range(data.shape[1]):
            y = np.dot(w, data[:, j]) 
            err = labels[j]-y[0]
            sqerr+=err**2
            w_accum += eta*err*data[:,j]
        w += w_accum 
        mse_hist.append(sqerr / data.shape[1])
    return mse_hist

#DRAW PLOTS

def draw_mse_dl(epochs, etas,mse):
    plt.figure()
    nepochs = np.arange(1, epochs + 1)
    for eta in etas:    
        plt.plot(nepochs, np.array(mse[eta]), label=f"η={eta}")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Delta Rule (Online) – MSE vs Epochs")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

