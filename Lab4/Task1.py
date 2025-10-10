from util import *
from rbm import RestrictedBoltzmannMachine
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ndim_visible = image_size[0]*image_size[1]
    ndim_hidden = 500
    batch_size= 20 # from the text
    rbm = RestrictedBoltzmannMachine(ndim_visible=ndim_visible,is_bottom=True, ndim_hidden=ndim_hidden, batch_size=batch_size,image_size=image_size)
    
    
    n_samples = train_imgs.shape[0]  # 60,000
    n_epochs = 20 # from the text
    iterations_per_epoch = n_samples // batch_size # 3000
    n_iterations = n_epochs * iterations_per_epoch  # 45,000

    rbm.cd1(train_imgs,n_iterations=n_iterations)
    
    plot_reconstruction_error(rbm)

    