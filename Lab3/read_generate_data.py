import numpy as np
import os
import matplotlib.pyplot as plt


def read_patterns(data_file_path = os.path.join('Lab3', 'data', 'pict.dat')):
    """
     Read data from file pict.dat
     
     Input: 
     data_file_path -> file path 

     Output:
     patterns -> np.array(11, 1024) 11 patterns with 1024 int valuse (1 or -1)

    """
    patterns = []
    file_name = data_file_path
    with open(file_name, 'r') as f:
        data = f.read()
        data = data.split(',')
        i = 0
        pattern = []
        for d in data:
            pattern.append(int(d))
            i = i+1
            if i == 1024:
                patterns.append(pattern)
                pattern = []
                i = 0
    return np.array(patterns)

def show_pattern(pattern_array, title, notShow=False, isTitle = True): 
    """ 
    Visualize pattern(array)

    Input: 
    pattern_array -> np.array (n_pixels)
    
    """
    side = int(np.sqrt(pattern_array.shape[0]))
    pattern_array = pattern_array.reshape((side, side))
    plt.figure(figsize=(3, 3))
    plt.imshow(pattern_array)
    if isTitle:
        plt.title(title)
    plt.axis('off')
    if not notShow:
        plt.show()
    else:
        return plt

def show_multiple_arrays(pattern_array, save_name = None):
    """ 
    Visualize all patterns(array)

    Input: 
    pattern_array -> np.array (n_patterns, m_pixels)
    
    """
    fig = plt.figure(figsize=(10, 3))
    for i in range(len(pattern_array)):
        side = int(np.sqrt(pattern_array[i].shape[0]))
        plt.subplot(1, len(pattern_array), i+1)
        plt.imshow(pattern_array[i].reshape((side, side)))
        plt.axis('off')
    if save_name != None:
        plt.tight_layout()
        plt.savefig(os.path.join('Lab3', 'plots_data', save_name))
    plt.show()

def add_noise(pattern, percent = 0.01):
    """
    Adding noise to one pattern.

    Input: 
    pattern -> np.array(1024,) one pattern
    percent -> int, percent for pixels to flip

    Output:
    noisy_pattern -> np.array(1024,) pattern with noise
    """
    noisy_pattern = pattern.copy()
    size = int(len(noisy_pattern)*percent)
    fllip_idx = np.random.choice(len(noisy_pattern), size = size, replace=False)
    noisy_pattern[fllip_idx] *= -1
    return noisy_pattern


def generate_sparse_patterns(N, P, rho):
    """
    Generate P sparse binnary patterns (0, 1) with size N, and activity rho.

    Input:
        N (int): Size of pattern
        P (int): Number of patterns
        rho (float): activity 

    Outputs:
        np.array: Tablica 2D o wymiarach (P, N) zawierająca wzorce (0, 1).
    """
    random_matrix = np.random.rand(P, N)
    sparse_patterns = (random_matrix < rho).astype(int)
    return sparse_patterns

def data_small_model():
    """
    Return data from background section of lab instruction.

    x1, x2, x3 -> np.arrays (8, ) 

    """
    x1 = np.array([-1,  -1,  1,  -1,  1,  -1,  -1,  1])
    x2 = np.array([-1,  -1,  -1,  -1,  -1,  1,  -1,  -1])
    x3 = np.array([-1,  1,  1,  -1,  -1,  1,  -1,  1])
    return x1, x2, x3


def data_small_model_new():
    """
    Return data from 3.1 section of lab instruction.

    x1d, x2d, x3d -> np.arrays (8, ) 
    
    """
    x1d = np.array([1,  -1,  1,  -1,  1,  -1,  -1,  1])
    x2d = np.array([1,  1,  -1,  -1,  -1,  1,  -1,  -1])
    x3d = np.array([1,  1,  1,  -1,  1,  1,  -1,  1])
    return x1d, x2d, x3d 

