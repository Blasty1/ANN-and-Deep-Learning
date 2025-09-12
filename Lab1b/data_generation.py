import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#* FOR LAB1b PART I
def split_points(n = 100, class_array=None, percent=0.25):
    """
    Spliting datapoints for train and valid by a percentage of points from a class.

    Inputs:
    n : int - number of points in the class
    class_array : np.array - points of the class
    percent : float - percentage of points to delete

    Outputs:
    train_set : np.array - train points of the class after spliting
    valid_set : np.array - valid points of the class after spliting
    """
    num_delete = int(percent * n)
    np.random.seed(42)
    # Determine number of points (columns for 2D, elements for 1D)
    if class_array.ndim == 1:
        total_points = class_array.shape[0]
    else:
        total_points = class_array.shape[1]
    ids = np.random.choice(total_points, num_delete, replace=False)
    mask = np.ones(total_points, dtype=bool)
    mask[ids] = False
    if class_array.ndim == 1:
        train_set = class_array[mask]
        valid_set = class_array[~mask]
    else:
        train_set = class_array[:, mask]
        valid_set = class_array[:, ~mask]
    return train_set, valid_set

def generate_splited_data(n, percent_of_A = 0.25, percent_of_B = 0.25, task_d = False):
    """
    Generate two non-linear separable sets of points, class A and class B.
    ClassA is splitet in two groups. Function enable to split points from both classes to train and valid .

    Inputs:
    n : int - number of points per class
    percent_of_A : float - percentage of points to move from class A to valid set
    percent_of_B : float - percentage of points to move from class B to valid set
    task_d : bool - if True, move to valid set points from the two groups of class A 
        ? task d definition form pdf file: 20% from a subset of classA for which 
        ? classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0

    Outputs:
    trainA : np.array - points of class A in train set
    validA : np.array - points of class A in valid set
    trainB : np.array - points of class B in train set
    validB : np.array - points of class B in valid set

    """
    sigmaA = 0.2
    mA = np.array([1.0, 0.3])
    sigmaB = 0.3
    mB = np.array([0.0, -0.1])

    np.random.seed(32)
    groupA1 = np.random.randn(1, round(0.5 * n)) * sigmaA - mA[0]
    np.random.seed(42)
    groupA2 = np.random.randn(1, round(0.5 * n)) * sigmaA + mA[0]
    if task_d:
        new_n = int(0.5 * n)
        trainA1, validA1 = split_points(new_n, groupA1, 0.2)
        trainA2, validA2 = split_points(new_n, groupA2, 0.8)
        classA_x1 = np.concatenate((trainA1, trainA2), axis=1)
        classA_x1_valid = np.concatenate((validA1, validA2), axis=1)
    else:
        classA_x1 = np.concatenate((groupA1, groupA2), axis=1)

    

    np.random.seed(42)
    if task_d:        
        classA_x2 = np.random.randn(1, n) * sigmaA + mA[1]
        print(classA_x2)
        classA_x2_train = classA_x2[0][: new_n]
        classA_x2_valid = classA_x2[0][new_n :]
        trainA = np.vstack((classA_x1, classA_x2_train))
        valid_A = np.vstack((classA_x1_valid, classA_x2_valid))

    else:
        classA_x2 = np.random.randn(1, n) * sigmaA + mA[1]
        classA = np.vstack((classA_x1, classA_x2))

    np.random.seed(42)
    classB = np.random.randn(2, n) * sigmaB
    classB[0, :] += mB[0]
    classB[1, :] += mB[1]

    if percent_of_A > 0:
        trainA, valid_A = split_points(n, classA, percent_of_A)
    elif not task_d:
        trainA = classA
        valid_A = None
    if percent_of_B > 0:
        trainB, valid_B = split_points(n, classB, percent_of_B)
    else:
        trainB = classB
        valid_B = None

    return trainA.T, valid_A.T if valid_A is not None else valid_A, trainB.T, valid_B.T if valid_B is not None else valid_B

def plot_data(classA, classB, type = 'Splited data'):
    plt.figure(figsize=(10, 8))
    plt.scatter(classA[0, :], classA[1, :], c='red', alpha=0.7, label='Class A', s=50)
    plt.scatter(classB[0, :], classB[1, :], c='blue', alpha=0.7, label='Class B', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{type} Linearly Non-Separable Data for Binary Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


#* FOR Lab1B PART II
def generate_time_series_data():
    """
    Generate Mackey-Glass time series data.
    SK-learn MLPREGRESOR link https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    Outputs:
    X : array (1200, 5) - input vector in MLPRegressor compatible format
    y : array (1200, ) - output vector in MLPRegressor compatible format

    """
    N = 2000
    t_start = 300
    t_stop = 1500
    b   = 0.1
    c   = 0.2
    tau = 17

    time_series = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
        1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

    for n in range(17,N+99):
        time_series.append(time_series[n] - b*time_series[n] + c*time_series[n-tau]/(1+time_series[n-tau]**10))
    
    X = []
    y = []
    for t in range(t_start, t_stop):
        output_data = np.array(time_series[t])
        input_data = np.array([time_series[t-25], time_series[t-20], time_series[t-15], time_series[t-10], time_series[t-5]])
        X.append(input_data)
        y.append(output_data)

    #* change for sk-learn MLPRegressor compatible format
    #* X must be shape (n_samples, n_features) X.shape = (1200, 5)
    #* y must be shape (n_samples, ) y.shape = (1200, )
    X = np.array(X)
    y = np.array(y)
    # plt.plot(np.arange(300, 1500), X[:, 0])
    # plt.show()
    return X, y

def split_data_for_train_valid_test(X, y, n = 1200, nr_valid=200, nr_test=200):
    """
    Split data for subsets: train 800 samples, valid 200 samples, test 200 samples.
    
    Inputs:
    X : array (1200, 5) - input vector in MLPRegressor compatible format
    y : array (1200, ) - output vector in MLPRegressor compatible format
    n : int - total number of samples
    nr_valid : int - number of samples in valid set
    nr_test : int - number of samples in test set
    
    Outpust:
    train_X : array (800, 5) - input vector for train set
    train_y : array (800, ) - output vector for train set
    valid_X : array (200, 5) - input vector for valid set
    valid_y : array (200, ) - output vector for valid set
    test_X : array (200, 5) - input vector for test set
    test_y : array (200, ) - output vector for test set
    
    """
    #shuffle=False to keep the time series order
    train_X, test_X, train_y, test_y = train_test_split( X,y , random_state=104,test_size=float(nr_test/n), shuffle=False)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, random_state=104,test_size=float(nr_valid/(n-nr_test)), shuffle=False)
    
    # print("train X shape: ", train_X.shape, "valid X shape: ", valid_X.shape, " test X shape: ", test_X.shape)
    return train_X, train_y, valid_X, valid_y, test_X, test_y

#! EXAMPLE OF USEGE
# if __name__ == "__main__":

# #     X, y = generate_time_series_data()
#     # train_X, train_y, valid_X, valid_y, test_X, test_y = split_data_for_train_valid_test(X, y)
#     n = 100
#     trainA, validA, trainB, validB = generate_splited_data(n, 0.0, 0.0)
#     classA = np.hstack((trainA, validA)) if validA is not None else trainA
#     classB = np.hstack((trainB, validB)) if validB is not None else trainB
#     print(classA.shape, classB.shape)
#     plot_data(classA, classB, type='Splited ALL')

# #     trainA, validA, trainB, validB = generate_splited_data(n, 0.25, 0.25)
#     # For show all points
#     classA = np.hstack((trainA, validA)) if validA is not None else trainA
#     classB = np.hstack((trainB, validB)) if validB is not None else trainB
#     plot_data(classA, classB, type='Splited 75%A 75%B')
#     # For show only train points
#     plot_data(trainA, trainB, type='Splited 75%A 75%B')
    

#     trainA, validA, trainB, validB = generate_splited_data(n, 0.5, 0.0)
    
#     classA = np.hstack((trainA, validA)) if validA is not None else trainA
#     classB = np.hstack((trainB, validB)) if validB is not None else trainB
#     plot_data(classA, classB, type='Splited 50%A')

#     trainA, validA, trainB, validB = generate_splited_data(n, 0.0, 0.5)
    
#     classA = np.hstack((trainA, validA)) if validA is not None else trainA
#     classB = np.hstack((trainB, validB)) if validB is not None else trainB
#     plot_data(classA, classB, type='Splited 50%B')

#     trainA, validA, trainB, validB = generate_splited_data(n, 0.0, 0.0, task_d=True)
    
#     classA = np.hstack((trainA, validA)) if validA is not None else trainA
#     classB = np.hstack((trainB, validB)) if validB is not None else trainB
#     plot_data(trainA, trainB, type='Splited Task d')