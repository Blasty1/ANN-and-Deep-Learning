import numpy as np
import matplotlib.pyplot as plt



def generate_overlaping_data(n):
    """
    Generate two overlaping sets of points, class A and class B.

    Inputs:
    n : int - number of points per class

    Outputs:
    classA : np.array - points of class A
    classB : np.array - points of class B

    """
    sigmaA = 1.5
    mA = np.array([3.5, 1])
    sigmaB = 1.2
    mB = np.array([7.0, 2])

    np.random.seed(42)
    classA = np.random.randn(2, n) * sigmaA
    classA[0, :] += mA[0]
    classA[1, :] += mA[1]

    np.random.seed(42)
    classB = np.random.randn(2, n) * sigmaB
    classB[0, :] += mB[0]
    classB[1, :] += mB[1]

    return classA, classB

def delete_points(n = 100, class_array=None, percent=0.25):
    """
    Delete a percentage of points from a class.

    Inputs:
    n : int - number of points in the class
    class_array : np.array - points of the class
    percent : float - percentage of points to delete

    Outputs:
    new_class : np.array - points of the class after deletion
    """
    num_delete = int(percent * n)
    np.random.seed(42)
    ids = np.random.choice(n, num_delete, replace=False)
    new_class = np.delete(class_array, ids, axis=1)
    return new_class

def generate_splited_data(n, deleteA = 0.25, deleteB = 0.25, task_d = False):
    """
    Generate two non-linear separable sets of points, class A and class B.
    ClassA is splitet in two groups. Function enable to delete points from both classes.

    Inputs:
    n : int - number of points per class
    deleteA : float - percentage of points to delete from class A
    deleteB : float - percentage of points to delete from class B
    task_d : bool - if True, delete points from the two groups of class A 
        ? task d definition form pdf file: 20% from a subset of classA for which 
        ? classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0

    Outputs:
    classA : np.array - points of class A
    classB : np.array - points of class B

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
        groupA1 = delete_points(new_n, groupA1, 0.2)
        groupA2 = delete_points(new_n, groupA2, 0.8)

    classA_x1 = np.concatenate((groupA1, groupA2), axis=1)
    np.random.seed(42)
    if task_d:        
        classA_x2 = np.random.randn(1, new_n) * sigmaA + mA[1]
    else:
        classA_x2 = np.random.randn(1, n) * sigmaA + mA[1]
    classA = np.vstack((classA_x1, classA_x2))

    np.random.seed(42)
    classB = np.random.randn(2, n) * sigmaB
    classB[0, :] += mB[0]
    classB[1, :] += mB[1]

    if deleteA > 0:
        classA = delete_points(n, classA, deleteA)
    if deleteB > 0:
        classB = delete_points(n, classB, deleteB)
    
    return classA, classB

def plot_data(classA, classB, type = 'Overlaping'):
    plt.figure(figsize=(10, 8))
    plt.scatter(classA[0, :], classA[1, :], c='red', alpha=0.7, label='Classe A', s=50)
    plt.scatter(classB[0, :], classB[1, :], c='blue', alpha=0.7, label='Classe B', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{type} Non Linearly-Separable Data for Binary Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

def suffle_and_create_labels(classA, classB):
    """
    Create labels for the two classes. Suffle the points and the labels.

    Inputs:
    classA : np.array - points of class A
    classB : np.array - points of class B

    Outputs:
    dataset : np.array - suffled points of both classes
    labels : np.array - suffled labels for the points
    """
    nA = classA.shape[1]
    nB = classB.shape[1]
    labelsA = np.ones(nA)
    labelsB = -np.ones(nB)
    labels = np.hstack((labelsA, labelsB))
    dataset = np.hstack((classA, classB))
    np.random.seed(42)
    indices = np.random.permutation(nA + nB)
    dataset = dataset[:, indices]
    labels = labels[indices]
    return dataset, labels
    
#! Example of usage
if __name__ == "__main__":
    n = 100
    classA, classB = generate_overlaping_data(n)
    plot_data(classA, classB, type='Overlaping')

    classA, classB = generate_splited_data(n, 0.0, 0.0)
    plot_data(classA, classB, type='Splited ALL')

    classA, classB = generate_splited_data(n, 0.25, 0.25)
    plot_data(classA, classB, type='Splited 75%A 75%B')

    classA, classB = generate_splited_data(n, 0.5, 0.0)
    plot_data(classA, classB, type='Splited 50%A')

    classA, classB = generate_splited_data(n, 0.0, 0.5)
    plot_data(classA, classB, type='Splited 50%B')

    classA, classB = generate_splited_data(n, 0.0, 0.0, task_d=True)
    plot_data(classA, classB, type='Splited Task d')