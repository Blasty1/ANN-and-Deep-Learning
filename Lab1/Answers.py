#ANSWERS 3-1-2:

"""
1. Perceptron vs Delta Learning Rule

        Perceptron Learning Rule: The error quickly stabilizes around 0.5, showing that the perceptron rule does not effectively 
        minimize the mean squared error. This happens because it only corrects classification mistakes in a binary way,
        without considering how far data points are from the decision boundary.

        Delta Learning Rule: The error decreases from 0.075 to about 0.05. Since the delta rule is based on gradient 
        descent, it directly minimizes the squared error, which results in better convergence.

2. Delta Rule: Batch vs Online
    We compare batch and online learning under two different initial weight conditions:

        -Higher initial weights:
            -During the first epochs (1–6), the batch mode converges faster than the online mode.
            -After epoch 6, the two curves overlap and converge to MSE ≈ 0.05 within 10 epochs.

        -Low initial weights:
            -Both models converge smoothly to MSE ≈ 0.05, starting around epoch 7.
            -Convergence is slightly faster than in the case of higher initial weights.

3. Effect of Bias and Data Distribution

        -If we remove the bias but data points remain in different quadrants of the coordinate system, the system is 
        still able to converge to a low MSE value (= 0.05), because a boundary line passing through the origin (0,0) 
        can still separate the two classes.

        -However, when we modify class parameters so that both class samples lie in the same quadrant, the model fails
        to converge to a low MSE. Even after 200 epochs, the error remains high, showing that no linear boundary through
        the origin can separate the classes in this case.
		
"""