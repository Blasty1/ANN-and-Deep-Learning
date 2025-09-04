#ANSWERS 3-1-2:

"""
1- Perceptron Learning rule:
    - For all the learning rate values tested the  the model converge to 1
    - This confirms the Perceptron Convergence Theorem, that guarantees the model convergence if a solution exists after a finite number of steps independent of the learning rate values
    - The effect of the learning rate value remains on the speed of convergence: small Lr => slower convergence , big Lr => faster convergence
    Delta Leraning rule:
        - More learning rate value is small, more the algorithm reaches a low final MSE 
        - With larger learning rate the algorithm still finds a separating boundary, but it tends to overshoot the minimum of the error surface leading to a higher value in the MSE curve.
        - This is a result of the delta rule form of gradient descent: 
            small Lr => small steps => closer to the true minimum with slower convergence
            big Lr => bigger steps =>fast converge but risk of missing the exact minimum or oscillating around it

2- Delta Rule: Batch vs Online:
    In this part we draw plots with two different initialisation weights:
    - Small initial weights:  
        - [1, 2] epochs: batch mode converges faster than online mode and by epoch 2, both methods reach the same MSE
	    - After epoch 2: the curves overlap with the same performance and converges after 8 epochs

    - High initial weights:
        - The curves ovelap and converge smoothly after 100 epochs
    - Batch mode update weights using accumulated error over the whole dataset, which make the updates faster on the first epochs
    - With high initial weights the model generates values that are far from the expected ones, which makes the convergence slower 

3- 
    - If we remove the bias but the data points are still on different quadrants on the coordinate system, the system still able to converge into a low value of MSE (=0.05) because we still able to find a boundary line that cross the coordinate origin (0, 0)  and separates the two classes.
    - After the modification of mB in order to center the class B on the coordinate origin, the model wasn’t able to converge into a low MSE value. After 50 epochs the model still got MSE=0.502
		
"""