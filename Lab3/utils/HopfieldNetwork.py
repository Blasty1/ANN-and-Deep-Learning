from read_generate_data import show_pattern
import numpy as np
class HopfieldNetwork:
    def __init__(self,num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons,num_neurons))
    
    def energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))

    @staticmethod
    def binary_to_bipolar(x):
        return np.where(x == 0, -1, 1)

    @staticmethod
    def bipolar_to_binary(x):
        return np.where(x == -1, 0, 1)
    
    def train(self,patterns,self_connections=False):
        """ Train the Hopfield Network using Hebbian learning rule.

        Args:
            patterns (_type_): _description_
            self_connections (bool, optional): Diagonal of the weight matrix ( 0 by default ). Defaults to False.
        """
        for p in patterns:
            p = p.reshape(-1,1)
            self.weights += np.dot(p,p.T)
        
        self.weights /= len(patterns)
        
        if(self_connections==False):
            np.fill_diagonal(self.weights,0) # No self-connections
    
    ## little_model
    def recall_synchronously(self,pattern,max_iterations=100,bipolar_coding=True):
        state = pattern.copy()
        for _ in range(max_iterations):
            new_state = state.copy()
            if bipolar_coding:
                new_state = np.sign(np.dot(self.weights,state))
                new_state[new_state==0] = 1 # Handle zero case
            else:
                new_state = np.where(np.dot(self.weights,state)>=0,1,0)
                
            if np.array_equal(new_state,state):
                #convergence reached
                break
            state = new_state
        return state


    def recall_asynchronously(self,pattern,max_iterations=100,random_order=False,bipolar_coding=True):
        state = pattern.copy()
        indices = np.arange(self.num_neurons)
            
        for iteration in range(max_iterations):
            if random_order:
                indices = np.random.permutation(self.num_neurons)

            prev_state = state.copy()
            for i in indices:
                net_input = np.dot(self.weights[i],state)
                if bipolar_coding:
                    state[i] = 1 if net_input >= 0 else -1
                else:
                    state[i] = 1 if net_input >= 0 else 0
                    
            if random_order and iteration%100 == 0:
                show_pattern(state,f"Result at {iteration}th iteration")
                
            if np.array_equal(state,prev_state):
                #convergence reached
                return state
        return state
    
    def recall(self, pattern,max_iterations=100,random_order=False,synchronous=True,bipolar_coding=True):
        """ Recall a pattern from the Hopfield Network.
        Args:
            pattern (_type_): _description_
            max_iterations (int, optional): _description_. Defaults to 100.
            random_order (bool, optional): True if we want to shuffle the indices at each iteration of the asynchronous recall method. Defaults to False.
            synchronous (bool, optional): True for synchronous recall or false for asynchronous. Defaults to True.
            bipolar_coding (bool, optional): True if patterns are in bipolar coding otherwise false for binary coding Defaults to True.

        Returns:
            _type_: return the new state
        """
        if synchronous:
            return self.recall_synchronously(pattern,max_iterations,bipolar_coding)
        else:
            return self.recall_asynchronously(pattern,max_iterations,random_order,bipolar_coding)
    
    def find_all_attractors(hopfield_net, num_random_tests=5000):
        """
        Systematically search for attractors by testing many patterns.
        An attractor is a stable state - a pattern that doesn't change when recalled.
        """
        
        attractor_set = set()
        # Test many random patterns
        np.random.seed(42)
        
        for _ in range(num_random_tests):
            random_pattern = np.random.choice([-1, 1], size=8)
            result = hopfield_net.recall(random_pattern)
            attractor_set.add(tuple(result))
        
        # Convert to list
        attractors = [np.array(attr) for attr in attractor_set]
        return attractors

    # for future tasks
    def add_noise(self,pattern,noise_level):
        pass