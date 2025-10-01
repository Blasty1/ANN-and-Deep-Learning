import numpy as np
import matplotlib.pyplot as plt
from read_generate_data import read_patterns, generate_sparse_patterns, show_multiple_arrays
from utils.HopfieldNetwork import HopfieldNetwork
import os

print("======= 3.6 Sparse Patterns =======\n")

bias_values = [0.05, 0.1, 0.2, 0.3, 0.5]
colors = ['purple', 'yellow', 'blue', 'red', 'green']
#***********************************************************************************

print("======= Sparse patterns with activity 10% =======\n")

size = 1024
nr_of_patterns = 20
rho10 = 0.1
binary_patterns_10 = generate_sparse_patterns(size, nr_of_patterns, rho10)
show_multiple_arrays(binary_patterns_10[:6], 'sparse_data_10.png')
capacity_results_10 = {theta: [] for theta in bias_values}

for i in range(1, nr_of_patterns+1):    
    print(f'--- Test {i} patterns ---')
    training_patterns = binary_patterns_10[:i]
    hopfieldNN = HopfieldNetwork(size)
    hopfieldNN.train_sparse(training_patterns)
    biases_stabel = 0
    for bias in bias_values:
        stable_count = 0

        for pattern in training_patterns:
            next_state = hopfieldNN.recall_sparse(pattern, bias)
            if np.array_equal(pattern, next_state):
                stable_count += 1
        print(f'Bias {bias}, stable patterns: {stable_count}, is network stable {'Yes' if stable_count == i else 'No'}') 
        biases_stabel += 1 if stable_count == i else 0
        capacity_results_10[bias].append(stable_count)


plt.figure(figsize=(10, 6))
i = 0
for theta, stable_counts in capacity_results_10.items():
    P_axis = np.arange(1, nr_of_patterns + 1)
    plt.plot(P_axis, stable_counts, marker='.', label=f'θ = {theta:.2f}', color = colors[i])
    i += 1

plt.plot(P_axis, P_axis, 'k--', label='P_stable = P_trained (Ideal)') 

# plt.title(f'Sparse Hopfield Capacity vs. Bias (N={size}, ρ={rho10})')
plt.xlabel('Number of Trained Patterns, P')
plt.ylabel('Number of Stable Patterns')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join('Lab3', 'plots_3_6', 'sparse_capacity_for_10.png'))
plt.show()

#***********************************************************************************

print("\n======= Sparse patterns with activity 5% =======\n")


size = 1024
nr_of_patterns = 20
rho5 = 0.05
binary_patterns_5 = generate_sparse_patterns(size, nr_of_patterns, rho5)
show_multiple_arrays(binary_patterns_5[:6], 'sparse_data_5.png')
capacity_results_5 = {theta: [] for theta in bias_values}

for i in range(1, nr_of_patterns+1):    
    print(f'--- Test {i} patterns ---')
    training_patterns = binary_patterns_5[:i]
    hopfieldNN = HopfieldNetwork(size)
    hopfieldNN.train_sparse(training_patterns)
    biases_stabel = 0
    for bias in bias_values:
        stable_count = 0

        for pattern in training_patterns:
            next_state = hopfieldNN.recall_sparse(pattern, bias)
            if np.array_equal(pattern, next_state):
                stable_count += 1
        print(f'Bias {bias}, stable patterns: {stable_count}, is network stable {'Yes' if stable_count == i else 'No'}') 
        biases_stabel += 1 if stable_count == i else 0
        capacity_results_5[bias].append(stable_count)

plt.figure(figsize=(10, 6))
i=0
for theta, stable_counts in capacity_results_5.items():
    P_axis = np.arange(1, nr_of_patterns + 1)
    plt.plot(P_axis, stable_counts, marker='.', label=f'θ = {theta:.2f}', color = colors[i])
    i += 1

plt.plot(P_axis, P_axis, 'k--', label='P_stable = P_trained (Ideal)') 

# plt.title(f'Sparse Hopfield Capacity vs. Bias (N={size}, ρ={rho5})')
plt.xlabel('Number of Trained Patterns, P')
plt.ylabel('Number of Stable Patterns')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join('Lab3', 'plots_3_6', 'sparse_capacity_for_5.png'))
plt.show()

#***********************************************************************************

print("\n======= Sparse patterns with activity 1% =======\n")
 
size = 1024
nr_of_patterns = 20
rho1 = 0.01
binary_patterns_1 = generate_sparse_patterns(size, nr_of_patterns, rho1)
show_multiple_arrays(binary_patterns_1[:6], 'sparse_data_1.png')
capacity_results_1 = {theta: [] for theta in bias_values}

for i in range(1, nr_of_patterns+1):    
    print(f'--- Test {i} patterns ---')
    training_patterns = binary_patterns_1[:i]
    hopfieldNN = HopfieldNetwork(size)
    hopfieldNN.train_sparse(training_patterns)
    biases_stabel = 0
    for bias in bias_values:
        stable_count = 0

        for pattern in training_patterns:
            next_state = hopfieldNN.recall_sparse(pattern, bias)
            if np.array_equal(pattern, next_state):
                stable_count += 1
        print(f'Bias {bias}, stable patterns: {stable_count}, is network stable {'Yes' if stable_count == i else 'No'}') 
        biases_stabel += 1 if stable_count == i else 0
        capacity_results_1[bias].append(stable_count)

plt.figure(figsize=(10, 6))
i=0
for theta, stable_counts in capacity_results_1.items():
    P_axis = np.arange(1, nr_of_patterns + 1)
    plt.plot(P_axis, stable_counts, marker='.', label=f'θ = {theta:.2f}', color = colors[i])
    i += 1

plt.plot(P_axis, P_axis, 'k--', label='P_stable = P_trained (Ideal)') # Diagonal line

# plt.title(f'Sparse Hopfield Capacity vs. Bias (N={size}, ρ={rho5})')
plt.xlabel('Number of Trained Patterns, P')
plt.ylabel('Number of Stable Patterns')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join('Lab3', 'plots_3_6', 'sparse_capacity_for_1.png'))
plt.show()