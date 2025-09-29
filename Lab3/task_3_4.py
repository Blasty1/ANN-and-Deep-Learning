import numpy as np
import matplotlib.pyplot as plt
from read_generate_data import read_patterns, add_noise, show_pattern, show_multiple_arrays
from utils.HopfieldNetwork import HopfieldNetwork
import os


def count_convergence_probabilty(orginal_p, hopfield_network, iterations = 1000, noise = 0):
    correct = 0
    reverse = 0
    reverse_orginal = orginal_p * -1
    for i in range(iterations):
        noisy_p = add_noise(orginal_p, noise/100)
        recalled = hopfield_network.recall(noisy_p, 100)
        if np.array_equal(recalled, orginal_p):
            correct +=1
        if np.array_equal(recalled, reverse_orginal):
            reverse +=1
    return float(correct/iterations), float(reverse/iterations)


print("======= 3.4 Distortion Resistance =======\n")

patterns = read_patterns()
p1 = patterns[0]
p2 = patterns[1]
p3 = patterns[2]

show_multiple_arrays(patterns[:3])

print("==== Iteration on all noise levels ====\n")


hopfieldNN = HopfieldNetwork(len(p1))
hopfieldNN.train(patterns[:3])

noise_levels = np.array(range(101))
retults = {'p1': {'P': [], 'RP': []}, 'p2': {'P': [], 'RP': []}, 'p3': {'P': [], 'RP': []}}
for noise in noise_levels:
    p1_probability, p1_reverse = count_convergence_probabilty(p1, hopfieldNN, 100, noise)
    p2_probability, p2_reverse = count_convergence_probabilty(p2, hopfieldNN, 100, noise)
    p3_probability, p3_reverse = count_convergence_probabilty(p3, hopfieldNN, 100, noise)
    retults['p1']['P'].append(p1_probability)
    retults['p2']['P'].append(p2_probability)
    retults['p3']['P'].append(p3_probability)
    retults['p1']['RP'].append(p1_reverse)
    retults['p2']['RP'].append(p2_reverse)
    retults['p3']['RP'].append(p3_reverse)
    print(f'Noise: {noise}% -> p1: {p1_probability}, p2: {p2_probability}, p3: {p3_probability}')
   


fig = plt.figure(figsize=(10, 6))
plt.plot(noise_levels, retults['p1']['P'], color = 'green', label = 'p1')
plt.plot(noise_levels, retults['p2']['P'], color = 'red', label = 'p2')
plt.plot(noise_levels, retults['p3']['P'], color = 'blue', label = 'p3')
plt.xlabel('Noise percent %', fontsize = 14)
plt.ylabel('Recovery probabilty', fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize = 10)
plt.grid(True)
save_path = os.path.join('Lab3', 'plots_3_4', f'probability_of_recovery.png')
plt.savefig(save_path)
plt.show()



fig = plt.figure(figsize=(10, 6))
plt.plot(noise_levels, retults['p1']['RP'], color = 'green', label = 'p1')
plt.plot(noise_levels, retults['p2']['RP'], color = 'red', label = 'p2')
plt.plot(noise_levels, retults['p3']['RP'], color = 'blue', label = 'p3')
plt.xlabel('Noise percent %', fontsize = 14)
plt.ylabel('Reverse pattern probabilty', fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize = 10)
plt.grid(True)
save_path = os.path.join('Lab3', 'plots_3_4', f'probability_of_reverse_pattern.png')
plt.savefig(save_path)
plt.show()