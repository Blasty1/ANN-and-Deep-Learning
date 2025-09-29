import numpy as np
import matplotlib.pyplot as plt
from Lab3.read_generate_data import read_patterns, add_noise, show_pattern, show_multiple_arrays


print("======= 3.4 Distortion Resistance =======\n")

patterns = read_patterns()
p1 = patterns[0]
p2 = patterns[1]
p3 = patterns[2]

# show_pattern(p1)
# show_pattern(p2)
# show_pattern(p3)
show_multiple_arrays(patterns[:3])

print("==== Iteration on all noise levels ====\n")

noise_levels = np.array(range(101))
for noise in noise_levels:
    # print(f'Noise level {noise}%')
    p1_noise = add_noise(p1, noise/100)
    p2_noise = add_noise(p2, noise/100)
    p3_noise = add_noise(p3, noise/100)
    
    #TODO train Hopfield network be Little synchronius update

    #TODO show results 

