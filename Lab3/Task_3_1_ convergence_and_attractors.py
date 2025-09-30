from read_generate_data import data_small_model_new, data_small_model
from utils.HopfieldNetwork import HopfieldNetwork
import numpy as np

patterns_to_store = data_small_model()
input = data_small_model_new()

# the number of neurons is 8 because the patterns have size 8
hopfieldNN = HopfieldNetwork(8)

hopfieldNN.train(patterns_to_store)

print("QUESTION 1: Testing convergence of distorted patterns")
print("="*70)

# Test recall for each distorted pattern
print("\nRecalling distorted patterns:")
for i in range(len(input)):
    recalled = hopfieldNN.recall(input[i])
    errors = (patterns_to_store[i] != recalled).sum()
    converged_correctly = np.array_equal(recalled, patterns_to_store[i])
    
    print(f"\nx{i+1}d: {input[i]}")
    print(f"  → Recalled: {recalled}")
    print(f"  → Target x{i+1}: {patterns_to_store[i]}")
    print(f"  → Errors: {errors}")
    print(f"  → Converged correctly: {converged_correctly}")

attractors = hopfieldNN.find_all_attractors(2)
print(f"The total number of attractors are: {len(attractors)}")


x1_very_distorted = np.array([1, 1, -1, 1, -1, 1, 1, -1]) # Highly distorted x1 (flip 5-6 bits out of 8)
x2_very_distorted = np.array([1, 1, 1, 1, 1, -1, 1, 1])  # Highly distorted x2 (flip 6 bits out of 8)
x3_very_distorted = np.array([1, -1, -1, 1, -1, -1, 1, -1]) # Highly distorted x3 (flip 5 bits out of 8)
input = [x1_very_distorted,x2_very_distorted,x3_very_distorted]
print("\nRecalling VERY distorted patterns:")
for i in range(len(input)):
    recalled = hopfieldNN.recall(input[i])
    errors = (patterns_to_store[i] != recalled).sum()
    converged_correctly = np.array_equal(recalled, patterns_to_store[i])
    
    print(f"\nx{i+1}d: {input[i]}")
    print(f"  → Recalled: {recalled}")
    print(f"  → Target x{i+1}: {patterns_to_store[i]}")
    print(f"  → Errors: {errors}")
    print(f"  → Converged correctly: {converged_correctly}")

