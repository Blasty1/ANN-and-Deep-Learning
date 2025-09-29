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

