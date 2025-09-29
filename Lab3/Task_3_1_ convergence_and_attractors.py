from Lab3.read_generate_data import data_small_model_new, data_small_model
from Lab3.utils import HopfieldNetwork

patterns_to_store = data_small_model()
input = data_small_model_new()

# the number of neurons is 8 because the patterns have size 8
hopfieldNN = HopfieldNetwork(8)

hopfieldNN.train(patterns_to_store)
output = hopfieldNN.recall_synchronously(input)
print("Output pattern after recall (synchronous):", output)
