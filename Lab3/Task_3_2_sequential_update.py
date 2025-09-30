from read_generate_data import show_pattern, read_patterns
from utils.HopfieldNetwork import HopfieldNetwork
import numpy as np
import matplotlib.pyplot as plt


patterns_to_store = read_patterns()

# the number of neurons is 8 because the patterns have size 8
hopfieldNN = HopfieldNetwork(1024)
hopfieldNN.train(patterns_to_store[0:3,:])

#show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[0]),"p1 reconstruction",True)
#show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[1]),"p2 reconstruction",True)
#show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[2]),"p3 reconstruction",True)

#show_pattern(patterns_to_store[0],"p1 pattern",True)
#show_pattern(patterns_to_store[1],"p2 pattern",True)
#show_pattern(patterns_to_store[2],"p3 pattern",True)

#plt.show()

#show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[9]),"p10 reconstruction",True)
#show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[10]),"p11 reconstruction",True)

#show_pattern(patterns_to_store[0],"p1 pattern",True)
#show_pattern(patterns_to_store[1],"p2 pattern",True)
#show_pattern(patterns_to_store[2],"p3 pattern",True)
#plt.show()




## with randomly units order selection

show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[0],random_order=True),"p1 reconstruction",True)
show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[1],random_order=True),"p2 reconstruction",True)
show_pattern(hopfieldNN.recall_asynchronously(patterns_to_store[2],random_order=True),"p3 reconstruction",True)
plt.show()