import numpy as np
import matplotlib.pyplot as plt
from Lab3.read_generate_data import read_patterns, generate_sparse_patterns

print("======= 3.6 Sparse Patterns =======\n")




#TODO Train on Modified train rule

#TODO Train on modifi update rule

#TODO Generete sparse paterns 10% activity 

#TODO test capacity on different bias values 
print("======= Sparse patterns with activity 10% =======\n")

size = 1024
nr_of_patterns = 300
rho10 = 0.1
binary_patterns_10 = generate_sparse_patterns(size, nr_of_patterns, rho10)
bias_vales = []
for bias in bias_vales:
    pass

#TODO Generete sparse paterns 5% activity 
#TODO repeat
print("======= Sparse patterns with activity 5% =======\n")


size = 1024
nr_of_patterns = 300
rho5 = 0.05
binary_patterns_5 = generate_sparse_patterns(size, nr_of_patterns, rho5)
bias_vales = []
for bias in bias_vales:
    pass


#TODO Generete sparse paterns 1% activity 
#TODO repeat
print("======= Sparse patterns with activity 1% =======\n")
 
size = 1024
nr_of_patterns = 300
rho1 = 0.01
binary_patterns_1 = generate_sparse_patterns(size, nr_of_patterns, rho1)
bias_vales = []
for bias in bias_vales:
    pass
