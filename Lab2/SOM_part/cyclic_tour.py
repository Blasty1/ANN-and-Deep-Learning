import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.SOM import SOM
import numpy as np




# Import data
with open("./data_lab2/cities.dat", "r") as file:
    data_read_from_file = []
    for line in file:
        line = line.strip()  # Remove whitespace first
        if(not line or line[0] == "%"):
            continue
        
        values = line.split(",")
        values[0] = float(values[0].strip())
        values[1] = float(values[1].replace(";", "").strip())
        data_read_from_file.append(values)

        

X = np.array(data_read_from_file).T

# Initialize the model
epochs = 200
grid_dimension = 1
input_dimension = 2
units_number = 10 # Number of cities

neighborhood_sizes = [2, 1, 0]
epochs_per_phase = epochs // len(neighborhood_sizes)

print("Training SOM with progressive neighborhood reduction...")
print(f"Phase 1: Neighborhood size = 2 (epochs 1-{epochs_per_phase})")
print(f"Phase 2: Neighborhood size = 1 (epochs {epochs_per_phase+1}-{2*epochs_per_phase})")  
print(f"Phase 3: Neighborhood size = 0 (epochs {2*epochs_per_phase+1}-{epochs})")

all_bmu_history = {}
epoch_counter = 0

for phase, sigma in enumerate(neighborhood_sizes):
    print(f"\n--- Training Phase {phase+1}: sigma0 = {sigma} ---")
    
    # Create new SOM for this phase with current neighborhood size
    model = SOM(grid_dimension=grid_dimension, 
               input_dimension=input_dimension, 
               units_number=units_number, 
               sigma0=sigma,  # Current neighborhood size
               epochs=epochs_per_phase,
               lr=0.5, 
               circular=True)
    
    # If not first phase, initialize with weights from previous phase
    if phase > 0:
        model.W = previous_weights.copy()
    
    # Train for this phase
    phase_bmu_history = model.train(X)
    
    # Store weights for next phase
    previous_weights = model.W.copy()
    
    # Combine BMU history
    for epoch_in_phase, bmus in phase_bmu_history.items():
        all_bmu_history[epoch_counter + epoch_in_phase] = bmus
    
    epoch_counter += epochs_per_phase
    
print(f"\nTraining completed with final neighborhood size = 0")
bmu_history = all_bmu_history

# Get final positions of neurons (these represent the tour)
final_weights = model.W  # Shape: (2, 10)

print(f"Final weights shape: {final_weights.shape}")

# Find which neuron wins each city
city_to_neuron = []
for i in range(X.shape[1]):  # For each city
    city = X[:, i]
    distances = [np.sqrt(np.sum((city - final_weights[:, j])**2)) for j in range(units_number)]
    winning_neuron = np.argmin(distances)
    city_to_neuron.append(winning_neuron)
    
# Create tour order based on neuron sequence
neuron_order = list(range(units_number))  # [0,1,2,3,4,5,6,7,8,9]
city_tour_order = []

print(f"\nTour construction:")
for neuron_idx in neuron_order:
    # Find cities assigned to this neuron
    cities_for_this_neuron = [i for i, winner in enumerate(city_to_neuron) if winner == neuron_idx]
    if cities_for_this_neuron:
        print(f"Neuron {neuron_idx}: Cities {cities_for_this_neuron}")
        city_tour_order.extend(cities_for_this_neuron)
    else:
        print(f"Neuron {neuron_idx}: No cities assigned")
print(f"\nFinal city tour order (by city indices): {city_tour_order}")

### 1 part --> Show the TOUR

# Plot 1: Show both the tour and training points
plt.subplot(1, 3, 1)
# Plot cities (training points)
plt.scatter(X[0, :], X[1, :], c='red', s=100, marker='o', label='Cities (Training Points)', zorder=3)

# Plot SOM neurons 
plt.scatter(final_weights[0, :], final_weights[1, :], c='blue', s=80, marker='s', label='SOM Neurons', zorder=2)

# Plot the tour connecting cities in the determined order
if len(city_tour_order) > 0:
    tour_x = [X[0, city_idx] for city_idx in city_tour_order] + [X[0, city_tour_order[0]]]  # Close loop
    tour_y = [X[1, city_idx] for city_idx in city_tour_order] + [X[1, city_tour_order[0]]]  # Close loop
    plt.plot(tour_x, tour_y, 'g-', linewidth=3, label='City Tour', zorder=1, alpha=0.8)
    
    # Add arrows to show direction
    for i in range(len(city_tour_order)):
        start_city = city_tour_order[i]
        end_city = city_tour_order[(i + 1) % len(city_tour_order)]
        dx = X[0, end_city] - X[0, start_city]
        dy = X[1, end_city] - X[1, start_city]
        plt.arrow(X[0, start_city], X[1, start_city], dx*0.7, dy*0.7, 
                 head_width=0.015, head_length=0.015, fc='green', ec='green', alpha=0.7)

# Add city labels with tour order
for i in range(X.shape[1]):
    tour_position = city_tour_order.index(i) if i in city_tour_order else "?"
    plt.annotate(f'C{i}({tour_position})', (X[0, i], X[1, i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add neuron labels
for i in range(final_weights.shape[1]):
    plt.annotate(f'N{i}', (final_weights[0, i], final_weights[1, i]), xytext=(5, -15), textcoords='offset points', color='blue', fontsize=8)

plt.title('SOM Cyclic Tour - Tour & Training Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()