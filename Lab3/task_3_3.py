from utils.HopfieldNetwork import HopfieldNetwork
import read_generate_data as d
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


patterns = d.read_patterns()                      
N = 1024
hop = HopfieldNetwork(N)
hop.train(patterns)

# ---- Energies of stored patterns ----
print("\nEnergies at stored patterns:")
for i, p in enumerate(patterns, start=1):
    print(f"  Energy at p{i}: {hop.energy(p)}")

# ---- Make noisy versions & energies ----
noisy_patterns = [d.add_noise(p, percent=0.2) for p in patterns]

print("\nEnergies at distorted (20% noise) patterns:")
for i, p in enumerate(noisy_patterns, start=1):
    print(f"  Energy at distorted p{i}: {hop.energy(p)}")

# ---- Energy evolution for each clean pattern ----
plt.figure()
for i, p in enumerate(patterns, start=1):
    _, E = hop.recall_asynchronously(p)
    plt.plot(np.arange(1, len(E) + 1), E, label=f"p{i}")
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Energy evolution for one noisy pattern (example: p7) ----
plt.figure()
_, E = hop.recall_asynchronously(noisy_patterns[6])  # index 6 == p7
plt.plot(np.arange(1, len(E) + 1), E)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.tight_layout()

Path("plots").mkdir(exist_ok=True, parents=True)
plt.savefig("plots/energy_evolution_distorted.png", dpi=300, bbox_inches="tight")
plt.show()

# Re-initialize a fresh network (untrained) and run recall on p6 to observe behavior
hop2 = HopfieldNetwork(N) 
_, E = hop2.recall_asynchronously(patterns[5])  # p6
plt.figure()
plt.plot(np.arange(1, len(E) + 1), E)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("plots/symmetric_random_weights.png", dpi=300, bbox_inches="tight")
plt.show()
