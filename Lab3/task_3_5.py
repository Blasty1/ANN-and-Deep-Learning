from utils.HopfieldNetwork import HopfieldNetwork
import read_generate_data as d
import matplotlib.pyplot as plt
import numpy as np
import os




def performance(patterns, noise_rate, nstored):
    to_store = patterns[:nstored, ]
    model = HopfieldNetwork(1024)
    model.train(to_store)
    noisy_patterns, accuracies = [], []
    for p in to_store:
        p_noisy = d.add_noise(p, noise_rate)
        noisy_patterns.append(p_noisy)

    for i, p in enumerate(noisy_patterns):
        p_final= model.recall(p)
        accuracy = (to_store[i] == p_final).sum() /len(p_final)
        accuracies.append(accuracy)
    return np.mean(accuracies)

patterns = d.read_patterns()
nstored = 11
np.random.seed(42)
plt.figure() 

stored = np.arange(1, nstored+1)
performances = [performance(patterns, 0.2, s) for s in stored]
x=stored
plt.plot(stored, performances)
plt.xlabel("number of stored patterns")
plt.ylabel("performance")
plt.xticks(np.arange(min(x)-1, max(x)+1))
plt.grid()
plt.legend()
plt.show()


####     using random patterns

def random_patterns(number, size=1024):
    patterns = np.zeros((number, size))
    for i in range(number):
        patterns[i] = np.random.choice([-1, 1], size=size)
    return patterns
np.random.seed(42)
nstored = 150
patterns = random_patterns(150)
plt.figure() 
stored = np.arange(1, nstored+1)
performances = [performance(patterns, 0.2, s) for s in stored]
plt.plot(stored, performances)
plt.xlabel("number of stored patterns")
plt.ylabel("performance")
plt.grid()
plt.savefig("plots/random_patterns_more.png", dpi=300, bbox_inches="tight")
plt.show()


###        PART II

def biased_patterns(num_patterns, size, bias=0.7):
    """
    Generate patterns of ±1 with a given bias toward +1.
    """
    # draw from {+1, -1} with probability [bias, 1-bias]
    return np.random.choice([1, -1], size=(num_patterns, size), p=[bias, 1-bias])

def stable_patterns(model, patterns):
    stable = 0
    for i in range(len(patterns)):
        p_noisy = d.add_noise(patterns[i], 0.2)          
        p_final, _ = model.recall_asynchronously(p_noisy.copy())
        if np.array_equal(p_final, patterns[i]) or np.array_equal(p_final, -patterns[i]):
            stable += 1
    return stable



###    recall of noisy patterns
np.random.seed(45)          
# patterns = random_patterns(300, size = 100)
patterns = biased_patterns(num_patterns=300, size=100)
network = HopfieldNetwork(100)
stable = []
for i in range(50):
    network.train(patterns[i:i+1])
    if i>0:
        stable.append(stable_patterns(network, patterns[:i]))
print(stable)

y = stable
x = np.arange(2, len(y)+2)

plt.figure()
plt.plot(x, y, marker="o")  # marker="o" helps see points better

plt.xlabel("number of stored patterns")
plt.ylabel("number of stable patterns")

# show all x values
plt.xticks(x, rotation=90)

# show all y values
plt.yticks(np.arange(min(y), max(y)+1))

plt.grid(True, linestyle="--", alpha=0.6)  # optional, makes it clearer
plt.savefig("plots/biased_noisy_network100.png", dpi=300, bbox_inches="tight")

plt.show() 


# # ---- PART II: noisy recall, averaged over trials, with/without self-connections ----






def stable_patterns(model, patterns, noise_rate=0.2):
    """Count how many stored patterns are recovered from noisy cues."""
    stable = 0
    for i in range(len(patterns)):
        # p_noisy = d.add_noise(patterns[i], noise_rate)        # flips but keeps ±1
        p = patterns[i].copy()
        p_final, _ = model.recall_asynchronously(p)
        # accept negative as valid recall
        if np.array_equal(p_final, patterns[i]) or np.array_equal(p_final, -patterns[i]):
            stable += 1
    return stable

def run_noisy_curve(N=100, K=50, noise_rate=0.2, self_connections=False, trials=10):
    """
    Incrementally store up to K patterns (length N).
    After each addition, count # of noisy recalls that return to their true pattern (or its neg).
    Repeat 'trials' times and return mean and std across trials.
    """
    counts = np.zeros((trials, K-1), dtype=float)  # we start appending from i>0 in your loop
    for t in range(trials):
        pats = random_patterns(max(300, K), size=N)  # your helper; generates ±1
        net = HopfieldNetwork(N)
        tmp = []
        for i in range(K):
            net.train(pats[i:i+1,], self_connections=self_connections)  # accumulate
            if i > 0:
                tmp.append(stable_patterns(net, pats[:i], noise_rate))
        counts[t, :] = np.array(tmp)
    return counts.mean(axis=0), counts.std(axis=0)

# parameters
N = 200
K = 100
noise = 0.2
trials = 10

# run both settings
m_no_diag, s_no_diag = run_noisy_curve(N, K, noise_rate=noise, self_connections=False, trials=trials)
m_diag,    s_diag    = run_noisy_curve(N, K, noise_rate=noise, self_connections=True,  trials=trials)

# x-axis (your plotting convention starts at 2 because we append from i>0)
x = np.arange(2, K+1)

# plotting
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8,6))
plt.plot(x, m_no_diag, label="no self-connections", marker="o")
plt.fill_between(x, m_no_diag - s_no_diag, m_no_diag + s_no_diag, alpha=0.2)

plt.plot(x, m_diag, label="with self-connections", marker="o")
plt.fill_between(x, m_diag - s_diag, m_diag + s_diag, alpha=0.2)

plt.xlabel("number of stored patterns")
plt.ylabel("noisy recall: # recovered (noise=0.2)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("plots/recall_compare_mean_std200_100p.png", dpi=300, bbox_inches="tight")
plt.show()
