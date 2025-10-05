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
from read_generate_data import show_pattern, read_patterns
from utils.HopfieldNetwork import HopfieldNetwork
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import io

def recall_with_gif_creation(hopfield_net, pattern, max_iterations=10000, show_every=100, gif_filename="recall_evolution.gif"):
    """
    TRUE asynchronous recall where each iteration updates only ONE neuron
    Creates a GIF of the pattern evolution with custom colors
    """
    state = pattern.copy()
    num_neurons = hopfield_net.num_neurons
    
    print(f"Starting TRUE asynchronous recall (1 neuron per iteration)")
    print(f"Max iterations: {max_iterations}, capturing every {show_every} iterations")
    
    # Create custom colormap: purple for -1, yellow for 1
    colors = ['#4B0082', '#FFD700']  # Purple and Gold/Yellow
    cmap = ListedColormap(colors)
    
    # List to store frames for GIF
    frames = []
    
    # Capture initial state
    fig, ax = plt.subplots(figsize=(5, 5))
    # Convert from bipolar (-1, 1) to (0, 1) for colormap indexing
    display_state = (state.reshape(32, 32) + 1) / 2
    ax.imshow(display_state, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f"Iteration 0", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Track convergence
    last_state = state.copy()
    stable_iterations = 0
    check_interval = 100
    
    for iteration in range(1, max_iterations + 1):
        # Randomly select ONE neuron to update
        i = np.random.randint(0, num_neurons)
        
        # Update this single neuron
        net_input = np.dot(hopfield_net.weights[i], state)
        state[i] = 1 if net_input >= 0 else -1
        
        # Capture frame at regular intervals
        if iteration % show_every == 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            display_state = (state.reshape(32, 32) + 1) / 2
            ax.imshow(display_state, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f"Iteration {iteration} (~{iteration/num_neurons:.2f} passes)", 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            plt.close(fig)
            
            print(f"Captured frame at iteration {iteration}")
        
        # Check convergence periodically
        if iteration % check_interval == 0:
            if np.array_equal(state, last_state):
                stable_iterations += check_interval
                if stable_iterations >= 2 * check_interval:
                    print(f"✓ Converged at iteration {iteration} (~{iteration/num_neurons:.2f} passes)")
                    print(f"   Network has been stable for {stable_iterations} neuron updates")
                    
                    # Capture final frame if not just captured
                    if iteration % show_every != 0:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        display_state = (state.reshape(32, 32) + 1) / 2
                        ax.imshow(display_state, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                        ax.set_title(f"Final (Iteration {iteration})", fontsize=14, fontweight='bold')
                        ax.axis('off')
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
                        buf.seek(0)
                        frames.append(Image.open(buf).copy())
                        plt.close(fig)
                    
                    # Add a few duplicate frames at the end to pause on final result
                    for _ in range(5):
                        frames.append(frames[-1])
                    
                    # Save GIF
                    print(f"Creating GIF with {len(frames)} frames...")
                    frames[0].save(
                        gif_filename,
                        save_all=True,
                        append_images=frames[1:],
                        duration=200,  # 200ms per frame
                        loop=0
                    )
                    print(f"✓ GIF saved as: {gif_filename}")
                    
                    return state, iteration
            else:
                stable_iterations = 0
                last_state = state.copy()
    
    print(f"✗ Reached max iterations: {max_iterations} (~{max_iterations/num_neurons:.1f} passes)")
    
    # Add duplicate frames at the end
    for _ in range(5):
        frames.append(frames[-1])
    
    # Save GIF even if max iterations reached
    print(f"Creating GIF with {len(frames)} frames...")
    frames[0].save(
        gif_filename,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print(f"✓ GIF saved as: {gif_filename}")
    
    return state, max_iterations


def display_patterns_with_color(patterns_to_store):
    """
    Display the original patterns and test patterns with custom colors
    """
    # Create custom colormap
    colors = ['#4B0082', '#FFD700']  # Purple and Gold/Yellow
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Display p1, p2, p3
    for idx, ax in enumerate(axes[0]):
        display_state = (patterns_to_store[idx].reshape(32, 32) + 1) / 2
        ax.imshow(display_state, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"p{idx+1} pattern", fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Display p10, p11, and p1 for comparison
    display_state = (patterns_to_store[9].reshape(32, 32) + 1) / 2
    axes[1, 0].imshow(display_state, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].set_title("p10 (degraded p1)", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    display_state = (patterns_to_store[10].reshape(32, 32) + 1) / 2
    axes[1, 1].imshow(display_state, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].set_title("p11 (mixture p2/p3)", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Leave the last subplot empty or show p1 again
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("patterns_overview.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


# Main execution
patterns_to_store = read_patterns()
hopfieldNN = HopfieldNetwork(1024)
hopfieldNN.train(patterns_to_store[0:3,:])

print("=" * 60)
print("Displaying Patterns with Custom Colors")
print("=" * 60)

# Display all patterns first
display_patterns_with_color(patterns_to_store)

print("\n" + "=" * 60)
print("Creating GIFs for Pattern Recall with Random Async Updates")
print("=" * 60)

# Test with degraded pattern p10
print("\n" + "=" * 60)
print("Test 1: Recalling degraded pattern p10")
print("=" * 60)

result_p10, iters_p10 = recall_with_gif_creation(
    hopfieldNN, 
    patterns_to_store[9], 
    max_iterations=10000,
    show_every=100,
    gif_filename="p10_recall_evolution.gif"
)

# Test with mixed pattern p11
print("\n" + "=" * 60)
print("Test 2: Recalling mixed pattern p11")
print("=" * 60)

result_p11, iters_p11 = recall_with_gif_creation(
    hopfieldNN, 
    patterns_to_store[10], 
    max_iterations=10000,
    show_every=100,
    gif_filename="p11_recall_evolution.gif"
)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Pattern p10: {iters_p10} neuron updates (~{iters_p10/1024:.2f} passes)")
print(f"  → GIF saved as: p10_recall_evolution.gif")
print(f"Pattern p11: {iters_p11} neuron updates (~{iters_p11/1024:.2f} passes)")
print(f"  → GIF saved as: p11_recall_evolution.gif")
print(f"\nOverview image saved as: patterns_overview.png")
print("=" * 60)