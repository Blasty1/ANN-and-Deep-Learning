import matplotlib.pyplot as plt
from matplotlib import animation
import Lab1b.function_approximation.part3_1_3 as f

def animate_wireframe(xx, yy, z_true, z_pred_list, interval=200):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-0.7, 0.7)

    # Draw true surface once
    ax.plot_wireframe(xx, yy, z_true, linewidth=0.5, color="blue")

    # Keep track of prediction surface
    pred_wf = {"artist": None}

    def init():
        pred = z_pred_list[0]
        pred_wf["artist"] = ax.plot_wireframe(xx, yy, pred, linewidth=0.5, color="red")
        ax.set_title("Epoch 1")
        return pred_wf["artist"],

    def update(frame):
        if pred_wf["artist"] is not None:
            pred_wf["artist"].remove()
        pred = z_pred_list[frame]
        pred_wf["artist"] = ax.plot_wireframe(xx, yy, pred, linewidth=0.5, color="red")
        ax.set_title(f"Epoch {frame+1} - Hidden Nodes: {f.nnodes}")
        return pred_wf["artist"],

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(z_pred_list), interval=interval, blit=False, repeat=False
    )
    # ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)))
    plt.show()   
    return ani
# ani = animate_wireframe(xx, yy, z, z_predictions, interval=200)

_ = animate_wireframe(f.xx, f.yy, f.z, f.z_predictions, interval=200)
####
