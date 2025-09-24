import numpy as np
import matplotlib.pyplot as plt
import votes
# Import the parties
with open('./data_lab2/mpparty.dat', "r") as f:
    party_number = []
    for line in f:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        party_number.append(int(line))

coding = {0:"no party", 1:'m', 2:'fp', 3:'s', 4:'v', 5:'mp', 6:'kd', 7:'c'}
parties = [coding[i] for i in party_number]





rows = cols = 10
bmus = np.asarray(votes.bmus)
coords = np.stack(np.meshgrid(range(rows), range(cols), indexing='ij'), -1).reshape(-1, 2)
xy = coords[bmus]  # shape (349, 2)
print(bmus.min(), bmus.max())       

unique_parties = sorted(set(parties))
cmap = plt.cm.get_cmap('tab20', len(unique_parties))
party2color = {p: cmap(i) for i, p in enumerate(unique_parties)}

cell_samples = {(r, c): [] for r in range(rows) for c in range(cols)}
for i, (r, c) in enumerate(xy):
    cell_samples[(r, c)].append(i)  

# Plot background grid 
fig, ax = plt.subplots(figsize=(7, 7))
color_grid = np.ones((rows, cols, 4))  # white RGBA
ax.imshow(color_grid, origin='upper')

max_marker = 30  
for (r, c), indices in cell_samples.items():
    k = len(indices)
    if k == 0:
        continue

   
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))
    xs = np.linspace(-0.35, 0.35, ncols)  
    ys = np.linspace(-0.35, 0.35, nrows) 
    xv, yv = np.meshgrid(xs, ys)
    offsets = np.column_stack([xv.ravel(), yv.ravel()])[:k]

    for idx, (off_x, off_y) in zip(indices, offsets):
        p = parties[idx]
        color = party2color[p]
        ax.scatter(c + off_x, r + off_y,
                   s=40, marker='o',
                   facecolor=color, edgecolor='k', linewidth=0.3, zorder=3, alpha=0.9)

    ax.text(c - 0.35, r - 0.35, str(k), ha='left', va='top', fontsize=6, color='black', zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.6))

ax.set_title("SOM 10×10 — All samples plotted inside their BMU cell")
ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(rows - 0.5, -0.5)

handles = [plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                      markerfacecolor=party2color[p], markeredgecolor='k', label=p)
           for p in unique_parties]
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="party")
plt.tight_layout()
plt.show()
