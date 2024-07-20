import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Load the Excel file and the specific sheet
file_path = 'final_3D_matrix.xlsx'
sheet_name = 'simHV'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract the data
matrix = data.values

# Define the labels
Li = ["HH", "HV", "VH", "VV"]
Ls = ["VV", "VH", "HV", "hh"]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the bar3 plot
_x = np.arange(matrix.shape[1])
_y = np.arange(matrix.shape[0])
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = matrix.ravel()
bottom = np.zeros_like(top)
width = depth = 0.8

# Normalize the values to range [0, 1] for the colormap
norm = plt.Normalize(top.min(), top.max())
colors = cm.viridis(norm(top))

# Plot bars
for i in range(len(x)):
    ax.bar3d(x[i], y[i], bottom[i], width, depth, top[i], color=colors[i], shade=True)

# Add labels
ax.set_xticks(np.arange(len(Li)))
ax.set_xticklabels(Li)
ax.set_yticks(np.arange(len(Ls)))
ax.set_yticklabels(Ls)

# Title and axis labels
titles_size = 25
ax.set_title('Simulation base HVVH' +sheet_name+'bits', fontsize=titles_size, fontweight='bold', fontname='Times New Roman')
ax.set_xlabel('Source', fontsize=15, fontweight='bold', fontname='Times New Roman')
ax.set_ylabel('detectors', fontsize=15, fontweight='bold', fontname='Times New Roman')
ax.set_zlabel(' ', fontsize=15, fontweight='bold', fontname='Times New Roman')

# Colormap
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Z ticks
high_num = matrix.max()
ax.set_zticks(np.round(np.linspace(0, high_num, 4), 2))

# Add color bar which maps values to colors
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(matrix)
cbar = plt.colorbar(mappable, pad=0.15, shrink=0.5, aspect=5)  # Adjust pad to move colorbar
cbar.set_label('', fontsize=15, fontweight='bold', fontname='Times New Roman')

# Show plot
plt.show()

# # Save the figure
# fig.savefig('Simulation_base_HVVH_50_bits.png')
