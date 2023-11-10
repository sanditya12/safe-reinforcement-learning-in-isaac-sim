




import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from racing_data.left_boundary_points import left_boundary_points
from racing_data.right_boundary_points import right_boundary_points
from plotdata import trajectory_filter, speeds_filter
left_points_2d = [sublist[:2] for sublist in left_boundary_points]
right_points_2d = [sublist[:2] for sublist in right_boundary_points]

x_l = [point[0] for point in left_points_2d]
y_l = [point[1] for point in left_points_2d]
 
x_r = [point[0] for point in right_points_2d]
y_r = [point[1] for point in right_points_2d]

x_filter = [point[0] for point in trajectory_filter]
y_filter = [point[1] for point in trajectory_filter] 

norm_speeds = colors.Normalize(vmin=0.8, vmax=1)

scalar_map = cm.ScalarMappable(cmap=cm.viridis, norm= norm_speeds)
scalar_map.set_array([])

# x_conv = [point[0] for point in trajectory_conventional]
# y_conv = [point[1] for point in trajectory_conventional]

# x_data = [point[0] for point in data]
# y_data = [point[1] for point in data]
 
# Plot the line segments
plt.plot(x_l, y_l, color = '#808080')
plt.plot(x_r, y_r, color = '#808080')
for i in range(len(trajectory_filter)-1):
    plt.plot([x_filter[i], x_filter[i+1]], [y_filter[i], y_filter[i+1]], color = scalar_map.to_rgba(speeds_filter[i]))

plt.colorbar(scalar_map, orientation='vertical')
# plt.plot(x_conv, y_conv, color = 'red')
# plt.plot(x_data, y_data, color = 'yellow')
 
# Add labels and show the plot
plt.xlabel("x")
plt.ylabel("y")
plt.title("Line Segments")
# plt.grid(True)
plt.show()