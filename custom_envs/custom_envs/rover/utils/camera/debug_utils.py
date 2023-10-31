import torch
import numpy as np
# from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.isaac.debug_draw")
from omni.isaac.debug_draw import _debug_draw
import random



def draw_depth(heightmap_points: torch.tensor, depth_points: torch.tensor,):
    draw = _debug_draw.acquire_debug_draw_interface()
    rover_distribution = heightmap_points.tolist()
    depth_points = depth_points.tolist()
    N = len(rover_distribution)

    rover_distributionZ = []
    rover_distribution2 = []
    depth_pointsZ = []
    depth_points2 = []
    for i in range(N):
        rover_distributionZ.append(rover_distribution[i][2]+0.1)

    for i in range(N):
        rover_distribution2.append([rover_distribution[i][0], rover_distribution[i][1], rover_distributionZ[i]])

    for i in range(N):
        depth_pointsZ.append(depth_points[i][2]+0.1)
    for i in range(N):
        depth_points2.append([depth_points[i][0], depth_points[i][1], depth_pointsZ[i]])

    colors = [3 for _ in range(N)]
    sizes = [[3] for _ in range(N)]
    draw.clear_lines()
    draw.draw_lines(rover_distribution, depth_points, [(1, 0.0, 0.0, 0.9)]*N, [3]*N)