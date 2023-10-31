import os
import pymeshlab
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from .usd_utils import trimesh_to_usd

# Get the directory containing the script
directory_terrain_utils = os.path.dirname(os.path.abspath(__file__))


class TerrainManager():

    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        terrain_path = os.path.join(self.dir_path, "terrain_data/map.ply")
        rock_mesh_path = os.path.join(self.dir_path, "terrain_data/big_stones.ply")

        self.heightmap = None
        self.meshes = [terrain_path, rock_mesh_path]


    def load_mesh(self, path) -> Tuple[np.ndarray, np.ndarray]:

        # Assert that the specified path exists in the list of meshes.
        assert path in self.meshes, f"The provided path '{path}' must exist in the 'self.meshes' list."

        # create an empty meshset
        ms = pymeshlab.MeshSet() 

        # load the mesh
        ms.load_new_mesh(path) 

        # get the mesh
        mesh = ms.current_mesh() # get the mesh

        # Get vertices as float32 array
        vertices = mesh.vertex_matrix().astype('float32')

        # Get faces as uint32 array
        faces = mesh.face_matrix().astype('uint32')

        return vertices, faces
    
    def mesh_to_omni_stage(self, position = None, orientation = None, ground_only = False):
        vertices, faces = self.load_mesh(self.meshes[0])
        trimesh_to_usd(vertices, faces, position, orientation)

        if not ground_only:
            for mesh in self.meshes[1:]:
                vertices, faces = self.load_mesh(mesh)
                trimesh_to_usd(vertices, faces, position, orientation)
    
    def mesh_to_heightmap(vertices, faces, grid_size_in_m, resolution_in_m=0.1):

        # Calculate the grid size
        grid_size = grid_size_in_m / resolution_in_m

        # Initialize the heightmap
        heightmap = np.zeros((int(grid_size+1), int(grid_size+1)), dtype=np.float32)

        # Define bounding box
        min_x, min_y, _ = np.min(vertices, axis=0)
        max_x, max_y, _ = np.max(vertices, axis=0)

        # Calculate the size of a grid cell
        cell_size_x = (max_x - min_x) / (grid_size)
        cell_size_y = (max_y - min_y) / (grid_size)
        
        # Iterate over each face to update the heightmap
        for face in faces:
            # Get the vertices
            v1, v2, v3 = vertices[face]
            # Project to 2D grid
            min_i = int((min(v1[0], v2[0], v3[0]) - min_x) / cell_size_x)
            max_i = int((max(v1[0], v2[0], v3[0]) - min_x) / cell_size_x)
            min_j = int((min(v1[1], v2[1], v3[1]) - min_y) / cell_size_y)
            max_j = int((max(v1[1], v2[1], v3[1]) - min_y) / cell_size_y)

            # Update the heightmap
            for i in range(min_i, max_i + 1):
                for j in range(min_j, max_j + 1):
                    heightmap[i, j] = max(heightmap[i, j], v1[2], v2[2], v3[2])
        
        return heightmap


    def find_rocks_in_heightmap(heightmap, threshold=1.0):
        from scipy.signal import convolve2d
        from scipy import ndimage
        from scipy.ndimage import binary_dilation
        import cv2

        # Sobel operators for gradient in x and y directions
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        
        # Compute the gradient components
        grad_x = convolve2d(heightmap, sobel_x, mode='same', boundary='wrap')
        grad_y = convolve2d(heightmap, sobel_y, mode='same', boundary='wrap')

        # Compute the overall gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Initialize a rock mask with zeros
        rock_mask = np.zeros_like(heightmap, dtype=np.int32)

        # Mark the areas where gradient magnitude is greater than the threshold as rocks
        rock_mask[grad_magnitude > threshold] = 1

        # Perform dilation to add a safety margin around the rocks
        closed_rock_mask = binary_dilation(rock_mask, iterations=1)
        filled_rock_mask = ndimage.binary_fill_holes(closed_rock_mask).astype(int)

        # Safety margin around the rocks
        kernel = np.ones((5, 5), np.uint8)
        safe_rock_mask = cv2.dilate(filled_rock_mask.astype(np.uint8), kernel, iterations=1)

        return safe_rock_mask

    def show_heightmap(heightmap, name="2D Heightmap"):
        plt.figure(figsize=(10, 10))

        # Display the heightmap
        plt.imshow(heightmap, cmap='terrain', origin='lower')

        # Add a color bar for reference
        plt.colorbar(label='Height')

        # Add labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f"{name}")

        # Show the plot
        plt.show()

def load_mesh(path=os.path.join(directory_terrain_utils, '../terrain_data/map.ply')) -> Tuple[np.ndarray, np.ndarray]:
    ms = pymeshlab.MeshSet() # create an empty meshset

    # load the mesh
    ms.load_new_mesh(path)

    # get the mesh
    mesh = ms.current_mesh()

    # Get vertices as float32 array
    vertices = mesh.vertex_matrix().astype('float32')
    
    # Get faces as uint32 array 
    faces = mesh.face_matrix().astype('uint32')

    return vertices, faces

def mesh_to_omni_stage(position = None, orientation = None):
    vertices, faces = load_mesh()

    
    trimesh_to_usd(vertices, faces)


    trimesh_to_heightmap(vertices, faces, grid_size_in_m=50, resolution_in_m=0.1)



def trimesh_to_heightmap(vetices, faces, grid_size_in_m, resolution_in_m=0.1):
    
    # Calculate the grid size
    grid_size = grid_size_in_m / resolution_in_m

    # Initialize the heightmap
    heightmap = np.zeros((int(grid_size+1), int(grid_size+1)), dtype=np.float32)

    # Define bounding box
    min_x, min_y, _ = np.min(vetices, axis=0)
    max_x, max_y, _ = np.max(vetices, axis=0)

    # Calculate the size of a grid cell
    cell_size_x = (max_x - min_x) / (grid_size)
    cell_size_y = (max_y - min_y) / (grid_size)
    print(f"Bounding box: x[{min_x}, {max_x}], y[{min_y}, {max_y}]")
    print(f"Cell sizes: x = {cell_size_x}, y = {cell_size_y}")

    # Iterate over each face to update the heightmap
    for face in faces:
        # Get the vertices
        v1, v2, v3 = vetices[face]
        #print(f"Z-values: {v1[2]}, {v2[2]}, {v3[2]}")
        # Project to 2D grid
        min_i = int((min(v1[0], v2[0], v3[0]) - min_x) / cell_size_x)
        max_i = int((max(v1[0], v2[0], v3[0]) - min_x) / cell_size_x)
        min_j = int((min(v1[1], v2[1], v3[1]) - min_y) / cell_size_y)
        max_j = int((max(v1[1], v2[1], v3[1]) - min_y) / cell_size_y)

        # Update the heightmap
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                heightmap[i, j] = max(heightmap[i, j], v1[2], v2[2], v3[2])

    return heightmap


def find_rocks_in_heightmap(heightmap, threshold=1.0):
    from scipy.signal import convolve2d
    from scipy import ndimage
    from scipy.ndimage import binary_dilation
    import cv2

    kernel = np.ones((9,9),np.uint8)
    
    # Sobel operators for gradient in x and y directions
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Compute the gradient components
    grad_x = convolve2d(heightmap, sobel_x, mode='same', boundary='wrap')
    grad_y = convolve2d(heightmap, sobel_y, mode='same', boundary='wrap')

    # Compute the overall gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Initialize a rock mask with zeros
    rock_mask = np.zeros_like(heightmap, dtype=np.int32)

    # Mark the areas where gradient magnitude is greater than the threshold as rocks
    rock_mask[grad_magnitude > threshold] = 1

    closed_rock_mask = cv2.morphologyEx(rock_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    filled_rock_mask = ndimage.binary_fill_holes(closed_rock_mask).astype(int)

    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation to add a safety margin around the filled rocks
    dilated_mask = cv2.dilate(filled_rock_mask.astype(np.uint8), kernel, iterations=1)
    return dilated_mask

def show_heightmap(heightmap):
    plt.figure(figsize=(10, 10))

    # Display the heightmap
    plt.imshow(heightmap, cmap='terrain', origin='lower')

    # Add a color bar for reference
    plt.colorbar(label='Height')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Heightmap')

    # Show the plot
    plt.show()


if __name__ == "__main__":

    vertices, faces = load_mesh()
    heightmap = trimesh_to_heightmap(vertices, faces, grid_size_in_m=60, resolution_in_m=0.1)
    rocks = find_rocks_in_heightmap(heightmap, threshold=0.7)
    show_heightmap(rocks)
    show_heightmap(heightmap)