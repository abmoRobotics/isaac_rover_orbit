import os
import pymeshlab
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch 
try:
    from custom_envs.rover.utils.terrain_utils.usd_utils import trimesh_to_usd
except Exception as e:
    print(f'Error importing trimesh_to_usd: {e}')
#from .usd_utils import trimesh_to_usd

# Get the directory containing the script
directory_terrain_utils = os.path.dirname(os.path.abspath(__file__))


class TerrainManager():

    def __init__(self, device):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        terrain_path = os.path.join(self.dir_path, "../terrain_data/map.ply")
        rock_mesh_path = os.path.join(self.dir_path, "../terrain_data/big_stones.ply")

        
        self.meshes = [terrain_path, rock_mesh_path]

        self.meshes = {
            "terrain": terrain_path,
            "rock": rock_mesh_path
        }

        self.heightmap = None
        self.resolution_in_m = 0.1

        vertices, faces = self.load_mesh(self.meshes["terrain"])
        self.heightmap = self.mesh_to_heightmap(vertices, faces, grid_size_in_m=60, resolution_in_m=self.resolution_in_m)
        self.rock_mask = self.find_rocks_in_heightmap(self.heightmap, threshold=0.7)
        self.spawn_locations = self.random_rover_spawns(rock_mask=self.rock_mask, n_spawns=100, seed=41, )
        if device == 'cuda:0':
            self.spawn_locations = torch.from_numpy(self.spawn_locations).cuda()

    def load_mesh(self, path) -> Tuple[np.ndarray, np.ndarray]:

        # Assert that the specified path exists in the list of meshes.
        #assert path in self.meshes, f"The provided path '{path}' must exist in the 'self.meshes' list."

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
        vertices, faces = self.load_mesh(self.meshes["terrain"])
        trimesh_to_usd(vertices, faces, position, orientation)

        if not ground_only:
            for key, value in self.meshes.items():
                if key != "terrain":
                    vertices, faces = self.load_mesh(value)
                    trimesh_to_usd(vertices, faces, position, orientation, name=key)

            # for mesh in self.meshes[1:]:
            #     vertices, faces = self.load_mesh(mesh)
            #     trimesh_to_usd(vertices, faces, position, orientation,)
    
    def mesh_to_heightmap(self, vertices, faces, grid_size_in_m, resolution_in_m=0.1):

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


    def find_rocks_in_heightmap(self, heightmap, threshold=0.5):
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
        kernel = np.ones((11, 11), np.uint8)
        safe_rock_mask = cv2.dilate(filled_rock_mask.astype(np.uint8), kernel, iterations=1)

        return safe_rock_mask

    def show_heightmap(self, heightmap, name="2D Heightmap"):
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

    def random_rover_spawns(self, rock_mask: np.ndarray, n_spawns: int = 100, min_xy: float = 13.0, max_xy: float = 47, seed = None) -> np.ndarray:
        """Generate random rover spawn locations. Calculates random x,y checks if it is a rock, if not, 
        add to list of spawn locations with corresponding z value from heightmap.

        Args:
            rock_mask (np.ndarray): A binary mask indicating the locations of rocks.
            n_spawns (int, optional): The number of spawn locations to generate. Defaults to 1.
            min_dist (float, optional): The minimum distance between two spawn locations. Defaults to 1.0.

        Returns:
            np.ndarray: An array of shape (n_spawns, 3) containing the spawn locations.
        """
        max_xy = int(max_xy / self.resolution_in_m)
        min_xy = int(min_xy / self.resolution_in_m)

        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Get the heightmap dimensions
        height, width = rock_mask.shape

        assert max_xy < width, f"max_xy ({max_xy}) must be less than width ({width})"
        assert max_xy < height, f"max_xy ({max_xy}) must be less than height ({height})"
        
        # Initialize the spawn locations array
        spawn_locations = np.zeros((n_spawns, 3), dtype=np.float32)

        # Generate spawn locations
        for i in range(n_spawns):

            valid_location = False
            while not valid_location:
                # Generate a random x and y
                x = np.random.randint(min_xy, max_xy)
                y = np.random.randint(min_xy, max_xy)

                # Check if the location is too close to a previous location
                if rock_mask[y, x] == 0:
                    valid_location = True
                    spawn_locations[i, 0] = x
                    spawn_locations[i, 1] = y
                    spawn_locations[i, 2] = self.heightmap[y, x]
        
        # Scale xy
        spawn_locations[:, 0:2] = spawn_locations[:, 0:2] * self.resolution_in_m
        return spawn_locations

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


def find_rocks_in_heightmap(heightmap, threshold=0.5):
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


def random_rover_spawns(rock_mask: np.ndarray, n_spawns: int = 10, min_dist: float = 1.0, seed = None) -> np.ndarray:
    """Generate random rover spawn locations. Calculates random x,y checks if it is a rock, if not, 
    add to list of spawn locations with corresponding z value from heightmap.

    Args:
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.
        n_spawns (int, optional): The number of spawn locations to generate. Defaults to 1.
        min_dist (float, optional): The minimum distance between two spawn locations. Defaults to 1.0.

    Returns:
        np.ndarray: An array of shape (n_spawns, 3) containing the spawn locations.
    """

    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Get the heightmap dimensions
    height, width = rock_mask.shape

    # Initialize the spawn locations array
    spawn_locations = np.zeros((n_spawns, 3), dtype=np.float32)

    # Generate spawn locations
    for i in range(n_spawns):

        valid_location = False
        while not valid_location:
            # Generate a random x and y
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)

            # Check if the location is too close to a previous location
            if rock_mask[y, x] == 0:
                valid_location = True
                spawn_locations[i, 0] = x
                spawn_locations[i, 1] = y
                spawn_locations[i, 2] = heightmap[y, x]

        # Check if the location is a rock
        # if rock_mask[y, x] == 0:
        #     # Add the location to the spawn locations array
        #     spawn_locations[i, 0] = x
        #     spawn_locations[i, 1] = y
        #     spawn_locations[i, 2] = heightmap[y, x]
        # else:
        #     # If the location is a rock, generate a new location
        #     i -= 1
        #     continue

        # # Check if the location is too close to a previous location
        # if i > 0:
        #     # Calculate the distance between the current location and all previous locations
        #     distances = np.linalg.norm(spawn_locations[:i, 0:2] - spawn_locations[i, 0:2], axis=1)

        #     # Check if any of the distances are less than the minimum distance
        #     if np.any(distances < min_dist):
        #         # If so, generate a new location
        #         i -= 1
        #         continue

    return spawn_locations

def visualize_spawn_points(spawn_locations: np.ndarray, heightmap: np.ndarray, rock_mask: np.ndarray):
    """
    Visualize the spawn locations on the heightmap.

    Args:
        spawn_locations (np.ndarray): An array of shape (n_spawns, 3) containing the spawn locations.
        heightmap (np.ndarray): A 2D array representing the heightmap.
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.

    Returns:
        None
    """
    
    fig = plt.figure()

    # Plotting 3D scatter plot for spawn locations
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D Visualization of Spawn Locations')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')
    ax1.scatter(spawn_locations[:, 0], spawn_locations[:, 1], spawn_locations[:, 2], c='r', marker='o')

    # Plotting 2D heightmap with spawn points
    ax2 = fig.add_subplot(122)
    ax2.set_title('2D Heightmap with Spawn Locations')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    
    ax2.imshow(heightmap, cmap='terrain', origin='lower')
    
    # Overlay rock mask
    ax2.imshow(np.ma.masked_where(rock_mask == 0, rock_mask), cmap='coolwarm', alpha=0.4)

    # Overlay spawn points
    ax2.scatter(spawn_locations[:, 0], spawn_locations[:, 1], c='r', marker='o')
    #plt.colorbar(label='Height')
    plt.show()

def visualize_spawn_points2(spawn_locations: np.ndarray, heightmap: np.ndarray, rock_mask: np.ndarray):
    """
    Visualize the spawn locations on the heightmap.

    Args:
        spawn_locations (np.ndarray): An array of shape (n_spawns, 3) containing the spawn locations.
        heightmap (np.ndarray): A 2D array representing the heightmap.
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.

    Returns:
        None
    """
    
    fig = plt.figure()

    # Create a 3D axis object
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D Visualization of Spawn Locations and Heightmap')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')

    # Create meshgrid for surface plot
    x = np.linspace(0, heightmap.shape[1] - 1, heightmap.shape[1])
    y = np.linspace(0, heightmap.shape[0] - 1, heightmap.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plotting the heightmap as a surface
    ax1.plot_surface(X, Y, heightmap, alpha=0.5, cmap='viridis')
    
    # Overlay spawn locations as scatter plot
    ax1.scatter(spawn_locations[:, 0], spawn_locations[:, 1], spawn_locations[:, 2], c='r', marker='o', s=50)

    # Plotting 2D heightmap with spawn points
    ax2 = fig.add_subplot(122)
    ax2.set_title('2D Heightmap with Spawn Locations')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.imshow(heightmap, cmap='terrain', origin='lower')
    
    # Overlay rock mask
    ax2.imshow(np.ma.masked_where(rock_mask == 0, rock_mask), cmap='coolwarm', alpha=0.4)

    # Overlay spawn points
    ax2.scatter(spawn_locations[:, 0], spawn_locations[:, 1], c='r', marker='o')
    
    plt.show()

def visualize_spawn_points3(spawn_locations: np.ndarray, heightmap: np.ndarray, rock_mask: np.ndarray):
    """
    Visualize the spawn locations on the heightmap as separate plots.

    Args:
        spawn_locations (np.ndarray): An array of shape (n_spawns, 3) containing the spawn locations.
        heightmap (np.ndarray): A 2D array representing the heightmap.
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.

    Returns:
        None
    """
    
    # Create a 3D plot for heightmap and spawn locations
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('3D Visualization of Spawn Locations and Heightmap')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')

    x = np.linspace(0, heightmap.shape[1] - 1, heightmap.shape[1])
    y = np.linspace(0, heightmap.shape[0] - 1, heightmap.shape[0])
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(X, Y, heightmap, alpha=0.5, cmap='viridis')
    ax1.scatter(spawn_locations[:, 0], spawn_locations[:, 1], spawn_locations[:, 2], c='r', marker='o', s=50)
    
    plt.show()

    # Create a 2D plot for heightmap and spawn locations
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('2D Heightmap with Spawn Locations')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    
    ax2.imshow(heightmap, cmap='terrain', origin='lower')
    #ax2.imshow(np.ma.masked_where(rock_mask == 0, rock_mask), cmap='coolwarm', alpha=0.4)
    ax2.scatter(spawn_locations[:, 0], spawn_locations[:, 1], c='r', marker='o')

    plt.show()

def visualize_spawn_points4(spawn_locations: np.ndarray, heightmap: np.ndarray, rock_mask: np.ndarray):
    """
    Visualize the spawn locations on the heightmap as separate plots.

    Args:
        spawn_locations (np.ndarray): An array of shape (n_spawns, 3) containing the spawn locations.
        heightmap (np.ndarray): A 2D array representing the heightmap.
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.

    Returns:
        None
    """

    # Create a 3D plot for heightmap and spawn locations
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('3D Visualization of Spawn Locations and Heightmap')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')

    x = np.linspace(0, heightmap.shape[1] - 1, heightmap.shape[1])
    y = np.linspace(0, heightmap.shape[0] - 1, heightmap.shape[0])
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(X, Y, heightmap, alpha=0.5, cmap='viridis')
    ax1.scatter(spawn_locations[:, 0], spawn_locations[:, 1], spawn_locations[:, 2], c='r', marker='o', s=50)

    # Create a 2D plot for heightmap and spawn locations
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('2D Heightmap with Spawn Locations')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')

    ax2.imshow(heightmap, cmap='terrain', origin='lower')
    #ax2.imshow(np.ma.masked_where(rock_mask == 0, rock_mask), cmap='coolwarm', alpha=0.4)
    ax2.scatter(spawn_locations[:, 0], spawn_locations[:, 1], c='r', marker='o')

    plt.show()

def visualize_spawn_points5(spawn_locations: np.ndarray, heightmap: np.ndarray):
    """
    Visualize the spawn locations on the heightmap as separate plots.

    Args:
        spawn_locations (np.ndarray): An array of shape (n_spawns, 3) containing the spawn locations.
        heightmap (np.ndarray): A 2D array representing the heightmap.
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.

    Returns:
        None
    """

    # Create a 3D plot for heightmap and spawn locations
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('3D Visualization of Spawn Locations and Heightmap')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')
    #ax1.set_xlim([0, 600])  # Set the limits for the X-axis
    ax1.set_zlim([0, 10])
    x = np.linspace(0, 600, heightmap.shape[1])  # Adjust the scale for the X-axis
    y = np.linspace(0, heightmap.shape[0] - 1, heightmap.shape[0])
    X, Y = np.meshgrid(x, y)

    

    ax1.plot_surface(X, Y, heightmap, alpha=0.5, cmap='viridis')
    ax1.scatter(spawn_locations[:, 0], spawn_locations[:, 1], spawn_locations[:, 2]+0.5, c='r', marker='o', s=50)

    # Create a 2D plot for heightmap and spawn locations
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('2D Heightmap with Spawn Locations')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')

    ax2.imshow(heightmap, cmap='terrain', origin='lower')
    #ax2.imshow(np.ma.masked_where(rock_mask == 0, rock_mask), cmap='coolwarm', alpha=0.4)
    ax2.scatter(spawn_locations[:, 0], spawn_locations[:, 1], c='r', marker='o')

    plt.show()

if __name__ == "__main__":
    terrain = TerrainManager(device='cpu')
    # vertices, faces = terrain.load_mesh(terrain.meshes[0])
    # heightmap = terrain.mesh_to_heightmap(vertices, faces, grid_size_in_m=60, resolution_in_m=0.1)
    # rock_mask = terrain.find_rocks_in_heightmap(heightmap, threshold=0.7)
    # spawn_locations = terrain.random_rover_spawns(rock_mask=rock_mask, n_spawns=100, min_dist=1.0, seed=41)
    # vertices, faces = load_mesh()
    # heightmap = trimesh_to_heightmap(vertices, faces, grid_size_in_m=60, resolution_in_m=0.1)
    # rock_mask = find_rocks_in_heightmap(heightmap, threshold=0.7)
    # # show_heightmap(rock_mask)
    # # show_heightmap(heightmap)

    # # heightmap = np.random.rand(100, 100)  # Replace with your actual heightmap
    # # rock_mask = np.random.randint(0, 2, (100, 100))  # Replace with your actual rock mask

    # Generate spawn locations using the random_rover_spawns function
    #spawn_locations = random_rover_spawns(rock_mask=rock_mask, n_spawns=1000, min_dist=1.0, seed=41)

    # Visualize the spawn locations using the visualize_spawn_points function
    show_heightmap(terrain.rock_mask)
    spawns = terrain.spawn_locations
    spawns[:, 0:2] = spawns[:, 0:2] * 10
    #terrain.spawn_locations[0:2] = terrain.spawn_locations[0:2] * 10
    visualize_spawn_points5(spawns, terrain.heightmap)