import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Check if line is not empty
                coords = line.strip().split()
                if len(coords) >= 3:  # Expect at least x, y, z coordinates
                    point = [float(coord) for coord in coords[:3]]
                    points.append(point)
    return np.array(points)

def visualize_point_cloud_with_axis(points, symmetry_axis=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Point Cloud')

    # Plot symmetry axis if provided
    if symmetry_axis is not None:
        symmetry_axis = symmetry_axis
        origin = np.mean(points, axis=0)
        arrow_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))  # Length of arrow based on point cloud spread
        ax.quiver(*origin, symmetry_axis[0], symmetry_axis[1], symmetry_axis[2], color='k', label='Symmetry Axis', length=arrow_length)

    if symmetry_axis is not None:
        symmetry_axis = -symmetry_axis
        origin = np.mean(points, axis=0)
        arrow_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))  # Length of arrow based on point cloud spread
        ax.quiver(*origin, symmetry_axis[0], symmetry_axis[1], symmetry_axis[2], color='k', label='Symmetry Axis', length=arrow_length)



    origin = np.zeros(3)
    axes_length = arrow_length
    ax.quiver(*origin, axes_length, 0, 0, color='r', label='X axis')
    ax.quiver(*origin, 0, axes_length, 0, color='g', label='Y axis')
    ax.quiver(*origin, 0, 0, axes_length, color='b', label='Z axis')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud with Symmetry Axis')
    ax.legend()

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])
    ax.set_box_aspect([1,1,1])  # Set equal scaling for all axes



    plt.show()


def find_symmetry_axes(points, num_axes=2):
    # Calculate PCA to find principal axes
    pca = PCA(n_components=num_axes)
    pca.fit(points)

    # Retrieve principal axes (components) from PCA
    symmetry_axes = pca.components_

    return symmetry_axes

def normalize_vector(v):
    norm = np.linalg.norm(v)  # Compute the Euclidean norm (magnitude) of the vector
    if norm == 0:
        return v  # Handle zero-length vector
    return v / norm  # Return the normalized vector

# Example usage
if __name__ == "__main__":
    # Generate example 3D point cloud
    np.random.seed(0)

    print("51_large_clamp")
    points = load_xyz_file("/content/51_large_clamp.xyz")

    # Find symmetry axes
    symmetry_axes = find_symmetry_axes(points, num_axes=1)

    for i, axis in enumerate(symmetry_axes):

        print(f"Symmetry Axis {i+1}: {axis}")
        print(f"Normilized: {normalize_vector(axis)}")
        visualize_point_cloud_with_axis(points, axis)


    print("52_extra_large_clamp")
    points = load_xyz_file("/content/52_extra_large_clamp.xyz")

    # Find symmetry axes
    symmetry_axes = find_symmetry_axes(points, num_axes=1)

    for i, axis in enumerate(symmetry_axes):

        print(f"Symmetry Axis {i+1}: {axis}")
        print(f"Normilized: {normalize_vector(axis)}")
        visualize_point_cloud_with_axis(points, axis)


    print("36_wood_block")
    points = load_xyz_file("/content/36_wood_block.xyz")

    # Find symmetry axes
    symmetry_axes = find_symmetry_axes(points, num_axes=2)

    for i, axis in enumerate(symmetry_axes):

        print(f"Symmetry Axis {i+1}: {axis}")
        print(f"Normilized: {normalize_vector(axis)}")
        visualize_point_cloud_with_axis(points, axis)







