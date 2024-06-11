import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm 


def normalize(data):
    # ==================================================
    # complete the code
    # ==================================================

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    data_norm = (data - mean) / std

    return data_norm 

 
def compute_covariance(data):
    # ==================================================
    # complete the code
    # ==================================================
    Sigma = np.cov(np.transpose(data))
    return Sigma


def get_principal_component_first(data):
    # ==================================================
    # complete the code
    # ==================================================
    cov_matrix = compute_covariance(data)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    pc = sorted_eigenvectors[:, 0]
    return pc
    
    
def get_principal_component_second(data):
    # ==================================================
    # complete the code
    # ==================================================
    cov_matrix = compute_covariance(data)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    pc = sorted_eigenvectors[:, 1]
    return pc 
    

def plot_principal_component(data, pc1, pc2):   
    # ==================================================
    # complete the code
    # ==================================================

    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], label='Data points')

    eigvals, eigvecs = np.linalg.eig(compute_covariance(data))
    idx_1 = np.argmax(eigvals)
    idx_2 = np.argsort(eigvals)[-2]

    direction_1 = eigvecs[:,idx_1] * eigvals[idx_1]
    direction_2 = eigvecs[:,idx_2] * eigvals[idx_2]
    plt.annotate('', xy=direction_1, xytext=(0,0), arrowprops=dict(facecolor='red', edgecolor='red', width=1.0))
    plt.annotate('', xy=direction_2, xytext=(0,0), arrowprops=dict(facecolor='blue', edgecolor='blue', width=1.0))
    # plt.quiver(*direction_1, *pc1, scale=3, color='r', label='First Principal Component')
    # plt.quiver(*direction_2, *pc2, scale=3, color='b', label='Second Principal Component')

    plt.legend()
    plt.axis('equal')

    plt.show()


def plot_projection_principal_component(data, pc):
    # ==================================================
    # complete the code
    # ==================================================

    projection_1st_pc = data @ pc

    # Reconstruct the projected points in the original space
    projected_data_1st_pc = np.outer(projection_1st_pc, pc)

    # Plot the original data and the projection on the first principal component
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.5)
    plt.scatter(projected_data_1st_pc[:, 0], projected_data_1st_pc[:, 1], label='Projection on 1st PC', alpha=0.5)

    # Draw lines from original points to projected points
    for i in range(data.shape[0]):
        plt.plot([data[i, 0], projected_data_1st_pc[i, 0]], 
                [data[i, 1], projected_data_1st_pc[i, 1]], 'r--')

    plt.legend()
    plt.axis('equal')

    plt.show()
