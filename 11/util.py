import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm


def compute_distance(data, centroid):
    # ==================================================
    # complete the code
    # ==================================================

    distance = np.zeros((np.size(data, 0), np.size(data, 0)))
    distance = np.sqrt(np.sum((data[:, np.newaxis] - centroid) ** 2, axis=2))

    return distance


def compute_centroid(data, label_data, num_cluster, centroid_prev):
    # ==================================================
    # complete the code
    # ==================================================
    
    centroid = np.zeros((num_cluster, np.size(data, 1)))

    for i in range(num_cluster):
        centroid[i] = np.mean(data[label_data == i], axis=0)
    return centroid


def compute_label_data(distance):
    # ==================================================
    # complete the code
    # ==================================================
    
    data_label = np.argmin(distance, axis=1)

    return data_label


def compute_loss(distance, data_label):
    # ==================================================
    # complete the code
    # ==================================================

    loss = 0

    for i in range(len(data_label)):
        loss += distance[i, data_label[i]]

    loss = np.sum(loss)
    loss /= len(data_label)

    return loss


def plot_data_label(data, label_data, num_cluster):
    # ==================================================
    # complete the code
    # ==================================================

    plt.figure(figsize=(10, 10))

    plt.scatter(data[:, 0], data[:, 1], c=label_data)

    plt.tight_layout()

    plt.show()


def plot_centroid(data, centroid_iter):
    # ==================================================
    # complete the code
    # ==================================================

    plt.figure(figsize=(10, 10))

    plt.scatter(data[:, 0], data[:, 1], c='gray')
    plt.scatter(centroid_iter[0, :, 0], centroid_iter[0, :, 1], c='blue', marker='o', label='initial')
    plt.scatter(centroid_iter[-1, :, 0], centroid_iter[-1, :, 1], c='red', marker='s', label='final')

    for i in range(len(centroid_iter[0])):
        plt.plot(centroid_iter[:, i, 0], centroid_iter[:, i, 1], '-')

    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()    