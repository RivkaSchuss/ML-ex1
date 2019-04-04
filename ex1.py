from math import floor

import numpy as np

from init_centroids import init_centroids
import load
import matplotlib.pyplot as plt


def eucl_dist(a, b):
    return np.linalg.norm(a - b)


def print_centroids(centroids, iter):
    print("iter " + str(iter) + ": ", end=" ")
    for centroid in centroids:
        print(str(centroid) + " ", end=" ")
    print()


def create_centroids(k):
    centroids = []
    indexes = []
    for i in range(k):
        indexes.append(0)
        centroids.append([])
        for j in range(3):
            centroids[i].append(round(0, 2))
    return indexes, centroids


def update_centroids(centr_indexes, centroids):
    updated_centroids = []
    for i in range(len(centroids)):
        if centr_indexes[i] == 0:
            temp_centroids = [round(centr, 2) for centr in centroids[i]]
            updated_centroids.append(temp_centroids)
        else:
            centroid = []
            for j in centroids[i]:
                update = j / centr_indexes[i]

                centroid.append(round(update, 2))
            updated_centroids.append(centroid)
    return updated_centroids


def run_k_means(num_iterations, centroids_current, k, x):
    loss = []
    for iteration in range(num_iterations):
        iteration_loss = 0
        labels_pic = []

        centr_indexes, centroids_updated = create_centroids(k)
        num_centroids = len(centroids_current)
        for example in x:
            distance = []
            for i in range(num_centroids):
                distance.append(0)
            for centroid in range(num_centroids):
                distance[centroid] = eucl_dist(example, centroids_current[centroid])

            index = distance.index(min(distance))
            iteration_loss += distance[index]
            for i in range(len(centroids_updated[index])):
                centroids_updated[index][i] += example[i]
            centr_indexes[index] += 1
            labels_pic.append(centroids_current[index])

        updated_centroids = update_centroids(centr_indexes, centroids_updated)
        print_centroids(centroids_current, iteration)
        centroids_current = np.array(updated_centroids)
        iteration_loss = iteration_loss / float(len(x))
        loss.append(iteration_loss)

    return labels_pic, loss


def plot(x, vector):
    a = load.A
    a_norm = load.A_norm
    result = np.zeros(x.shape)
    for i in range(len(vector)):
        result[i] = vector[i]
    new_img = result.reshape(a.shape)
    plt.imshow(new_img)
    plt.show()
    plt.imshow(a_norm)
    plt.grid(False)
    plt.show()


def run_for_each_value():
    num_iterations = 11
    k_values = [2, 4, 8, 16]
    x = load.X

    for k in k_values:
        print("k=" + str(k))
        centroids = init_centroids(x, k)
        vector, loss = run_k_means(num_iterations, centroids, k, x)
        plot(x, vector)


if __name__ == '__main__':
    run_for_each_value()
