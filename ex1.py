
import numpy as np

from init_centroids import init_centroids
import load
import matplotlib.pyplot as plt


def round_floor(num):
    return np.floor(num * 100) / 100


def eucl_dist(a, b):
    return np.linalg.norm(a - b)


def print_centroids(centroids, iter):
    to_print = "iter " + str(iter) + ": "
    length = len(centroids)
    i = 0
    for centroid in centroids:
        to_print += "["
        num_centroids = len(centroid) - 1
        for index in range(num_centroids):
            to_print += str(round_floor(centroid[index])) + ", "
        index += 1
        to_print += str(round_floor(centroid[index])) + "]"
        if i < length - 1:
            to_print += ", "
        i += 1
    print(to_print)


def create_centroids(k):
    centroids = []
    indexes = []
    for i in range(k):
        indexes.append(0)
        centroids.append([])
        for j in range(3):
            centroids[i].append(round(0, 2))
    return centroids, indexes


def update_centroids(centr_indexes, centroids):
    updated_centroids = []
    for i in range(len(centroids)):
        if centr_indexes[i] == 0:
            updated_centroids.append(centroids)
        else:
            centroid = []
            for j in centroids[i]:
                centroid.append(j / centr_indexes[i])
            updated_centroids.append(centroid)
    return updated_centroids


def run_k_means(num_iterations, centroids_current, k, x):
    loss = []

    for iteration in range(num_iterations):
        loss_per_iter = 0
        labels = []
        centroids_updated, centr_indexes = create_centroids(k)
        num_centroids = len(centroids_current)
        for example in x:
            distance = []
            for i in range(num_centroids):
                distance.append(0)
            for centroid in range(num_centroids):
                distance[centroid] = eucl_dist(example, centroids_current[centroid])

            index = distance.index(min(distance))
            loss_per_iter += distance[index]
            current_centroid = centroids_updated[index]
            for i in range(len(current_centroid)):
                current_centroid[i] += example[i]
            centr_indexes[index] += 1
            labels.append(centroids_current[index])

        updated_centroids = update_centroids(centr_indexes, centroids_updated)
        print_centroids(centroids_current, iteration)
        centroids_current = np.array(updated_centroids)
        iteration_loss = loss_per_iter / float(len(x))
        loss.append(iteration_loss)

    return labels, loss


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
        # plot(x, vector)


if __name__ == '__main__':
    run_for_each_value()
