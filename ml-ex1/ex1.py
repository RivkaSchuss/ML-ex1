
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
            to_add = str(round_floor(centroid[index]))
            if to_add == '0.0':
                to_add = '0.'
            to_print += to_add + ", "
        index += 1
        to_add = str(round_floor(centroid[index]))
        if to_add == '0.0':
            to_add = '0.'
        to_print += to_add + "]"
        if i < length - 1:
            to_print += ", "
        i += 1
    print(to_print)


def create_centroids(k):
    centroids = []
    indexes = []

    # basic initializing of the centroids, and indexes
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

    # running each iteration
    for iteration in range(num_iterations):
        loss_per_iter = 0
        labels = []

        # creating the centroids and indexes
        centroids_updated, centr_indexes = create_centroids(k)
        num_centroids = len(centroids_current)

        # iterating over the data of the image
        for example in x:
            distance = []
            # for each centroid, calculate the distance
            for centroid in range(num_centroids):
                distance.append(0)
                # calculating the euclides distance from each value of the image, to the current centroids
                distance[centroid] = eucl_dist(example, centroids_current[centroid])

            # setting the index of each centroid to be the min distance
            index = distance.index(min(distance))

            # adding the min distance to the loss
            loss_per_iter += distance[index]

            # declaring the current centroids to be the minimum centroids in the list
            current_centroid = centroids_updated[index]

            # iterating over the minimum centroids
            for centr in range(len(current_centroid)):
                current_centroid[centr] += example[centr]

            # adding 1 to the indexes
            centr_indexes[index] += 1

            # adding the appropriate centroids to the labels for the image
            labels.append(centroids_current[index])

        # updating the centroids with the new calculated centroids
        updated_centroids = update_centroids(centr_indexes, centroids_updated)

        # printing
        print_centroids(centroids_current, iteration)

        # resetting the centroids to be the centroids recently calculated, for the next iteration
        centroids_current = np.array(updated_centroids)

        # calculating the loss
        loss_per_iter = loss_per_iter / float(len(x))
        loss.append(loss_per_iter)

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

    # getting the x from the image
    x = load.X

    # running the k means algorithm for each value of k
    for k in k_values:
        print("k=" + str(k) + ":")

        # getting the initial centroids
        centroids = init_centroids(x, k)
        vector, loss = run_k_means(num_iterations, centroids, k, x)
        # print(loss)
        # plot(x, vector)


if __name__ == '__main__':
    run_for_each_value()
