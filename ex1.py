from math import floor

import numpy as np

from init_centroids import init_centroids
import load
import matplotlib.pyplot as plt


def eucl_dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def run_k_means():
    num_iterations = 10
    k_values = [2, 4, 8, 16]
    x = load.X
    cluster = np.zeros(x.shape[0])
    for k in k_values:
        print("k=" + str(k))
        centroids_arr = init_centroids(x, k)
        print('iter 0 ', end=" ")
        for item in centroids_arr:
            print("[", end='')
            for center in item:
                print(center, end=", ")
            print("]", end=",")
        print("")
        # centroids_old = np.zeros(centroids_arr.shape)
        for iteration in range(num_iterations):

            # err = eucl_dist(centroids_arr, centroids_old, None)
            # while err != 0:

            for i in range(len(x)):
                distances = eucl_dist(x[i], centroids_arr)
                clust = np.argmin(distances)
                cluster[i] = clust

            # centroids_old = np.copy(centroids_arr)

            for i in range(k):
                points = [x[j] for j in range(len(x)) if cluster[j] == i]
                if points:
                    centroids_arr[i] = np.mean(points, axis=0)

            # calculation difference between new centroid and old centroid values
            # err = eucl_dist(centroids_arr, centroids_old, None)
            #
            # for i in range(k):
            #     d = [eucl_dist(x[j], centroids_arr[i], None) for j in range(len(x)) if cluster[j] == i]
            #     error += np.sum(d)

            # print('iter 0' + " " + str(centroids_arr))
            print('iter ' + str(iteration + 1), end=" ")
            for item in centroids_arr:
                print("[", end='')
                for center in item:
                    print(center, end=", ")

                print("]", end=",")
                # print(item, end=" ")
            print("")




        plt.imshow(x)
        plt.grid(False)
        plt.show()


if __name__ == '__main__':
    run_k_means()
