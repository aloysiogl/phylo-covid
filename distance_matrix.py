import zlib
import sys
import pickle
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm

def Z(x):
    compressed = zlib.compress(x.encode('utf-8'))
    return sys.getsizeof(compressed)

def NCD(x, y):
    z_x = Z(x)
    z_y = Z(y)

    z_x_y = Z(x + y)
    z_y_x = Z(y + x)
    z_xy = (z_x_y + z_y_x) / 2

    return (z_xy - min(z_x, z_y)) / max(z_x, z_y)

def compute_distances(dataset, i, n):
    distances = []
    for j in range(i+1, n):
            distance = NCD(dataset[i]['sequence'], dataset[j]['sequence'])
            distances.append(distance)
    return distances

def main():
    with open('dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    n = len(dataset)
    distance_matrix = np.zeros((n, n))

    # for i in tqdm(range(n)):
    #     distances = compute_distances(dataset, distance_matrix, i, n)

    #     for j in range(i+1, n):
    #         distance_matrix[i, j] = distances[j - i - 1]

    distances = Parallel(n_jobs=4)(delayed(compute_distances)(dataset, i, n) for i in tqdm(range(n)))

    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = distances[i][j-i-1]
            distance_matrix[j, i] = distances[i][j-i-1]

    np.save('distance_matrix.npy', distance_matrix)

if __name__ == '__main__':
    main()