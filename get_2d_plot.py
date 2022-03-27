import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from clustering import get_clusters, get_cluster_info, who_variant

def main():
    distance_matrix = np.load('distance_matrix.npy')
    with open('dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    cluster_distance_matrix, considered_clusters, labels = get_clusters(distance_matrix, dataset)

    embedding = MDS(n_components=2, n_jobs=-1, dissimilarity='precomputed')
    embedded_clusters = embedding.fit_transform(cluster_distance_matrix)

    colors = []
    for i, c in enumerate(considered_clusters):
        info = get_cluster_info(considered_clusters[i], labels, dataset)
        name = list(info['classes'].keys())[np.argmax(info['classes'].values())]
        who_name = who_variant(name)
        if who_name:
            name = who_name
        plt.annotate(name, (embedded_clusters[i, 0], embedded_clusters[i, 1]))
        colors.append(name)

    colors_numbers = []
    color_map = {}
    for c in colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        colors_numbers.append(color_map[c])
    plt.scatter(embedded_clusters[:, 0], embedded_clusters[:, 1], c=colors_numbers, cmap='hsv')
    plt.title('MDS embedded clusters')
    plt.show()

if __name__ == '__main__':
    main()
