import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from clustering import get_complete_clustering, get_cluster_info, who_variant

def main():
    distance_matrix = np.load('distance_matrix.npy')
    with open('dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    clustering, considered_clusters, labels = get_complete_clustering(distance_matrix, dataset)

    def label_from_id(id):
        info = get_cluster_info(considered_clusters[id], labels, dataset)
        name = list(info['classes'].keys())[np.argmax(info['classes'].values())]
        who_name = who_variant(name)
        if who_name:
            name = who_name
        return name

    def plot_dendrogram(model):
        # Create linkage matrix and then plot the dendrogram
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            current_children = []
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 0  # leaf node
                    current_children.append(child_idx)
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        dendrogram(linkage_matrix, truncate_mode="level", leaf_label_func=label_from_id)

    plt.title("Hierarchical clustering complete-linkage Dendrogram")
    plot_dendrogram(clustering)
    plt.xlabel("Simplified lineage name")
    plt.show()

if __name__ == '__main__':
    main()
