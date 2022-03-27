import numpy as np
import pickle
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo

from clustering import get_clusters, get_cluster_info, who_variant

def main():
    distance_matrix = np.load('distance_matrix.npy')
    with open('dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    cluster_distance_matrix, considered_clusters, labels = get_clusters(distance_matrix, dataset)

    def label_from_id(clade):
        clade_str = str(clade)
        try:
            id = int(clade_str)
        except ValueError:
            return ""
        info = get_cluster_info(id, labels, dataset)
        name = list(info['classes'].keys())[np.argmax(info['classes'].values())]
        who_name = who_variant(name)
        if who_name:
            name = who_name
        return name

    names = [str(i) for i in considered_clusters]
    cluster_distance_matrix = cluster_distance_matrix.tolist()
    lower_triagular_matrix = []
    for i in range(len(cluster_distance_matrix)):
        current_row = []
        for j in range(i+1):
            current_row.append(cluster_distance_matrix[i][j])
        lower_triagular_matrix.append(current_row)
    phylo_distance_matrix = DistanceMatrix(names, matrix=lower_triagular_matrix)
    tree_constructor = DistanceTreeConstructor()
    tree = tree_constructor.nj(phylo_distance_matrix)
    Phylo.draw(tree, label_func=label_from_id)


if __name__ == '__main__':
    main()
