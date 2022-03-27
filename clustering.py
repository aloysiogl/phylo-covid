import numpy as np

from sklearn.cluster import AgglomerativeClustering


def get_cluster_info(cluster_id, labels, dataset):
    indices = np.where(labels == cluster_id)[0]
    info = {}
    info['start_date'] = min(dataset[i]['date'] for i in indices)
    info['end_date'] = max(dataset[i]['date'] for i in indices)
    info['size'] = len(indices)
    classes = {}
    for ind in indices:
        pango_name = dataset[ind]['info']['virus']['pangolinClassification']
        simplified_name = simplify_name(pango_name)
        if simplified_name != 'unclassifiable':
            if simplified_name not in classes:
                classes[simplified_name] = 1
            else:
                classes[simplified_name] += 1
    info['classes'] = classes
    return info

def cluster_distance(cluster_id_1, cluster_id_2, labels, distance_matrix):
    indices_1 = np.where(labels == cluster_id_1)[0]
    indices_2 = np.where(labels == cluster_id_2)[0]
    return np.mean([distance_matrix[i, j] for i in indices_1 for j in indices_2])

def simplify_name(name):
    split_name = name.split('.')
    if name.startswith('AY.'):
        return 'δ (B.1.617.2 like)'
    elif name.startswith('P.1.') or name == 'P.1':
        return 'γ (B.1.1.28.1 like)'
    elif name.startswith('BA.') or name == 'B.1.1.529':
        return 'ο (B.1.1.529 like)'
    elif name == 'B.1.351':
        return 'β (B.1.351 like)'
    elif name == 'B.1.1.7' or name.startswith('Q.'):
        return 'α (B.1.1.7 like)'
    elif name == 'B.1.1.316' or name.startswith('R.'):
        return 'B.1.1.316 like'
    elif name == 'B.1.1.25' or name.startswith('D.'):
        return 'B.1.1.25 like'
    elif name == 'B.1.1.306' or name.startswith('AE.'):
        return 'B.1.1.306 like'
    elif name == 'B.1.1.33' or name.startswith('N.'):
        return 'B.1.1.33 like'
    elif name.startswith('C.'):
        return 'B.1.1.1 like'
    elif name.startswith('B.1.595.'):
        return 'B.1.595 like'
    elif len(split_name) == 3 and split_name[0] == 'B' and split_name[1] == '1':
        return 'B.1 like'
    elif len(split_name) == 4 and split_name[0] == 'B' and split_name[1] == '1' and split_name[2] == '1':
        return 'B.1.1 like'
    elif len(split_name) == 2 and split_name[0] == 'B':
        return 'B like'
    elif name == 'B':
        return 'B'
    elif name == 'A':
        return 'A'
    elif len(split_name) == 2 and split_name[0] == 'A':
        return 'A like'
    elif len(split_name) == 3 and split_name[0] == 'A' and split_name[1] == '1':
        return 'A.1 like'
    elif name.startswith('B.1.258'):
        return 'B.1.258 like'
    elif name.startswith('B.1.36'):
        return 'B.1.36 like'
    elif name.startswith('B.1'):
        return 'B.1 like'
    elif name.startswith('B'):
        return 'B like'
    elif name.startswith('A.2.5'):
        return 'A.2.5 like'
    else:
        return 'unclassifiable'

def who_variant(name):
    if name.startswith(('α', 'β', 'γ', 'δ','ο')):
        return name.split()[0]
    return False

def get_clusters(distance_matrix, dataset, n_clusters=100, min_cluster_size=8):     
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
    clustering.fit(distance_matrix)
    labels = clustering.labels_

    considered_clusters = [cluster for cluster in range(n_clusters) if get_cluster_info(cluster, labels, dataset)['size'] >= min_cluster_size]
    cluster_distance_matrix = np.array([[cluster_distance(i, j, labels, distance_matrix) for j in considered_clusters] for i in considered_clusters])
    mask = np.ones(cluster_distance_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    min_cluster_distance = np.min(cluster_distance_matrix[mask])
    max_cluster_distance = np.max(cluster_distance_matrix[mask])
    cluster_distance_matrix = cluster_distance_matrix - min_cluster_distance
    cluster_distance_matrix = cluster_distance_matrix / (max_cluster_distance - min_cluster_distance) + 0.1
    np.fill_diagonal(cluster_distance_matrix, 0)
    return cluster_distance_matrix, considered_clusters, labels

def get_complete_clustering(distance_matrix, dataset):
    cluster_distance_matrix, considered_clusters, labels = get_clusters(distance_matrix, dataset)
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='complete')
    clustering.fit(cluster_distance_matrix)

    return clustering, considered_clusters, labels
