import numpy as np
import pickle

from sklearn.cluster import AgglomerativeClustering

distance_matrix = np.load('distance_matrix.npy')
with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

def get_cluster_info(cluster_id, labels):
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

def cluster_distance(cluster_id_1, cluster_id_2, labels):
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

n_clusters = 100
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
clustering.fit(distance_matrix)
labels = clustering.labels_
unlabeled_names = set()
labeled_names = set()
for cluster in range(n_clusters):
    info = get_cluster_info(cluster, labels)
    new_classes = set()
    for pango_name in info['classes']:
        name = simplify_name(pango_name)
        new_classes.add(name)
    info['classes'] = list(new_classes)
    # if info['size'] >= 0:
    #     print(info)
# print(unlabeled_names)
# print("Unlabeled_total", len(unlabeled_names), "Labeled_total", len(labeled_names))
# exit()

# Clusters with size greater that or equal to 8
considered_clusters = [cluster for cluster in range(n_clusters) if get_cluster_info(cluster, labels)['size'] >= 8]
cluster_distance_matrix = np.array([[cluster_distance(i, j, labels) for j in considered_clusters] for i in considered_clusters])
mask = np.ones(cluster_distance_matrix.shape, dtype=bool)
np.fill_diagonal(mask, 0)
min_cluster_distance = np.min(cluster_distance_matrix[mask])
max_cluster_distance = np.max(cluster_distance_matrix[mask])
cluster_distance_matrix = cluster_distance_matrix - min_cluster_distance
cluster_distance_matrix = cluster_distance_matrix / (max_cluster_distance - min_cluster_distance) + 0.1
np.fill_diagonal(cluster_distance_matrix, 0)
print(cluster_distance_matrix)
# Plot cluster sizes
import matplotlib.pyplot as plt

sizes = [get_cluster_info(cluster, labels)['size'] for cluster in range(n_clusters)]
# plt.plot(sizes)
# plt.show()

# exit()

from scipy.cluster.hierarchy import dendrogram
clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='complete')
clustering.fit(cluster_distance_matrix)

from sklearn.manifold import MDS
embedding = MDS(n_components=2, n_jobs=-1, dissimilarity='precomputed')
transformed = embedding.fit_transform(cluster_distance_matrix)

import matplotlib.pyplot as plt

colors = []
for i, c in enumerate(considered_clusters):
    info = get_cluster_info(considered_clusters[i], labels)
    name = list(info['classes'].keys())[np.argmax(info['classes'].values())]
    if name.startswith(('α', 'β', 'γ', 'δ','ο')):
        name = name.split()[0]
    plt.annotate(name, (transformed[i, 0], transformed[i, 1]))
    colors.append(name)
# Map letters to numbers
colors_numbers = []
color_map = {}
for c in colors:
    if c not in color_map:
        color_map[c] = len(color_map)
    colors_numbers.append(color_map[c])
plt.scatter(transformed[:, 0], transformed[:, 1], c=colors_numbers, cmap='hsv')
plt.show()
exit()


def llf(children, n_samples):
    def l(id):
        # print(id)
        # print(len(children))
        # print(children[id-1275])
        info = get_cluster_info(considered_clusters[id], labels)
        name = list(info['classes'].keys())[np.argmax(info['classes'].values())]
        if name.startswith(('α', 'β', 'γ', 'δ','ο')):
            return name.split()[0]
        return name
    return l
def plot_dendrogram(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    children = []
    for i, merge in enumerate(model.children_):
        current_count = 0
        current_children = []
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
                current_children.append(child_idx)
            else:
                current_count += counts[child_idx - n_samples]
                current_children += children[child_idx - n_samples]
        counts[i] = current_count
        children.append(current_children)

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    # print(linkage_matrix.shape, n_samples, len(model.children_))
    # exit()
    dendrogram(linkage_matrix, truncate_mode="level", leaf_label_func=llf(children, n_samples))
    # dendrogram(linkage_matrix, truncate_mode="level")
from matplotlib import pyplot as plt
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(clustering)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

for cluster in considered_clusters:
    print("Id:", considered_clusters.index(cluster), "Info:", get_cluster_info(cluster, labels))

plt.show()