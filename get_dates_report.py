import numpy as np
import pickle
from tabulate import tabulate

from clustering import get_clusters, get_cluster_info, who_variant

def main():
    distance_matrix = np.load('distance_matrix.npy')
    with open('dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    _, considered_clusters, labels = get_clusters(distance_matrix, dataset)

    variants = {}
    for i in range(len(considered_clusters)):
        info = get_cluster_info(considered_clusters[i], labels, dataset)
        name = list(info['classes'].keys())[np.argmax(info['classes'].values())]
        who_name = who_variant(name)
        if who_name:
            name = who_name
            if name not in variants:
                variants[name] = {'start': info['start_date'], 'end': info['end_date']}
            else:
                variants[name]['start'] = min(variants[name]['start'], info['start_date'])
                variants[name]['end'] = max(variants[name]['end'], info['end_date'])
    
    variants_keys = list(variants.keys())
    starts = [variants[v]['start'] for v in variants_keys]
    approx_starts = ['2021-05-XX', '2020-11-XX', '2021-11-XX', '2021-02-XX']
    ends = [variants[v]['end'] for v in variants_keys]
    table = zip(variants, starts, approx_starts, ends)

    print(tabulate(table, headers=['Variant', 'Start date', 'Approx onset date', 'End date']))
    
if __name__ == '__main__':
    main()
