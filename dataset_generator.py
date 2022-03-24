import json
import pickle
from tqdm import tqdm

from Bio import SeqIO

###############################################################################
#                       Dataset dependent variables                           #
###############################################################################

fasta_sequences = SeqIO.parse(open('ncbi_dataset/data/genomic.fna', 'r'),'fasta')
extra_information = open('ncbi_dataset/data/data_report.jsonl', 'r')
n_sequences = 883020
max_per_day = 3

###############################################################################

def main():
    """
    Go through the dataset, select at most max_per_day sequences per day
    and save as a pickle file
    """
    total_in_new_dataset = 0
    count_by_day = {}
    selected_sequences = []
    info_iter = iter(extra_information)
    for fasta in tqdm(fasta_sequences, total=n_sequences): 
        found_accession = False
        while not found_accession:
            line = next(info_iter)
            l = json.loads(line)
            if l['accession'] == fasta.id:
                found_accession = True
                new_entry = {}
                date = l['releaseDate']
                if date not in count_by_day:
                    count_by_day[date] = 1
                else:
                    count_by_day[date] += 1
                if count_by_day[date] <= max_per_day:
                    new_entry['sequence'] = str(fasta.seq)
                    new_entry['date'] = date
                    new_entry['info'] = l
                    total_in_new_dataset += 1

    print("Total on new dataset: ", total_in_new_dataset)
    with open('dataset.pickle', 'wb') as out_file:
        pickle.dump(selected_sequences, out_file)

if __name__ == '__main__':
    main()
