'''
Description: Converts USDA-Freebase matched data into seq2seq learning format.
Author: Mandy Korpusik
Date: 4/26/16
'''

import csv
from save_tagged_data import write_out


def load_data(usdaf='../keras/USDA-encoder/foodsWithNutrients.csv',
              freebasef='../keras/USDA-encoder/freebaseEquiv.csv', val_split=0.2):
    seqs = [] # list of Freebase descriptions
    labels = [] # list of USDA ids

    with open(freebasef, 'rb') as csvf:
        reader = csv.reader(csvf, delimiter=';')
        for row in reader:
            freebase = row[0]
            usda_id = row[2]
            seqs.append(freebase)
            labels.append(usda_id)

    # split into train/val sets
    train_seq = seqs[:int(len(seqs) * (1 - val_split))]
    train_labels = labels[:int(len(seqs) * (1 - val_split))]
    val_seq = seqs[int(len(seqs) * (1 - val_split)):]
    val_labels = labels[int(len(seqs) * (1 - val_split)):]
    
    # write to files
    for fname, seqs in [('data_usda/src-train.txt', train_seq),
                        ('data_usda/src-val.txt', val_seq),
                        ('data_usda/targ-train.txt', train_labels),
                        ('data_usda/targ-val.txt', val_labels)]:
        write_out(fname, seqs)
                

if __name__=='__main__':
    load_data()
