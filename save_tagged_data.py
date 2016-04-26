'''
Description: Converts semantic tagging data into data for seq2seq learning.
Author: Mandy Korpusik
Date: 4/26/16
'''

import codecs
import spacy.en
from collections import deque

nlp = spacy.en.English()

label_map = {0:'Food', 1:'Brand', 2:'Quantity', 3:'Description', 4:'Other'}


def write_out(fname, seqs):
    outf = open(fname, 'w')
    for seq in seqs:
        outf.write(seq.encode('utf-8')+'\n')
    outf.close()


def load_data(path="/usr/users/korpusik/PyNutrition-data/semlab.train.tagged_22,000",
              val_split=0.2):
    seqs = [] # list of sentences (strings)
    labels = [] # list of labels (as space-separated strings)
    stringbuf = [] # current sentence
    lindex = 0
    labelindices = deque() # current labels

    def processlabels(tokens):
        sindex = 0
        labels = []
        clabel = labelindices.popleft()[1]
        for token in tokens:
            if labelindices and sindex >= labelindices[0][0]:
                clabel = labelindices.popleft()[1]
            if token.pos_ == 'PUNCT' and 'x' not in token.shape_.lower():
                label_index = 4
            else:
                label_index = int(clabel) - 1
            labels.append(label_map[label_index])

            sindex += len(token.string)
        return labels

    
    with codecs.open(path, 'r', 'utf-8') as f:
            for line in f.readlines():
                # end of meal, append to list
                if line == '\n':
                    if stringbuf:
                        tokens = nlp(' '.join(stringbuf))
                        curr_labels = processlabels(tokens)

                        seqs.append(' '.join([token.orth_ for token in tokens]))
                        labels.append(' '.join(curr_labels))

                        stringbuf = []
                        lindex = 0
                        labelindices.clear()

                    continue

                # remove newlines, funky unicode stuff that sometimes appears
                line = line.strip().lstrip('\u200b')
                words, category = line.partition('|')[::2]

                labelindices.append((lindex, category))
                lindex += len(words) + 1
                stringbuf.append(words)

    # split into train/val sets
    train_seq = seqs[:int(len(seqs) * (1 - val_split))]
    train_labels = labels[:int(len(seqs) * (1 - val_split))]
    val_seq = seqs[int(len(seqs) * (1 - val_split)):]
    val_labels = labels[int(len(seqs) * (1 - val_split)):]
    
    # write to files
    for fname, seqs in [('data_semtag/src-train.txt', train_seq),
                        ('data_semtag/src-val.txt', val_seq),
                        ('data_semtag/targ-train.txt', train_labels),
                        ('data_semtag/targ-val.txt', val_labels)]:
        write_out(fname, seqs)
                

if __name__=='__main__':
    load_data()
