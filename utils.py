import csv
from random import *
from collections import defaultdict
import pickle
from bigumbel import BiGumbelBox
from os.path import join
import torch

random_seed = 1


def get_vocab(filename):
    word2idx = defaultdict()
    with open(filename) as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split('\t')
            word2idx[parts[1]] = parts[0]
    return word2idx


def get_new_model(params):
    model = BiGumbelBox(params.device, params.VOCAB_SIZE, params.DIM, params.NEG_PER_POS,
                        [1e-4, 0.01], [-0.1, -0.001], params).to(params.device)
    return model


def load_hr_map(data_dir):
    file = join(data_dir, 'ndcg_test.pickle')
    with open(file, 'rb') as f:
        hr_map = pickle.load(f)
    return hr_map


def get_subset_of_given_relations(ids, rel_list):
    subs = []
    for r in rel_list:
        sub = ids[(ids[:, 1] == r).nonzero().squeeze(1)]  # sub triple set
        subs.append(sub)
    subset = torch.cat(subs, dim=0)
    return subset

def load_hr_map(data_dir):
    file = join(data_dir, 'ndcg_test.pickle')
    with open(file, 'rb') as f:
        hr_map = pickle.load(f)
    return hr_map

