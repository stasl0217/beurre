import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset, TensorDataset


class TripleDataset(TensorDataset):
    """Pairwise Probability dataset"""

    def __init__(self, filenames):
        read = False
        for filename in filenames:
            with open(filename, 'rb') as f:
                if read:
                    temp = pickle.load(f)
                    data = np.concatenate((data, temp), axis=0)
                else:
                    read = True
                    data = pickle.load(f)

        self.ids = torch.from_numpy(data[:, :3].astype(np.long))
        if data.shape[1]>4:
            self.probs = torch.from_numpy(data[:, 3:].astype(np.float32))

        else:
            self.probs = torch.from_numpy(data[:, 3].astype(np.long))
        self.length = self.ids.shape[0]
        super().__init__(self.ids, self.probs)

    def __getitem__(self, index):
        return self.ids[index], self.probs[index]

    def __len__(self):
        return self.length



class UncertainTripleDataset(TensorDataset):
    def __init__(self, data_dir, filename):
        df = pd.read_csv(join(data_dir, filename), sep='\t', names=['h', 'r', 't', 'p'])

        data = df[['h', 'r', 't']].values

        prob = df['p'].values

        self.ids = torch.from_numpy(data.astype(np.long)).long()
        self.probs = torch.from_numpy(prob.astype(np.float32))
        self.length = self.ids.shape[0]
        super().__init__(self.ids, self.probs)

        self.true_head, self.true_tail = self.get_true_head_and_tail(self.ids)


    def __getitem__(self, index):
        return self.ids[index], self.probs[index]

    def __len__(self):
        return self.length

    def get_true_head_and_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head0, relation0, tail0 in triples:
            head, relation, tail = int(head0), int(relation0), int(tail0)
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

