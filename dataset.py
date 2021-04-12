import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset, TensorDataset

class PairDataset(TensorDataset):
    """Pairwise Probability dataset"""

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            data =  pickle.load(f)
        self.ids = torch.from_numpy(data[:, :2].astype(np.long))
        self.probs = torch.from_numpy(data[:, 2].astype(np.float32))
        self.length = self.ids.shape[0]

    def __getitem__(self, index):
        return self.ids[index], self.probs[index]

    def __len__(self):
        return self.length

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
                    # data analysis
                    # temp = data[661:662]
                    # temp = np.concatenate((temp, data[694:695]), axis=0)
                    # temp = np.concatenate((temp, data[718:719]), axis=0)
                    # data = temp
                    # data analysis
                    # print('data', data)
        # print('Data shape', data.shape)
        self.ids = torch.from_numpy(data[:, :3].astype(np.long))
        if data.shape[1]>4:
            self.probs = torch.from_numpy(data[:, 3:].astype(np.float32))
            # print(self.probs.shape)
            # self.probs = torch.from_numpy(np.asarray([i/10 for i in range(10)]).reshape(-1, 1))
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
        # positive_sample = self.ids[index]
        # pos_prob = self.probs[index]

        # negative_samples1 = positive_sample.repeat_interleave(self.negative_sample_size, 1)
        # negative_samples2 = positive_sample.repeat_interleave(self.negative_sample_size, 1)
        #
        # corrupted_heads = [self.get_negative_samples(pos, mode='corrupt_head') for pos in positive_sample]
        # corrupted_tails = [self.get_negative_samples(pos, mode='corrupt_tail') for pos in positive_sample]
        #
        # negative_samples1[:, 0] = torch.cat(corrupted_heads)
        # negative_samples2[:, 2] = torch.cat(corrupted_tails)
        # negative_samples = torch.cat((negative_samples1, negative_samples2), 0)
        # neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_prob.dtype)
        #
        # return positive_sample, pos_prob, negative_samples, neg_probs

        return self.ids[index], self.probs[index]

    def __len__(self):
        return self.length

    # @staticmethod
    # def collate_fn(data):
    #     positive_samples = torch.stack([_[0] for _ in data], dim=0)
    #     positive_probs = torch.stack([_[1] for _ in data], dim=0)
    #     negative_samples = torch.stack([_[2] for _ in data], dim=0)
    #     negative_probs = torch.cat([_[3] for _ in data], dim=0)
    #     return positive_samples, positive_probs, negative_samples, negative_probs

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

