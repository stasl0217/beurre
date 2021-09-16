
from utils import get_new_model
import os
print(os.getcwd())

from param import *
from trainer import run_train
import numpy as np
import torch
import random
import argparse
from dataset import *

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='main.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data', type=str, help="cn15k or nl27k")
    parser.add_argument('--task', type=str, help="mse or ndcg")

    return parser.parse_args(args)


def main(args):
    params = set_params(args.data, args.task)

    train_dataset = UncertainTripleDataset(params.data_dir, 'train.tsv')
    train_test_dataset = UncertainTripleDataset(params.data_dir, 'train.tsv')  # obsolete, not used
    dev_dataset = UncertainTripleDataset(params.data_dir, 'val.tsv')
    test_dataset = UncertainTripleDataset(params.data_dir, 'test_with_neg.tsv')


    print(params.whichmodel)
    print(params.early_stop)
    run = wandb_initialize(params)

    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    model = get_new_model(params)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.LR)

    run_train(
        model, run, train_dataset, train_test_dataset, dev_dataset, test_dataset,
        optimizer, params
    )

    print('done')


if __name__ =="__main__":
    main(parse_args())





