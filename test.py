
import torch
import argparse
from dataset import *
from traintest import test_func, evaluate_ndcg
from utils import load_hr_map

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='test-pretrained.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data', type=str, help="cn15k or nl27k")
    parser.add_argument('--task', type=str, help="mse or ndcg")
    parser.add_argument('--model_path', type=str, help="trained model file.")

    return parser.parse_args(args)


def main(args):
    model_path = args.model_path
    data_name = args.data
    task = args.task

    data_dir = join('./data', data_name)
    if data_name == 'cn15k':
        num_entity = 15000
    else:
        num_entity = 27221
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = UncertainTripleDataset(data_dir, 'test.tsv')

    model = torch.load(model_path)

    if task == 'mse':
        test_MSE, test_MAE, _, _ = test_func(test_dataset, device, model, {}, ndcg_also=False)
        print('test MSE', test_MSE)
        print('test MAE', test_MAE)
    elif task == 'ndcg':
        linear_ndcg, exp_ndcg = evaluate_ndcg(model, load_hr_map(data_dir), num_entity)
        print('test linear ndcg', linear_ndcg)
        print('test exponential ndcg', exp_ndcg)


if __name__ =="__main__":
    main(parse_args())
