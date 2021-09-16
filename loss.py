from param import *
from utils import get_subset_of_given_relations
import torch.nn as nn
import param

def get_logic_loss(model, ids, params):
    # transitive rule loss regularization
    transitive_coff = torch.tensor(params.regularization['transitive']).to(params.device)
    if transitive_coff > 0:
        transitive_rule_reg = transitive_coff * model.transitive_rule_loss(ids)
    else:
        transitive_rule_reg = 0

    # composite rule loss regularization
    composite_coff = torch.tensor(params.regularization['composite']).to(params.device)
    if composite_coff > 0:
        composition_rule_reg = composite_coff * model.composition_rule_loss(ids)
    else:
        composition_rule_reg = 0

    return (transitive_rule_reg + composition_rule_reg) / len(ids)


def main_mse_loss(model, ids, cls):
    criterion = nn.MSELoss(reduction='mean')
    prediction, truth = model(ids, cls, train=True)
    mse = criterion(torch.exp(prediction), truth)
    return mse


def main_msle_loss(model, ids, cls):
    criterion = nn.MSELoss(reduction='mean')
    prediction, truth = model(ids, cls, train=True)
    mse = criterion(prediction + 1, torch.log(truth + 1))  # prediction is already log
    return mse


def main_mle_loss(model, ids, cls):
    criterion = nn.L1Loss(reduction='mean')
    prediction, truth = model(ids, cls, train=True)
    mse = criterion(prediction + 1, torch.log(truth + 1))  # prediction is already log
    return mse


def L2_regularization(model, ids, params):
    regularization = params.regularization
    device = params.device
    # regularization on delta
    delta_coff, min_coff = torch.tensor(regularization['delta']).to(device), torch.tensor(regularization['min']).to(
        device)
    delta_reg1 = delta_coff * torch.norm(torch.exp(model.delta_embedding[ids[:, 0]]), dim=1).mean()
    delta_reg2 = delta_coff * torch.norm(torch.exp(model.delta_embedding[ids[:, 2]]), dim=1).mean()

    min_reg1 = min_coff * torch.norm(model.min_embedding[ids[:, 0]], dim=1).mean()
    min_reg2 = min_coff * torch.norm(model.min_embedding[ids[:, 2]], dim=1).mean()

    rel_trans_coff = torch.tensor(regularization['rel_trans']).to(device)
    rel_trans_reg = rel_trans_coff * (
            torch.norm(torch.exp(model.rel_trans_for_head[ids[:, 1]]), dim=1).mean() + \
            torch.norm(torch.exp(model.rel_trans_for_tail[ids[:, 1]]), dim=1).mean()
    )

    rel_scale_coff = torch.tensor(regularization['rel_scale']).to(device)
    rel_scale_reg = rel_scale_coff * (
            torch.norm(torch.exp(model.rel_scale_for_head[ids[:, 1]]), dim=1).mean() + \
            torch.norm(torch.exp(model.rel_scale_for_tail[ids[:, 1]]), dim=1).mean()
    )

    L2_reg = delta_reg1 + delta_reg2 + min_reg1 + min_reg2 + rel_trans_reg + rel_scale_reg


    return L2_reg


def my_loss(model, ids, cls, params):
    NEG_RATIO = 1
    pos_loss = main_mse_loss(model, ids, cls)

    negative_samples, neg_probs = model.random_negative_sampling(ids, cls)
    neg_loss = main_mse_loss(model, negative_samples, neg_probs)

    main_loss = pos_loss + NEG_RATIO * neg_loss

    logic_loss = get_logic_loss(model, ids, params)

    L2_reg = L2_regularization(model, ids, params)

    loss = main_loss + L2_reg + logic_loss

    return loss, pos_loss, neg_loss, logic_loss
