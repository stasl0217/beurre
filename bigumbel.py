import torch
import torch.nn as nn
# from param import *
from tensor_dataloader import *
import torch.nn.functional as F
from torch.distributions import uniform
import copy
import numpy as np

class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed


class BiGumbelBox(nn.Module):
    def __init__(self, device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params):
        super(BiGumbelBox, self).__init__()
        # super(BiGumbelBox, self).__init__(device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params)

        self.euler_gamma = 0.57721566490153286060
        self.min_init_value = min_init_value
        self.delta_init_value = delta_init_value

        min_embedding = self.init_embedding(vocab_size, embed_dim, min_init_value)
        delta_embedding = self.init_embedding(vocab_size, embed_dim, delta_init_value)
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)

        rel_trans_for_head = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        rel_scale_for_head = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        torch.nn.init.normal_(rel_trans_for_head, mean=0, std=1e-4)  # 1e-4 before
        torch.nn.init.normal_(rel_scale_for_head, mean=1, std=0.2)  # 0.2 before

        rel_trans_for_tail = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        rel_scale_for_tail = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        torch.nn.init.normal_(rel_trans_for_tail, mean=0, std=1e-4)
        torch.nn.init.normal_(rel_scale_for_tail, mean=1, std=0.2)

        # make nn.Parameter
        self.rel_trans_for_head, self.rel_scale_for_head = nn.Parameter(rel_trans_for_head.to(device)), nn.Parameter(
            rel_scale_for_head.to(device))
        self.rel_trans_for_tail, self.rel_scale_for_tail = nn.Parameter(rel_trans_for_tail.to(device)), nn.Parameter(
            rel_scale_for_tail.to(device))

        self.true_head, self.true_tail = None, None  # for negative sample filtering
        self.gumbel_beta = params.GUMBEL_BETA
        self.params = params
        self.device = device
        self.ratio = ratio
        self.vocab_size = vocab_size
        self.alpha = 1e-16
        self.clamp_min = 0.0
        self.clamp_max = 1e10
        self.REL_VOCAB_SIZE = params.REL_VOCAB_SIZE



    def forward(self, ids, probs, train=True):

        head_boxes = self.transform_head_boxes(ids)
        tail_boxes = self.transform_tail_boxes(ids)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        pos_predictions = log_prob
        return pos_predictions, probs


    def transform_head_boxes(self, ids):
        head_boxes = self.get_entity_boxes(ids[:, 0])

        rel_ids = ids[:, 1]
        relu = nn.ReLU()

        translations = self.rel_trans_for_head[rel_ids]
        scales = relu(self.rel_scale_for_head[rel_ids])

        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def transform_tail_boxes(self, ids):
        tail_boxes = self.get_entity_boxes(ids[:, 2])

        rel_ids = ids[:, 1]
        relu = nn.ReLU()

        translations = self.rel_trans_for_tail[rel_ids]
        scales = relu(self.rel_scale_for_tail[rel_ids])

        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes


    def intersection(self, boxes1, boxes2):
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(boxes1.min_embed, boxes2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(boxes1.max_embed, boxes2.max_embed)
        )

        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    def log_volumes(self, boxes, temp=1., gumbel_beta=1., scale=1.):
        eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        log_vol = torch.sum(
            torch.log(
                F.softplus(boxes.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)

        return log_vol

    def get_entity_boxes(self, entities):
        min_rep = self.min_embedding[entities]  # batchsize * embedding_size
        delta_rep = self.delta_embedding[entities]
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def init_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed

    def random_negative_sampling(self, positive_samples, pos_probs, neg_per_pos=None):
        if neg_per_pos is None:
            neg_per_pos = self.ratio
        negative_samples1 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)
        negative_samples2 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)

        corrupted_heads = [self.get_negative_samples_for_one_positive(pos, neg_per_pos, mode='corrupt_head') for pos in positive_samples]
        corrupted_tails = [self.get_negative_samples_for_one_positive(pos, neg_per_pos, mode='corrupt_tail') for pos in positive_samples]

        negative_samples1[:, 0] = torch.cat(corrupted_heads)
        negative_samples2[:, 2] = torch.cat(corrupted_tails)
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(self.device)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_probs.dtype).to(self.device)

        return negative_samples, neg_probs

    def random_negative_sampling0(self, positive_samples, pos_probs, neg_per_pos=None):
        if neg_per_pos is None:
            neg_per_pos = self.ratio
        negative_samples1 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)
        negative_samples2 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)

        # corrupt tails
        corrupted_heads = torch.randint(self.vocab_size, (negative_samples1.shape[0],)).to(self.device)
        corrupted_tails = torch.randint(self.vocab_size, (negative_samples1.shape[0],)).to(self.device)

        #filter
        bad_heads_idxs = (corrupted_heads == negative_samples1[:,0])
        bad_tails_idxs = (corrupted_tails == negative_samples2[:,2])
        corrupted_heads[bad_heads_idxs] = torch.randint(self.vocab_size, (torch.sum(bad_heads_idxs),)).to(self.device)
        corrupted_tails[bad_tails_idxs] = torch.randint(self.vocab_size, (torch.sum(bad_tails_idxs),)).to(self.device)

        negative_samples1[:, 0] = corrupted_heads
        negative_samples2[:, 2] = corrupted_tails
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(self.device)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_probs.dtype).to(self.device)

        return negative_samples, neg_probs


    def get_negative_samples_for_one_positive(self, positive_sample, neg_per_pos, mode):
        head, relation, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < neg_per_pos:
            negative_sample = np.random.randint(self.params.VOCAB_SIZE, size=neg_per_pos * 2)

            # filter true values
            if mode == 'corrupt_head' and (int(relation), int(tail)) in self.true_head:  # filter true heads
                # For test data, some (relation, tail) pairs may be unseen and not in self.true_head
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(int(relation), int(tail))],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
            elif mode == 'corrupt_tail' and (int(head), int(relation)) in self.true_tail:
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(int(head), int(relation))],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:neg_per_pos]

        negative_sample = torch.from_numpy(negative_sample)
        return negative_sample


    def head_transformation(self, head_boxes, rel_ids):
        relu = nn.ReLU()
        translations = self.rel_trans_for_head[rel_ids]
        scales = relu(self.rel_scale_for_head[rel_ids])
        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def tail_transformation(self, tail_boxes, rel_ids):
        relu = nn.ReLU()
        translations = self.rel_trans_for_tail[rel_ids]
        scales = relu(self.rel_scale_for_tail[rel_ids])
        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes

    def get_entity_boxes_detached(self, entities):
        """
        For logic constraint. We only want to optimize relation parameters, so detach entity parameters
        """
        min_rep = self.min_embedding[entities].detach()
        delta_rep = self.delta_embedding[entities].detach()
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def transitive_rule_loss(self, ids):
        subsets = [ids[(ids[:,1] == r).nonzero().squeeze(1),:] for r in self.params.RULE_CONFIGS['transitive']['relations']]
        sub_ids = torch.cat(subsets, dim=0)

        # only optimize relation parameters
        head_boxes = self.get_entity_boxes_detached(sub_ids[:, 0])
        tail_boxes = self.get_entity_boxes_detached(sub_ids[:, 2])
        head_boxes = self.head_transformation(head_boxes, sub_ids[:,1])
        tail_boxes = self.tail_transformation(tail_boxes, sub_ids[:,1])

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # P(f_r(epsilon_box)|g_r(epsilon_box)) should be 1
        vol_loss = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(tail_boxes)))
        return vol_loss

    def composition_rule_loss(self, ids):
        def rels(size, rid):
            # fill a tensor with relation id
            return torch.full((size,), rid, dtype=torch.long)

        def biconditioning(boxes1, boxes2):
            intersection_boxes = self.intersection(boxes1, boxes2)
            log_intersection = self.log_volumes(intersection_boxes)
            # || 1-P(Box2|Box1) ||
            condition_on_box1 = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(boxes1)))
            # || 1-P(Box1|Box2) ||
            condition_on_box2 = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(boxes2)))
            loss = condition_on_box1 + condition_on_box2
            return loss

        vol_loss = 0
        for rule_combn in self.params.RULE_CONFIGS['composite']['relations']:
            r1, r2, r3 = rule_combn
            r1_triples = ids[(ids[:, 1] == r1).nonzero().squeeze(1), :]
            r2_triples = ids[(ids[:, 1] == r2).nonzero().squeeze(1), :]

            # use heads and tails from r1, r2 as reasonable entity samples to help optimize relation parameters
            if len(r1_triples) > 0 and len(r2_triples) > 0:
                entities = torch.cartesian_prod(r1_triples[:,0], r2_triples[:,2])
                head_ids, tail_ids = entities[:,0], entities[:,1]
                size = len(entities)

                # only optimize relation parameters
                head_boxes_r1r2 = self.get_entity_boxes_detached(head_ids)
                tail_boxes_r1r2 = self.get_entity_boxes_detached(tail_ids)
                r1r2_head = self.head_transformation(head_boxes_r1r2, rels(size, r1))
                r1r2_head = self.head_transformation(r1r2_head, rels(size, r2))
                r1r2_tail = self.tail_transformation(tail_boxes_r1r2, rels(size, r1))
                r1r2_tail = self.tail_transformation(r1r2_tail, rels(size, r2))

                # head_boxes_r1r2 have been modified in transformation
                # so make separate box objects with the same parameters
                head_boxes_r3 = self.get_entity_boxes_detached(head_ids)
                tail_boxes_r3 = self.get_entity_boxes_detached(tail_ids)
                r3_head = self.head_transformation(head_boxes_r3, rels(size, r3))
                r3_tail = self.tail_transformation(tail_boxes_r3, rels(size, r3))

                head_transform_loss = biconditioning(r1r2_head, r3_head)
                tail_transform_loss = biconditioning(r1r2_tail, r3_tail)
                vol_loss += head_transform_loss
                vol_loss += tail_transform_loss
        return vol_loss


