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


def get_subset_of_given_relations(ids, rel_list):
    subs = []
    for r in rel_list:
        sub = ids[(ids[:, 1] == r).nonzero().squeeze(1)]  # sub triple set
        subs.append(sub)
    subset = torch.cat(subs, dim=0)
    return subset


class SoftBox(nn.Module):
    def __init__(self, device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params):
        super(SoftBox, self).__init__()

        self.min_init_value = min_init_value
        self.delta_init_value = delta_init_value

        min_embedding = self.init_word_embedding(vocab_size, embed_dim, min_init_value)
        delta_embedding = self.init_word_embedding(vocab_size, embed_dim, delta_init_value)
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)

        rel_min_embedding = self.init_word_embedding(params.REL_VOCAB_SIZE, embed_dim, min_init_value)
        rel_delta_embedding = self.init_word_embedding(params.REL_VOCAB_SIZE, embed_dim, delta_init_value)
        self.rel_min_embedding = nn.Parameter(rel_min_embedding)
        self.rel_delta_embedding = nn.Parameter(rel_delta_embedding)

        self.device = device
        self.ratio = ratio
        self.vocab_size = vocab_size
        self.temperature = 1.0

        self.alpha = 1e-16
        self.clamp_min = 0.0
        self.clamp_max = 1e10

        self.REL_VOCAB_SIZE = params.REL_VOCAB_SIZE

    def forward(self, ids, probs, train=False):
        """Returns box embeddings for ids (batchsize*3)"""
        #         # generate negative examples if we need random examples
        #         if self.ratio >=1 and train:
        #             idx_with_negatives, probs_with_negative = self.random_negative_sampling(ids, probs)
        #             ids = idx_with_negatives
        #             probs = probs_with_negative

        min_rep = self.min_embedding[ids]  # batchsize * 3 * embedding_size
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep + torch.exp(delta_rep)

        #  
        rel_min_rep = self.rel_min_embedding[ids[:, 1]]
        rel_delta_rep = self.rel_delta_embedding[ids[:, 1]]
        rel_max_rep = rel_min_rep + torch.exp(rel_delta_rep)

        boxes1 = Box(min_rep[:, 0, :], max_rep[:, 0, :])
        boxes2 = Box(rel_min_rep[:, :], rel_max_rep[:, :])
        boxes3 = Box(min_rep[:, 2, :], max_rep[:, 2, :])
        # calculate intersection
        three_way_intersection = self.three_way_intersection(boxes1, boxes2, boxes3)
        log_intersection = self.log_volumes(three_way_intersection)

        #  log_prob = log_intersection-self.universe_log_volume()

        #   - conditioning
        # condition on small
        log_prob = log_intersection - self.smaller_log_volumes(boxes1, boxes3)

        # condtion on subject
        # log_prob = log_intersection-self.log_volumes(boxes1)

        # condition on object
        # log_prob = log_intersection-self.log_volumes(boxes3)

        pos_predictions = log_prob
        return pos_predictions, probs

    def smaller_log_volumes(self, boxes1, boxes2):
        volumes1 = self.log_volumes(boxes1)
        volumes2 = self.log_volumes(boxes2)
        smaller, _ = torch.min(torch.stack([volumes1, volumes2], dim=1), dim=1)
        return smaller

    def universe_log_volume(self):
        universe_box_min = torch.min(self.min_embedding, dim=0)[0].view(1, -1)
        max_embedding = self.min_embedding + torch.exp(self.delta_embedding)
        universe_box_max = torch.max(max_embedding, dim=0)[0].view(1, -1)
        # print(universe_box_min.size())
        # print(universe_box_max.size())
        # print(universe_box_min)
        # print(universe_box_max)
        universe_box = Box(universe_box_min, universe_box_max)
        return self.log_volumes(universe_box)

    def get_entity_boxes(self, entities):
        #  
        min_rep = self.min_embedding[entities]  # batchsize * embedding_size
        delta_rep = self.delta_embedding[entities]
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def log_volumes(self, boxes):
        # MIN_VOL added by  
        MIN_VOL = -10
        # vol = torch.log(torch.clamp(F.softplus(boxes.delta_embed), self.clamp_min, self.clamp_max)+torch.tensor(self.alpha)).sum(1)

        #  
        vol = torch.log(F.softplus(boxes.delta_embed) + torch.tensor(self.alpha)).sum(1)
        # vol[vol != vol] = MIN_VOL  # replace nan
        return vol

    def intersection(self, boxes1, boxes2):
        intersections_min = torch.max(boxes1.min_embed, boxes2.min_embed)
        intersections_max = torch.min(boxes1.max_embed, boxes2.max_embed)
        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    def three_way_intersection(self, boxes1, boxes2, boxes3):
        middle_intersection = self.intersection(boxes1, boxes2)
        three_way_intersection = self.intersection(middle_intersection, boxes3)
        return three_way_intersection

    def get_cond_probs(self, boxes1, boxes2):
        log_intersection = self.log_volumes(self.intersection(boxes1, boxes2))
        log_box2 = self.log_volumes(boxes2)
        return torch.exp(log_intersection - log_box2)

    def init_word_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed



    def random_negative_sampling1(self, pos_triples, probs):
        batch_size, k = pos_triples.shape
        num_neg_sample = pos_triples.shape[0] * self.ratio
        negative_samples1 = pos_triples.repeat(self.ratio, 1)
        negative_samples2 = pos_triples.repeat(self.ratio, 1)

        # corrupt tails
        negative_samples1[:, 2] = torch.randint(self.vocab_size, (num_neg_sample,))
        # corrupt head
        negative_samples2[:, 0] = torch.randint(self.vocab_size, (num_neg_sample,))
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(device)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=probs.dtype).to(self.device)

        return negative_samples, neg_probs

    def random_negative_sampling0(self, batch_in, batch_out):
        with torch.no_grad():
            batch_size, k = batch_in.shape
            num_neg_sample = batch_in.shape[0] * self.ratio
            negative_samples = batch_in.repeat(self.ratio, 1)  # shape (batch_size * ratio, k)

            #  
            neg = torch.randint(self.vocab_size, (num_neg_sample, k))
            neg[:, 1] = torch.randint(self.REL_VOCAB_SIZE, (num_neg_sample,))
            negative_samples = neg.to(device)

            # original
            # negative_samples.scatter_(1, torch.randint(k,(num_neg_sample,1)).to(self.device),
            #                          torch.randint(self.vocab_size, (num_neg_sample,1)).to(self.device))
            # # We remove indices which happened to be the same
            # negative_samples = negative_samples[negative_samples[:,0] != negative_samples[:,1]]

            negative_probs = torch.zeros(negative_samples.shape[0], dtype=batch_out.dtype).to(self.device)
            batch_in = torch.cat((batch_in, negative_samples), 0)
            batch_out = torch.cat((batch_out, negative_probs), 0)
            return (batch_in, batch_out)

    def loss(self, input, target):
        # cross entropy loss for log probabilities
        """
        :param input: log probabilities
        :param target: target probabilities
        """
        return -((target * input) + ((1 - target) * torch.log(1 - torch.exp(input)))).mean(dim=0)

    def get_triple_boxes(self, ids):
        min_rep = self.min_embedding[ids]  # batchsize * 3 * embedding_size
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep + torch.exp(delta_rep)
        boxes1 = Box(min_rep[:, 0, :], max_rep[:, 0, :])
        boxes2 = Box(min_rep[:, 1, :], max_rep[:, 1, :])
        boxes3 = Box(min_rep[:, 2, :], max_rep[:, 2, :])
        # calculate intersection
        triple_boxes = self.three_way_intersection(boxes1, boxes2, boxes3)
        return triple_boxes, boxes1, boxes2, boxes3

    def get_svo_boxes(ids):
        min_rep = self.min_embedding[ids]  # batchsize * 3 * embedding_size
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep + torch.exp(delta_rep)
        boxes1 = Box(min_rep[:, 0, :], max_rep[:, 0, :])
        boxes2 = Box(min_rep[:, 1, :], max_rep[:, 1, :])
        boxes3 = Box(min_rep[:, 2, :], max_rep[:, 2, :])
        return boxes1, boxes2, boxes3

    #  , logic rules

    # not fixed, don't use it
    def contrastive_loss(self, ids, small_col=0, large_col=2):
        min_rep = self.min_embedding[ids]  # batchsize * 3 * embedding_size
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep + torch.exp(delta_rep)
        small_boxes = Box(min_rep[:, small_col, :], max_rep[:, small_col, :])
        large_boxes = Box(min_rep[:, large_col, :], max_rep[:, large_col, :])

        hinge_loss_min = large_boxes.min_embed - small_boxes.min_embed
        hinge_loss_min[hinge_loss_min < 0] = 0
        hinge_loss_min = hinge_loss_min.sum(dim=1)

        hinge_loss_max = small_boxes.max_embed - large_boxes.max_embed
        hinge_loss_max[hinge_loss_max < 0] = 0
        hinge_loss_max = hinge_loss_max.sum(dim=1)

        hinge_loss = (hinge_loss_min + hinge_loss_max).mean()

        return hinge_loss

    # not fixed, don't use it
    def contrastive_loss_vol(self, ids, small_col=0, large_col=2):
        min_rep = self.min_embedding[ids]  # batchsize * 3 * embedding_size
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep + torch.exp(delta_rep)
        small_boxes = Box(min_rep[:, small_col, :], max_rep[:, small_col, :])
        large_boxes = Box(min_rep[:, large_col, :], max_rep[:, large_col, :])

        log_vol_small = torch.exp(self.log_volumes(small_boxes))
        log_vol_large = torch.exp(self.log_volumes(large_boxes))

        hinge_loss = log_vol_small - log_vol_large
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss = hinge_loss.mean()

        return hinge_loss

    # not fixed, don't use it
    def and_rule_loss(self, ids1, ids2):
        # (A,r,B)^(B,r,C)=>(A,r,C)
        triple_boxes1, A_boxes, r_boxes, _ = self.get_triple_boxes(ids1)
        triple_boxes2, _, _, C_boxes = self.get_triple_boxes(ids2)

        # AND operation
        and_boxes = self.intersection(triple_boxes1, triple_boxes2)
        and_prob = torch.exp(self.log_volumes(and_boxes))

        # implied boxes (A,r,C)
        implied_boxes = self.three_way_intersection(A_boxes, r_boxes, C_boxes)
        implied_prob = torch.exp(self.log_volumes(and_boxes))

        # implied_prob >= and_prob
        hinge_loss = implied_prob - and_prob
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss = hinge_loss.mean()
        return hinge_loss

    # not fixed, don't use it
    def implication_rule_loss(self, ids1, ids2):
        # ids1 => ids2
        triple_boxes1, _, _, _ = self.get_triple_boxes(ids1)
        triple_boxes2, _, _, _ = self.get_triple_boxes(ids2)

        rule_body_prob = torch.exp(self.log_volumes(triple_boxes1))
        rule_head_prob = torch.exp(self.log_volumes(triple_boxes2))

        # rule_head_prob >= rule_body_prob
        hinge_loss = rule_body_prob - rule_head_prob
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss = hinge_loss.mean()

        return hinge_loss


# ## AffineBox

# In[37]:


class AffineBox(SoftBox):
    def __init__(self, device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params):
        super(AffineBox, self).__init__(device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params)

        self.scale_is_log = params.AFFINR_SCALE_IS_LOG  # how to parameterize scale

        # redefine relation
        rel_dim = 1

        rel_trans = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        torch.nn.init.normal_(rel_trans, mean=0, std=1e-3)
        rel_scale = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        if self.scale_is_log:
            torch.nn.init.normal_(rel_scale, mean=0, std=1)
        else:
            torch.nn.init.normal_(rel_scale, mean=1, std=0.2)

        self.rel_trans, self.rel_scale = nn.Parameter(rel_trans.to(device)), nn.Parameter(rel_scale.to(device))

    def forward(self, ids, probs, train=True):
        """Returns box embeddings for ids (batchsize*3)"""

        head_boxes = self.transform_head_boxes(ids)
        tail_boxes = self.get_entity_boxes(ids[:, 2])

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        pos_predictions = log_prob
        return pos_predictions, probs

    def get_relation_scales(self, rel_ids):
        """
        Return scales based on different parameterization ways
        """
        if self.scale_is_log:
            scales = torch.exp(self.rel_scale[rel_ids])
        else:
            relu = nn.ReLU()
            scales = relu(self.rel_scale[rel_ids])
            # scales = F.softplus(self.rel_scale[rel_ids])  # softplus performance is worse than relu
        return scales

    def transform_head_boxes(self, ids):
        #  
        head_boxes = self.get_entity_boxes(ids[:, 0])

        rel_ids = ids[:, 1]

        translations = self.rel_trans[rel_ids]
        scales = self.get_relation_scales(rel_ids)

        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def contrastive_loss_vol(self, ids, small='head'):
        """For monotonic relations"""
        head_boxes = self.get_entity_boxes(ids[:, 0])
        tail_boxes = self.get_entity_boxes(ids[:, 2])

        transformed_head_boxes = self.transform_head_boxes(ids)

        log_vol_transformed_head = torch.exp(self.log_volumes(transformed_head_boxes))
        log_vol_tail = torch.exp(self.log_volumes(tail_boxes))

        if small == 'head':
            log_vol_small, log_vol_large = log_vol_transformed_head, log_vol_tail
        else:
            log_vol_small, log_vol_large = log_vol_tail, log_vol_transformed_head

        hinge_loss = log_vol_small - log_vol_large
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss = hinge_loss.mean()

        return hinge_loss

    def transitive_rule_loss(self, rel_ids, l2_coefficient=1):
        """
        For transitivity, constrain that there is no transformation for such relations.
        Add L2 regularization for such relations
        """

        translations = self.rel_trans[rel_ids]

        # translation should be 0
        L2_on_translation = torch.tensor(l2_coefficient).to(self.params.device) * torch.norm(translations, dim=1).mean()

        # Should not constrain the relation scales as 1, otherwise much worse performance

        return L2_on_translation

    def implication_rule_loss(self, relation_pairs, l2_coefficient=1):
        """
        :param relation_pairs: list[(r1, r2)]  (a, r1, b) => (a, r2, b)
        """
        # (a, r1, b) => (a, r2, b)
        # r2 (rule head relation) should have larger scale than r1 (rule body relation)
        rule_body_relations = [p[0] for p in relation_pairs]
        rule_head_relations = [p[1] for p in relation_pairs]
        scales1 = self.get_relation_scales(rule_body_relations)
        scales2 = self.get_relation_scales(rule_head_relations)

        scale_diff = torch.sum(
            torch.stack((
                scales1,
                -scales2
            )),
            dim=0
        )
        scale_diff[scale_diff < 0] = 0  # hinge
        implication_loss = torch.tensor(l2_coefficient).to(self.params.device) * scale_diff.mean()
        return implication_loss


# ## BiAffineBox

# In[38]:


class BiAffineBox(SoftBox):
    def __init__(self, device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params):
        super(BiAffineBox, self).__init__(device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params)

        # redefine relation
        rel_dim = 1
        rel_trans_for_head = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        rel_scale_for_head = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        torch.nn.init.normal_(rel_trans_for_head, mean=0, std=1e-3)
        torch.nn.init.normal_(rel_scale_for_head, mean=1, std=0.2)

        rel_trans_for_tail = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        rel_scale_for_tail = torch.empty(params.REL_VOCAB_SIZE, params.DIM)
        torch.nn.init.normal_(rel_trans_for_tail, mean=0, std=1e-3)
        torch.nn.init.normal_(rel_scale_for_tail, mean=1, std=0.2)

        # make nn.Parameter
        self.rel_trans_for_head, self.rel_scale_for_head = nn.Parameter(rel_trans_for_head.to(device)), nn.Parameter(
            rel_scale_for_head.to(device))
        self.rel_trans_for_tail, self.rel_scale_for_tail = nn.Parameter(rel_trans_for_tail.to(device)), nn.Parameter(
            rel_scale_for_tail.to(device))

    def forward(self, ids, probs, train=True):
        """Returns box embeddings for ids (batchsize*3)"""

        head_boxes = self.transform_head_boxes(ids)
        tail_boxes = self.transform_tail_boxes(ids)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        pos_predictions = log_prob
        return pos_predictions, probs

    def transform_head_boxes(self, ids):
        #  
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


# ## GumbelBox

# In[39]:


class GumbelBox(AffineBox):
    def __init__(self, device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params):
        super(GumbelBox, self).__init__(device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params)

        self.euler_gamma = 0.57721566490153286060

    def forward(self, ids, probs, train=True):
        head_boxes = self.transform_head_boxes(ids)
        tail_boxes = self.get_entity_boxes(ids[:, 2])

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        pos_predictions = log_prob
        return pos_predictions, probs

    # override SoftBox
    def intersection(self, boxes1, boxes2):
        # logsumexp: multivariable softplus, smooth way to compute max

        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)),
            0
        )

        # print('head min', boxes1.min_embed[:5,:5])
        # print('tail min', boxes2.min_embed[:5,:5])
        # print('intersections_min', intersections_min[:5,:5])

        intersections_min = torch.max(
            intersections_min,
            torch.max(boxes1.min_embed, boxes2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)),
            0
        )
        # print('head max', boxes1.max_embed[:5,:5])
        # print('tail max', boxes2.max_embed[:5,:5])
        # print('intersections_max', intersections_max[:5,:5])
        intersections_max = torch.min(
            intersections_max,
            torch.min(boxes1.max_embed, boxes2.max_embed)
        )
        # print('intersections_max', intersections_max)

        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    # override SoftBox
    # _log_soft_volume_adjusted
    def log_volumes(self, boxes, temp=1., gumbel_beta=1., scale=1.):
        eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        # print('box delta', boxes.delta_embed)

        log_vol = torch.sum(
            torch.log(
                F.softplus(boxes.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)

        # print('log_vol', log_vol)

        return log_vol



class BiGumbelBox(GumbelBox):
    def __init__(self, device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params):
        super(BiGumbelBox, self).__init__(device, vocab_size, embed_dim, ratio, min_init_value, delta_init_value, params)

        # redefine relation
        rel_dim = 1
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

        self.inverse_dict = get_inverse_relations()

        self.true_head, self.true_tail = None, None

        self.gumbel_beta = params.GUMBEL_BETA

        self.params = params




    def forward(self, ids, probs, train=True):
        """Returns box embeddings for ids (batchsize*3)"""

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
        relu = nn.ReLU()  # relu worked better than softplus

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


    def transitive_rule_loss0(self, rel_ids, epsilon=0.1, l2_coefficient=1):
        """
        epsilon: small box size. Boxes that are larger than this size will be penalized
        """
        num_epsilon_box = len(rel_ids)

        epsilon_box_min = self.init_word_embedding(num_epsilon_box, self.params.DIM, self.min_init_value)
        epsilon_box_delta = torch.full((num_epsilon_box, self.params.DIM), fill_value=epsilon)
        epsilon_box_max = epsilon_box_min + epsilon_box_delta

        boxes1 = Box(epsilon_box_min.clone().to(self.params.device), epsilon_box_max.clone().to(self.params.device))
        boxes2 = Box(epsilon_box_min.clone().to(self.params.device), epsilon_box_max.clone().to(self.params.device))

        head_boxes = self.head_transformation(boxes1, rel_ids)
        tail_boxes = self.tail_transformation(boxes2, rel_ids)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # P(f_r(epsilon_box)|g_r(epsilon_box)) should be 1
        log_vol_loss = (1 - torch.exp(log_intersection - self.log_volumes(tail_boxes))).mean()
        return log_vol_loss

    def transitive_rule_loss1(self, rel_ids, l2_coefficient=1):
        """
        For transitivity, constrain that there is no transformation for such relations.
        Add L2 regularization for such relations
        """

        translations1 = self.rel_trans_for_head[rel_ids]
        translations2 = self.rel_trans_for_tail[rel_ids]

        # translation should be 0
        L2_on_translation = torch.tensor(l2_coefficient).to(self.params.device) * (
                    torch.norm(translations1, dim=1).mean() + torch.norm(translations2, dim=1).mean())

        # Should not constrain the relation scales as 1, otherwise much worse performance

        return L2_on_translation

    def inverse_rule_loss(self, batch_rel_ids):
        # find inverse relation pairs from this batch
        pairs = [[int(r1.detach()), self.inverse_dict[int(r1.detach())]] for r1 in batch_rel_ids if
                 int(r1.detach()) in self.inverse_dict]
        pairs = torch.Tensor(pairs).type(torch.LongTensor).to(self.device)

        left = pairs[:, 0]
        right = pairs[:, 1]
        left_trans_head = self.rel_trans_for_head[left]
        left_trans_tail = self.rel_trans_for_tail[left]
        right_trans_head = self.rel_trans_for_head[right]
        right_trans_tail = self.rel_trans_for_tail[right]

        diff = torch.sum(
            torch.norm(left_trans_head + left_trans_tail, dim=1) + torch.norm(right_trans_head + right_trans_tail,
                                                                               dim=1)) / self.params.BATCH_SIZE

        return diff


    def transitive_rule_loss(self, batch_rel_ids, epsilon=0.1):
        """
        epsilon: small box size. Boxes that are larger than this size will be penalized
        """
        # print(batch_rel_ids)
        # print(batch_rel_ids == RULE_CONFIGS['transitive']['relations'][0])
        subsets = [batch_rel_ids[(batch_rel_ids == r).nonzero().squeeze(1)] for r in self.params.RULE_CONFIGS['transitive']['relations']]
        rel_ids = torch.cat(subsets, dim=0)

        num_epsilon_box = len(rel_ids)

        epsilon_box_min = self.init_word_embedding(num_epsilon_box, self.params.DIM, self.min_init_value)
        epsilon_box_delta = torch.full((num_epsilon_box, self.params.DIM), fill_value=epsilon)
        epsilon_box_max = epsilon_box_min + epsilon_box_delta

        boxes1 = Box(epsilon_box_min.clone().to(self.params.device), epsilon_box_max.clone().to(self.params.device))
        boxes2 = Box(epsilon_box_min.clone().to(self.params.device), epsilon_box_max.clone().to(self.params.device))

        head_boxes = self.head_transformation(boxes1, rel_ids)
        tail_boxes = self.tail_transformation(boxes2, rel_ids)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # P(f_r(epsilon_box)|g_r(epsilon_box)) should be 1
        vol_loss = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(tail_boxes))) / self.params.BATCH_SIZE
        return vol_loss

    def random_negative_sampling0(self, positive_samples, pos_probs, neg_per_pos=None):
        if neg_per_pos is None:
            neg_per_pos = self.ratio
        negative_samples1 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)
        negative_samples2 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)

        corrupted_heads = [self.get_negative_samples_for_one_positive(pos, mode='corrupt_head') for pos in positive_samples]
        corrupted_tails = [self.get_negative_samples_for_one_positive(pos, mode='corrupt_tail') for pos in positive_samples]

        negative_samples1[:, 0] = torch.cat(corrupted_heads)
        negative_samples2[:, 2] = torch.cat(corrupted_tails)
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(self.device)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_probs.dtype).to(self.device)

        return negative_samples, neg_probs

    def random_negative_sampling(self, positive_samples, pos_probs, neg_per_pos=None):
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


    def get_negative_samples_for_one_positive(self, positive_sample, mode):
        head, relation, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.ratio:
            negative_sample = np.random.randint(self.params.VOCAB_SIZE, size=self.ratio * 2)

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

        negative_sample = np.concatenate(negative_sample_list)[:self.ratio]

        negative_sample = torch.from_numpy(negative_sample)
        return negative_sample
