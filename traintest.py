from tensor_dataloader import TensorDataLoader
from dataset import *
from torch.optim.lr_scheduler import StepLR
import time

def evaluate_mse(prediction, truth):
    pred = prediction.detach().cpu()
    truth_np = truth.detach().cpu().numpy()

    mse = (np.square(pred - truth_np)).mean()

    mae = (np.absolute(pred - truth_np)).mean()
    return mse, mae


def evaluate_ndcg(model, hr_map, num_entity):
    tester = Tester(model, num_entity)
    mean_linear_ndcg, mean_exp_ndcg = tester.mean_ndcg(hr_map)
    return mean_linear_ndcg, mean_exp_ndcg


def test_func(test_data, device, model, params, threshold=None, neg_mse_also=False, ndcg_also=False):
    # return mse, mae, neg_mse, ndcg
    # neg also: to test negative samples separately (used for validation)
    data = TensorDataLoader(test_data, batch_size=test_data.length)
    for ids, cls in data:
        ids, cls = ids.to(device), cls.to(device)

        # special_subset_indices = (ids[:, 1] == special_rel_index).nonzero().squeeze(1)

        with torch.no_grad():
            prediction, truth = model(ids, cls)

            score = prediction
            label = truth
            mse, mae = evaluate_mse(torch.exp(score), label)

            ndcg = None
            if ndcg_also:
                ndcg = evaluate_ndcg(model, params.hr_map, params.VOCAB_SIZE)

            if not neg_mse_also:
                return mse, mae, None, ndcg

            # test for negative samples
            negative_samples, neg_probs = model.random_negative_sampling(ids, cls, neg_per_pos=1)
            neg_prediction, _ = model(negative_samples, neg_probs)
            neg_mse, neg_mae = evaluate_mse(torch.exp(neg_prediction), neg_probs)

            combined_mae = (mae+neg_mae)/(1+params.NEG_RATIO)  # for validation

            return mse, combined_mae, neg_mse, ndcg


def train_func(train_data, train_test_data, dev_data, test_data,
               best_metric, optimizer, rule_configs, device, model, params,
               verbose=True,
               sub_dataset=None, verb_dataset=None, obj_dataset=None,
               sub_pair_dataset=None, verb_pair_dataset=None, obj_pair_dataset=None):
    # Train the model
    train_loss = 0
    batch_size = 2 * params.BATCH_SIZE
    data = TensorDataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.true_head, model.true_tail = train_data.true_head, train_data.true_tail  # for negative sampling

    step = 0
    for ids, cls in data:
        model.train()
        ids, cls = ids.to(device), cls.to(device)

        loss, pos_loss, neg_loss, logic_loss = my_loss(model, ids, cls)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        step += 1

        if step % 20 == 0 or (batch_size > 4096 and step % 10 ==0):
            wandb.log({'Train Loss': loss})

            # test with neg MSE
            model.eval()

            # test
            test_MSE, test_MAE, _, _ = test_func(test_data, device, model, params, ndcg_also=False)
            wandb.log({'Test MSE': test_MSE})
            wandb.log({'Test MAE': test_MAE})

            # validation
            valid_pos_mse, valid_mae, valid_neg_mse, _ = test_func(dev_data, device, model, params, neg_mse_also=True)
            valid_mse = (valid_pos_mse + params.NEG_RATIO * valid_neg_mse) / (1 + params.NEG_RATIO)
            wandb.log({'Valid MSE': valid_mse})
            wandb.log({'Valid pos MSE': valid_pos_mse})
            wandb.log({'Valid neg MSE': valid_neg_mse})
            wandb.log({'Valid MAE': valid_mae})

            if verbose:
                print(
                    f'\tLoss: {loss:.10f}(train)\t|pos_loss: {pos_loss:.3f}\t|neg_loss: {neg_loss:.3f}\t|logic_loss:{logic_loss:.3f}')
                print(f'\t\tTest (with neg) MSE: {test_MSE:.3f}')
                print(f'\t\tValid MSE: {valid_mse:.3f}\t|Valid pos: {valid_pos_mse:.3f}|Valid neg: {valid_neg_mse:.3f}')


            if test_MSE < best_metric['test_mse']:
                best_metric['test_mse'] = test_MSE
                wandb.log({'Best Test MSE': best_metric['test_mse']})

            if test_MAE < best_metric['test_mae']:
                best_metric['test_mae'] = test_MAE
                wandb.log({'Best Test MAE': best_metric['test_mae']})

            if valid_mse < best_metric['valid_mse']:
                best_metric['valid_mse'] = valid_mse
                wandb.log({'Best Valid MSE': best_metric['valid_mse']})

            if valid_mae < best_metric['valid_mae']:
                best_metric['valid_mae'] = valid_mae
                wandb.log({'Best Valid MAE': best_metric['valid_mae']})

    # at the end of each epoch, test nDCG
    test_MSE, test_MAE, _, _ = test_func(test_data, device, model, params, ndcg_also=False)
    print(f'Test MSE: {test_MSE}')
    print(f'Test MAE: {test_MAE}')

    return train_loss, best_metric


def run_train(
        model, run, train_dataset, train_test_dataset, dev_dataset, test_dataset,
        optimizer, params, verbose=True):

    if params.lrschedule:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # lr linear decay


    best_metric = {
        'test_mse': 1,
        'valid_mse': 1,
        'ndcg': 0,
        'test_mae': 100,
        'valid_mae': 100
    }

    last_best_metric = best_metric.copy()
    last_best_ndcg = 0

    last_best_epoch = 0  # for early stopping

    start_time = time.time()

    for epoch in range(params.EPOCH):

        loss, best_metric = train_func(train_dataset, train_test_dataset, dev_dataset, test_dataset,
                                       best_metric, optimizer, params.RULE_CONFIGS, params.device, model, params=params, verbose=verbose)
        # with open('../pytorch_svo/data/probability/embedding'+str(epoch)+'.pkl', 'wb') as f:
        # 	mydict = {'min_embedding': model.min_embedding.cpu(),
        # 			  'delta_embedding': model.delta_embedding.cpu()}
        # 	pickle.dump(mydict, f)
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))

        if params.early_stop == 'ndcg' and epoch % 10 == 0:
            print('####NDCG####')
            linear_ndcg, exp_ndcg = evaluate_ndcg(model)
            print(f'Test ndcg (linear, exp): {linear_ndcg:.3f}, {exp_ndcg:.3f}')
            wandb.log({'ndcg': linear_ndcg})
            wandb.log({'exp_ndcg': exp_ndcg})

            if linear_ndcg > last_best_ndcg:
                last_best_ndcg = linear_ndcg
                last_best_epoch = epoch
                wandb.log({'best_ndcg': linear_ndcg})
                wandb.log({'best_exp_ndcg': exp_ndcg})
                wandb.log({'epoch': last_best_epoch})

                torch.save(model, join(params.model_dir, f'{params.whichmodel}-{wandb.run.id}.pt'))
            else:
                if epoch >= 1 and epoch-last_best_epoch >= 20:
                    print('***best epoch:***', last_best_epoch)
                    print('***best metric:***', last_best_metric)
                    wandb.log({'epoch': last_best_epoch})
                    break  # early stop

        # early stopping
        # stop if best_metric didn't improve in last 10 epochs

        print('best', best_metric['valid_mse'], 'last best', last_best_metric['valid_mse'])
        if params.early_stop == 'valid_mse':
            if epoch >= 1 and best_metric['valid_mse'] >= last_best_metric['valid_mse']:  # no improvement or already overfit
                print('epoch', epoch, 'last_best_epoch', last_best_epoch)
                if epoch - last_best_epoch >= 50:  # patience
                    print('***best epoch:***', last_best_epoch)
                    print('***best metric:***', last_best_metric)
                    wandb.log({'epoch': last_best_epoch})

                    # run.finish()  # end wandb watch
                    break
            else:
                last_best_metric = best_metric.copy()
                last_best_epoch = epoch

                torch.save(model, join(params.model_dir, f'{params.whichmodel}-{wandb.run.id}.pt'))

        if params.early_stop == 'valid_mae':
            if epoch >= 1 and best_metric['valid_mae'] >= last_best_metric['valid_mae']:  # no improvement or already overfit
                print('epoch', epoch, 'last_best_epoch', last_best_epoch)
                if epoch - last_best_epoch >= 50:  # patience
                    print('***best epoch:***', last_best_epoch)
                    print('***best metric:***', last_best_metric)
                    wandb.log({'epoch': last_best_epoch})

                    # run.finish()  # end wandb watch
                    break
            else:
                last_best_metric = best_metric.copy()
                last_best_epoch = epoch

                torch.save(model, join(params.model_dir, f'{params.whichmodel}-{wandb.run.id}.pt'))


        if params.lrschedule:
            scheduler.step()


class NDCGRankingTestDataset(TensorDataset):
    def __init__(self, h, r, num_entities):
        self.h, self.r = h, r
        self.num_entities = num_entities
        self.length = num_entities

        # make candidate list for ranking task
        self.candidate_triples = self.get_all_candidate_triples()

    def get_all_candidate_triples(self):
        # candidate triples:
        # (h, r, 0), (h, r, 1), (h, r, 2) ...
        candidates = torch.zeros((self.num_entities, 3), dtype=torch.long)
        candidates[:, 0] = self.h
        candidates[:, 1] = self.r
        candidates[:, 2] = torch.arange(0, self.num_entities)
        return candidates

    def __getitem__(self, index):
        return self.candidate_triples[index, :]

    def __len__(self):
        return self.length


class Tester:
    class IndexScore:
        """
        The score of a tail when h and r is given.
        It's used in the ranking task to facilitate comparison and sorting.
        Print w as 3 digit precision float.
        """

        def __init__(self, index, score):
            self.index = index
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __repr__(self):
            # return "(index: %d, w:%.3f)" % (self.index, self.score)
            return "(%d, %.3f)" % (self.index, self.score)

        def __str__(self):
            return "(index: %d, w:%.3f)" % (self.index, self.score)

    def __init__(self, model, num_entity):
        """
        :type test_dataset: ShirleyTripleDataset
        """
        self.model = model
        self.num_entity = num_entity

    def get_score(self, h, r, i):
        ids = torch.LongTensor([[h, r, i]])
        cls = torch.Tensor([0])  # dummy
        log_score, _ = self.model(ids, cls)
        return torch.exp(log_score).detach().cpu().numpy()[0]

    def get_t_ranks(self, h, r, ts):
        """
        Given some t index, return the ranks for each t
        :return:
        """
        ranking_dataset = NDCGRankingTestDataset(
            h, r, self.num_entity
        )  # for one hr
        candidates_data = TensorDataLoader(
            ranking_dataset,
            batch_size=ranking_dataset.length,
            shuffle=False
        )
        with torch.no_grad():
            for ids in candidates_data:  # only one batch

                ids = ids  # [[h,r,0],[h,r,1]...]
                cls = torch.zeros(ids.shape[0])
                log_scores, _ = self.model(ids, cls)
                scores = log_scores
                grt_scores = scores[ts]
                ranks = np.array([(scores > s).sum().detach().cpu().numpy() for s in grt_scores])
                # print('ranks', ranks)
                break

        return ranks

    def ndcg(self, h, r, tw_truth):
        """
        Compute nDCG(normalized discounted cummulative gain)
        sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
        :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
        :return:
        """
        # prediction
        ts = [tw.index for tw in tw_truth]
        ranks = self.get_t_ranks(h, r, ts)

        # linear gain
        gains = np.array([tw.score for tw in tw_truth])
        discounts = np.log2(ranks + 2)  # avoid division by 0
        discounted_gains = gains / discounts
        dcg = np.sum(discounted_gains)  # discounted cumulative gain

        # normalize
        best_possible_ranks = np.array([(gains >= g).sum() for g in gains])  # gains [0.9, 0.8, 0.8, 0.7] -> [1,3,3,4]
        # max_possible_dcg = np.sum(gains / np.log2(best_possible_ranks + 1))
        max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))

        ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

        # exponential gain
        exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
        exp_discounted_gains = exp_gains / discounts
        exp_dcg = np.sum(exp_discounted_gains)
        # normalize
        exp_best_possible_ranks = np.array([(exp_gains >= g).sum() for g in exp_gains])
        # exp_max_possible_dcg = np.sum(exp_gains / np.log2(exp_best_possible_ranks + 1))
        exp_max_possible_dcg = np.sum(exp_gains / np.log2(np.arange(len(gains)) + 2))
        exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

        return ndcg, exp_ndcg, ranks

    def mean_ndcg(self, hr_map):
        """
        :param hr_map: {h:{r:{t:w}}}
        :return:
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0
        count = 0

        t0 = time.time()

        # debug ndcg
        res = []  # [(h,r,tw_truth, ndcg)]

        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ndcg, exp_ndcg, ranks = self.ndcg(h, r, tw_truth)  # nDCG with linear gain and exponential gain

                ndcg_sum += ndcg
                exp_ndcg_sum += exp_ndcg
                count += 1

                # debug
                res.append((h, r, tw_truth, ndcg, ranks))

        return ndcg_sum / count, exp_ndcg_sum / count



