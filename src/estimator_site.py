# coding=utf-8
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from torch.utils.tensorboard import SummaryWriter
import torch
import yaml
import time
import math
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from utils import *
from base_estimator import BaseEstimator
from sklearn.linear_model import LogisticRegression

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class SITE(nn.Module):
    """
    TARNet which combines T-learner and S-learner.
    """
    def __init__(self, input_dim, hparams):

        super().__init__()
        d_backbone = [input_dim] + str2lst(hparams['dim_backbone'])
        d_task = [d_backbone[-1]] + str2lst(hparams['dim_task'])
        self.treat_embed = hparams.get('treat_embed', True)

        self.backbone = torch.nn.Sequential()
        for i in range(1, len(d_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(d_backbone[i-1], d_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            # self.backbone.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=d_backbone[i]))
            self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        if self.treat_embed is True: # 拼接treatment带来的
            d_task[0] += 2

        self.tower_1 = torch.nn.Sequential()
        for i in range(1, len(d_task)):
            self.tower_1.add_module(f"tower_dense{i}", torch.nn.Linear(d_task[i-1], d_task[i]))
            self.tower_1.add_module(f"tower_relu{i}", torch.nn.ELU())
            # self.tower_1.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=out_dim[i]))
            self.tower_1.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.tower_1.add_module("tower_output", torch.nn.Linear(d_task[-1], 1))

        self.tower_0 = deepcopy(self.tower_1)
        self.embedding = nn.Embedding(2, 2)

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]
        self.rep = self.backbone(covariates)
        self.rep_1 = self.rep[t == 1]
        self.rep_0 = self.rep[t == 0]
        rep_t = torch.cat([self.rep, self.embedding(t.int())], dim=-1) if self.treat_embed is True else self.rep

        self.out_1 = self.tower_1(rep_t)
        self.out_0 = self.tower_0(rep_t)

        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0

        return output_f

class PDDM(nn.Module):
    def __init__(self, input_dim, hparams):

        super().__init__()
        self.U = nn.Linear(input_dim, hparams['dim_pddm'])
        self.V = nn.Linear(input_dim, hparams['dim_pddm'])
        self.C = nn.Linear(hparams['dim_pddm']*2, hparams['dim_c'])
        self.S = nn.Linear(hparams['dim_c'], hparams['dim_s'])

    def forward(self, _xi, _xj):

        u = torch.abs(_xi - _xj)
        v = (_xi + _xj) / 2
        u1 = torch.nn.functional.elu(self.U(u))
        v1 = torch.nn.functional.elu(self.V(v))
        c = torch.concat([u1, v1], dim=-1).reshape([1, -1])
        c = torch.nn.functional.elu(self.C(c)).reshape([1, -1])
        s = self.S(c)
        return s
    
def sigmoid(x):
	return 1 / float(1 + math.exp(-x))


def three_pair_extration(x, t, yf, propensity_estimator):
    '''
    :param x: pre-treatment covariates
    :param t: treatment
    :param yf: factual outcome
    :param propensity_dir: the directory that saves propensity model
    :return: the selected three pairs' pre-treatment covariates, index, treatment,
    factual_outcome, and the similarity score ((_xk, _xl), (_xm, _xn), (_xk, _xm), (_xi, _xk), (_xj, _xm))
    '''
    three_pairs, I_three_pairs = find_three_pairs(x, t, propensity_estimator)
    t_three_pairs = t[I_three_pairs]
    y_three_pairs = yf[I_three_pairs]
    three_pairs_simi = get_three_pair_simi(three_pairs, propensity_estimator)
    return three_pairs, I_three_pairs, t_three_pairs, y_three_pairs, three_pairs_simi
def find_three_pairs(x, t, propensity_estimator):
    def find_middle_pair(x, y):
        # min = np.abs(x[0]-0.5) + np.abs(y[0]-0.5)
        # index_1 = 0
        # index_2 = 0
        # for i in range(x.shape[0]):
        #     for j in range(y.shape[0]):
        #         value = np.abs(x[i]-0.5) + np.abs(y[j]-0.5)
        #         if  value < min:
        #             min = value
        #             index_1 = i
        #             index_2 = j
        # return index_1, index_2
        index_x = np.argmin(np.abs(x - 0.5))
        index_y = np.argmin(np.abs(y - 0.5))
        return index_x, index_y
    def find_nearest_point(x, p):
        diff = np.abs(x-p)
        I_diff = np.argmin(diff[diff>0])
        I_diff = I_diff[0] if I_diff.size>1 else I_diff
        return I_diff

    x_propensity_score = propensity_estimator.predict_proba(x)[:, 1]
    x_return = np.ones([6, x.shape[1]])
    I_x_return = np.zeros(6,dtype=int)
    I_t = np.where(t>0)[0]
    I_c = np.where(t<1)[0]

    prop_t = x_propensity_score[I_t]
    prop_c = x_propensity_score[I_c]
    x_t = x[I_t]
    x_c = x[I_c]
    index_t, index_c = find_middle_pair(prop_t, prop_c)
    index_k = np.argmax(np.abs(prop_c - prop_t[index_t]))
    index_l = find_nearest_point(prop_c, prop_c[index_k])

    # find _xn, _xm
    index_m = np.argmax(np.abs(prop_t - prop_c[index_c]))
    index_n = find_nearest_point(prop_t, prop_t[index_m,])
    x_return[0,:] = x_t[index_t,:]
    x_return[1, :] = x_c[index_c, :]
    x_return[2, :] = x_c[index_k, :]
    x_return[3, :] = x_c[index_l, :]
    x_return[4, :] = x_t[index_m, :]
    x_return[5, :] = x_t[index_n, :]
    I_x_return[0] = int(I_t[index_t])
    I_x_return[1] = int(I_c[index_c])
    I_x_return[2] = int(I_c[index_k])
    I_x_return[3] = int(I_c[index_l])
    I_x_return[4] = int(I_t[index_m])
    I_x_return[5] = int(I_t[index_n])
    # debug: see the index matching

    return x_return, I_x_return
def get_three_pair_simi(three_pairs, propensity_estimator):
    def get_simi_ground(x, propensity_estimator):
        x_propensity_score = propensity_estimator.predict_proba(x)[:, 1]
        n_train = x.shape[0]
        s_x_matrix = np.ones([n_train, n_train])
        for i in range(n_train):
            for j in range(n_train):
                s_x_matrix[i, j] = similarity_score(x_propensity_score[i], x_propensity_score[j])
        return s_x_matrix
    
    three_pairs_simi = get_simi_ground(three_pairs, propensity_estimator)
    simi = np.ones([5,1])
    '''
    S(k, l), S(m, n), S(k, l), S(i, k), S(j, m)
    '''
    simi[0, 0] = three_pairs_simi[2, 3]
    simi[1, 0] = three_pairs_simi[4, 5]
    simi[2, 0] = three_pairs_simi[2, 4]
    simi[3, 0] = three_pairs_simi[0, 2]
    simi[4, 0] = three_pairs_simi[1, 4]
    return simi

def similarity_score(s_i, s_j, mode = 'linear'):
    if mode == 'sigmoid':
        _mid = (s_i + s_j)/float(2)
        _dis = abs(s_j - s_i)/float(2)
        score = 2*sigmoid(abs(_mid-0.5)) - 3*sigmoid(_dis)+1
    if mode == 'linear':
        _mid = (s_i + s_j) / float(2)
        _dis = abs(s_j - s_i) / float(2)
        score = (1.5 * abs(_mid - 0.5) - 2 * _dis + 1)/float(2)
    return score



class Estimator(BaseEstimator):

    def __init__(self, board=None, hparams={}, path=None):
        super().__init__(board=board, hparams=hparams, path=path)

    def model_construct(self):
        hparams = self.hparams
        self.model = SITE(self.x_dim, hparams).to(DEVICE)
        self.pddm = PDDM(input_dim=str2lst(hparams['dim_backbone'])[-1], hparams=self.hparams).to(DEVICE)
        print(hparams.get('model'))

    def _fit_site(self, _data, _three_pairs_idx, three_pairs_simi):
        _xt, _t, _yf, _ = _data[:, :-2], _data[:, [-3]], _data[:, [-2]], _data[:, -1] 
        self.model.zero_grad()
        _pred_f = self.model(_xt)
        _loss_fit = ((_pred_f-_yf)**2).mean()
        
        _xi, _xj, _xk, _xl, _xm, _xn = self.model.rep[_three_pairs_idx]
        _s_kl = self.pddm(_xk, _xl)
        _s_mn = self.pddm(_xm, _xn)
        _s_km = self.pddm(_xk, _xm)
        _s_ik = self.pddm(_xi, _xk)
        _s_jm = self.pddm(_xj, _xm)

        _simi_kl = three_pairs_simi[0]
        _simi_mn = three_pairs_simi[1]
        _simi_km = three_pairs_simi[2]
        _simi_ik = three_pairs_simi[3]
        _simi_jm = three_pairs_simi[4]
        _loss_pddm = (_simi_kl - _s_kl)**2 + (_simi_mn - _s_mn)**2 + (_simi_km - _s_km)**2 + (_simi_ik - _s_ik)**2 + (_simi_jm - _s_jm)**2
        _loss_pddm = _loss_pddm.sum()

        '''mid_point distance minimization'''
        _mid_jk = (_xj + _xk) / 2.0
        _mid_im = (_xi + _xm) / 2.0
        _loss_mpdm = (_mid_jk - _mid_im)**2
        _loss_mpdm = _loss_mpdm.sum()

        _loss = _loss_fit + self.hparams['pddm_w'] * _loss_pddm + self.hparams['mpdm_w'] * _loss_mpdm
        _loss.backward()
        self.optimizer.step()
       
        # Metric update
        self.board.add_scalar('loss/loss_fit', _loss_fit.item(), global_step=self.iter_num)
        self.board.add_scalar('loss/loss_pddm', _loss_pddm.item(), global_step=self.iter_num)
        self.board.add_scalar('loss/loss_mpdm', _loss_mpdm.item(), global_step=self.iter_num)
        self.iter_num += 1
    
    def _fit(self, _data):
        _xt, _t, _yf, _ = _data[:, :-2], _data[:, [-3]], _data[:, [-2]], _data[:, -1] 
        self.model.zero_grad()
        _pred_f = self.model(_xt)
        _loss_fit = ((_pred_f-_yf)**2).mean()
        _loss = _loss_fit 
        _loss.backward()
        self.optimizer.step()
       
        # Metric update
        self.board.add_scalar('loss/loss_fit', _loss_fit.item(), global_step=self.iter_num)
        self.iter_num += 1

    def fit(self):


        self.model_construct()
        self.optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "lr": self.hparams['lr'], "weight_decal": self.hparams['l2_reg']},
            {"params": self.pddm.parameters(), "lr": self.hparams['lr'], "weight_decal": self.hparams['l2_reg']}])
        earlystop = EarlyStopping(patience=self.hparams.get('patience'), delta=0, maximize=True)
        self.iter_num = 0
        self.hparams['start_time'] = time.time()
        batch_size = self.hparams['batchSize']
        self.propensity_estimator = LogisticRegression('l2', class_weight='balanced', C=3.0)

        self.propensity_estimator.fit(self.trainset[:, :-3], self.trainset[:, [-3]].astype(int))

        for epoch in tqdm(range(1, self.epochs), ncols=10):
            self.epoch = epoch
            self.model.train()
            all_idx = np.arange(len(self.trainset))            
            np.random.shuffle(all_idx)
            total_batch =  len(all_idx) // batch_size

            for idx in range(total_batch):  # train_loader
                _idx = all_idx[idx*batch_size : (idx+1)*batch_size]
                _data = self.trainset[_idx]
                _t = _data[:, -3]
                if (len(_t[_t==1]) > 4 and len(_t[_t==0]) > 4): # 
                    _three_pairs, _three_pairs_idx, _, _, _three_pairs_simi = three_pair_extration(
                        _data[:, :-3], _data[:, [-3]], _data[:, [-2]], self.propensity_estimator)
                    _data = torch.tensor(_data, device=DEVICE, dtype=torch.float32)
                    _three_pairs_simi = torch.tensor(_three_pairs_simi, device=DEVICE, dtype=torch.float32)
                    self._fit_site(_data, _three_pairs_idx, _three_pairs_simi)
                else:
                    _data = torch.tensor(_data, device=DEVICE, dtype=torch.float32)
                    self._fit(_data)
                

            # Evaluation and model selection
            if self.epoch % 10 == 0:
                _train_metric = self.evaluation('train')
                self.train_metric = metric_update(self.train_metric, _train_metric, self.epoch)
                [self.board.add_scalar(f"train/{key}", _train_metric[key], global_step=self.epoch) for key in _train_metric.keys()]

            if self.epoch % 1 == 0:
                _eval_metric = self.evaluation('eval')
                self.eval_metric = metric_update(self.eval_metric, _eval_metric, self.epoch)
                [self.board.add_scalar(f"eval/{key}", _eval_metric[key], global_step=self.epoch) for key in _eval_metric.keys()]

                update = earlystop(abs(_eval_metric['auuc']), deepcopy(self.model.state_dict()))
                print(_eval_metric) if update else None
                if (earlystop.early_stop is True) and (self.epoch > 100):
                    break
            self.epoch += 1



        self.model.load_state_dict(earlystop.state_dict)
        self.train_best_metric = self.evaluation('train')
        self.eval_best_metric = self.evaluation('eval')
        self.test_best_metric = self.evaluation('test')   
        print("[Eval]" + ", ".join([f"{key}={self.eval_best_metric[key]}" for key in self.eval_best_metric]))
        print("[Test]" + ", ".join([f"{key}={self.test_best_metric[key]}" for key in self.test_best_metric]))

        
        [self.board.add_scalar(f"train_best/{key}", self.train_best_metric[key], global_step=self.epoch) for key in self.train_best_metric.keys()]
        [self.board.add_scalar(f"eval_best/{key}", self.eval_best_metric[key], global_step=self.epoch) for key in self.eval_best_metric.keys()]
        [self.board.add_scalar(f"test_best/{key}", self.test_best_metric[key], global_step=self.epoch) for key in self.test_best_metric.keys()]
        self.board.add_graph(self.model, _data[:, :-2])
        metric_export(self.path, self.train_best_metric, self.eval_best_metric, self.test_best_metric)
        self.hparams['end_time'] = time.time()
        with open(f"{self.path}/config.yaml", 'w') as f:
            yaml.dump(self.hparams, f, encoding='utf-8', allow_unicode=True)
        torch.save(self.model, f"{self.path}/model.pt")


if __name__ == "__main__":
    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--model', type=str, default='site')
    hparams.add_argument('--device', type=str, default='cuda')
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--epoch', type=int, default=400)
    hparams.add_argument('--seed', type=int, default=2)
    hparams.add_argument('--patience', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')
    hparams.add_argument('--root', type=str, default='debug0522')

    hparams.add_argument('--dim_backbone', type=str, default='16,16')
    hparams.add_argument('--dim_pddm', type=int, default=8)
    hparams.add_argument('--dim_c', type=int, default=8)
    hparams.add_argument('--dim_s', type=int, default=8)
    hparams.add_argument('--pddm_w', type=float, default=0.5)
    hparams.add_argument('--mpdm_w', type=float, default=0.5)
    hparams.add_argument('--dim_task', type=str, default='32,32')
    hparams.add_argument('--batchSize', type=int, default=32)
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)

    hparams.add_argument('--lambda', type=float, default=0)
    hparams.add_argument('--tune', type=int, default=0)

    hparams = vars(hparams.parse_args())
    if hparams['tune'] == 1:
        path = f"{hparams['root']}/{hparams['data']}/{hparams['model']}/" + '_'.join([str(i) for i in [hparams['dim_backbone'], hparams['dim_pddm'], hparams['dim_c'], hparams['dim_s'], hparams['pddm_w'], hparams['mpdm_w'], hparams['seed']]])
    else:
        path = f"{hparams['root']}/{hparams['data']}/{hparams['model']}_{hparams['seed']}"
    if os.path.exists(f"{path}/model.pt"):
        print(f"{path}/model.pt exists!")
    else:
        writer = SummaryWriter(path)
        seed_everything(hparams['seed'])
        estimator = Estimator(board=writer, hparams=hparams, path=path)
        estimator.fit()
        writer.close()
