# coding=utf-8
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from torch.utils.tensorboard import SummaryWriter
import torch
import math
import torch.nn as nn
from utils import *
from base_estimator import BaseEstimator

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        m.bias.data.fill_(0)


def wasserstein_torch(X,t ):
    """ Returns the Wasserstein distance between treatment groups """
    from geomloss import SamplesLoss
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)

    return imbalance_loss

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))

class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, dropout):
        super(ShareNetwork, self).__init__()
        # self.DNN = nn.Sequential(
        #     nn.Linear(input_dim, share_dim),
        #     nn.ELU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(share_dim, share_dim),
        #     nn.ELU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(share_dim, base_dim),
        #     nn.ELU(),
        # )
        self.DNN = nn.Sequential(

                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout),
            )

        self.DNN.apply(init_weights)

    def forward(self, x):
        h_rep = self.DNN(x)
        h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, dropout):
        super().__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits

class BaseModel4MetaLearner(nn.Module):
    def __init__(self, input_dim, base_dim, dropout):
        super().__init__()
        self.DNN = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            # nn.ELU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(base_dim, 1),
            # nn.ELU()
            # nn.BatchNorm1d(base_dim),
        )

    def forward(self, x):
        logit = self.DNN(x)
        return logit


class PrpsyNetwork(nn.Module):
    """propensity network"""
    def __init__(self, base_dim, dropout):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, dropout)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, dropout):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, dropout)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, dropout):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, dropout)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""
    def __init__(self, base_dim, dropout):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, dropout)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit

class ESX(nn.Module):
    """ESX"""
    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, tau_network: TauNetwork, shareNetwork: ShareNetwork):
        super().__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork
        self.prpsy_network = prpsy_network
        self.mu1_network = mu1_network
        self.mu0_network = mu0_network
        self.tau_network = tau_network

    def forward(self, inputs):

        x, t = inputs[:, :-1], inputs[:, -1]
        shared_h = self.shareNetwork(x)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = mu1_logit
        p_mu0 = mu0_logit
        p_h1 = p_mu1 # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0 # Refer to the naming in TARnet/CFR


        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h
    

class Estimator(BaseEstimator):

    def __init__(self, board=None, hparams={}, path=None):
        super().__init__(board=board, hparams=hparams, path=path)

    def model_construct(self):
        hparams = self.hparams
        dropout, share_dim, base_dim = hparams['dropout'], hparams['dim_share'], hparams['dim_base']
       
        shareNetwork = ShareNetwork(input_dim=self.x_dim, share_dim=share_dim, base_dim=base_dim, dropout=dropout)
        prpsy_network = PrpsyNetwork(base_dim, dropout)
        mu1_network = Mu1Network(base_dim, dropout)
        mu0_network = Mu0Network(base_dim, dropout)
        tau_network = TauNetwork(base_dim, dropout)
        self.model = ESX(prpsy_network, mu1_network, mu0_network, tau_network, shareNetwork).to(DEVICE)

    def _fit(self, _data, wass_indicator):
        
        _x, _t, _yf, _ = _data[:, :-2], _data[:, -3], _data[:, -2], _data[:, -1]  # do not use the ycf
        self.model.zero_grad()
        p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, _, _, _, p_h1, p_h0, shared_h = self.model(
        _x)
        p_tau = p_h1 - p_h0
        p_yf = p_h1 * _t + p_h0 * (1 - _t)
        p_ycf = p_h0 * _t + p_h1 * (1 - _t)

        prpsy_loss = self.hparams['prpsy_w'] * torch.nn.functional.binary_cross_entropy_with_logits(p_prpsy_logit.reshape(-1), _t)
        estr_loss = self.hparams['escvr1_w'] * torch.nn.functional.mse_loss(p_estr.reshape(-1), (_yf * _t))
        escr_loss = self.hparams['escvr0_w'] * torch.nn.functional.mse_loss(p_escr.reshape(-1), (_yf * (1 - _t)))

        #loss for TR, CR
        tr_loss = self.hparams['h0_w'] * torch.nn.functional.mse_loss(p_h1[_t.bool()].reshape(-1),
                                        _yf[_t.bool()])  # * (1 / (2 * p_t))
        cr_loss = self.hparams['h1_w'] * torch.nn.functional.mse_loss(p_h0[~_t.bool()].reshape(-1),
                                        _yf[~_t.bool()])  # * (1 / (2 * (1 - p_t)))


        #loss for cross TR: mu1_prime, cross CR: mu0_prime
        cross_tr_loss = self.hparams['mu1hat_w'] * torch.nn.functional.mse_loss((p_mu0_logit + p_tau_logit)[_t.bool()].reshape(-1),
                                                _yf[_t.bool()])
        cross_cr_loss = self.hparams['mu0hat_w'] * torch.nn.functional.mse_loss((p_mu1_logit - p_tau_logit)[~_t.bool()].reshape(-1),
                                                _yf[~_t.bool()])

        # imb_dist_loss = self.hparams['lambda'] * wasserstein_torch(X=shared_h, t=_t)

        _loss = prpsy_loss + estr_loss + escr_loss \
                + tr_loss + cr_loss \
                + cross_tr_loss + cross_cr_loss
                # + imb_dist_loss
        
        _loss.backward()
        self.optimizer.step()

        # Metric update
        _loss_wass = _loss_wass.item() if wass_indicator else 0
        # self.board.add_scalar('loss/fit_loss', _loss_fit.item(), global_step=self.iter_num)
        # self.board.add_scalar('loss/total_loss', _loss.item(), global_step=self.iter_num)
        self.iter_num += 1

    def predict(self, data):

        data = torch.tensor(data, device=DEVICE, dtype=torch.float32)
        x, t, yf, ycf = data[:, :-2], data[:, -3], data[:, -2], data[:, -1]
        self.model.eval()
        p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h = self.model(x)
        pred_0 = p_h0.reshape([-1]).detach().cpu().numpy()
        pred_1 = p_h1.reshape([-1]).detach().cpu().numpy()
        yf = yf.cpu().numpy()
        ycf = ycf.cpu().numpy()
        t = t.cpu().numpy()
        return pred_0, pred_1, yf, ycf, t
    


if __name__ == "__main__":
    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--model', type=str, default='descn')
    hparams.add_argument('--device', type=str, default='cuda')
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--epoch', type=int, default=400)
    hparams.add_argument('--seed', type=int, default=2)
    hparams.add_argument('--patience', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')
    hparams.add_argument('--root', type=str, default='debug0522')

    hparams.add_argument('--dim_share', type=int, default=16)
    hparams.add_argument('--dim_base', type=int, default=16)
    hparams.add_argument('--batchSize', type=int, default=32)
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)

    hparams.add_argument('--prpsy_w', type=float, default=1)
    hparams.add_argument('--escvr1_w', type=float, default=0.5)
    hparams.add_argument('--escvr0_w', type=float, default=0.5)
    hparams.add_argument('--h1_w', type=float, default=0.5)
    hparams.add_argument('--h0_w', type=float, default=0.5)
    hparams.add_argument('--mu1hat_w', type=float, default=1)
    hparams.add_argument('--mu0hat_w', type=float, default=1)
    
    hparams.add_argument('--lambda', type=float, default=0)
    hparams.add_argument('--tune', type=int, default=0)

    hparams = vars(hparams.parse_args())

    if hparams['tune'] == 1:
        path = f"{hparams['root']}/{hparams['data']}/{hparams['model']}/" + '_'.join([str(i) for i in [hparams['dim_share'], hparams['dim_base'], hparams['mu1hat_w'], hparams['mu0hat_w'],hparams['prpsy_w'], hparams['escvr1_w'], hparams['escvr0_w'], hparams['h1_w'], hparams['h0_w'], hparams['seed']]])
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
