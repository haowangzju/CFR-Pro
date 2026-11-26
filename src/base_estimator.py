# coding=utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from torch.utils.tensorboard import SummaryWriter
import ot
import torch.nn
import copy
from tqdm import tqdm
from models import *
from utils import *
import yaml
import time
from tqdm import trange

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
METRICS = ['mae_ate', 'mae_att', 'pehe', 'r2_f', 'rmse_f', 'r2_cf', 'rmse_cf', 'auuc', 'rauuc']


class BaseEstimator:

    def __init__(self, board=None, hparams={}, path=None):
        self.board = board
        self.hparams = hparams
        self.path = path
        data_name = hparams.get('data')
        print("Current data:", data_name)
        if data_name == "IHDP":
            self.trainset = np.loadtxt(f"Datasets/{data_name}/train_{hparams['seed']}.csv", delimiter=',', dtype=np.float32)
            self.trainevalset = np.loadtxt(f"Datasets/{data_name}/traineval_{hparams['seed']}.csv", delimiter=',', dtype=np.float32)
            self.evalset = np.loadtxt(f"Datasets/{data_name}/eval_{hparams['seed']}.csv", delimiter=',', dtype=np.float32)
            self.testset = np.loadtxt(f"Datasets/{data_name}/test_{hparams['seed']}.csv", delimiter=',', dtype=np.float32)
        else:
            self.trainset = np.loadtxt(f"Datasets/{data_name}/train.csv", delimiter=',', dtype=np.float32)
            self.trainevalset = np.loadtxt(f"Datasets/{data_name}/traineval.csv", delimiter=',', dtype=np.float32)
            self.evalset = np.loadtxt(f"Datasets/{data_name}/eval.csv", delimiter=',', dtype=np.float32)
            self.testset = np.loadtxt(f"Datasets/{data_name}/test.csv", delimiter=',', dtype=np.float32)

        self.x_dim = self.trainset.shape[-1] - 3
        # self.train_scaler = train_set.scaler
        # self.eval_scaler = eval_set.scaler
        # self.test_scaler = test_set.scaler


        self.train_metric = {metric: np.array([]) for metric in METRICS}
        self.eval_metric = deepcopy(self.train_metric)
        self.test_metric = deepcopy(self.train_metric)

        self.train_best_metric = {metric: None for metric in METRICS}
        self.eval_best_metric = deepcopy(self.train_best_metric)
        self.eval_best_metric['r2_f'] = -10  
        self.eval_best_metric["pehe"] = 100
        self.eval_best_metric['auuc'] = 0
        self.loss_metric = {metric: np.array([]) for metric in ['loss', 'loss_f', 'loss_c']}

        self.epochs = hparams.get('epoch', 200)
        self.criterion = torch.nn.MSELoss()
        self.epoch = 0
    
    def _fit(self, _data, wass_indicator):
        pass

    def model_construct(self):
        hparams = self.hparams
        if hparams.get('model') in ['bnn-wass', 'bnn-mmd']:
            self.model = TLearner(self.x_dim, hparams).to(DEVICE)
        elif hparams.get('model') in ['cfr-mmd', 'cfr-wass']:
            self.model = YLearner(self.x_dim, hparams).to(DEVICE)
        elif hparams.get('model') in ['res-mmd', 'res-wass']:
            self.model = IResLearner(self.x_dim, hparams).to(DEVICE)
        else:
            print(f"Model {hparams.get('model')} is not found")
        print(hparams.get('model'))

    def fit(self):
        self.model_construct()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['l2_reg'])
        earlystop = EarlyStopping(patience=self.hparams.get('patience'), delta=0, maximize=True)
        self.iter_num = 0
        self.hparams['start_time'] = time.time()
        batch_size = self.hparams['batchSize']
        # pbar = trange(self.epochs , desc='Progress', unit='step', ncols=120)
        for epoch in tqdm(range(1, self.epochs), ncols=10):
        # for epoch in pbar:
            self.epoch = epoch
            self.model.train()
            all_idx = np.arange(len(self.trainset))            
            np.random.shuffle(all_idx)
            total_batch =  len(all_idx) // batch_size

            for idx in range(total_batch):  # train_loader
                _idx = all_idx[idx*batch_size : (idx+1)*batch_size]
                _data = torch.tensor(self.trainset[_idx], device=DEVICE, dtype=torch.float32)
                wass_indicator = (self.hparams['lambda'] != 0 and epoch > 20)
                self._fit(_data, wass_indicator)

            # Evaluation and model selection
            if self.epoch % 10 == 0:
                _train_metric = self.evaluation('train')
                self.train_metric = metric_update(self.train_metric, _train_metric, self.epoch)
                [self.board.add_scalar(f"train/{key}", _train_metric[key], global_step=self.epoch) for key in _train_metric.keys()]

            if self.epoch % 1 == 0:
                _eval_metric = self.evaluation('eval')
                self.eval_metric = metric_update(self.eval_metric, _eval_metric, self.epoch)
                [self.board.add_scalar(f"eval/{key}", _eval_metric[key], global_step=self.epoch) for key in _eval_metric.keys()]

                update = earlystop(abs(_eval_metric['auuc']), copy.deepcopy(self.model.state_dict()))
                print(_eval_metric) if update else None
                if (earlystop.early_stop is True) and (self.epoch > 100):
                    break

            self.epoch += 1
        self.model.load_state_dict(earlystop.state_dict)
        self.train_best_metric = self.evaluation('train')
        self.eval_best_metric = self.evaluation('eval')
        self.test_best_metric = self.evaluation('test')
        print(self.eval_best_metric)

        
        [self.board.add_scalar(f"train_best/{key}", self.train_best_metric[key], global_step=self.epoch) for key in self.train_best_metric.keys()]
        [self.board.add_scalar(f"eval_best/{key}", self.eval_best_metric[key], global_step=self.epoch) for key in self.eval_best_metric.keys()]
        [self.board.add_scalar(f"test_best/{key}", self.test_best_metric[key], global_step=self.epoch) for key in self.test_best_metric.keys()]
        self.board.add_graph(self.model, _data[:, :-2])
        metric_export(self.path, self.train_best_metric, self.eval_best_metric, self.test_best_metric)
        self.hparams['end_time'] = time.time()
        with open(f"{self.path}/config.yaml", 'w') as f:
            yaml.dump(self.hparams, f, encoding='utf-8', allow_unicode=True)
        torch.save(self.model, f"{self.path}/model.pt")

    def predict(self, data):

        data = torch.tensor(data, device=DEVICE, dtype=torch.float32)
        x, t, yf, ycf = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
        x_0 = torch.cat([x, torch.zeros((len(t), 1), device=DEVICE)], dim=-1)
        x_1 = torch.cat([x, torch.ones((len(t), 1), device=DEVICE)], dim=-1)

        self.model.eval()
        pred_0 = self.model(x_0).reshape([-1]).detach().cpu().numpy()
        pred_1 = self.model(x_1).reshape([-1]).detach().cpu().numpy()

        yf = yf.cpu().numpy()
        ycf = ycf.cpu().numpy()
        t = t.detach().cpu().numpy()
        return pred_0, pred_1, yf, ycf, t

    def evaluation(self, mode: str):

        data = {
            'train': self.trainevalset,
            'eval': self.evalset,
            'test': self.testset}[mode]
        # scaler = {
        #     'train': self.train_scaler,
        #     'eval': self.eval_scaler,
        #     'test': self.test_scaler}[data]

        pred_0, pred_1, yf, ycf, t = self.predict(data)
        # pred_0, pred_1, yf = scaler.reverse_y(pred_0), scaler.reverse_y(pred_1), scaler.reverse_y(yf)  # 标签反归一化
        mode = 'in-sample' if mode == 'train' else 'out-sample'
        metric = metrics(pred_0, pred_1, yf, ycf, t, mode)

        return metric