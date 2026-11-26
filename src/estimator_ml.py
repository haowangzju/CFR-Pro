import os
import argparse
from utils import *
import numpy as np

from utils import *
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from data_processor import *
from sklearn.neighbors import KNeighborsRegressor
import datetime
from econml.dml import CausalForestDML
from econml.orf import DMLOrthoForest


class BaseEstimator:

    def __init__(self, params={}):
        # Section: LoadData
        data_name = params.get('data')
        # self.eval_path = f"Datasets/{data_name}/eval_{params['seed']}.csv"
        self.train_data = np.loadtxt(f"Datasets/{data_name}/train_{params['seed']}.csv", delimiter=',', dtype=np.float32)
        self.test_data = np.loadtxt(f"Datasets/{data_name}/test_{params['seed']}.csv", delimiter=',', dtype=np.float32)

        # Section: Normalize
        # self.train_scaler = StandardScaler()
        # self.test_scaler = StandardScaler()
        # self.train_data = self.train_scaler.fit_transform(self.train_data)
        # self.test_data = self.test_scaler.fit_transform(self.test_data)

        # Section: Metrics
        self.train_metric = None
        self.test_metric_in_sample = None
        self.test_metric_out_sample = None

    def fit(self):
        pass

    def predict(self, data: np.ndarray):
        pass

    def evaluation(self, data=None):
        # in-sample metrics over training data
        pred_0, pred_1, yf, ycf, t = self.predict(self.train_data)
        # pred_0, pred_1, yf = self.train_scaler.reverse_y(pred_0), self.train_scaler.reverse_y(pred_1), self.train_scaler.reverse_y(yf)
        self.train_metric = metrics(pred_0, pred_1, yf, ycf, t, mode='in-sample')
        # out-sample metrics over test data
        pred_0, pred_1, yf, ycf, t = self.predict(self.test_data)
        # pred_0, pred_1, yf = self.test_scaler.reverse_y(pred_0), self.test_scaler.reverse_y(pred_1), self.test_scaler.reverse_y(yf)
        self.test_metric_in_sample = metrics(pred_0, pred_1, yf, ycf, t, mode='in-sample')
        self.test_metric_out_sample = metrics(pred_0, pred_1, yf, ycf, t, mode='out-sample')


class EstimatorT(BaseEstimator):  # T-learner

    def __init__(self, model=None, params={}):
        super(EstimatorT, self).__init__(params)
        self.model_0 = LinearRegression if model is None else model
        self.model_1 = deepcopy(self.model_0)

    def fit(self):
        x, t, yf = self.train_data[:, :-3], self.train_data[:, -3], self.train_data[:, -2]
        self.model_0.fit(x[t == 0], yf[t == 0])  # fit: y | x, t=0
        self.model_1.fit(x[t == 1], yf[t == 1])  # fit: y | x, t=1

    def predict(self, data):
        x, t, yf, ycf = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
        pred_0 = self.model_0.predict(x)  # predict: y | x, do(t=0)
        pred_1 = self.model_1.predict(x)  # predict: y | x, do(t=1)

        return pred_0, pred_1, yf, ycf, t


class EstimatorS(BaseEstimator):  # S-learner

    def __init__(self, model, params={}):
        super(EstimatorS, self).__init__(params)
        from sklearn import linear_model
        self.model = linear_model.LinearRegression() if model is None else model

    def fit(self):
        x = self.train_data[:, :-2]  # [x,t]
        yf = self.train_data[:, -2]
        self.model.fit(x, yf)

    def predict(self, data):
        x, t, yf, ycf = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
        x_0 = np.c_[deepcopy(x), np.zeros_like(t)]  # 生成数据(x, do(t=0))
        x_1 = np.c_[deepcopy(x), np.ones_like(t)]  # 生成数据(x, do(t=1))

        pred_0 = self.model.predict(x_0)  # 预测y | x, do(t=0)
        pred_1 = self.model.predict(x_1)  # 预测y | x, do(t=1)

        return pred_0, pred_1, yf, ycf, t


class KNNEstimatorT(EstimatorT):

    def __init__(self, params={}):
        super(KNNEstimatorT, self).__init__(params)
        self.model_0 = KNeighborsRegressor(n_neighbors=4)
        self.model_1 = KNeighborsRegressor(n_neighbors=4)


class PSMEstimator(BaseEstimator):

    def __init__(self, model=None, params={}):
        super(PSMEstimator, self).__init__(params)
        from sklearn import linear_model
        self.model_t = linear_model.LinearRegression() if model is None else model

    def fit(self):
        x = self.train_data[:, :-3]
        t = self.train_data[:, -3]
        self.model_t.fit(x, t)

    def predict(self, data):
        x, t, yf, ycf = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
        ps = self.model_t.predict(x)
        x_0, x_1 = x[t == 0], x[t == 1]
        ps_0, ps_1 = ps[t == 0], ps[t == 1]
        y_0, y_1 = yf[t == 0], yf[t == 1]

        pred_0 = np.array(
            [np.mean(y_0[np.argsort(abs(ps[i] - ps_0))[:2]]) for i in range(len(t))])
        pred_1 = np.array(
            [np.mean(y_1[np.argsort(abs(ps[i] - ps_1))[:2]]) for i in range(len(t))])

        return pred_0, pred_1, yf, ycf, t

    def evaluation(self, data=None):
        from sklearn.metrics import r2_score
        pred_0, pred_1, yf, ycf, t = self.predict(self.train_data)
        self.train_metric = metrics(pred_0, pred_1, yf, ycf, t, mode='in-sample')
        self.train_metric['ps'] = r2_score(t, self.model_t.predict(self.train_data[:, :-3]))
        pred_0, pred_1, yf, ycf, t = self.predict(self.test_data)
        self.test_metric_in_sample = metrics(pred_0, pred_1, yf, ycf, t, mode='in-sample')
        self.test_metric_in_sample['ps'] = r2_score(t, self.model_t.predict(self.test_data[:, :-3]))
        self.test_metric_out_sample = metrics(pred_0, pred_1, yf, ycf, t, mode='out-sample')
        self.test_metric_out_sample['ps'] = r2_score(t, self.model_t.predict(self.test_data[:, :-3]))


class EstimatorDML(BaseEstimator):

    def __init__(self, model=None, params={}):
        super(EstimatorDML, self).__init__(params)

        self.model = CausalForestDML(max_depth=1,
                                     model_t=LinearRegression(),
                                     model_y=LinearRegression()) if model is None else model

    def fit(self):
        x, t, yf = self.train_data[:, :-3], self.train_data[:, -3], self.train_data[:, -2]
        self.model.fit(yf, t, X=x, W=x)

    def predict(self, data):
        x, t, yf, ycf = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
        ite_pred = self.model.effect(X=x, T0=np.zeros_like(t), T1=np.ones_like(t))
        return ite_pred, yf, ycf, t

    def evaluation(self, data=None):
        # in-sample metrics over training data
        ite_pred, yf, ycf, t = self.predict(self.train_data)
        self.train_metric = metrics_tree(ite_pred, yf, ycf, t)
        # out-sample metrics over test data
        ite_pred, yf, ycf, t = self.predict(self.test_data)
        self.test_metric_in_sample = metrics_tree(ite_pred, yf, ycf, t)
        self.test_metric_out_sample = metrics_tree(ite_pred, yf, ycf, t)

if __name__ == "__main__":

    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--data', type=str, default='IHDP', help="['IHDP', 'ACIC']")
    hparams.add_argument('--root', type=str, default='debug')

    params = vars(hparams.parse_args())
    time_stamp = datetime.datetime.now().strftime('%m%d%H%M%S')
    for seed in range(1, 11):
        params['seed'] = seed
        if params['data'] == 'ACIC':
            acic2016_processor(random_state=seed)
        elif params['data'] == 'IHDP':
            ihdp_processor(exp_number=seed)
        estimator_dict = {
            'OForest': EstimatorDML(DMLOrthoForest(max_depth=5, n_trees=60, model_T=LinearRegression(), model_Y=LinearRegression(), random_state=seed), params=params),
            'OLS-S': EstimatorS(model=LinearRegression(), params=params),
            'OLS-T': EstimatorT(model=LinearRegression(), params=params),
            'KNN': EstimatorT(model=KNeighborsRegressor(n_neighbors=2), params=params),
            'RF-S': EstimatorS(model=RandomForestRegressor(n_estimators=5, max_depth=3, random_state=seed), params=params),
            'PSM': PSMEstimator(params=params),
            'CForest': EstimatorDML(CausalForestDML(max_depth=3, n_estimators=60, model_t=LinearRegression(), model_y=LinearRegression(), random_state=seed), params=params),
        }

        for estimator_name in estimator_dict:
            estimator = estimator_dict[estimator_name]
            path = f"./{params['root']}/{params['data']}/{estimator_name}_{seed}"
            if os.path.exists(path) is False:
                os.makedirs(path)
            estimator.fit()
            estimator.evaluation()
            metric_export(path, estimator.train_metric, estimator.test_metric_in_sample, estimator.test_metric_out_sample)

        print(f"complete: {seed}")
