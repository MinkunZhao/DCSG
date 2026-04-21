import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from encoder.config.configurator import configs
from encoder.data_utils.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data


class DataHandlerGeneralCF:
    def __init__(self):
        self.sem_item_adj = None
        self.sem_user_adj = None
        if configs['data']['name'] == 'amazon':
            predir = '../data/amazon/'
        elif configs['data']['name'] == 'yelp':
            predir = '../data/yelp/'
        elif configs['data']['name'] == 'steam':
            predir = '../data/steam/'
        else:
            raise NotImplementedError
        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'

    def _build_semantic_graph(self):
        usrprf = configs['usrprf_embeds']
        itmprf = configs['itmprf_embeds']

        user_norm = normalize(usrprf, norm='l2', axis=1)
        item_norm = normalize(itmprf, norm='l2', axis=1)
        sim_matrix = user_norm @ item_norm.T  # (user_num, item_num)

        k = configs['model']['sem_graph_topk']
        rows = []
        cols = []
        data = []
        for i in range(sim_matrix.shape[0]):
            topk_idx = np.argpartition(sim_matrix[i], -k)[-k:]
            rows.extend([i] * k)
            cols.extend(topk_idx)
            data.extend(sim_matrix[i][topk_idx])

        adj_size = len(usrprf) + len(itmprf)
        extended_rows = rows
        extended_cols = [col + len(usrprf) for col in cols]

        idxs = t.from_numpy(np.vstack([extended_rows, extended_cols])).long().cuda()
        vals = t.from_numpy(np.array(data)).float().cuda()
        self.semantic_adj = t.sparse_coo_tensor(
            idxs, vals,
            (adj_size, adj_size)
        ).coalesce()

    def _load_one_mat(self, file):
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat

    def _normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat, self_loop=False):
        if not self_loop:
            a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
            b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        else:
            data = np.ones(configs['data']['user_num'])
            row_indices = np.arange(configs['data']['user_num'])
            column_indices = np.arange(configs['data']['user_num'])
            a = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['user_num'], configs['data']['user_num']))

            data = np.ones(configs['data']['item_num'])
            row_indices = np.arange(configs['data']['item_num'])
            column_indices = np.arange(configs['data']['item_num'])
            b = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['item_num'], configs['data']['item_num']))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)

        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
        self.torch_adj = self._make_torch_adj(trn_mat)

        if configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        if configs['model']['name'][-4:] == 'dcsg':
            self._build_semantic_graph()
