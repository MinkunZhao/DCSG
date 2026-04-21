import torch as t
from torch import nn
from encoder.config.configurator import configs

class BaseModel(nn.Module):
    def __init__(self, data_handler):
        super(BaseModel, self).__init__()

        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        if configs['data']['name'] in configs['model']:
            self.hyper_config = configs['model'][configs['data']['name']]
        else:
            self.hyper_config = configs['model']

    def forward(self):
        pass

    def cal_loss(self, batch_data):
        """return losses and weighted loss to training"""
        pass

    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        """return all-rank predictions to evaluation process, should call _mask_predict for masking the training pairs"""
        pass