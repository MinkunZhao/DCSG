import random
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop, NodeDrop
import json
import requests
import os
from tqdm import tqdm
import time

init = nn.init.xavier_uniform_

class LLMGuideManager:
    """管理LLM API调用，为门控融合提供监督信号"""
        if os.path.exists(self.label_cache_path):
            try:
                self.gate_labels = t.load(self.label_cache_path)
                print(f"Loaded gate labels from cache with shape {self.gate_labels.shape}")
                return
            except:
                print("Failed to load cached labels, will create new ones")

        self.gate_labels = t.ones(self.user_num + self.item_num, self.embedding_size) * 0.5
        self.gate_labels = self.gate_labels.cuda()

    def save_labels(self):
        """保存标签到缓存"""
        Args:
            node_id: 节点ID
            is_user: 是否为用户节点
            adj: 稀疏邻接矩阵
        Returns:
            结构特征的文本描述
        if is_user:
            profile_data = self.user_profiles.get(str(node_id), {})
            profile = profile_data.get("profile", "No profile available")
            return f"User profile: {profile}"
        else:
            profile_data = self.item_profiles.get(str(node_id), {})
            profile = profile_data.get("profile", "No profile available")
            return f"Item profile: {profile}"

    def call_llm_api(self, prompt):
        """调用LLM API获取响应"""
        Args:
            node_id: 节点ID
            is_user: 是否为用户节点
            adj: 邻接矩阵
        Returns:
            一个0-1之间的值，表示偏向协同信息的程度

        response = self.call_llm_api(prompt)
        print(response)
        if not response:
            return 0.5  # 默认平衡值

        try:
            value = float(response.strip())
            value = max(0.0, min(1.0, value))
            return value
        except:
            print(f"Failed to parse LLM response: {response}")
            return 0.5

    def batch_process_nodes(self, adj, batch_size=10, max_nodes=30000, save_interval=100):
        """批量处理节点获取LLM建议的门控权重"""
            edge_index = self.semantic_adj.indices()

            edge_values = self.semantic_adj.values()

            edge_attr = t.stack([edge_values, edge_values], dim=1)

            for gat in self.gat_layers:
                embeds = gat(embeds, edge_index, edge_attr=edge_attr)
                embeds = F.leaky_relu(embeds)
            return embeds'''

    def forward(self, keep_rate=1.0):
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        co_embeds = self._propagate_co(self.adj, co_embeds, keep_rate)

        sem_embeds = t.cat([
            self.semantic_mlp(self.usrprf_embeds),
            self.semantic_mlp(self.itmprf_embeds)
        ], 0)
        sem_embeds = self._propagate_sem(sem_embeds)

        gate_input = t.cat([co_embeds, sem_embeds], dim=1)
        gate = self.sigmoid(self.gate(gate_input))
        fused_embeds = gate * co_embeds + (1 - gate) * sem_embeds

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:], gate

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1, gate1 = self.forward(self.keep_rate)
        user_embeds2, item_embeds2, gate2 = self.forward(self.keep_rate)
        user_embeds3, item_embeds3, gate3 = self.forward(1.0)

        user_sem_embeds, item_sem_embeds, gate_sem = self.forward(1.0)

        anc1, pos1, neg1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc2, pos2, neg2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc3, pos3, neg3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        anc_sem, pos_sem, _ = self._pick_embeds(user_sem_embeds, item_sem_embeds, batch_data)

        cl_loss = cal_infonce_loss(anc1, anc2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos1, pos2, item_embeds2, self.cl_temperature)
        cl_loss /= anc1.shape[0]
        cl_loss *= self.cl_weight

        contrast_loss = cal_infonce_loss(anc3, anc_sem, user_sem_embeds, self.contrast_temp) + \
                        cal_infonce_loss(pos3, pos_sem, item_sem_embeds, self.contrast_temp)
        contrast_loss *= self.contrast_weight

        bpr_loss = cal_bpr_loss(anc3, pos3, neg3) / anc3.shape[0]

        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        anc_prf, pos_prf, neg_prf = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc3, anc_prf, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos3, pos_prf, itmprf_embeds, self.kd_temperature)
        kd_loss /= anc3.shape[0]
        kd_loss *= self.kd_weight

        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) + t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        anc, pos, neg = batch_data
        gate_labels = self.llm_guide.gate_labels

        anc_gate_labels = gate_labels[anc]
        pos_gate_labels = gate_labels[self.user_num + pos]
        neg_gate_labels = gate_labels[self.user_num + neg]

        anc_gate_pred = gate3[anc]
        pos_gate_pred = gate3[self.user_num + pos]
        neg_gate_pred = gate3[self.user_num + neg]

        gate_loss = F.mse_loss(anc_gate_pred, anc_gate_labels) + \
                    F.mse_loss(pos_gate_pred, pos_gate_labels) + \
                    F.mse_loss(neg_gate_pred, neg_gate_labels)
        gate_loss /= 3.0  # 平均三个部分的损失
        gate_loss *= self.gate_supervision_weight  # 应用权重系数

        total_loss = bpr_loss + reg_loss + cl_loss + kd_loss + contrast_loss + gate_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'gate_loss': gate_loss  # 门控监督损失记录
        }
        return total_loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds, _ = self.forward(1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds