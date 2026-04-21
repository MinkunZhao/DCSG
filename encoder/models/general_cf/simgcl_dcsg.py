import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop
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
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise

    def _propagate_co(self, adj, embeds, perturb=False):
        """协同通道传播（带扰动）"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, perturb=False, keep_rate=1.0):
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        dropped_adj = self.edge_dropper(self.adj, keep_rate)
        co_embeds = self._propagate_co(dropped_adj, co_embeds, perturb)

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

        user_embeds1, item_embeds1, gate1 = self.forward(perturb=True, keep_rate=self.keep_rate)
        user_embeds2, item_embeds2, gate2 = self.forward(perturb=True, keep_rate=self.keep_rate)
        user_embeds3, item_embeds3, gate3 = self.forward(perturb=False, keep_rate=self.keep_rate)
        user_embeds_sem, item_embeds_sem, gate_sem = self.forward(perturb=False, keep_rate=1.0)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3)

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        user_contrast_loss = cal_infonce_loss(anc_embeds3, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds3, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds3, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds3, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds3, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds3.shape[0]
        kd_loss *= self.kd_weight

        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        gate_labels = self.llm_guide.gate_labels
        anc, pos, neg = batch_data

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

        loss = bpr_loss + reg_loss + kd_loss + cl_loss + contrast_loss + gate_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'cl_loss': cl_loss,
            'contrast_loss': contrast_loss,
            'gate_loss': gate_loss  # 门控监督损失记录
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds, _ = self.forward(perturb=False, keep_rate=1.0)

        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        pck_user_embeds = user_embeds[pck_users]

        full_preds = pck_user_embeds @ item_embeds.T

        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds