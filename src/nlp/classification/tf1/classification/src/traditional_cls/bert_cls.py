# !/user/bin/python
# -*- coding:utf-8 -*-

from model_base import ModelBase
from bert import BertModel
import torch
import torch.nn as nn


class BertClassifier(ModelBase):
    def __init__(self, num_labels, bret_pretrainded_path):
        """
        在定义task model 的时候必须包含的两部分: self.train_state 和 self.device
        如果不包含着两部分将task model放入Training中进行训练的时候就会报错
        另外模型还要包含两部分，一部分是模型的结构部分，一部分是loss function
        :param num_labels:
        :param bret_pretrainded_path:
        """
        # 初始化
        super().__init__()

        # 构建深度学习的网络结构
        self.bert = BertModel.from_pretrained(bret_pretrainded_path)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, data, device):
        """
        模型的前项计算过程，传入的data数据为从迭代器中得到的一个batch数据,
            如果只是predict过程，data["y_batch"] 得到的值为None
        :param data:
            if task_type == "cls":
                data = {"text_a_batch": text_a_batch, "x_a_batch": x_a_batch,
                            "label_batch": label_batch, "y_batch": y_batch}
            elif task_type == "seq":
                data = {"text_a_batch": text_a_batch, "x_a_batch": x_a_batch,
                            "label_batch": label_batch, "y_batch": y_batch}
            elif task_type == "sim":
                data = {"text_a_batch": text_a_batch, "x_a_batch": x_a_batch,
                            "label_batch": label_batch, "y_batch": y_batch}
        :return: loss, logits, y, y_pred
        """
        # get data
        x = data["x_batch"].to(device)

        # token_type_ids_batch
        token_type_ids_batch = data["token_type_ids_batch"].to(device)

        if self.training:
            # 将bert 调整为train mode
            self.bert.train()
            _, pooled_encoded_layers = self.bert(x, token_type_ids_batch)
            logits = self.fc(pooled_encoded_layers)

        else:
            # 将bert 调整为eval mode
            self.bert.eval()
            with torch.no_grad():
                _, pooled_encoded_layers = self.bert(x, token_type_ids_batch)
                logits = self.fc(pooled_encoded_layers)

        return logits
