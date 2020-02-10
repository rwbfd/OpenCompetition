# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-03-14
Description :
"""
# import modules
import os, sys
from file_utils import TaskModelBase
from data_loader_factory import DataLoaderFactory
from model_runner import ModelRunner
import torch
from torch import nn
import argparse
from get_optim import get_optim
from .bert import BertModel
from daguan_util import convert_input_to_list, convert_test_input_to_list
from focalloss import FocalLoss

curr_path = os.getcwd()
sys.path.append(curr_path)

# 设置使用哪些gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# argparser
parser = argparse.ArgumentParser()

# path
parser.add_argument('--train_data_path', type=str, default="./train_data/train_input.txt")
parser.add_argument('--dev_data_path', type=str, default="./train_data/dev_input.txt")
parser.add_argument('--test_data_path', type=str, default="./train_data/test_input.txt")
parser.add_argument('--bert_pretrain_path', type=str, default="step_64.5w")
parser.add_argument('--model_save_path', type=str, default="test_model.bin")
parser.add_argument('--eval_file_save_name', type=str, default="eval_out.txt")


# train process
parser.add_argument('--total_step', type=int, default=10000)
parser.add_argument('--eval_per_step', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_seq_length', type=int, default=96)

# optimizer
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--warm_up', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--amsgrad', type=bool, default=False)

# model
parser.add_argument('--label_to_id', type=dict, default={'<PAD>': 0, 'O': 1,
                                                         'B-a': 2, 'I-a': 3, 'B-b': 4,
                                                         'I-b': 5, 'B-c': 6, 'I-c': 7})
parser.add_argument('--dropout_prob', type=float, default=0.1)

# others
parser.add_argument('--language', type=str, default="english")

args = parser.parse_args()


# define class
class TaskModel(TaskModelBase):
    def __init__(self, num_labels, dropout_prob, bret_pretrainded_path):
        """
        :param num_labels:
        :param dropout_prob:
        :param bret_pretrainded_path:
        """
        # 初始化
        super().__init__()

        # 构建深度学习的网络结构
        self.fc = nn.Linear(768, num_labels)
        # self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fn = FocalLoss()

    def forward(self, data):
        """
        :param data:
        :return:
        """
        # get data
        x = data["x"]
        y = data["y"]

        logits = self.fc(x)

        if y is not None:
            loss = self.loss_fn.forward(torch.reshape(logits, [-1, logits.shape[-1]]), y.view(-1))
        else:
            loss = None

        return {"loss": loss, "logits": logits, "predict": logits.argmax(dim=-1)}


# define function
def main(args):
    # create dataloader
    data_loader = DataLoaderFactory()

    train_data_loader = data_loader.get_input_for_ccf(file_path,
                                                      batch_size,
                                                      max_seq_length,
                                                      shuffle,
                                                      drop_last)

    test_data_loader = data_loader.get_input_for_ccf(file_path,
                                                      batch_size,
                                                      max_seq_length,
                                                      shuffle,
                                                      drop_last)

    dev_data_loader = data_loader.get_input_for_ccf(file_path,
                                                      batch_size,
                                                      max_seq_length,
                                                      shuffle,
                                                      drop_last)

    # 设置task model
    task_model = TaskModel(num_labels=len(args.label_to_id), dropout_prob=args.dropout_prob,
                           bret_pretrainded_path=args.bert_pretrain_path)

    # 重新加载模型参数
    # print("从test_model.bin 加载参数")
    # task_model.load_state_dict(torch.load(os.path.join(os.path.dirname(curr_path), "model_save", "model_89.15.bin"),
    #                                       "cuda" if torch.cuda.is_available() else None))

    # 设置优化器
    optimizer = get_optim(task_model.parameters(), args)

    # print config
    print("args", args)

    # 开始模型训练
    cls_app = ModelRunner(task_type="cls", is_bert=False, label_to_id=args.label_to_id)

    cls_app.train(total_step=args.total_step, eval_per_step=args.eval_per_step, task_model=task_model,
                  model_save_path=args.model_save_path, optimizer=optimizer,
                  train_data_loader=train_data_loader, dev_data_loader=dev_data_loader,
                  eval_label_list=list(args.label_to_id.values()), compare_param="f1",
                  eval_file_save_name=args.eval_file_save_name)

    cls_app.predict_for_ccf(dataiter=dev_data_loader, model=task_model,
                                    save_file_path="test_predict_out.txt",
                                    model_path=args.model_save_path, load_from_onnx=False)

    print("模型运行完成")


# main
if __name__ == '__main__':
    main(args)
