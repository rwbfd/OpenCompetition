# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-03-17
Description :  数据处理的dataset
"""
# import modules
import os, sys

curr_path = os.path.dirname(__file__)
sys.path.append(curr_path)

from torch.utils import data
import torch


__all__ = ["DataIter", "BertDataset"]


# define class
class BertDataset(object):
    def __init__(self, data_dict):
        """
        :param data_dict:
        """
        super().__init__()
        self.data_dict = data_dict

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        try:
            return {k: v[index] for k, v in self.data_dict.items()}
        except:
            print(index)
            raise 0

    def __len__(self):
        """
        获取数据的长度
        :return:
        """
        return len(list(self.data_dict.values())[0])


class DataIter(object):
    def __init__(self, batch_size, max_seq_length, num_works=4):
        """"""
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_works = num_works

    def get_iter_for_bert(self, data_dict, shuffle, drop_last):
        dataset = BertDataset(data_dict)
        return data.DataLoader(dataset=dataset,
                               batch_size=self.batch_size,
                               shuffle=shuffle,
                               num_workers=self.num_works,
                               collate_fn=self.batch_dict_to_padded_dict_batch,
                               drop_last=drop_last)

    def get_double_sentence_iter_for_bert(self, data_dict, shuffle, drop_last):
        dataset = BertDataset(data_dict)
        return data.DataLoader(dataset=dataset,
                               batch_size=self.batch_size,
                               shuffle=shuffle,
                               num_workers=self.num_works,
                               collate_fn=self.batch_dict_to_padded_dict_batch_double_sentence,
                               drop_last=drop_last)

    def batch_dict_to_padded_dict_batch(self, batch_dict):
        """
        [{}, {}, {}] --> {k1: [], k2: [], ....}
        :param batch_dict:
        :return:
        """
        # 创建 batch_dict
        keys = batch_dict[0].keys()
        dict_batch = dict.fromkeys(keys)
        for k in keys:
            dict_batch[k] = list()

        # 将传入的 batch [{}, {}, {}] 存储到 batch_dict 中 {k1: [], k2: [], ....}
        for bat in batch_dict:
            for k, v in bat.items():
                dict_batch[k].append(v)

        try:
            # for k, v in dict_batch.items():
            #     a = self.padding_batch_list(v, self.max_seq_length)
            #     padded_dict_batch[k] = torch.LongTensor(a)
            padded_dict_batch = dict()
            for k, v in dict_batch.items():
                if k not in ["features_list", "raw_x", "raw_y"]:
                    padded_dict_batch[k] = torch.LongTensor(self.padding_batch_list(v, self.max_seq_length))
                else:
                    padded_dict_batch[k] = v
            # padded_dict_batch = {k: torch.LongTensor(self.padding_batch_list(v, self.max_seq_length))
            #                      for k, v in dict_batch.items()}
            return padded_dict_batch
        except:
            return None

    def batch_dict_to_padded_dict_batch_double_sentence(self, batch_dict):
        """
        [{}, {}, {}] --> {k1: [], k2: [], ....}
        :param batch_dict:
        :return:
        """
        # 创建 batch_dict
        keys = batch_dict[0].keys()
        dict_batch = dict.fromkeys(keys)
        for k in keys:
            dict_batch[k] = list()

        # 将传入的 batch [{}, {}, {}] 存储到 batch_dict 中 {k1: [], k2: [], ....}
        for bat in batch_dict:
            for k, v in bat.items():
                dict_batch[k].append(v)

        try:
            # for k, v in dict_batch.items():
            #     a = self.padding_batch_list(v, self.max_seq_length)
            #     padded_dict_batch[k] = torch.LongTensor(a)
            # padded_dict_batch = {k: torch.LongTensor(self.padding_batch_list(v, self.max_seq_length))
            #                      for k, v in dict_batch.items()}
            padded_dict_batch = dict()
            for k, v in dict_batch.items():
                padded_dict_batch[k] = torch.LongTensor(self.padding_batch_list(v, self.max_seq_length))

            return padded_dict_batch
        except:
            return None

    def padding_batch_list(self, batch_list, max_sentence_length, pad_value=0):
        """
        对输入的 [[list1], [list2], ....] 进行padding操作
        注意：
            该函数只会对 [[]] 形式的代码进行 padding，如不符合格式则返回原数据
        :param batch_list:
        :param max_sentence_length:
        :param pad_value:
        :return:
        """
        try:
            paded_batch_list = [b_l + [pad_value] * (max_sentence_length - len(b_l)) for b_l in batch_list]
            return paded_batch_list
        except:
            return batch_list


class CCFDataIter(object):
    def __init__(self, batch_size, max_seq_length, num_works=4):
        """"""
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_works = num_works

    def get_iter_for_bert(self, data_dict, shuffle, drop_last):
        dataset = BertDataset(data_dict)
        return data.DataLoader(dataset=dataset,
                               batch_size=self.batch_size,
                               shuffle=shuffle,
                               num_workers=self.num_works,
                               collate_fn=self.batch_dict_to_padded_dict_batch,
                               drop_last=drop_last)

    def batch_dict_to_padded_dict_batch(self, batch_dict):
        """
        [{}, {}, {}] --> {k1: [], k2: [], ....}
        :param batch_dict:
        :return:
        """
        # 创建 batch_dict
        keys = batch_dict[0].keys()
        dict_batch = dict.fromkeys(keys)
        for k in keys:
            dict_batch[k] = list()

        # 将传入的 batch [{}, {}, {}] 存储到 batch_dict 中 {k1: [], k2: [], ....}
        for bat in batch_dict:
            for k, v in bat.items():
                dict_batch[k].append(v)

        return dict_batch


# main
if __name__ == "__main__":
    data_dict = {"sentence": [[1, 2], [2, 3], [1, 2], [2, 3]],
                 "label": [1, 2, 1, 2],
                 "token_type_ids": [[0, 0], [0, 0], [0, 0], [0, 0]],
                 "mask": [[1, 1], [1, 1], [1, 1], [1, 1]]}
    dataiter = DataIter(4, 7).get_iter_for_bert(data_dict, True)
    for i in dataiter:
        print(i)
