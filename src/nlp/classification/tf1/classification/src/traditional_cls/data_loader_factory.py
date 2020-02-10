# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-04-12
Description :
auther : wcy
"""
# import modules
import os, sys

curr_path = os.path.dirname(__file__)
sys.path.append(curr_path)

import pandas as pd
from tokenization import BertTokenizerAddBywcy
from dataiter import DataIter, CCFDataIter
from itertools import chain
import pickle


__all__ = ["DataLoaderFactory"]


# define class
class DataLoaderFactory(object):
    def __init__(self, num_works=4):
        self.num_works = num_works

    def get_bert_input_for_cls(self, sentence_series, label_series, batch_size, max_seq_length,
                          bert_prained_path, unique_label_list=None, shuffle=False, language="chinese", drop_last=True):
        """
        :param sentence_series: pandas.Series
        :param label_series: pandas.Series
        :param batch_size: int
        :param max_seq_length: int
        :param bert_prained_path: bert 预训练模型的路径
        :param unique_label_list: 所有不同的label的集合
        :param shuffle: 是否对数据的顺序进行打乱
        :param language: 处理的文本的语言类型
        :return:
        """
        sentence_list, label_list = list(sentence_series), list(label_series)

        # get label_to_id
        if unique_label_list is not None:
            label_to_id = {k: v for v, k in enumerate(unique_label_list)}
        else:
            label_to_id = {k: v for v, k in enumerate(self._get_unique_lable_list(label_list))}

        # get bert tokenize
        tokenizer = self._get_bert_tokenizer(bert_prained_path)

        # convert sentences to token_id & token type id & attention mask
        token_ids_list, label_ids_list = list(), list()
        token_type_ids_list, attention_masks_list = list(), list()
        for sentence in sentence_list:
            dict_ids = self._deal_one_sentence_for_bert(sentence, language, max_seq_length, tokenizer)
            token_ids_list.append(dict_ids["token_ids"])
            token_type_ids_list.append(dict_ids["token_type_ids"])
            attention_masks_list.append(dict_ids["attention_masks"])

        # convert label id
        label_ids_list = [label_to_id[label] for label in label_list]

        # data dict
        data_dict = {
            "token_ids": token_ids_list,
            "token_type_ids": token_type_ids_list,
            "attention_masks": attention_masks_list,
            "label_ids": label_ids_list
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle, drop_last=drop_last)
        return dataiter

    def get_input_for_ccf(self, file_path, batch_size, max_seq_length, shuffle, drop_last):
        """
        :param file_path:
        :param batch_size:
        :param max_seq_length:
        :param shuffle:
        :param drop_last:
        :return:
        """
        data = pickle.load(file_path)
        x = list()
        y = list()
        id_list = list()

        # data dict
        data_dict = {"x": x, "y": y, "id": id_list}
        dataiter = CCFDataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle, drop_last=drop_last)
        return dataiter

    def get_bert_input_for_cls_add_feature(self, sentence_series, features_list, label_series, batch_size, max_seq_length,
                                           bert_prained_path, unique_label_list=None, shuffle=False, language="chinese", drop_last=True):
        """
        :param sentence_series: pandas.Series
        :param label_series: pandas.Series
        :param batch_size: int
        :param max_seq_length: int
        :param bert_prained_path: bert 预训练模型的路径
        :param unique_label_list: 所有不同的label的集合
        :param shuffle: 是否对数据的顺序进行打乱
        :param language: 处理的文本的语言类型
        :return:
        """
        sentence_list, label_list, features_list = list(sentence_series), list(label_series), list(features_list)

        # get label_to_id
        if unique_label_list is not None:
            label_to_id = {k: v for v, k in enumerate(unique_label_list)}
        else:
            label_to_id = {k: v for v, k in enumerate(self._get_unique_lable_list(label_list))}

        # get bert tokenize
        tokenizer = self._get_bert_tokenizer(bert_prained_path)

        # convert sentences to token_id & token type id & attention mask
        token_ids_list, label_ids_list = list(), list()
        token_type_ids_list, attention_masks_list = list(), list()
        for sentence in sentence_list:
            dict_ids = self._deal_one_sentence_for_bert(sentence, language, max_seq_length, tokenizer)
            token_ids_list.append(dict_ids["token_ids"])
            token_type_ids_list.append(dict_ids["token_type_ids"])
            attention_masks_list.append(dict_ids["attention_masks"])

        # convert label id
        label_ids_list = [label_to_id[label] for label in label_list]

        # data dict
        data_dict = {
            "token_ids": token_ids_list,
            "token_type_ids": token_type_ids_list,
            "attention_masks": attention_masks_list,
            "label_ids": label_ids_list,
            "features_list": features_list
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle, drop_last=drop_last)
        return dataiter

    def get_word2vec_input_for_seq(self, sentence_series, label_series, batch_size, max_seq_length,
                                   word2id, label_to_id=None, shuffle=True, drop_last=True, language="chinese"):
        """
        :param sentence_series:
        :param label_series:
        :param batch_size:
        :param max_seq_length:
        :param word2id:
        :param label_to_id:
        :param shuffle:
        :param drop_last:
        :param language:
        :return:
        """
        sentence_list, label_list = list(sentence_series), list(label_series)

        # get label_to_id
        if label_to_id is not None:
            pass
        else:
            label_to_id = {k: v+1 for v, k in enumerate(self._get_unique_lable_list(list(chain.from_iterable(label_list))))}
            label_to_id["<PAD>"] = 0

        # get bert tokenize
        # convert sentences to token_id & token type id & attention mask
        # convert label id
        token_ids_list, label_ids_list, token_len_list = list(), list(), list()
        for sentence, label in zip(sentence_list, label_list):
            dict_ids = self._deal_one_sentence_one_label_for_word2vec(sentence, label, max_seq_length, word2id, label_to_id, language)
            token_ids_list.append(dict_ids["token_ids"])
            label_ids_list.append(dict_ids["label_ids"])
            token_len_list.append(dict_ids["token_len"])

        # 对原始数据的句子长度进行控制
        raw_sentence = list()
        raw_label = list()
        for sentence, label in zip(sentence_list, label_list):
            sentence = sentence.split(" ")
            label = label.split(" ")
            if len(sentence) > max_seq_length - 2:
                raw_sentence.append(sentence[:max_seq_length - 2])
                raw_label.append(label[:max_seq_length - 2])
            else:
                raw_sentence.append(sentence)
                raw_label.append(label)

        # data dict
        data_dict = {
            "raw_x": raw_sentence,
            "raw_y": raw_label,
            "token_ids": token_ids_list,
            "label_ids": label_ids_list,
            "token_len": token_len_list
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle, drop_last)
        return dataiter

    def get_bert_input_for_double_sentence_add_feature(self, sentence_series1, sentence_series2, features_list,
                                                       label_series, batch_size, max_seq_length, bert_prained_path,
                                                       unique_label_list=None, shuffle=False, language="chinese",
                                                       drop_last=True):
        """
        :param sentence_series1:
        :param sentence_series2:
        :param features_list:
        :param label_series:
        :param batch_size:
        :param max_seq_length:
        :param bert_prained_path:
        :param unique_label_list:
        :param shuffle:
        :param language:
        :param drop_last:
        :return:
        """
        sentence_list1, sentence_list2, label_list, features_list = list(sentence_series1), list(sentence_series2), list(label_series), list(features_list)

        # get label_to_id
        if unique_label_list is not None:
            label_to_id = {k: v for v, k in enumerate(unique_label_list)}
        else:
            label_to_id = {k: v for v, k in enumerate(self._get_unique_lable_list(label_list))}

        # get bert tokenize
        tokenizer = self._get_bert_tokenizer(bert_prained_path)

        # convert sentences to token_id & token type id & attention mask
        token_ids_list, label_ids_list = list(), list()
        token_type_ids_list, attention_masks_list = list(), list()

        for sentence1, sentence2 in zip(sentence_list1, sentence_list2):

            data_ids = self._deal_two_sentence_for_bert(sentence1, sentence2, language, max_seq_length, tokenizer)

            # data
            token_ids_list.append(data_ids["token_ids"])
            token_type_ids_list.append(data_ids["token_type_ids"])
            attention_masks_list.append(data_ids["attention_masks"])

        # convert label id
        label_ids_list = [label_to_id[label] for label in label_list]

        # data dict
        data_dict = {
            "token_ids": token_ids_list,
            "token_type_ids": token_type_ids_list,
            "attention_masks": attention_masks_list,
            "label_ids": label_ids_list,
            "features_list": features_list
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle,
                                                                                       drop_last=drop_last)
        return dataiter

    def get_bert_input_for_context_cls(self, sentence_list_f, sentence_list_m, sentence_list_b, label_list, batch_size,
                                       max_seq_length, bert_prained_path, unique_label_list, shuffle=True, language="chinese", drop_last=False):
        """
        :param sentence_list_f:
        :param sentence_list_m:
        :param sentence_list_b:
        :param label_list:
        :param batch_size:
        :param max_seq_length:
        :param bert_prained_path:
        :param unique_label_list:
        :param shuffle:
        :param language:
        :return:
        """
        # get label_to_id
        label_to_id = {k: v for v, k in enumerate(unique_label_list)}

        # get bert tokenize
        tokenizer = self._get_bert_tokenizer(bert_prained_path)

        # data dict
        data_dict = dict()

        # convert sentences to token_id & token type id & attention mask
        if sentence_list_f is not None:
            token_ids_list_f = list()
            token_type_ids_list_f, attention_masks_list_f = list(), list()
            for sentence in sentence_list_f:
                dict_ids = self._deal_one_sentence_for_bert(sentence, language, max_seq_length, tokenizer)
                token_ids_list_f.append(dict_ids["token_ids"])
                token_type_ids_list_f.append(dict_ids["token_type_ids"])
                attention_masks_list_f.append(dict_ids["attention_masks"])
            data_dict["token_ids_f"] = token_ids_list_f
            data_dict["token_type_ids_f"] = token_type_ids_list_f
            data_dict["attention_masks_f"] = attention_masks_list_f

        # convert sentences to token_id & token type id & attention mask
        if sentence_list_m is not None:
            token_ids_list_m = list()
            token_type_ids_list_m, attention_masks_list_m = list(), list()
            for sentence in sentence_list_m:
                dict_ids = self._deal_one_sentence_for_bert(sentence, language, max_seq_length, tokenizer)
                token_ids_list_m.append(dict_ids["token_ids"])
                token_type_ids_list_m.append(dict_ids["token_type_ids"])
                attention_masks_list_m.append(dict_ids["attention_masks"])
            data_dict["token_ids_m"] = token_ids_list_m
            data_dict["token_type_ids_m"] = token_type_ids_list_m
            data_dict["attention_masks_m"] = attention_masks_list_m

        # convert sentences to token_id & token type id & attention mask
        if sentence_list_b is not None:
            token_ids_list_b = list()
            token_type_ids_list_b, attention_masks_list_b = list(), list()
            for sentence in sentence_list_b:
                dict_ids = self._deal_one_sentence_for_bert(sentence, language, max_seq_length, tokenizer)
                token_ids_list_b.append(dict_ids["token_ids"])
                token_type_ids_list_b.append(dict_ids["token_type_ids"])
                attention_masks_list_b.append(dict_ids["attention_masks"])
            data_dict["token_ids_b"] = token_ids_list_b
            data_dict["token_type_ids_b"] = token_type_ids_list_b
            data_dict["attention_masks_b"] = attention_masks_list_b

        # convert label id
        label_ids_list = [label_to_id[label] for label in label_list]

        data_dict["label_ids"] = label_ids_list

        dataiter = DataIter(batch_size, max_seq_length, num_works=self.num_works).get_iter_for_bert(data_dict, shuffle, drop_last=drop_last)
        return dataiter

    def get_bert_input_for_seq(self, sentence_series, label_series, batch_size, max_seq_length,
                               bert_prained_path, label_to_id, shuffle=True, drop_last=True, language="chinese"):
        """
        :param sentence_series:
        :param label_series:
        :param batch_size:
        :param max_seq_length:
        :param bert_prained_path:
        :param label_to_id:
        :param shuffle:
        :param drop_last:
        :param language:
        :return:
        """
        sentence_list, label_list = list(sentence_series), list(label_series)

        # get label_to_id
        if label_to_id is not None:
            pass
        else:
            label_to_id = {k: v+1 for v, k in enumerate(self._get_unique_lable_list(list(chain.from_iterable(label_list))))}
            label_to_id["[PAD]"] = 0

        # get bert tokenize
        tokenizer = self._get_bert_tokenizer(bert_prained_path)



        # convert sentences to token_id & token type id & attention mask
        # convert label id
        token_ids_list, label_ids_list = list(), list()
        token_type_ids_list, attention_masks_list = list(), list()
        for sentence, label in zip(sentence_list, label_list):
            dict_ids = self._deal_one_sentence_one_label_for_bert(
                sentence, label, language, max_seq_length, tokenizer, label_to_id)
            token_ids_list.append(dict_ids["token_ids"])
            label_ids_list.append(dict_ids["label_ids"])
            token_type_ids_list.append(dict_ids["token_type_ids"])
            attention_masks_list.append(dict_ids["attention_masks"])

        # 对原始数据的句子长度进行控制
        raw_sentence = list()
        raw_label = list()
        for sentence, label in zip(sentence_list, label_list):
            sentence = sentence.split(" ")
            label = label.split(" ")
            if len(sentence) > max_seq_length -2:
                raw_sentence.append(sentence[:max_seq_length-2])
                raw_label.append(label[:max_seq_length-2])
            else:
                raw_sentence.append(sentence)
                raw_label.append(label)

        # data dict
        data_dict = {
            "raw_x": raw_sentence,
            "raw_y": raw_label,
            "token_ids": token_ids_list,
            "token_type_ids": token_type_ids_list,
            "attention_masks": attention_masks_list,
            "label_ids": label_ids_list,
            "token_len": [len(t) for t in token_ids_list]
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle, drop_last)
        return dataiter

    def get_bert_input_for_sim(self, sentence_series1, sentence_series2, label_series, batch_size, max_seq_length,
                          bert_prained_path, unique_label_list=None, shuffle=True, language="chinese"):
        """

        :param sentence_series1:
        :param sentence_series2:
        :param label_series:
        :param batch_size:
        :param max_seq_length:
        :param bert_prained_path:
        :param unique_label_list:
        :param shuffle:
        :param language:
        :return:
        """
        sentence_list1, sentence_list2, label_list = list(sentence_series1), list(sentence_series2), list(label_series)

        # get label_to_id
        if unique_label_list is not None:
            label_to_id = {k: v for v, k in enumerate(unique_label_list)}
        else:
            label_to_id = {k: v for v, k in enumerate(self._get_unique_lable_list(label_list))}

        # get bert tokenizer
        tokenizer = self._get_bert_tokenizer(bert_prained_path)

        # deal data for sentence1
        token_ids_list1, token_type_ids_list1, attention_masks_list1 = list(), list(), list()
        for sentence in sentence_list1:
            data_ids = self._deal_one_sentence_for_bert(sentence, language, max_seq_length, tokenizer)
            # data
            token_ids_list1.append(data_ids["token_ids"])
            token_type_ids_list1.append(data_ids["token_type_ids"])
            attention_masks_list1.append(data_ids["attention_masks"])

        # deal data for sentence2
        token_ids_list2, token_type_ids_list2, attention_masks_list2 = list(), list(), list()
        for sen in sentence_series2:
            dict_ids = self._deal_one_sentence_for_bert(sen, language, max_seq_length, tokenizer)
            # data
            token_ids_list2.append(dict_ids["token_ids"])
            token_type_ids_list2.append(dict_ids["token_type_ids"])
            attention_masks_list2.append(dict_ids["attention_masks"])
        # deal data for label
        label_id_list = [label_to_id[lab] for lab in label_series]

        # data dict
        data_dict = {
            "token_ids1": token_ids_list1,
            "token_type_ids1": token_type_ids_list1,
            "attention_masks1": attention_masks_list1,
            "token_ids2": token_ids_list2,
            "token_type_ids2": token_type_ids_list2,
            "attention_masks2": attention_masks_list2,
            "label_ids": label_id_list,
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_iter_for_bert(data_dict, shuffle)
        return dataiter

    def get_bert_input_for_double_sen(self, sentence_series1, sentence_series2, label_series, batch_size, max_seq_length,
                          bert_prained_path, unique_label_list=None, shuffle=True, language="chinese", drop_last=False):
        """
        :param sentence_series1:
        :param sentence_series2:
        :param label_series:
        :param batch_size:
        :param max_seq_length:
        :param bert_prained_path:
        :param unique_label_list:
        :param shuffle:
        :param language:
        :return:
        """
        sentence_list1, sentence_list2, label_list = list(sentence_series1), list(sentence_series2), list(label_series)

        # get label_to_id
        if unique_label_list is not None:
            label_to_id = {k: v for v, k in enumerate(unique_label_list)}
        else:
            label_to_id = {k: v for v, k in enumerate(self._get_unique_lable_list(label_list))}

        # get bert tokenizer
        tokenizer = self._get_bert_tokenizer(bert_prained_path)

        # deal data for sentence1
        token_ids_list, token_type_ids_list, attention_masks_list = list(), list(), list()
        for sentence1, sentence2 in zip(sentence_list1, sentence_list2):

            data_ids = self._deal_two_sentence_for_bert(sentence1, sentence2, language, max_seq_length, tokenizer)

            # data
            token_ids_list.append(data_ids["token_ids"])
            token_type_ids_list.append(data_ids["token_type_ids"])
            attention_masks_list.append(data_ids["attention_masks"])

        # deal data for label
        label_id_list=list()
        for lab in label_series:
            if isinstance(lab, int):
                label_id_list.append(label_to_id[lab])
            else:
                label_id_list.append(label_to_id[int(lab)])

        # data dict
        data_dict = {
            "token_ids": token_ids_list,
            "token_type_ids": token_type_ids_list,
            "attention_masks": attention_masks_list,
            "label_ids": label_id_list,
        }
        dataiter = DataIter(batch_size, max_seq_length, num_works=4).get_double_sentence_iter_for_bert(data_dict, shuffle, drop_last=drop_last)
        return dataiter

    def _deal_one_sentence_for_bert(self, sentence, language, max_seq_length, tokenizer):
        """
        :param sentence:
        :param language:
        :return:
        """
        token = list()
        for word in self._split_sentence(sentence, language):
            # convert word to token
            token.extend(tokenizer.tokenize(word))

        # 对句子的长度进行控制
        if len(token) > max_seq_length - 2:
            token = ["[CLS]"] + token[:max_seq_length - 2] + ["[SEP]"]
        else:
            token = ["[CLS]"] + token + ["[SEP]"]

        # convert token to id
        token_id = tokenizer.convert_tokens_to_ids(token)
        return {"token_ids": token_id, "token_type_ids": [0] * len(token),
                "attention_masks": [1] * len(token)}

    def _deal_two_sentence_for_bert(self, sentence1, sentence2, language, max_seq_length, tokenizer):
        """
        :param sentence1:
        :param sentence2:
        :param language:
        :param max_seq_length:
        :param tokenizer:
        :return:
        """
        token1 = list()
        for word in self._split_sentence(sentence1, language):
            # convert word to token
            token1.extend(tokenizer.tokenize(word))

        token2 = list()
        for word in self._split_sentence(sentence2, language):
            # convert word to token
            token2.extend(tokenizer.tokenize(word))

        # 对句子的长度进行控制
        token = token1 + ["[SEP]"] + token2
        if len(token) > max_seq_length - 1:
            token = ["[CLS]"] + token[:max_seq_length - 1]
        else:
            token = ["[CLS]"] + token

        token_type_ids = [0] * (len(token1) + 1) + [1] * (len(token2) + 1)
        if len(token_type_ids) > max_seq_length:
            token_type_ids = token_type_ids[:max_seq_length]

        # convert token to id
        token_id = tokenizer.convert_tokens_to_ids(token)
        return {"token_ids": token_id, "token_type_ids": token_type_ids,
                "attention_masks": [1] * len(token)}

    def _deal_one_sentence_one_label_for_bert(self, sentence, label, language,
                                              max_seq_length, tokenizer, label_to_id):
        """
        :param sentence:
        :param language:
        :return:
        """
        token, label_id = list(), list()
        for word, lab in zip(self._split_sentence(sentence, language), label.split(" ")):
            # convert word to token
            work_token = tokenizer.tokenize(word)
            token.extend(work_token)
            label_id.extend([label_to_id[l] for l in [lab] + ["<PAD>"] * (len(work_token)-1)])

        # 对句子的长度进行控制
        if len(token) > max_seq_length - 2:
            token = ["[CLS]"] + token[:max_seq_length - 2] + ["[SEP]"]
            label_id = [0] + label_id[:max_seq_length-2] + [0]
        else:
            token = ["[CLS]"] + token + ["[SEP]"]
            label_id = [0] + label_id + [0]

        # convert token to id
        token_id = tokenizer.convert_tokens_to_ids(token)
        return {"token_ids": token_id, "label_ids": label_id,
                "token_type_ids": [0] * len(token), "attention_masks": [1] * len(token)}

    def _deal_one_sentence_one_label_for_word2vec(self, sentence, label, max_seq_length, word2id, label_to_id, language):
        """
        :param sentence:
        :param language:
        :return:
        """
        token = self._split_sentence(sentence, language)
        label = label.split(" ")

        # 对句子的长度进行控制
        if len(token) > max_seq_length:
            token = token[:max_seq_length]
            label = label[:max_seq_length]
        else:
            token = token
            label = label

        label_id = [label_to_id[l] for l in label]

        # convert token to id
        token_id = list()
        for tok in token:
            try:
                token_id.append(word2id[tok])
            except:
                print("存在此表中不存在的字符", tok)
                token_id.append(word2id["[UNK]"])

        # get sentence length
        token_len = len(token_id)

        return {"token_ids": token_id, "label_ids": label_id, "token_len": token_len}

    def _get_unique_lable_list(self, all_label):
        """
        获取unique_lable_list
        :param all_label:
        :return:
        """
        assert isinstance(all_label, list)
        return set(all_label)

    def _get_bert_tokenizer(self, path):
        """
        :return:
        """
        return BertTokenizerAddBywcy.from_pretrained(path)

    def _split_sentence(self, sentence, language):
        """
        :param language:
        :return:
        """
        assert language in ["chinese", "english"]
        if language == "english":
            if not isinstance(sentence, str):
                sentence = str(sentence)
            return sentence.split(" ")
        elif language == "chinese":
            try:
                return list(sentence)
            except:
                print("有问题的句子" + str(sentence))
                return list(".")


# main
if __name__ == '__main__':
    # bert pretrained path
    BERT_PRETRAINED_PATH = os.path.join(os.path.dirname(os.path.dirname(curr_path)),
                                        "bert_pretrained/model_ckpt/multi_cased_L-12_H-768_A-12")
    # 实例化类
    data_loader = DataLoaderFactory()
    # 读取数据
    df = pd.read_csv("test.csv")

    # get data loader for cls
    # dataiter = data_loader.get_bert_input_for_cls(df.sentence1, df.label1, language="english", batch_size=3,
    #                                               max_seq_length=10, bert_prained_path=BERT_PRETRAINED_PATH)

    # get data loader for seq
    # dataiter = data_loader.get_bert_input_for_seq(df.sentence1, df.label1, language="english", batch_size=3,
    #                                               max_seq_length=10, bert_prained_path=BERT_PRETRAINED_PATH)

    # get data loader for sim
    dataiter = data_loader.get_bert_input_for_sim(df.sentence1, df.sentence2, df.label1, language="english",
                                                  batch_size=3, max_seq_length=10, bert_prained_path=BERT_PRETRAINED_PATH)

    for i in dataiter:
        print(i)
