# !/user/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import logging
from torch.nn import DataParallel
from calculate_score import cal_score
from daguan_util import calculate_da_guan_f1, convert_output_to_submit


# define class
class ModelRunner(object):
    def __init__(self, task_type=None, is_bert=None, label_to_id=None):
        self.logger = self.set_logger()
        self.use_data_parallel = False
        self.task_type = task_type
        self.is_bert = is_bert
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def set_logger(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        return logging.getLogger(__name__)

    def train(self, total_step, eval_per_step, task_model, model_save_path, optimizer,
              train_data_loader, dev_data_loader, eval_label_list, compare_param,
              eval_file_save_name):
        """
        :param total_step:
        :param eval_per_step:
        :param task_model:
        :param model_save_path:
        :param optimizer:
        :param train_data_loader:
        :param dev_data_loader:
        :param eval_label_list:
        :param compare_param:
        :param eval_file_save_name:
        :return:
        """
        # 数据batch 数量
        self.logger.info("train 数据的batch 数量：" + str(train_data_loader.__len__()))

        # DataParallel
        [task_model] = self.config_devices([task_model])

        # 进行模型的训练
        self.logger.info("开始模型的训练。。。。")
        for step in range(total_step):
            try:
                data_batch = train_data_loader_iter.next()
            except:
                train_data_loader_iter = train_data_loader.__iter__()
                data_batch = train_data_loader_iter.next()

            self.train_one_step(step=step, data_batch=data_batch, task_model=task_model, optimizer=optimizer)

            if (step+1) % eval_per_step == 0:
                # model eval
                eval_out = self.eval_model(task_model, dev_data_loader, eval_label_list,
                                           eval_file_save_name)

                self.logger.info("----acc:" + str(eval_out["acc"]) + "----")
                self.logger.info("----f1:" + str(eval_out["f1"]) + "----")
                self.logger.info("----p:" + str(eval_out["precision"]) + "----")
                self.logger.info("----r:" + str(eval_out["recall"]) + "----")
                self.logger.info("----single_score:\n" + str(eval_out["single_score"]) + "----\n\n")

                if compare_param is "f1":
                    self.compare_f1_to_save_model(model=task_model,
                                                  f1=eval_out["f1"],
                                                  r=eval_out["recall"],
                                                  p=eval_out["precision"],
                                                  acc=eval_out["acc"],
                                                  single_score=eval_out["single_score"],
                                                  model_save_path=model_save_path)
                if compare_param is "acc":
                    self.compare_acc_to_save_model(model=task_model,
                                                   f1=eval_out["f1"],
                                                   r=eval_out["recall"],
                                                   p=eval_out["precision"],
                                                   acc=eval_out["acc"],
                                                   single_score=eval_out["single_score"],
                                                   model_save_path=model_save_path)

    def train_one_step(self, step, data_batch, task_model, optimizer):
        """
        :param step:
        :param data_batch:
        :param task_model:
        :param optimizer:
        :return:
        """
        # 进行 one epoch 的模型训练
        if data_batch is None:
            self.logger.info("存在错误的数据，跳过了第-" + str(step) + "-个数据!")
        else:
            task_model.train()
            optimizer.zero_grad()

            loss = task_model(data_batch)["loss"]
            if self.use_data_parallel:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                self.logger.info("--step-: " + str(step) + " -- " + "-- loss: " + str(loss))
        return None

    def eval_model(self, task_model, data_iter, eval_label_list, eval_file_save_name):
        """
        :param task_model:
        :param data_iter:
        :return:
        """
        # 输出数据的batch 数量
        self.logger.info("dev 数据集的batch数量： " + str(data_iter.__len__()))

        # 保存数据的列表
        y_true = list()
        y_predict = list()

        # 进行模型的测试
        task_model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(data_iter):
                if data_batch is None:
                    self.logger.info("存在错误的数据，跳过了第-" + str(i) + "-个batch!")
                else:

                    y_pred = task_model.forward(data_batch)["predict"].to("cpu").tolist()
                    y_tru = data_batch["y"].to("cpu").tolist()

                    y_predict.extend(y_pred)
                    y_true.extend(y_tru)

        # 计算模型的p, r, f1, cf
        self.logger.info("y_true 的总label数量为： " + str(len(y_true)))
        self.logger.info("y_predict 的总label数量为： " + str(len(y_predict)))
        [precision, recall, f1, single_score] = cal_score(y_true=y_true,
                                                          y_pred=y_predict,
                                                          all_label_ls=eval_label_list,
                                                          label_ls=eval_label_list,
                                                          average='micro')

        # 计算acc
        # calculate acc in this batch
        acc = (np.array(y_true) == np.array(y_predict)).sum().tolist() / len(y_true)

        return {"precision": precision, "recall": recall, "f1": f1,
                "acc": acc, "single_score": single_score}

    def save_model(self, model, save_model_path, save_to_onnx=False):
        if save_to_onnx:
            self.logger.info("save model , path: " + str(save_model_path))
            pass
        else:
            if self.use_data_parallel:
                torch.save(model.module.state_dict(), save_model_path)
                self.logger.info("save model , path: " + str(save_model_path))
            else:
                torch.save(model.state_dict(), save_model_path, )
                self.logger.info("save model , path: " + str(save_model_path))

    def load_model(self, model, model_path, load_from_onnx=False):
        if load_from_onnx:
            pass
        else:
            if self.use_data_parallel:
                model.load_state_dict(torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None))
                print("从" + str(model_path) + "加载模型成功")
                pass
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None))
                print("从" + str(model_path) + "加载模型成功")
        return None

    def config_devices(self, model_list=[]):
        """
        :param model_list:
        :return:
        """
        # 判断使用cpu 还是 gpu
        if torch.cuda.is_available():
            self.logger.info("使用的device为： cuda")

            # 如果是多块GPU 则进行并行化训练
            if torch.cuda.device_count() > 1:
                model_list = [DataParallel(mo) for mo in model_list]
                model_list = [mo.cuda() for mo in model_list]
                self.use_data_parallel = True
            else:
                model_list = [mo.cuda() for mo in model_list]
        return model_list

    def compare_acc_to_save_model(self, model, f1, r, p, acc, single_score, model_save_path):
        """
        :param model:
        :param f1:
        :param r:
        :param p:
        :param acc:
        :param single_score:
        :param model_save_path:
        :return:
        """
        print("使用 acc 值进行模型的选取")
        if self.use_data_parallel:
            # 如果使用了nn.DataParallel() 那么模型参数调用和模型保存会和没有调用过的情况有所不同
            # ask_model.module.train_state 分别对用模型 epoch/ best_epoch/ f1/ p/ r
            if acc > model.module.train_state[5]:
                model.module.train_state[1] = model.module.train_state[0]
                model.module.train_state[2] = f1
                model.module.train_state[3] = p
                model.module.train_state[4] = r
                model.module.train_state[5] = acc
                self.logger.info("----best_acc:" + str(model.module.train_state[5]) + "----")
                self.logger.info("----best_f1:" + str(model.module.train_state[2]) + "----")
                self.logger.info("----best_p:" + str(model.module.train_state[3]) + "----")
                self.logger.info("----best_r:" + str(model.module.train_state[4]) + "----")
                self.logger.info("----best_single_score:\n" + str(single_score) + "----\n\n")
                # save model
                self.save_model(model, model_save_path)

        else:
            # 如果没有使用nn.DataParallel()，那么模型就可以按照正常的方式调用
            if acc > model.train_state[5]:
                model.train_state[1] = model.train_state[0]
                model.train_state[2] = f1
                model.train_state[3] = p
                model.train_state[4] = r
                model.train_state[5] = acc
                self.logger.info("----best_acc:" + str(acc) + "----")
                self.logger.info("----best_f1:" + str(f1) + "----")
                self.logger.info("----best_p:" + str(p) + "----")
                self.logger.info("----best_r:" + str(r) + "----")
                self.logger.info("----best_single_score:\n" + str(single_score) + "----\n\n")

                # save model
                self.save_model(model, model_save_path)
        return None

    def compare_f1_to_save_model(self, model, f1, r, p, acc, single_score, model_save_path):
        """
        :param model:
        :param f1:
        :param r:
        :param p:
        :param acc:
        :param single_score:
        :param model_save_path:
        :return:
        """
        print("使用 f1 值进行模型的选取")
        if self.use_data_parallel:
            # 如果使用了nn.DataParallel() 那么模型参数调用和模型保存会和没有调用过的情况有所不同
            # ask_model.module.train_state 分别对用模型 epoch/ best_epoch/ f1/ p/ r
            if f1 > model.module.train_state[2]:
                model.module.train_state[1] = model.module.train_state[0]
                model.module.train_state[2] = f1
                model.module.train_state[3] = p
                model.module.train_state[4] = r
                model.module.train_state[5] = acc
                self.logger.info("----best_acc:" + str(model.module.train_state[5]) + "----")
                self.logger.info("----best_f1:" + str(model.module.train_state[2]) + "----")
                self.logger.info("----best_p:" + str(model.module.train_state[3]) + "----")
                self.logger.info("----best_r:" + str(model.module.train_state[4]) + "----")
                self.logger.info("----best_single_score:\n" + str(single_score) + "----\n\n")
                # save model
                self.save_model(model, model_save_path)

        else:
            # 如果没有使用nn.DataParallel()，那么模型就可以按照正常的方式调用
            if f1 > model.train_state[2]:
                model.train_state[1] = model.train_state[0]
                model.train_state[2] = f1
                model.train_state[3] = p
                model.train_state[4] = r
                model.train_state[5] = acc
                self.logger.info("----best_acc:" + str(acc) + "----")
                self.logger.info("----best_f1:" + str(f1) + "----")
                self.logger.info("----best_p:" + str(p) + "----")
                self.logger.info("----best_r:" + str(r) + "----")
                self.logger.info("----best_single_score:\n" + str() + "----\n\n")

                # save model
                self.save_model(model, model_save_path)
        return None

    def _remove_pad(self, y, length):
        y_removed = []
        if self.is_bert:
            for i, k in zip(y, length):
                y_removed.append(i[1:k-1])
            return y_removed

        else:
            for i, k in zip(y, length):
                y_removed.append(i[:k])
            return y_removed

    def predict_for_ccf(self, dataiter, model, save_file_path, model_path, load_from_onnx=False):
        """
        :param dataiter:
        :param model:
        :param save_file_path:
        :param model_path:
        :param load_from_onnx:
        :return:
        """
        self.load_model(model, model_path, load_from_onnx)

        id_list = list()
        y_pred = list()

        [model] = self.config_devices([model])

        model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(dataiter):
                if data_batch is None:
                    self.logger.info("存在错误的数据，跳过了第-" + str(i) + "-个batch!")
                else:
                    model_out = model(data_batch)

                    # 接受数据
                    y_p = model_out["predict"].to("cpu").tolist()
                    id_l = data_batch["id_list"].to("cpu").tolist()

                    # y_pred & y_true & text
                    y_pred.extend(y_p)
                    id_list.extend(id_l)

        with open(save_file_path, "w") as f:
            for x, y in zip(id_list, y_pred):
                assert len(x) == len(y)
                for word, i in zip(x, y):
                    f.writelines(str(word) + "\t" + str(self.id_to_label[i]) + "\n")
                f.writelines("\n")