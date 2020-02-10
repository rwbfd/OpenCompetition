from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd


# define function
def cal_score(y_true, y_pred, all_label_ls, label_ls=None, average='micro'):
    """
    :param y_true: list of true values
    :param y_pred: list of predict values
    :param all_label_ls: list of label types which you'd like to calculate average score
    :param label_ls: list of label types
    :param average: average method (micro, macro)
    :return: [precision_all_ls, recall_all_ls, f1_all_ls, score_each_df]
    """
    if not len(np.unique(np.asarray(y_true))) == len(all_label_ls):
        true_ls, pred_ls = [], []
        for i in range(len(y_true)):
            if y_true[i] in all_label_ls:
                true_ls.append(y_true[i])
                pred_ls.append(y_pred[i])
    else:
        true_ls, pred_ls = y_true, y_pred

    precision_all = precision_score(true_ls, pred_ls, average=average)
    recall_all = recall_score(true_ls, pred_ls, average=average)
    f1_all = f1_score(true_ls, pred_ls, average=average)

    if label_ls:
        precision_each_array = precision_score(y_true, y_pred, labels=label_ls, average=None)
        recall_each_array = recall_score(y_true, y_pred, labels=label_ls, average=None)
        f1_each_array = f1_score(y_true, y_pred, labels=label_ls, average=None)
        score_df = pd.DataFrame({"precision": precision_each_array, "recall": recall_each_array,
                                 "f1": f1_each_array}, index=label_ls)

        return [precision_all, recall_all, f1_all, score_df]

    return [precision_all, recall_all, f1_all]


if __name__ == '__main__':
    y_true = [0, 1, 2, 2, 3, 3, 3]
    y_pred = [0, 0, 0, 2, 3, 3, 3]

    # score = cal_score(y_true, y_pred)
    score=cal_score(y_true, y_pred, all_label_ls=[0, 1, 2, 3], label_ls=[0, 1, 2, 3])
    # print("precision_all: {}\nrecall_all: {}\nf1_all: {}".format(score[0], score[1], score[2]))
    print("precision_all: {}\nrecall_all: {}\nf1_all: {}\nscore_df:\n{}".format(
        score[0], score[1], score[2], score[3]))

