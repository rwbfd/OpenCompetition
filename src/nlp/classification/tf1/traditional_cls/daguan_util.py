import codecs
import os
import numpy as np

curr_path = os.getcwd()

# 0 install crf++ https://taku910.github.io/crfpp/
# 1 train data in
# 2 test data in
# 3 crf train
# 4 crf test
# 5 submit test


# step 1 train data in
def convert_raw_to_train(in_file, out_file):
    """
    :param in_file:
    :param out_file:
    :return:
    """
    with codecs.open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            tags = []
            samples = line.strip().split('  ')
            for sample in samples:
                sample_list = sample[:-2].split('_')
                tag = sample[-1]
                features.extend(sample_list)
                tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(['B-' + tag] + ['I-' + tag] * (len(sample_list)-1))
            results.append(dict({'features': features, 'tags': tags}))
        train_write_list = []
        with codecs.open(out_file, 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['tags'])):
                    train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
                train_write_list.append('\n')
            f_out.writelines(train_write_list)


# step 2 test data in
def convert_raw_to_test(in_file, out_file):
    """
    :param in_file:
    :param out_file:
    :return:
    """
    with codecs.open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            sample_list = line.split('_')
            features.extend(sample_list)
            results.append(dict({'features': features}))
        test_write_list = []
        with codecs.open(out_file, 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['features'])):
                    test_write_list.append(result['features'][i] + '\n')
                test_write_list.append('\n')
            f_out.writelines(test_write_list)


# # 3 crf train
# crf_train = "crf_learn -f 3 template.txt dg_train.txt dg_model"
# os.system(crf_train)
#
#
# # 4 crf test
# crf_test = "crf_test -m dg_model dg_test.txt -o dg_result.txt"
# os.system(crf_test)


# 5 submit data
def convert_output_to_submit(in_file, out_file):
    """
    :param in_file:
    :param out_file:
    :return:
    """
    f_write = codecs.open(out_file, 'w', encoding='utf-8')
    with codecs.open(in_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')
        for line in lines:
            if line == '':
                continue
            tokens = line.split('\n')
            features = []
            tags = []
            for token in tokens:
                feature_tag = token.split()
                features.append(feature_tag[0])
                tags.append(feature_tag[-1])
            samples = []
            i = 0
            while i < len(features):
                sample = []
                if tags[i] == 'O':
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j] == 'O':
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/o')
                else:
                    if tags[i][0] != 'B':
                        """
                        注释掉 print 不再输出
                        change by wcy
                        """
                        # print(tags[i][0] + ' error start')
                        j = i + 1
                    else:
                        sample.append(features[i])
                        j = i + 1
                        while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                            sample.append(features[j])
                            j += 1
                        samples.append('_'.join(sample) + '/' + tags[i][-1])
                i = j
            f_write.write('  '.join(samples) + '\n')


def convert_input_to_list(in_file):
    """
    :return:
    """
    with open(in_file) as f:
        con = f.readlines()
    con = [co.replace("\n", "").split("\t") for co in con]

    sentence_list = list()
    label_list = list()

    sentence = list()
    label = list()
    for co in con:
        if len(co) == 2:
            sentence.append(co[0])
            label.append(co[1])
        elif len(co) == 1:
            # 添加 sentence 和 label
            sentence_list.append(sentence)
            label_list.append(label)

            # 清空列表
            sentence = list()
            label = list()
        else:
            print("存在错误数据： " + str(co))

    sentence_list = [" ".join(s) for s in sentence_list]
    label_list = [" ".join(l) for l in label_list]
    return {"sentence_list": sentence_list, "label_list": label_list}


def convert_test_input_to_list(in_file):
    """
    :return:
    """
    with open(in_file) as f:
        con = f.readlines()
    con = [co.replace("\n", "") for co in con]

    sentence_list = list()
    label_list = list()

    sentence = list()
    label = list()
    for co in con:
        if co != '':
            sentence.append(co)
            label.append("O")
        elif co == "":
            # 添加 sentence 和 label
            if len(sentence) > 0:
                sentence_list.append(sentence)
                label_list.append(label)

            # 清空列表
            sentence = list()
            label = list()
        else:
            print("存在错误数据： " + str(co))

    sentence_list = [" ".join(s) for s in sentence_list]
    label_list = [" ".join(l) for l in label_list]
    return {"sentence_list": sentence_list, "label_list": label_list}


def calculate_da_guan_f1(true_file, predict_file, split="  "):
    """
    :param true_file:
    :param predict_file:
    :param split:
    :return:
    """
    # 读取 true file 数据
    with open(true_file) as f:
        true_lines = f.readlines()
    true_lines = [lines.replace("\n", "") for lines in true_lines]

    # 读取 predict file 数据
    with open(predict_file) as f:
        predict_lines = f.readlines()
    predict_lines = [lines.replace("\n", "") for lines in predict_lines]

    correct = []
    pred_len = 0
    true_len = 0
    for p, t in zip(predict_lines, true_lines):
        p = str(p).split(split)
        p = set([word for word in p if word[-1] != "o"])
        t = str(t).split(split)
        t = set([word for word in t if word[-1] != "o"])
        cor = sum(1 for word in p if word in t) if p != {""} else 0
        correct.append(cor)
        pred_len += len(p) if p != {""} else 0
        true_len += len(t) if t != {""} else 0

    precision = np.sum(correct) / pred_len if pred_len != 0 else 0
    recall = np.sum(correct) / true_len if true_len != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    """
    注释掉，不再输出
    change by wcy
    """
    # print("predict length:", pred_len)
    # print("true length:", true_len)
    # print("correct:", np.sum(correct), correct)
    # print("precision:", precision)
    # print("recall:", recall)
    # print("f1", f1)

    return [precision, recall, f1]


# mian
if __name__ == "__main__":
    # convert_raw_to_train("raw/baseline/train.txt", "raw/baseline/train_input.txt")
    # convert_output_to_submit("train_input.txt", "train_submit.txt")
    # pred = ["11/a 12/b 16/o 11/a", "13/a 14/a 15/b 16/o"]
    # true = ["11/a 12/b 16/o", "13/a 14/b 16/o 17/a"]
    # result_ls = metric(pred, true)
    a = convert_input_to_list(in_file=os.path.join(curr_path, "train_data", "dev_input.txt"))
    # [precision_ner, recall_ner, f1_ner] = calculate_da_guan_f1(true_file="train_data/dev.txt",
    #                                                            predict_file="output/eval_out_submit.txt")

    # a = convert_test_input_to_list(in_file=os.path.join(curr_path, "train_data", "test_input.txt"))
    print(11)
    pass





