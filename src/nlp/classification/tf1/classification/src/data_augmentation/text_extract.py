import numpy as np
import pandas as pd


def extract(input_file,maxlen,output_file,time):
    """

    :param input_file:
    :param maxlen: length of content after extracting
    :param output_file:
    :param time: time of extracting
    :return:
    """
    prob = np.random.randint(1, 10)/10

    process_content = []
    data = pd.read_csv(input_file)

    content = data['content']

    for i in content:
        i = str(i)

        if len(i) > maxlen:


            #前后被抽取的content长度
            split_id = int(len(i) * prob)

            pre_content = content[:split_id]
            post_content = content[split_id:]
            #前后抽取的长度
            pre_extract_len = int(maxlen * prob)
            post_extact_len = maxlen - pre_extract_len

            if split_id == pre_extract_len:
                begin_id1 = 0

            #随机初始化begin_id，用切片的方式从content中截取固定长度
            else:

                begin_id1 = np.random.randint(0, split_id - pre_extract_len)

                begin_id2 = np.random.randint(0, (len(i)-split_id)-post_extact_len)

            #抽取后的content
            pre = pre_content[begin_id1:begin_id1+pre_extract_len]
            post = post_content[begin_id2:begin_id2+post_extact_len]

            result = pre + post

        else:
            result = i

        process_content.append(result)


    data['content'] = process_content

    dataframe = pd.DataFrame({'id': data['id'], 'label': data['label'], 'title': data['title']})


    #输出
    dataframe.to_csv(output_file)
    for i in range(time - 1):
        dataframe.to_csv(output_file, mode='a', header=False)



if __name__ == '__main__':
    extract("ccf_data/dev_1.csv", 320, "ccf_data1/dev_1.csv", 5)








