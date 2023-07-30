import os
from sys import path
path.append(os.getcwd())
path.append('/home/Medical_Understanding/TSPMN')
# from jieba.analyse import extract_tags, set_idf_path
import re
import jieba
import math
import pandas as pd
from tqdm import tqdm
import multiprocessing as mul
import time
# import synonyms
from common_utils import read_txt, read_json, write_json, write_txt
medical_dict_path = '../medical_dict/all_medical_words.txt'
medical_words = read_txt(medical_dict_path)
jieba.load_userdict(medical_dict_path)


def func(dialogue_old, medical_words=medical_words):
    new_dialog = {}
    dialogue = dialogue_old.replace('<patient>', ' ')
    dialogue = dialogue.replace('<doctor>', ' ')
    dialogue = dialogue.replace('\n', ' ')
    seg_list = jieba.cut(dialogue)
    seg_list = list(set(seg_list))
    medical_word_list = []
    for word in seg_list:
        if word in medical_words:
            medical_word_list.append(word)

    new_dialog['text'] = dialogue_old
    new_dialog['medical_words'] = medical_word_list

    return new_dialog

start = time.time()
pool = mul.Pool(64)

dialogues = read_json(path='all_data.json')
# dialogues = dialogues[:1000]
# print(len(dialogues))
dialogues = pool.map(func, dialogues)
end = time.time()
print('总共用时：', (end-start)/60)
print('已处理数据：', len(dialogues)) # 前1000条样本处理时间为0.47分钟，总计3.5M条样本

medical_words = []
for dialog in dialogues:
    medical_words.extend(dialog['medical_words'])

# words_len = len(medical_words)
medical_words_list = list(set(medical_words))

write_json(data=dialogues, path='all_data_with_medical_words.json')
write_txt(data=medical_words_list, path='medical_words_list.txt')


"""
log:
总共用时： 16.47050006389618
已处理数据： 63754
总共用时： 21.11899951696396
已处理数据： 81615
总共用时： 38.59306695063909
已处理数据： 170557
已写入数据至文件data/data_for_pretrain/all_data_with_medical_words_100.json，数据量：100
已写入数据至文件data/data_for_pretrain/all_data_with_medical_words.json，数据量：170557
"""
     

