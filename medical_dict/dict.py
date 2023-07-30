import os
from sys import path
path.append(os.getcwd())
# path.append('/home/Medical_Understanding/TSPMN')

from common_utils import write_txt, read_json, write_json, read_scel

def read_word_from_graph(path):
    graph = read_json(path=path)
    words = []
    for triple in graph['graph']:
        words.append(triple[0])
        words.append(triple[2])
    words = list(set(words))
    return words

def read_txt(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        file = f.readlines()
    for line in file:
        lines.append(line.strip('\n').split('\t')[0])
    return lines

path_1 = 'medical_dict/THUOCL_medical.txt'
dict_1 = read_txt(path_1)

path_2 = 'medical_dict/sougou_dict.scel'
dict_2 = read_scel(os.path.join(path_2))

graph_pathes = ['data4pretrain/VRBot-sigir2021-datasets/kamed_joint_graph/graph.json', 
                'data4pretrain/VRBot-sigir2021-datasets/meddialog_joint_graph/graph.json']

graph_words = []
for path in graph_pathes:
    graph_words.extend(read_word_from_graph(path=path))

graph_words = list(set(graph_words))

all_words = []
all_words.extend(dict_1)
all_words.extend(graph_words)
all_words.extend(dict_2)

all_words = list(set(all_words))
single_words = []
for word in all_words:
    if len(word) == 1: # 去掉单字
        single_words.append(word)

all_words = list(set(all_words) - set(single_words))




write_txt(data=all_words, path='medical_dict/all_medical_words.txt')