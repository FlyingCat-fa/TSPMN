import os
from sys import path
path.append(os.getcwd())
root_path = '/home/Medical_Understanding'
path.append(root_path)

from common_utils import write_txt, read_txt, read_json, write_json

# data_path = 'data_chunyu.json'
# data_100_path = 'data_chunyu_100.json'
# data = read_json(data_path)
# write_json(data=data[:100], path=data_100_path)

def read_word_from_graph(path):
    graph = read_json(path=path)
    words = []
    for triple in graph['graph']:
        words.append(triple[0])
        words.append(triple[2])
    words = list(set(words))
    return words

def read_txt_v2(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        file = f.readlines()
    for line in file:
        lines.append(line.strip('\n').split('\t')[0])
    return lines

path_1 = 'medical_dict/THUOCL_medical.txt'
dict_1 = read_txt_v2(path_1)

path_2 = 'medical_dict/medical_words.txt'
dict_2 = read_txt(path=path_2)

# graph_pathes = ['VRBot-sigir2021-datasets/kamed_joint_graph/graph.json', 
#                 'VRBot-sigir2021-datasets/meddg_joint_graph/graph.json', 
#                 'VRBot-sigir2021-datasets/meddialog_joint_graph/graph.json']

graph_pathes = ['VRBot-sigir2021-datasets/kamed_joint_graph/graph.json', 
                'VRBot-sigir2021-datasets/meddialog_joint_graph/graph.json']

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
    if len(word) == 1:
        single_words.append(word)

all_words = list(set(all_words) - set(single_words))




write_txt(data=all_words, path='medical_dict/all_medical_words.txt')