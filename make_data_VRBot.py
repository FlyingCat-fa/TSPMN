from __future__ import absolute_import, division, print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

from sys import path
path.append(os.getcwd())
root_path = '/home/Medical_Understanding'
path.append(root_path)
import multiprocessing as mul
import time
from common_utils import read_json, write_json, read_dir_file_name
"""
将对话保存为纯文本序列
"""

def make_data_from_VRBot(dialog):
    new_dialog = ''
    dialog_len = 0
    old_role = ''
    for i, turn in enumerate(dialog):
        role = '<' + turn["role"] + '>'
        turn["sentence"] = turn["sentence"].strip()
        if role == old_role:
            dialog_len += len(turn["sentence"])
            if dialog_len > 400:
                break
            new_dialog += turn["sentence"]
        else:
            dialog_len += len(turn["sentence"]) + 1
            if dialog_len > 400:
                break
            old_role = role
            new_dialog += role + turn["sentence"] + '\n'
    return new_dialog

dir_path = 'VRBot-sigir2021-datasets'
# dir_sub_path = ['kamed_test','kamed_valid', 'kamed_train']
dir_sub_path = ['meddg_test','meddg_valid', 'meddg_train']

preprocess_dialogs = []

for sub_path in dir_sub_path:
    file_names = read_dir_file_name(path=os.path.join(dir_path, sub_path), suffix='json')
    dialogues = []
    for file_name in file_names:
        dialogue = read_json(path=os.path.join(dir_path, sub_path, file_name))
        dialogues.append(dialogue['dialogues'])

start = time.time()
pool = mul.Pool(64)
new_dialogs = pool.map(make_data_from_VRBot, dialogues)
preprocess_dialogs.extend(new_dialogs)

end = time.time()
print('Total time : ', (end-start)/60)

# write_json(data=preprocess_dialogs[:100], path=os.path.join('data/data_for_pretrain', 'meddg_100.json'))
write_json(data=preprocess_dialogs, path=os.path.join('data/data_for_pretrain', 'meddg.json'))






