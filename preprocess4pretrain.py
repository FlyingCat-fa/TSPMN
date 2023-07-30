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

def make_data_from_ReMedi(dialog):
    dialog = dialog['information']
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

def make_data_from_MedDialog(dialog):
    new_dialog = ''
    dialog_len = 0
    old_role = ''
    for i, sent in enumerate(dialog):
        turn = {"turn": i, "sentence": sent[3:]}
        if sent[:3] == '病人：':
            turn["role"] = "patient"
        elif sent[:3] == '医生：':
            turn["role"] = "doctor"

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


if __name__ == '__main__':

    preprocess_dialogs = []
    pool = mul.Pool(64)

    # ReMeDi-large数据处理
    dir_path = 'data4pretrain/ReMeDi-large'
    file_names = read_dir_file_name(path=dir_path, suffix='json')
    dialogues = []
    for file_name in file_names:
        dialogue = read_json(path=os.path.join(dir_path, file_name))
        dialogues.extend(dialogue)
    preprocess_dialogs.extend(pool.map(make_data_from_ReMedi, dialogues))

    # meddg数据处理
    dir_path = 'data4pretrain/VRBot-sigir2021-datasets'
    dir_sub_path = ['kamed_test','kamed_valid', 'kamed_train']
    for sub_path in dir_sub_path:
        file_names = read_dir_file_name(path=os.path.join(dir_path, sub_path), suffix='json')
        dialogues = []
        for file_name in file_names:
            dialogue = read_json(path=os.path.join(dir_path, sub_path, file_name))
            dialogues.append(dialogue['dialogues'])
    preprocess_dialogs.extend(pool.map(make_data_from_VRBot, dialogues))


    # MedDialog数据处理
    dir_path = 'data4pretrain/MedDialog'
    file_names = ['test_data.json','validate_data.json', 'train_data.json']
    for file_name in file_names:
        dialogues = read_json(path=os.path.join(dir_path, file_name))
        preprocess_dialogs.extend(pool.map(make_data_from_MedDialog, dialogues))

    write_json(data=preprocess_dialogs, path=os.path.join('data4pretrain', 'all_data.json'))






