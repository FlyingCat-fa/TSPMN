from __future__ import absolute_import, division, print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

from sys import path
path.append(os.getcwd())
path.append('/home/Medical_Understanding/MSL')

import json

from common_utils import read_txt, read_json, mkdir, write_json, read_dir_file_name

root_path = '/home/Medical_Understanding/MSL'
label_dir = 'data_origin/MSL/doc_label.dict'
data_dir = 'data_low_resource_MSL'

output_data_dir = 'data_processed/low_resource_MSL'

mkdir(os.path.join(root_path, output_data_dir))

label_map = {}

lines = read_txt(os.path.join(root_path, label_dir))
for line in lines:
    label = line.split('\t')[0]
    label_map[label] = len(label_map)

write_json(data=label_map, path=os.path.join(root_path, output_data_dir, 'label_map.json'))

file_names = read_dir_file_name(os.path.join(root_path, data_dir))
# file_names = ['train', 'dev', 'test']

for file_name in file_names:
    if file_name == 'label_map.json':
        continue
    file_path = os.path.join(root_path, data_dir, file_name)
    # dialogs = read_json(file_path)
    dialogs = read_txt(file_path)
    new_dialogs = []
    for dialog in dialogs:
        dialog = json.loads(dialog)
        utterances = []
        utterances.append("患者:" + ''.join(dialog["doc_token"]))
        label = dialog["doc_label"]
        new_dialog = {'utterances':utterances, 'label':label}
        new_dialogs.append(new_dialog)
    
    write_json(data=new_dialogs, path=os.path.join(root_path, output_data_dir, file_name))
    



