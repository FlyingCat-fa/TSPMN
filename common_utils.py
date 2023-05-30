import json, pickle
import logging
from time import gmtime, strftime
import sys
import os
import json5
import numpy as np


def mkdir(path):
    """
    创建文件夹
    """
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        # print(path + ' 创建成功')
        return True
    else:
        # print(path + ' 目录已存在')
        return False

def read_dir_file_name(path, suffix='json'):
    """
    读取文件夹下的所有文件名，并返回特定后缀的文件名
    """
    files_names = os.listdir(path)
    new_file_names = []
    for file_name in files_names:
        if file_name.split('.')[-1] == suffix:
            new_file_names.append(file_name)
    
    return new_file_names

def read_numpy(path):
    """
    读取npy文件
    """
    data = np.load(path, allow_pickle=True)
    return data

def write_numpy(path, data):
    """
    读取npy文件
    """
    np.save(file=path, arr=data)
    print('已写入数据至文件{}，数据量：{}'.format(path, data.shape[0]))

def read_json(path):
    """
    读取json文件
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """
    写入数据至json文件
    """
    with open(path, 'w', encoding='utf8') as f_write:
        json.dump(data, f_write, indent=2, ensure_ascii=False)
    
    print('已写入数据至文件{}，数据量：{}'.format(path, len(data)))


def read_txt(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        file = f.readlines()
    for line in file:
        lines.append(line.strip('\n'))
    return lines

def write_txt(data, path):
    lines = []
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line)
            f.write('\n')
    return lines

def write_pickle(data, path):
    """
    写入数据至pickle文件
    data = {"input_ids": input_ids_all, "token_type_ids": token_type_ids_all, "input_masks": input_mask_all, "labels": label_all}
    """
    with open(path, "wb") as f: 
        pickle.dump(data, f)
    
    print('已写入数据至文件{}'.format(path))


def read_pickle(data, path):
    """
    写入数据至pickle文件
    data = {"input_ids": input_ids_all, "token_type_ids": token_type_ids_all, "input_masks": input_mask_all, "labels": label_all}
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    return data

def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Logger wrapper"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = (
            log_file
            if log_file is not None
            else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


class Config(object):
    """Config load from json file
    """

    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json5.load(fin)

        self.dict = config
        if config:
            self._update(config)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)