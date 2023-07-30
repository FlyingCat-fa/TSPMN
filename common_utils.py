import json, pickle
import logging
from time import gmtime, strftime
import sys
import os
import json5
import numpy as np
import struct


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

'''
read_scel
'''

# 主要两部分
# 1.全局拼音表，貌似是所有的拼音组合，字典序
#       格式为(index,len,pinyin)的列表
#       index: 两个字节的整数 代表这个拼音的索引
#       len: 两个字节的整数 拼音的字节长度
#       pinyin: 当前的拼音，每个字符两个字节，总长len
#
# 2.汉语词组表
#       格式为(same,py_table_len,py_table,{word_len,word,ext_len,ext})的一个列表
#       same: 两个字节 整数 同音词数量
#       py_table_len:  两个字节 整数
#       py_table: 整数列表，每个整数两个字节,每个整数代表一个拼音的索引
#
#       word_len:两个字节 整数 代表中文词组字节数长度
#       word: 中文词组,每个中文汉字两个字节，总长度word_len
#       ext_len: 两个字节 整数 代表扩展信息的长度，好像都是10
#       ext: 扩展信息 前两个字节是一个整数(不知道是不是词频) 后八个字节全是0
#
#      {word_len,word,ext_len,ext} 一共重复same次 同音词 相同拼音表
 
 
# 拼音表偏移，
startPy = 0x1540;
 
# 汉语词组表偏移
startChinese = 0x2628;
 
# 全局拼音表
GPy_Table = {}
 
# 解析结果
# 元组(词频,拼音,中文词组)的列表
 
 
# 原始字节码转为字符串
def byte2str(data):
    pos = 0
    str = ''
    while pos < len(data):
        c = chr(struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0])
        if c != chr(0):
            str += c
        pos += 2
    return str
 
# 获取拼音表
def getPyTable(data):
    data = data[4:]
    pos = 0
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos],data[pos + 1]]))[0]
        pos += 2
        lenPy = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pos += 2
        py = byte2str(data[pos:pos + lenPy])
 
        GPy_Table[index] = py
        pos += lenPy
 
# 获取一个词组的拼音
def getWordPy(data):
    pos = 0
    ret = ''
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        ret += GPy_Table[index]
        pos += 2
    return ret
 
# 读取中文表
def getChinese(data):
    GTable = []
    pos = 0
    while pos < len(data):
        # 同音词数量
        same = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
 
        # 拼音索引表长度
        pos += 2
        py_table_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
 
        # 拼音索引表
        pos += 2
        py = getWordPy(data[pos: pos + py_table_len])
 
        # 中文词组
        pos += py_table_len
        for i in range(same):
            # 中文词组长度
            c_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 中文词组
            pos += 2
            word = byte2str(data[pos: pos + c_len])
            # 扩展数据长度
            pos += c_len
            ext_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 词频
            pos += 2
            count = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
 
            # 保存
            GTable.append((count, py, word))
 
            # 到下个词的偏移位置
            pos += ext_len
    return GTable
 
 
def read_scel(file_name):
    # print('-' * 60)
    with open(file_name, 'rb') as f:
        data = f.read()
 
    # print("词库名：", byte2str(data[0x130:0x338])) # .encode('GB18030')
    # print("词库类型：", byte2str(data[0x338:0x540]))
    # print("描述信息：", byte2str(data[0x540:0xd40]))
    # print("词库示例：", byte2str(data[0xd40:startPy]))
 
    getPyTable(data[startPy:startChinese])
    getChinese(data[startChinese:])
    output = []
    for word in getChinese(data[startChinese:]):
        output.append(word[2])
    return output