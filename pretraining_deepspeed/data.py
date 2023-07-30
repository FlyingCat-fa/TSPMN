import os
from sys import path
path.append(os.getcwd())
path.append('/home/Medical_Understanding')
import random
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
random.seed(42)
import numpy as np
from common_utils import read_json


def collate_fn_ent(batch):
    new_batch = []
    for example in batch:
        new_batch.extend(example)
    # a = batch[0]
    ent_labels, token_ids, token_type_ids, attention_mask, cur_choice_idx, lm_indexes, lm_labels = list(zip(*new_batch))
        
    token_ids = [torch.tensor(instance) for i, instance in enumerate(token_ids)]
    token_type_ids = [torch.tensor(instance) for i, instance in enumerate(token_type_ids)]
    attention_mask = [torch.tensor(instance) for i, instance in enumerate(attention_mask)]
    lm_indexes = [torch.tensor(instance).long() for i, instance in enumerate(lm_indexes)]
    lm_labels = [torch.tensor(instance).long() for i, instance in enumerate(lm_labels)]
    cur_choice_idx = torch.tensor(cur_choice_idx)
    ent_labels = torch.tensor(ent_labels)

    input_ids = rnn_utils.pad_sequence(token_ids, batch_first=True, padding_value=0) # pad_id
    input_masks = rnn_utils.pad_sequence(attention_mask, batch_first=True, padding_value=0) # pad_id
    token_type_ids = rnn_utils.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    lm_indexes = rnn_utils.pad_sequence(lm_indexes, batch_first=True, padding_value=0)
    lm_labels = rnn_utils.pad_sequence(lm_labels, batch_first=True, padding_value=-100)

    return input_ids, cur_choice_idx, token_type_ids, ent_labels, input_masks, lm_indexes, lm_labels


class EntDataset_v2(Dataset):
    """
    读取原始对话，数据处理及tokenize在本类中实现，同时实现实体匹配和mask预测
    """
    def __init__(
        self,
        args, 
        tokenizer, 
        all_medical_words, 
        all_medical_word_ids, 
        special_id,
        maxlen=512,
        printable=True,
    ):
        self.tokenizer = tokenizer
        self.all_medical_words = all_medical_words
        self.all_medical_word_ids = all_medical_word_ids
        self.args = args
        self.doctor_id, self.patient_id, self.ent_id, self.mask_id, self.cls_id, self.sep_id = special_id
        self.special_id = special_id
        data, _ = self.load(
            args.data_dir,
            maxlen,
            printable=printable,
        )
        if args.debug:
            data = data[:10000]
        self._data = data
        self.maxlen = maxlen

    def list_split(self, lst):
        n = self.args.split_num
        if len(lst) % n != 0:
            m = (len(lst) // n) + 1
        else:
            m = len(lst) // n
        sp_lst = []
        for i in range(n):
            sp_lst.append(lst[i*m:(i+1)*m])
        return sp_lst

    def word_mask(self, tokens, masked_words):
        lm_index_list = []
        lm_label_list = []
        mask = self.mask_id
        for word in masked_words:
            word_len = len(word) - 1 # 原word增加了<ent>特殊符
            start = 0
            flag = True
            while flag:
                try:
                    index = tokens.index(word[0], start)
                    if tokens[index:index+word_len] == word[:-1]:
                        for lm_index in range(index, index+word_len):
                            lm_index_list.append(lm_index)
                            lm_label_list.append(tokens[lm_index])
                        # 80% of the time, replace with [MASK]
                        if random.random() < 0.8:
                            tokens[index:index+word_len] = [mask] * word_len
                        # 10% of the time, keep original
                        elif random.random() < 0.5:
                            pass
                        # 10% of the time, replace with random word
                        else:
                            masked_token = random.sample(self.vocab_ids, word_len)
                            tokens[index:index+word_len] = masked_token
                    start = index + 1
                except:
                    flag = False

        return tokens, lm_index_list, lm_label_list

    def create_instances_from_dialogue(self, dialogue):
        """
        从对话文本中构造训练example
        """
        doctor, patient, ent, mask, cls_id, sep_id = self.special_id
        medical_words = dialogue['medical_words']
        dialogue = dialogue['text'].split('\n')

        tokens_1 = []
        tokens_2 = []
        for sentence in dialogue:
            if sentence.startswith('<patient>'):
                tokens_2.append(patient)
                sentence = sentence[len('<patient>'):]
            elif sentence.startswith('<doctor>'):
                tokens_2.append(doctor)
                sentence = sentence[len('<doctor>'):]
            tokens_2.extend(self.tokenizer.encode(sentence, add_special_tokens=False))
        tokens_2.append(sep_id)

        pos_word_num = len(medical_words)

        max_predictions = self.args.max_predictions_per_seq * self.args.split_num
        neg_medical_word_ids = random.sample(self.all_medical_word_ids, max_predictions + pos_word_num)

        medical_word_ids = []
        for word in medical_words:
            word_index = self.all_medical_words.index(word)
            word_ids = self.all_medical_word_ids[word_index]
            medical_word_ids.append(word_ids)
            # 避免随机采样到正例
            try:
                neg_word_index = neg_medical_word_ids.index(word_ids)
                neg_medical_word_ids.pop(neg_word_index)
            except:
                continue

        neg_medical_word_ids = neg_medical_word_ids[:max_predictions]

        if pos_word_num > max_predictions:
            medical_word_ids = medical_word_ids[:max_predictions]
        random.shuffle(medical_word_ids)
        pos_medical_word_ids = self.list_split(medical_word_ids)
        neg_medical_word_ids = self.list_split(neg_medical_word_ids)

        max_ent_len = self.args.max_seq_len - len(tokens_2) - 2

        tokenized_examples = []
        for i, medical_word_per_seq in enumerate(neg_medical_word_ids):
            tokens_2_i, lm_indexes_tokens_2, lm_labels= self.word_mask(tokens=tokens_2 + [], masked_words=pos_medical_word_ids[i])
            pos_num = len(pos_medical_word_ids[i])
            ent_labels = [1] * len(medical_word_per_seq)
            ent_labels[:pos_num] = [0] * pos_num
            medical_word_ids = medical_word_per_seq
            medical_word_ids[:pos_num] = pos_medical_word_ids[i]

            seq_len = sum([len(word) for word in medical_word_ids])
            while seq_len > max_ent_len:
                seq_len = seq_len - len(medical_word_ids[-1])
                medical_word_ids.pop()
                ent_labels.pop()
            medical_word_ids_with_label = list(zip(medical_word_ids, ent_labels))
            random.shuffle(medical_word_ids_with_label)
            medical_word_ids, ent_labels = zip(*medical_word_ids_with_label)
            medical_word_ids = list(medical_word_ids)
            ent_labels = list(ent_labels)
            cur_choice_idx = []
            word_seq = [cls_id]
            for word in medical_word_ids:
                word_seq.extend(word)
                cur_choice_idx.append(len(word_seq)-1)

            
            tokens_1 = word_seq + [sep_id]
            tokens_1_len = len(tokens_1)

            token_ids = tokens_1 + tokens_2_i
            token_type_ids = [0] * len(tokens_1) + [1] * len(tokens_2_i)
            attention_mask = [1] * len(token_ids)

            # lm_labels = [-100] * len(token_ids)
            lm_indexes = []
            for i, lm_index in enumerate(lm_indexes_tokens_2):
                lm_index += tokens_1_len
                lm_indexes.append(int(lm_index))
                # lm_labels[lm_index] = lm_label_list[i]



            ent_num = len(ent_labels)

            if ent_num < self.args.max_predictions_per_seq:
                ent_labels += [-100] * (self.args.max_predictions_per_seq - ent_num)
                cur_choice_idx += [0] * (self.args.max_predictions_per_seq - ent_num)
        

            # tokenized_example = {
            #         "label": ent_labels,
            #         "token_id": token_ids,
            #         "type_id": token_type_ids,
            #         "attention_mask": attention_mask,
            #         "cur_choice_idx": cur_choice_idx
            #     }
            tokenized_example = (
                    ent_labels,
                    token_ids,
                    token_type_ids,
                    attention_mask,
                    cur_choice_idx,
                    lm_indexes,
                    lm_labels,
                )
            tokenized_examples.append(tokenized_example)
#################################################################
        return tokenized_examples

    @staticmethod
    def load(
        path,
        debug=True,
        maxlen=512,
        printable=True,
        inputs=['token_ids', 'token_type_ids', 'attention_mask', 'cur_choice_idx', 'ent_labels', 'token_len']
    ):
        dialogues = read_json(path)

        if printable:
            print("Loaded {} samples".format(len(dialogues)))
        return dialogues, None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        dialog = self._data[idx]
        tokenized_examples = self.create_instances_from_dialogue(dialog)
        return tokenized_examples


class EntDataset(Dataset):
    """
    读取原始对话，数据处理及tokenize在本类中实现， 仅实现了实体匹配
    """
    def __init__(
        self,
        args, 
        tokenizer, 
        all_medical_words, 
        all_medical_word_ids, 
        special_id,
        maxlen=512,
        printable=True,
    ):
        self.tokenizer = tokenizer
        self.all_medical_words = all_medical_words
        self.all_medical_word_ids = all_medical_word_ids
        self.args = args
        self.doctor_id, self.patient_id, self.ent_id, self.mask_id, self.cls_id, self.sep_id = special_id
        self.special_id = special_id
        data, _ = self.load(
            args.data_dir,
            maxlen,
            printable=printable,
        )
        if args.debug:
            data = data[:10000]
        self._data = data
        self.maxlen = maxlen

    def list_split(self, lst):
        n = self.args.split_num
        if len(lst) % n != 0:
            m = (len(lst) // n) + 1
        else:
            m = len(lst) // n
        sp_lst = []
        for i in range(n):
            sp_lst.append(lst[i*m:(i+1)*m])
        return sp_lst

    def word_mask(self, tokens, masked_words):
        mask = self.mask_id
        for word in masked_words:
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                word_len = len(word) - 1 # 原word增加了<ent>特殊符
                start = 0
                flag = True
                while flag:
                    try:
                        index = tokens.index(word[0], start)
                        if tokens[index:index+word_len] == word[:-1]:
                            tokens[index:index+word_len] = [mask] * word_len
                        start = index + 1
                    except:
                        flag = False
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    pass
                # 10% of the time, replace with random word
                else:
                    word_len = len(word) - 1
                    start = 0
                    flag = True
                    while flag:
                        try:
                            index = tokens.index(word[0], start)
                            if tokens[index:index+word_len] == word[:-1]:
                                masked_token = random.sample(self.vocab_ids, word_len)
                                tokens[index:index+word_len] = masked_token
                            start = index + 1
                        except:
                            flag = False

        return tokens

    def create_instances_from_dialogue(self, dialogue):
        """
        从对话文本中构造训练example
        """
        doctor, patient, ent, mask, cls_id, sep_id = self.special_id
        medical_words = dialogue['medical_words']
        dialogue = dialogue['text'].split('\n')

        tokens_1 = []
        tokens_2 = []
        for sentence in dialogue:
            if sentence.startswith('<patient>'):
                tokens_2.append(patient)
                sentence = sentence[len('<patient>'):]
            elif sentence.startswith('<doctor>'):
                tokens_2.append(doctor)
                sentence = sentence[len('<doctor>'):]
            tokens_2.extend(self.tokenizer.encode(sentence, add_special_tokens=False))
        tokens_2.append(sep_id)

        pos_word_num = len(medical_words)

        max_predictions = self.args.max_predictions_per_seq * self.args.split_num
        neg_medical_word_ids = random.sample(self.all_medical_word_ids, max_predictions + pos_word_num)

        medical_word_ids = []
        for word in medical_words:
            word_index = self.all_medical_words.index(word)
            word_ids = self.all_medical_word_ids[word_index]
            medical_word_ids.append(word_ids)
            # 避免随机采样到正例
            try:
                neg_word_index = neg_medical_word_ids.index(word_ids)
                neg_medical_word_ids.pop(neg_word_index)
            except:
                continue

        neg_medical_word_ids = neg_medical_word_ids[:max_predictions]

        if pos_word_num > max_predictions:
            medical_word_ids = medical_word_ids[:max_predictions]
        random.shuffle(medical_word_ids)
        pos_medical_word_ids = self.list_split(medical_word_ids)
        neg_medical_word_ids = self.list_split(neg_medical_word_ids)

        max_ent_len = self.args.max_seq_len - len(tokens_2) - 2

        tokenized_examples = []
        for i, medical_word_per_seq in enumerate(neg_medical_word_ids):
            tokens_2_i = self.word_mask(tokens=tokens_2 + [], masked_words=pos_medical_word_ids[i])
            pos_num = len(pos_medical_word_ids[i])
            ent_labels = [1] * len(medical_word_per_seq)
            ent_labels[:pos_num] = [0] * pos_num
            medical_word_ids = medical_word_per_seq
            medical_word_ids[:pos_num] = pos_medical_word_ids[i]

            seq_len = sum([len(word) for word in medical_word_ids])
            while seq_len > max_ent_len:
                seq_len = seq_len - len(medical_word_ids[-1])
                medical_word_ids.pop()
                ent_labels.pop()
            medical_word_ids_with_label = list(zip(medical_word_ids, ent_labels))
            random.shuffle(medical_word_ids_with_label)
            medical_word_ids, ent_labels = zip(*medical_word_ids_with_label)
            medical_word_ids = list(medical_word_ids)
            ent_labels = list(ent_labels)
            cur_choice_idx = []
            word_seq = [cls_id]
            #### MLM
            # for word in medical_word_ids:
            #     word_seq.extend(word)
            #     cur_choice_idx.append(len(word_seq)-1)

            
            # tokens_1 = word_seq + [sep_id]

            # token_ids = tokens_1 + tokens_2_i
            token_ids = [cls_id] + tokens_2_i
            token_type_ids = [0] * len(tokens_1) + [1] * len(tokens_2_i)
            attention_mask = [1] * len(token_ids)

            ent_num = len(ent_labels)

            if ent_num < self.args.max_predictions_per_seq:
                ent_labels += [-100] * (self.args.max_predictions_per_seq - ent_num)
                cur_choice_idx += [0] * (self.args.max_predictions_per_seq - ent_num)
        

            # tokenized_example = {
            #         "label": ent_labels,
            #         "token_id": token_ids,
            #         "type_id": token_type_ids,
            #         "attention_mask": attention_mask,
            #         "cur_choice_idx": cur_choice_idx
            #     }
            tokenized_example = (
                    ent_labels,
                    token_ids,
                    token_type_ids,
                    attention_mask,
                    cur_choice_idx
                )
            tokenized_examples.append(tokenized_example)
#################################################################
        return tokenized_examples

    @staticmethod
    def load(
        path,
        debug=True,
        maxlen=512,
        printable=True,
        inputs=['token_ids', 'token_type_ids', 'attention_mask', 'cur_choice_idx', 'ent_labels', 'token_len']
    ):
        dialogues = read_json(path)

        if printable:
            print("Loaded {} samples".format(len(dialogues)))
        return dialogues, None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        dialog = self._data[idx]
        tokenized_examples = self.create_instances_from_dialogue(dialog)
        return tokenized_examples