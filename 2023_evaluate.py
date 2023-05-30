from __future__ import absolute_import, division, print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='3'
from sys import path
path.append(os.getcwd())
root_path = '/home/Medical_Understanding'
path.append(root_path)
from tqdm import tqdm
import argparse
import logging
import random
import json
import argparse
import torch
import logging
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from os.path import join, exists
from tqdm import tqdm
import pickle
import sys
from transformers import (BertConfig, WEIGHTS_NAME, CONFIG_NAME, 
                            AdamW, BertTokenizer, get_linear_schedule_with_warmup)

from modeling.modeling_bert import BertForMultiEntPrediction
from datetime import datetime

import numpy as np
import random

import torch.nn.utils.rnn as rnn_utils
from evaluate_MSL.classification_evaluate import \
    ClassificationEvaluator as cEvaluator

from common_utils import Config, mkdir, read_json, write_json


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def pruning_token(seqs, max_len=400):
    seq_max_len = 0
    max_len_seq_index = 0
    seqs_len = []
    for seq in seqs:
        seqs_len.append(len(seq))
    while sum(seqs_len) > max_len:
        max_len_seq_index = seqs_len.index((max(seqs_len)))
        seqs[max_len_seq_index].pop()
        seqs_len[max_len_seq_index] = seqs_len[max_len_seq_index] - 1

    new_seqs = []
    for seq in seqs:
        new_seqs.extend(seq)
    return new_seqs

def collate_fn_ent(batch):
    new_batch = []
    for feature in batch:
        new_batch.extend(feature)
    # a = batch[0]
    example_ids, entity_ids, labels, input_ids, token_type_ids, attention_masks, cur_choice_idxs = list(zip(*new_batch))
        
    input_ids = [torch.tensor(instance) for i, instance in enumerate(input_ids)]
    token_type_ids = [torch.tensor(instance) for i, instance in enumerate(token_type_ids)]
    attention_masks = [torch.tensor(instance) for i, instance in enumerate(attention_masks)]
    cur_choice_idxs = [torch.tensor(instance) for i, instance in enumerate(cur_choice_idxs)]
    labels = [torch.tensor(instance) for i, instance in enumerate(labels)]
    entity_ids = [torch.tensor(instance) for i, instance in enumerate(entity_ids)]
    example_ids = torch.tensor(example_ids)
    # labels = torch.tensor(labels)

    input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0) # pad_id
    input_masks = rnn_utils.pad_sequence(attention_masks, batch_first=True, padding_value=0) # pad_id
    token_type_ids = rnn_utils.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    cur_choice_idxs = rnn_utils.pad_sequence(cur_choice_idxs, batch_first=True, padding_value=0) # pad_id
    labels = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)
    entity_ids = rnn_utils.pad_sequence(entity_ids, batch_first=True, padding_value=-1)

    return example_ids, entity_ids, input_ids, cur_choice_idxs, token_type_ids, labels, input_masks


class EntDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer, 
        special_id,
        data_name = 'train',
        is_train=True,
        maxlen=512,
        max_sen_len=200,
        split_num=1,
        printable=True,
    ):
        self.tokenizer = tokenizer
        # self.doctor_id, self.patient_id, self.ent_id, self.mask_id, self.cls_id, self.sep_id = special_id
        self.special_id = special_id

        label_map_path = os.path.join(data_dir, 'label_map.json')
        self.label_map = read_json(path=label_map_path)
        self.all_entities = list(self.label_map.keys())

        data_path = os.path.join(data_dir, '{}.json'.format(data_name))
        self._data = read_json(path=data_path)
        self.maxlen = maxlen
        self.split_num = split_num
        self.max_sen_len = max_sen_len

    def convert_example_to_feature(
        self,
        example_id,
        example,
        entities,
    ):
        doctor_id, patient_id, ent_id, mask_id, cls_id, sep_id = self.special_id
        utterances = example["utterances"]
        # label = example["label"]

        hypothesis_ids = []
        for entity in entities:
            entity_id = self.tokenizer.encode(entity, add_special_tokens=False) + [ent_id]
            hypothesis_ids.extend(entity_id)
        hypothesis_ids = [cls_id] + hypothesis_ids + [sep_id]

        max_len = self.maxlen - len(hypothesis_ids) - 1
        if max_len > self.max_sen_len:
            max_sen_len = self.max_sen_len
        else:
            max_sen_len = max_len

        premise_id_list = []
        for sentence in utterances:
            if len(sentence) > 0:
                sentence_id = self.tokenizer.encode(sentence[3:], add_special_tokens=False) # 去掉说话人 医生： 病人：的前缀
                if len(sentence_id) > max_sen_len:
                    sentence_id = sentence_id[:max_sen_len]
                if sentence[:3] == "患者:":
                    premise_id_list.append([patient_id] + sentence_id)
                elif sentence[:3] == "医生:":
                    premise_id_list.append([doctor_id] + sentence_id)
                else:
                    print("文本中的说话人有错误！！！")
        # 避免对话长度过长
        premise_ids = pruning_token(premise_id_list, max_len=max_len)
        premise_ids = premise_ids + [sep_id]

        input_ids = hypothesis_ids + premise_ids
        token_type_ids = [0] * len(hypothesis_ids) + [1] * len(premise_ids)
        attention_mask = [1] * len(input_ids)

        cur_choice_idx = []
        for j, idx in enumerate(input_ids):
            if idx == ent_id:
                cur_choice_idx.append(j)
                # if len(cur_choice_idx) > 20:
                #     print('  ')
        entity_id = []
        label = []
        for entity in entities:
            entity_id.append(self.label_map[entity])
            if entity in example["label"]:
                label.append(0)
            else:
                label.append(1)
        # tokenized_example = {
        #         "example_id": example_id,
        #         "entity_id": entity_id,
        #         "label": label,
        #         "token_id": input_ids,
        #         "type_id": token_type_ids,
        #         "attention_mask": attention_mask,
        #         "cur_choice_idx": cur_choice_idx
        #     }
        feature = (
                example_id,
                entity_id,
                label,
                input_ids,
                token_type_ids,
                attention_mask,
                cur_choice_idx
        )
        return feature

    def list_split(self, lst):
            n = self.split_num
            if len(lst) % n != 0:
                m = (len(lst) // n) + 1
            else:
                m = len(lst) // n
            sp_lst = []
            for i in range(n):
                sp_lst.append(lst[i*m:(i+1)*m])
            return sp_lst

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        example = self._data[idx]
        example_id = idx
        features = []
        entities_list = self.list_split(self.all_entities)
        for entities in entities_list:
            feature = self.convert_example_to_feature(example_id=example_id, example=example, entities=entities)
            features.append(feature)
        return features


def infer(args, model, eval_dataset, logger):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    results = {}

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    # Note that DistributedSampler samples randomly

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size_eval)

    # epoch_iterator = tqdm(eval_dataset, desc="Evaluating")
    epoch_iterator = eval_dataset
    preds = None
    out_label_ids = None
    out_example_ids = None
    out_entity_ids = None
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        # if args.debug:
        #     if step > 5:
        #         break
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            example_ids, entity_ids, input_ids, cur_choice_idxes, token_type_ids, labels, input_masks = batch
            outputs = model(input_ids, cur_choice_idxes, labels=labels, attention_mask=input_masks, token_type_ids=token_type_ids)
            logits = outputs.logits

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            out_example_ids = example_ids.detach().cpu().numpy()
            out_entity_ids = entity_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            out_example_ids = np.append(out_example_ids, example_ids.detach().cpu().numpy(), axis=0)
            out_entity_ids = np.append(out_entity_ids, entity_ids.detach().cpu().numpy(), axis=0)

    return preds, out_label_ids, out_example_ids, out_entity_ids


def evaluate(args, model, entity_eval_dataset, logger):
    # entity infer
    # entity_eval_dataset = load_eval_dataset()
    predict_probs = []
    predicts = []
    standard_labels = []
    entity_preds, out_label_ids, out_example_ids, out_entity_ids = infer(args, model, eval_dataset=entity_eval_dataset, logger=logger)
    out_example_ids = (np.expand_dims(out_example_ids, 1).repeat(entity_preds.shape[1], axis=1)).reshape(-1)
    entity_preds = entity_preds.reshape(-1, 2)
    entity_preds = np.argmax(entity_preds, axis=-1)
    out_entity_ids = out_entity_ids.reshape(-1)
    out_label_ids = out_label_ids.reshape(-1)
    dataset_len = args.dataset_len
    label_map = args.label_map
    id2label = {}
    for key, value in label_map.items():
        id2label[value] = key
    for i in range(dataset_len):
        predicts.append([])
        standard_labels.append([])
    for i in range(entity_preds.shape[0]):
        entity_pred = entity_preds[i]
        entity_id = out_entity_ids[i]
        example_id = out_example_ids[i]
        label_id = out_label_ids[i]

        if entity_pred == 0 and entity_id >= 0:
            predicts[example_id].append(id2label[entity_id])
        if label_id == 0 and entity_id >= 0:
            standard_labels[example_id].append(entity_id)


    evaluator = cEvaluator('eval_dir')
    (_, precision_list, recall_list, fscore_list, right_list,
        predict_list, standard_list, turn_accuracy) = \
            evaluator.evaluate(
                predicts, standard_label_ids=standard_labels, label_map=label_map,
                threshold=0.5, top_k=2,
                is_flat=True, is_multi=True, is_prob=False)
    
    
    logger.warn(
        "Performance is precision: %f, "
        "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Turn accuracy: %f" % (
            precision_list[0][cEvaluator.MICRO_AVERAGE],
            recall_list[0][cEvaluator.MICRO_AVERAGE],
            fscore_list[0][cEvaluator.MICRO_AVERAGE],
            fscore_list[0][cEvaluator.MACRO_AVERAGE],
            right_list[0][cEvaluator.MICRO_AVERAGE],
            predict_list[0][cEvaluator.MICRO_AVERAGE],
            standard_list[0][cEvaluator.MICRO_AVERAGE], turn_accuracy))
    evaluator.save()
    precision = precision_list[0][cEvaluator.MICRO_AVERAGE]
    recall = recall_list[0][cEvaluator.MICRO_AVERAGE]
    fscore = fscore_list[0][cEvaluator.MICRO_AVERAGE]
    macro_fscore = fscore_list[0][cEvaluator.MACRO_AVERAGE]
    Turn_accuracy = turn_accuracy

    return precision, recall, fscore, macro_fscore, Turn_accuracy

def main():
    start = datetime.now()
    root_path = '/home/Medical_Understanding'
    args_from_json = Config(config_file=os.path.join(root_path, 'config_MSL.json'))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--entity_label_num", type=int, default=2)
    # parser.add_argument("--state_label_num", type=int, default=3)
    args = parser.parse_args()
    for key ,value in vars(args).items():
        args_from_json.add(key, value)
    # args_from_json.add('local_rank', args.local_rank)
    args = args_from_json
    if args.pretrained:
        args.output_dir += '_pretrained'
        args.logdir += '_pretrained'
    output_dir = args.output_dir
    mkdir(output_dir)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_seed(args)

    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # special_tokens = ['<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_ent>']
    special_tokens = ['<doctor>', '<patient>', '<ent>']
    vocab_size = len(tokenizer) + len(special_tokens)
    # Padding for divisibility by 8
    if vocab_size % 8 != 0:
        pad_len = 8 - (vocab_size % 8)
        pad_token = '<pad_{}>'
        for i in range(pad_len):
            special_tokens.append(pad_token.format(i))
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id
    doctor, patient, ent, mask = tokenizer.convert_tokens_to_ids(['<doctor>', '<patient>', '<ent>'] + ['[MASK]'])
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    special_id = [doctor, patient, ent, mask, cls_id, sep_id]
    if args.local_rank in [-1, 0]:
        printable=True
    else:
        printable = False

    # model = BertForMTLSequenceClassification.from_pretrained(args.model_name_or_path)
    model = BertForMultiEntPrediction.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    epoch = 15
    model_path = os.path.join(root_path, args.output_dir, 'checkpoint-{}.pth'.format(epoch))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(args.device)
    
    test_dataset = EntDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer, 
            special_id=special_id,
            data_name='test',
            split_num=args.split_num,
            printable=printable)
    args.label_map = test_dataset.label_map
    args.dataset_len = len(test_dataset)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size_eval, num_workers=4, collate_fn=collate_fn_ent)


    precision, recall, fscore, macro_fscore, Turn_accuracy = evaluate(args, model, test_dataloader, logger)
    end = datetime.now()
    print("运行时间：", end-start)

    if os.path.isfile(os.path.join(args.output_dir, 'test_results.json')):
        test_results = read_json(os.path.join(args.output_dir, 'test_results.json'))
    else:
        test_results = []
    test_result = {"epoch": epoch, 'precision': precision, 'recall': recall, 'fscore': fscore, 'macro_fscore': macro_fscore, 'Turn_accuracy': Turn_accuracy}
    test_results.append(test_result)
    write_json(data=test_results, path=os.path.join(args.output_dir, 'test_results.json'))


if __name__ == "__main__":
    main()





