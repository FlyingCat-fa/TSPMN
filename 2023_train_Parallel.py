from __future__ import absolute_import, division, print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

from sys import path
path.append(os.getcwd())
root_path = '/home/Medical_Understanding'
path.append(root_path)
import random
import json
import argparse
import torch
import logging
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (BertConfig, WEIGHTS_NAME, CONFIG_NAME, 
                            AdamW, BertTokenizer, get_linear_schedule_with_warmup)

from modeling.modeling_bert import BertForMultiEntPrediction

# import pandas as pd
import numpy as np
import random
import torch.nn.utils.rnn as rnn_utils
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from common_utils import Config, mkdir, read_json, write_json
# from Entity_parallel.evaluate import evaluate
from evaluate import evaluate


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_model_state(state_path, model, device):
    base_weights = torch.load(state_path,map_location=device)
    print('Loading base network...')
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k in base_weights:
            state_dict[k] = base_weights[k]
    model.load_state_dict(state_dict)
    return model
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


def train(args, model, tokenizer, train_dataset, eval_dataset=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=args.logdir)

    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2, collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=args.learning_rate, 
                        betas=(args.adam_beta_1, args.adam_beta_2), 
                        eps=args.adam_epsilon, 
                        weight_decay=args.weight_decay,)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=t_total // 30, \
        # num_warmup_steps=args.warmup_steps, \
        num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_epoch = -1
    best_performance = 0
    best_result = None

    for epoch in train_iterator:
        tr_batch_loss = 0.0
        # epoch_iterator = train_dataset
        epoch_iterator = tqdm(train_dataset, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if args.debug:
                if step > 5:
                    break
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            example_ids, entity_ids, input_ids, cur_choice_idxes, token_type_ids, labels, input_masks = batch
            outputs = model(input_ids, cur_choice_idxes, labels=labels, attention_mask=input_masks, token_type_ids=token_type_ids)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_batch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar('Loss', loss.item(), global_step)

        if args.local_rank in [-1, 0]: # Only evaluate when single GPU otherwise metrics may not average well

            # Save model checkpoint

            # output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            output_dir = args.output_dir
            mkdir(output_dir)

            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'checkpoint-{}.pth').format(epoch))

            logger.info("Saving model checkpoint to %s", output_dir)
            logger.info(" current_global_step = %s, average loss = %s", global_step, tr_loss / global_step)

            # evaluate
            if args.evaluate_during_training:

                precision, recall, fscore, macro_fscore, Turn_accuracy = evaluate(args, model, eval_dataset, logger)

                if os.path.isfile(os.path.join(args.output_dir, 'eval_results.json')):
                    eval_results = read_json(os.path.join(args.output_dir, 'eval_results.json'))
                else:
                    eval_results = []
                eval_result = {"epoch": epoch, 'precision': precision, 'recall': recall, 'fscore': fscore, 'macro_fscore': macro_fscore, 'Turn_accuracy': Turn_accuracy}
                if fscore > best_performance:
                    best_performance = fscore
                    best_epoch = epoch
                    best_result = eval_result
                eval_results.append(eval_result)
                write_json(data=eval_results, path=os.path.join(args.output_dir, 'eval_results.json'))
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        if args.evaluate_during_training:
            eval_results = read_json(os.path.join(args.output_dir, 'eval_results.json'))
            best_result['best_epoch'] = best_epoch
            eval_results.append(best_result)
            write_json(data=eval_results, path=os.path.join(args.output_dir, 'eval_results.json'))
    
    return global_step, tr_loss / global_step



def main():
    root_path = '/home/Medical_Understanding'
    args_from_json = Config(config_file=os.path.join(root_path, 'config_MSL.json'))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    parser.add_argument("--entity_label_num", type=int, default=2)
    # parser.add_argument("--state_label_num", type=int, default=3)
    args = parser.parse_args()
    for key ,value in vars(args).items():
        args_from_json.add(key, value)
    args = args_from_json
    if args.pretrained:
            args.output_dir += '_pretrained'
            args.logdir += '_pretrained'
    args.cuda = torch.cuda.is_available()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

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
    train_dataset = EntDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer, 
        special_id=special_id,
        data_name='train',
        split_num=args.split_num,
        printable=printable)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn_ent)

    if args.evaluate_during_training:
        dev_dataset = EntDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer, 
            special_id=special_id,
            data_name='dev',
            split_num=args.split_num,
            printable=printable)
        args.label_map = dev_dataset.label_map
        args.dataset_len = len(dev_dataset)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size_eval, num_workers=4, collate_fn=collate_fn_ent)
    else:
        dev_dataloader = None

    model = BertForMultiEntPrediction.from_pretrained(args.model_name_or_path)
    
    model.resize_token_embeddings(len(tokenizer))
    if args.pretrained:
        # epoch = 5
        # epoch = 9
        # epoch = 19
        # epoch = 4
        # model_path='/home/Medical_Understanding/MSL/model_files/pretraining/pretrain_deepspeed/checkpoint-{}.pth'
        # model_path = model_path.format(epoch)
        # model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_entity_and_mask/checkpoint-4.pth' # v0
        # model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_entity_and_mask_old/checkpoint-4.pth' # old v1
        # model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_entity_and_mask_old_2/checkpoint-4.pth' # old 2 v2
        model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_entity_and_mask_2023/checkpoint-4.pth' # 2023
        # model_path = '/home/Medical_Understanding/model_files/pretrain_MSL/checkpoint-19.pth'
        # model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_only_entity/checkpoint-4.pth'
        # model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_only_mask/checkpoint-9.pth'
        # model.load_state_dict(torch.load(model_path, map_location=args.device))
        # model_path = '/home/Medical_Understanding/model_files/pretrain_deepspeed_only_mask_mlm/checkpoint-4.pth'
        # model_path = '/home/SPIDER/data/medical_data/medical_ngram/pytorch_model_epoch_4.bin'
        model = load_model_state(state_path=model_path, model=model, device=device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataloader, eval_dataset=dev_dataloader)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()

# nohup CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 19600 --nproc_per_node=1 train_Parallel.py > run.log 2>&1 &