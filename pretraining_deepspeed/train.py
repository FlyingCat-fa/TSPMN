from __future__ import absolute_import, division, print_function
from distutils.log import debug
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

from sys import path
path.append(os.getcwd())
# root_path = '/home/Medical_Understanding'
# path.append(root_path)
from tqdm import tqdm
import argparse
import logging
import random
import json
import argparse
import torch
import logging
import os
import deepspeed
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from os.path import join, exists
import pickle
import sys
from transformers import (BertConfig, WEIGHTS_NAME, CONFIG_NAME, 
                            AdamW, BertTokenizer, get_linear_schedule_with_warmup)

from modeling.modeling_bert import BertForMultiEntPrediction_v2

# import pandas as pd
import numpy as np
import random
from modeling.dataset import (
    collate_fn_entity_parallel,
    SingleTaskDataset, 
    MultiTaskDataset,
    MultiTaskBatchSampler,
    DistMultiTaskBatchSampler,
    DistSingleTaskBatchSampler,
    DistTaskDataset
    )
from pretraining_deepspeed.data import EntDataset_v2, collate_fn_ent
import torch.nn.utils.rnn as rnn_utils
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from common_utils import Config, mkdir, read_numpy, read_txt
from utils import get_argument_parser


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def load_train_dataset(args, tokenizer, all_medical_words, all_medical_word_ids, special_id):
    printable = args.local_rank in [-1, 0]

    train_dataset = EntDataset_v2(
        args=args, 
        tokenizer=tokenizer, 
        all_medical_words=all_medical_words, 
        all_medical_word_ids=all_medical_word_ids, 
        special_id=special_id,
        printable=printable
    )
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=8, collate_fn=collate_fn_ent)

    return train_dataloader


def train(args, model, train_dataset):
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
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        dist_init_required=True)

    logger.info("propagate deepspeed-config settings to client settings")
    args.train_batch_size = model.train_micro_batch_size_per_gpu()
    args.gradient_accumulation_steps = model.gradient_accumulation_steps()
    args.fp16 = model.fp16_enabled()
    args.print_steps = model.steps_per_print()
    args.learning_rate = model.get_lr()[0]
    args.wall_clock_breakdown = model.wall_clock_breakdown()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    model.train()
    for epoch in train_iterator:
        tr_batch_loss = 0.0
        epoch_iterator = train_dataset
        # epoch_iterator = tqdm(train_dataset, desc="Iteration epoch: {}".format(epoch), disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if args.debug:
                if step > 5:
                    break
            if args.n_gpu == 1:
                batch = tuple(t.to(args.device) for t in batch) # multi-gpu does scattering it-self
            input_ids, cur_choice_idxes, token_type_ids, labels, input_masks, lm_indexes, lm_labels = batch
            # entity and mask
            outputs = model(input_ids, cur_choice_idxes, labels=labels, attention_mask=input_masks, token_type_ids=token_type_ids, lm_indexes=lm_indexes, lm_labels=lm_labels)
            # only entity
            # outputs = model(input_ids, cur_choice_idxes, labels=labels, attention_mask=input_masks, token_type_ids=token_type_ids, lm_indexes=lm_indexes, lm_labels=None)
            # only mask
            # outputs = model(input_ids, cur_choice_idxes, labels=None, attention_mask=input_masks, token_type_ids=token_type_ids, lm_indexes=lm_indexes, lm_labels=lm_labels)
            logits = outputs.logits
            loss = outputs.loss
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            model.backward(loss)

            tr_loss += loss.item()
            tr_batch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                lr_this_step = args.learning_rate * warmup_linear(
                    global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                model.step()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar('Loss', loss.item(), global_step)

        if args.local_rank in [-1, 0]: # Only evaluate when single GPU otherwise metrics may not average well

            # Save model checkpoint

            # output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))

            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'checkpoint-{}.pth').format(epoch))

            logger.info("Saving model checkpoint to %s", args.output_dir)
            logger.info(" current_global_step = %s, average loss = %s", global_step, tr_loss / global_step)
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    return global_step, tr_loss / global_step

def main():
    args_from_json = Config(config_file='pretraining_deepspeed/config.json')
    parser = argparse.ArgumentParser()
    for key, value in args_from_json.items():
        parser.add_argument("--{}".format(key), type=type(value), default=value)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="For distributed training: local_rank")
    parser.add_argument("--device", type=str, default="cuda",
                        help="For distributed training: local_rank")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--split_num", type=int, default=4,
                        help="将正确的医疗词汇分批预测的次数")
    parser.add_argument("--medical_word_path", type=str, default='data4pretrain/medical_words_list.txt',
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="For distributed training: local_rank")
    deepspeed.init_distributed(dist_backend='nccl')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    mkdir(args.output_dir)

    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = torch.cuda.is_available()
    # Set seed
    set_seed(args)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

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
    # [doctor, patient, ent, mask, cls_id, sep_id]
    doctor, patient, ent, mask = tokenizer.convert_tokens_to_ids(['<doctor>', '<patient>', '<ent>'] + ['[MASK]'])
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    special_id = [doctor, patient, ent, mask, cls_id, sep_id]

    all_medical_words = read_txt(path=args.medical_word_path)
    all_medical_word_ids = []
    for word in all_medical_words:
        word_id = tokenizer.encode(word, add_special_tokens=False) + [ent]
        all_medical_word_ids.append(word_id)

    train_data = load_train_dataset(args=args, 
                                    tokenizer=tokenizer, 
                                    all_medical_words=all_medical_words, 
                                    all_medical_word_ids=all_medical_word_ids, 
                                    special_id=special_id)

    model = BertForMultiEntPrediction_v2.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))



    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, train_data)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m torch.distributed.launch --master_port 29600 --nproc_per_node=4 MTL_train_Parallel.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 2960 --nproc_per_node=4 pretraining/train_Parallel.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 2960 --nproc_per_node=8 pretraining/train_Parallel.py
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 2960 --nproc_per_node=1 pretraining/train_Parallel.py
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 29600 --nproc_per_node=2 MTL/MTL_train_Parallel_v3.py
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 29600 --nproc_per_node=1 MTL_train_Parallel_v3.py