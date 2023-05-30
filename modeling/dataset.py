from ftplib import all_errors
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
import torch
import json
import random
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import math
import torch.nn.utils.rnn as rnn_utils


def collate_fn(batch):
    task_ids = torch.tensor([instance["task"]["task_id"] for instance in batch])
    dialog_ids = torch.tensor([instance["sample"]["dialog_id"] for instance in batch])
    window_ids = torch.tensor([instance["sample"]["window_id"] for instance in batch])
    entity_ids = torch.tensor([instance["sample"]["entity_id"] for instance in batch])
    input_ids = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["token_id"]) for instance in batch],
        batch_first=True, padding_value=0) # pad_id
    input_masks = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["attention_mask"]) for instance in batch],
        batch_first=True, padding_value=0) # pad_id
    token_type_ids = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["type_id"]) for instance in batch],
        batch_first=True, padding_value=0)
    labels = torch.tensor([instance["sample"]["label"] for instance in batch])

    return input_ids, token_type_ids, labels, input_masks, task_ids, dialog_ids, window_ids, entity_ids


def collate_fn_entity_parallel(batch):
    task_ids = torch.tensor([instance["task"]["task_id"] for instance in batch])
    dialog_ids = torch.tensor([instance["sample"]["dialog_id"] for instance in batch])
    window_ids = torch.tensor([instance["sample"]["window_id"] for instance in batch])
    entity_ids = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["entity_id"]) for instance in batch],
        batch_first=True, padding_value=-100)
    input_ids = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["token_id"]) for instance in batch],
        batch_first=True, padding_value=0) # pad_id
    input_masks = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["attention_mask"]) for instance in batch],
        batch_first=True, padding_value=0) # pad_id
    token_type_ids = rnn_utils.pad_sequence(
        [torch.tensor(instance["sample"]["type_id"]) for instance in batch],
        batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(
            [torch.tensor(instance["sample"]["label"]) for instance in batch],
            batch_first=True, padding_value=-100)
    cur_choice_idxes = rnn_utils.pad_sequence(
            [torch.tensor(instance["sample"]["cur_choice_idx"]) for instance in batch],
            batch_first=True, padding_value=0)

    return input_ids, cur_choice_idxes, token_type_ids, labels, input_masks, task_ids, dialog_ids, window_ids, entity_ids

class MyDataset(Dataset):

    def __init__(self, input_ids, token_type_ids, input_masks, labels, max_len):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.input_masks = input_masks
        self.lm_labels = labels
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = self.token_type_ids[index]
        token_type_ids = token_type_ids[:self.max_len]
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        lm_labels = [self.lm_labels[index]]
        lm_labels = torch.tensor(lm_labels, dtype=torch.long)

        input_masks = self.input_masks[index]
        input_masks = input_masks[:self.max_len]
        input_masks = torch.tensor(input_masks, dtype=torch.long)
        instance = {"input_ids": input_ids, "token_type_ids": token_type_ids, "lm_labels": lm_labels, "input_masks": input_masks}
        return instance

    def __len__(self):
        return len(self.input_ids)

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, task_id=2, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        self._task_id = task_id

        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)

        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        
    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return {
            "task": {"task_id": self._task_id},
            "sample": {
                "input_id": self.input_ids[item].astype(np.int64),
                "input_mask": self.input_masks[item].astype(np.int64),
                "segment_id": self.segment_ids[item].astype(np.int64),
                "lm_label_id": self.lm_label_ids[item].astype(np.int64),
                "is_next": self.is_nexts[item].astype(np.int64)
            }
        }
        # return (torch.tensor(self.input_ids[item].astype(np.int64)),
        #         torch.tensor(self.input_masks[item].astype(np.int64)),
        #         torch.tensor(self.segment_ids[item].astype(np.int64)),
        #         torch.tensor(self.lm_label_ids[item].astype(np.int64)),
        #         torch.tensor(self.is_nexts[item].astype(np.int64)))


class SingleTaskDataset(Dataset):
    def __init__(
        self,
        path,
        data=None,
        is_train=True,
        maxlen=512,
        task_id=0,
        printable=True,
    ):
        if not data:

            data, _ = self.load(
                path,
                is_train,
                maxlen,
                printable=printable,
            )
        self._data = data
        self._task_id = task_id
        self.maxlen = maxlen

    def get_task_id(self):
        return self._task_id

    @staticmethod
    def load(
        path,
        is_train=True,
        maxlen=512,
        printable=True,
    ):

        with open(path, "r", encoding="utf-8") as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                cnt += 1
                data.append(sample)
            if printable:
                print("Loaded {} samples out of {}".format(len(data), cnt))
        return data, None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        return {
            "task": {"task_id": self._task_id},
            "sample": self._data[idx],
        }


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, (
                "Duplicate task_id %s" % task_id
            )
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]

class MultiTaskDataset_Balanced(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, (
                "Duplicate task_id %s" % task_id
            )
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        max_len = 0
        for dataset in self._datasets:
            if max_len < len(dataset):
                max_len = len(dataset)
        all_len = 0
        for dataset in self._datasets:
            epoch = math.floor(all_len/len(dataset))
            all_len += epoch * len(dataset)
            
        return all_len

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]


class DistMultiTaskBatchSampler(Sampler):
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt,
        extra_task_ratio,
        rank=0,
        world_size=1,
        drop_last=False,
    ):
        self.rank = rank
        self.world_size = world_size
        self._datasets = datasets
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        self.drop_last = drop_last
        train_data_list = []
        self.dataset_max_len = 0
        for dataset in datasets:
            if self.dataset_max_len < len(dataset):
                self.dataset_max_len = len(dataset)
        for dataset in datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size, dataset_max_len=self.dataset_max_len)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size, dataset_max_len):
        epoch = math.floor(dataset_max_len / dataset_len)
        all_index_batches = []
        for _ in range(epoch):
            index_samples = list(range(dataset_len))
            random.shuffle(index_samples)
            index_batches = [
                index_samples[i:min(i + batch_size, dataset_len)]
                for i in range(0, dataset_len, batch_size)
            ]
            all_index_batches.extend(index_batches)
        random.shuffle(all_index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            batch = [(task_id, sample_id) for sample_id in batch]
            if len(batch) % self.world_size != 0:
                if self.drop_last:
                    break
                else:
                    batch.extend(
                        [
                            batch[0]
                            for _ in range(
                                self.world_size - len(batch) % self.world_size
                            )
                        ]
                    )
            chunk_size = len(batch) // self.world_size
            yield batch[self.rank * chunk_size : (self.rank + 1) * chunk_size]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices

class DistMultiTaskBatchSampler_Balanced(Sampler):
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt,
        extra_task_ratio,
        rank=0,
        world_size=1,
        drop_last=False,
    ):
        self.rank = rank
        self.world_size = world_size
        self._datasets = datasets
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        self.drop_last = drop_last
        train_data_list = []
        self.dataset_max_len = 0
        for dataset in datasets:
            if self.dataset_max_len < len(dataset):
                self.dataset_max_len = len(dataset)
        for dataset in datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size, dataset_max_len=self.dataset_max_len)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size, dataset_max_len):
        epoch = math.floor(dataset_max_len / dataset_len)
        epoch = math.floor(epoch/2)
        all_index_batches = []
        for _ in range(epoch):
            index_samples = list(range(dataset_len))
            random.shuffle(index_samples)
            index_batches = [
                index_samples[i:min(i + batch_size, dataset_len)]
                for i in range(0, dataset_len, batch_size)
            ]
            all_index_batches.extend(index_batches)
        random.shuffle(all_index_batches)
        return all_index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            batch = [(task_id, sample_id) for sample_id in batch]
            if len(batch) % self.world_size != 0:
                if self.drop_last:
                    break
                else:
                    batch.extend(
                        [
                            batch[0]
                            for _ in range(
                                self.world_size - len(batch) % self.world_size
                            )
                        ]
                    )
            chunk_size = len(batch) // self.world_size
            yield batch[self.rank * chunk_size : (self.rank + 1) * chunk_size]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class DistSingleTaskBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=0, world_size=1, drop_last=False):
        self.rank = rank
        self.world_size = world_size
        self._dataset = dataset
        self.drop_last = drop_last
        self._data = self._get_index_batches(len(dataset), batch_size)

    @staticmethod
    def _get_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        return index_batches

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        indices = iter(self._data)
        for batch in indices:
            task_id = self._dataset.get_task_id()
            batch = [(task_id, sample_id) for sample_id in batch]
            yield batch


class MultiTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt,
        extra_task_ratio,
    ):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:

            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_samples = list(range(dataset_len))
        random.shuffle(index_samples)
        index_batches = [
            index_samples[i:min(i + batch_size, dataset_len)]
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class DistTaskDataset(Dataset):
    def __init__(self, dataset, task_id):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        _, sample_id = idx
        return self._dataset[sample_id]

    def get_task_id(self):
        return self._dataset.get_task_id()
