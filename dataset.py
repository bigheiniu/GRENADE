import copy

# import graph_tool.all as gt
import os.path

import numpy
from random import random

import pandas as pd
import torch
import tqdm

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.utils import class_weight
import torch
from typing import List, Any
from collections import defaultdict


from torch_sparse import SparseTensor

print("FINISH Package LOADING!!!")
nltk.download('stopwords')
en_stop = stopwords.words('english')

class NodeMLMDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, train_type,no_sample=False):
        super(NodeMLMDataset, self).__init__()
        self.project = getattr(hparams, "project", 'ogbn-arxiv')
        self.txt_file = hparams.data_dir + f'/{self.project}/X.all.txt'
        self.model_name_or_path = getattr(hparams, 'model_name_or_path', None)
        self.is_link_pre = getattr(hparams, 'is_link_pre', False)
        self.is_bug = getattr(hparams, 'is_bug', False)
        self.is_lm2many = getattr(hparams, 'is_lm2many', False)
        self.k_hopcontrast = getattr(hparams, 'k_hopcontrast', 1)
        self.mlm_probability = getattr(hparams, 'mlm_probability', 0.15)
        self.max_length = hparams.max_seq_length

        self.train_type = train_type
        self.sample_neighbor_count = -1 if no_sample else hparams.sample_neighbor_count
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.catched_file = hparams.data_dir + f'/{self.project}/{type(self.tokenizer).__name__}.torch'
        # if os.path.exists(self.catched_file) is False:
        if os.path.exists(self.catched_file) is False and hparams.overwrite_cache is False:
            print("Reading Files!!!")
            with open(self.txt_file, 'r') as f1:
                data_list = f1.readlines()
            print("Reading Finished!!!")
            data_list = [i.strip() for i in data_list]
            abs_title_encode = self.tokenizer(data_list, truncation=True, max_length=self.max_length, padding=False)
            abs_title_encode = abs_title_encode['input_ids']

            torch.save(abs_title_encode, self.catched_file)
            self.titleabs = abs_title_encode
        else:
            print("LOADED FILE {}".format(self.catched_file))
            self.titleabs = torch.load(self.catched_file)

        graph_dataset = torch.load(hparams.data_dir + f"/{hparams.project}-ogbn.torch")
        split_idx = graph_dataset['split_idx']
        labels = graph_dataset['label']
        self.node_dict = graph_dataset['node_dict']

        self.num_classes = len(set(labels.reshape(-1).tolist()))
        print("There are {} labels!".format(self.num_classes))
        print("Start the Edges Creation!!!")
        print("Finish The Edges Creation!!!")
        # train, valid, test
        self.split_idx = list(chain.from_iterable(split_idx.values()))
        # nodes count:
        self.all_nodes_count = 2449029 if self.project == "ogbn-products" else 169343
        self.is_train = train_type == "train"
        if train_type == "train":
            # set the random seed for reproducibility for training
            print("!!!WE are setting the seed for the training dataset")
            random.seed(hparams.seed)
            np.random.seed(hparams.seed)
            train_labels = np.array([labels[i].item() for i in self.split_idx]).reshape(-1)
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels),
                                                              y=train_labels)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.labels = labels
        self.no_sample = no_sample

    def sample_neighbors(self, node_id, neighbor_count):
        if node_id == -1:
            return [-1] * neighbor_count
        neighbor_ids = list(self.node_dict[node_id])
        if len(neighbor_ids) == 0:
            # utilize itself as the pad neighbors
            neighbor_ids = [-1] * neighbor_count
        elif len(neighbor_ids) < neighbor_count:
            # pad the neighbors
            neighbor_ids += [-1] * (neighbor_count - len(neighbor_ids))
        if self.train_type == "train":
            np.random.shuffle(neighbor_ids)
        return neighbor_ids[:neighbor_count]

    def sample_khop_neighbors(self, node_id, neighbor_count, k_hop_left):
        neighbor_id = node_id
        for _ in range(k_hop_left):
            neighbor_id = self.sample_neighbors(neighbor_id, neighbor_count)[0]
        return [neighbor_id]

    def mae_mask(self, input_ids_list):
        # randomly mask words
        all_len = sum([len(i) for i in input_ids_list])
        # 0 means keep
        # 1 means mask
        mask_flag = np.random.binomial(1, self.mlm_probability, all_len)
        return_input_ids_list = []
        current_index = 0
        for input_ids in input_ids_list:
            new_input_ids = []
            for one_token in input_ids:
                if mask_flag[current_index] == 0:
                    new_input_ids.append(one_token)
                current_index += 1
            return_input_ids_list.append(new_input_ids)
        assert np.sum(1-mask_flag) == sum([len(i) for i in return_input_ids_list])
        return return_input_ids_list


    def __getitem__(self, i):
        item = self.split_idx[i]
        querys_no_pad_ids = self.titleabs[item]
        label = self.labels[item]
        if self.is_link_pre:
            # ATTENTION: K-hop selections
            if self.k_hopcontrast > 1:
                nei_list = self.sample_khop_neighbors(item, self.sample_neighbor_count, self.k_hopcontrast)
            else:
                nei_list = self.sample_neighbors(item, self.sample_neighbor_count)
            nei_list = [i if i != -1 else item for i in nei_list]
            nei_no_pad_ids_list = [self.titleabs[i] for i in nei_list]
        else:
            nei_no_pad_ids_list = []
            nei_list = []


        if self.no_sample is False:
            # 0 is padding
            sampled_neighbors = self.sample_neighbors(item, self.sample_neighbor_count)
            input_ids_list = [querys_no_pad_ids] + nei_no_pad_ids_list
            if self.is_link_pre:
                output = {
                    "input_ids": input_ids_list,
                    "q": [item + 1] + [i + 1 for i in nei_list],
                    "n": [[i + 1 for i in sampled_neighbors]],
                    "clf_label": label
                }
            else:
                two_hop_sampled_neighbors = [self.sample_neighbors(i, self.sample_neighbor_count) for i in sampled_neighbors]
                output = {
                    "input_ids": querys_no_pad_ids,
                    "q": item + 1,
                    "n": [i + 1 for i in sampled_neighbors],
                    "n1": [[i + 1 for i in j] for j in two_hop_sampled_neighbors],
                    "clf_label": label
                }
            return output
        else:
            return {
                "input_ids": querys_no_pad_ids,
                "q": item + 1,
                "clf_label": label
            }


    def __len__(self):
        return len(self.split_idx)

def create_adj_t(node_dict=None, node_count=169343):
    # if node_dict is not None:
    rows = []
    cols = []
# construct the edge index
    for key, values in node_dict.items():
        for v in values:
            rows.append(key)
            cols.append(v)
    # 1 for padding
    adj_t = SparseTensor(
        row=torch.tensor(np.array(rows)) + 1,
        col=torch.tensor(np.array(cols)) + 1,
        value=torch.ones(len(rows)),
        # set for ogbn-arxiv dataset only
        sparse_sizes=(node_count + 1, node_count + 1)
        )

    return adj_t

def pad_sequence(input_ids: List[List], pad_id: int):
    max_length = max([len(i) for i in input_ids])
    pad_input_ids = torch.tensor([i + [pad_id] * (max_length - len(i)) for i in input_ids])
    attention_mask = torch.tensor([[1] * len(i) + [0] * (max_length - len(i)) for i in input_ids])
    special_tokens_mask = torch.tensor([
        [1] + [0] * (len(i) - 2) + [1] + [1] * (max_length - len(i)) for i in input_ids]
    )
    return pad_input_ids, attention_mask, special_tokens_mask
from itertools import chain
def neighbor_mask_collator(batch, mlm_probability, tokenizer, is_link_pre=False, **kwargs):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    new_batch = {}
    for key in batch[0].keys():
        # elements in batch are list
        if "input_ids" == key:
            if is_link_pre:
                new_batch["input_ids"], new_batch["attention_mask"], new_batch["special_tokens_mask"] = pad_sequence(list(chain.from_iterable([i[key] for i in batch])), pad_id=tokenizer.pad_token_id)
            else:
                new_batch["input_ids"], new_batch["attention_mask"], new_batch["special_tokens_mask"] = pad_sequence([i[key] for i in batch], pad_id=tokenizer.pad_token_id)

        else:
            new_batch[key] = torch.tensor(numpy.array([i[key] for i in batch]))

    labels = new_batch['input_ids'].clone()
    original_input_ids = new_batch['input_ids'].clone()
    new_batch['original_input_ids'] = original_input_ids
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    special_tokens_mask = new_batch['special_tokens_mask']
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = special_tokens_mask.bool()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    new_batch['input_ids'][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # ATTENTION: reject the CLS and SEP special tokens
    random_words = torch.randint(103, len(tokenizer), labels.shape, dtype=torch.long)
    new_batch['input_ids'][indices_random] = random_words[indices_random]
    new_batch.update({"labels":labels})
    del new_batch['special_tokens_mask']
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return new_batch

from typing import NamedTuple, Tuple
class TensorCuda(NamedTuple):
    adj_t: Any
    size: Tuple[int, int]

    def to(self, device):
        adj_t = self.adj_t.to(device)
        return TensorCuda(adj_t, self.size)

def neighbor_mask_collator_gnn(batch,
                               mlm_probability,
                               tokenizer,
                               is_link_pre=False,
                               adj_t=None,
                               neighbor_sizes=(3, 5),
                               ):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    new_batch = neighbor_mask_collator(
        batch,
        mlm_probability,
        tokenizer,
        is_link_pre,
    )
    adjs = []
    n_id = new_batch['q']
    if len(n_id.shape) == 2:
        n_id = n_id.reshape(-1)
    if type(neighbor_sizes) is str:
        neighbor_sizes = list([int(i) for i in neighbor_sizes.split("_")])
    # 0 for padding
    for size in neighbor_sizes:
        adj_mini, n_id = adj_t.sample_adj(n_id,
                                    num_neighbors=size,
                                    replace=True)
        size = adj_mini.sparse_sizes()[::-1]
        adjs.append(TensorCuda(adj_mini, size))

    adjs = [adjs[0]] if len(adjs) == 1 else adjs[::-1]

    new_batch['n_ids'] = n_id
    new_batch['adjs'] = adjs

    return new_batch