import argparse

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from tqdm import tqdm
#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import collections

import numpy as np
import scipy.sparse as smat

import wandb
#
wandb.init(project="MLPFinetune", sync_tensorboard=True)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def load_matrix(src, dtype=None):
    """Load dense or sparse matrix from file.

    Args:
        src (str): path to load the matrix.
        dtype (numpy.dtype, optional): if given, convert matrix dtype. otherwise use default type.

    Returns:
        mat (numpy.ndarray or scipy.sparse.spmatrix): loaded matrix

    Notes:
        If underlying matrix is {"csc", "csr", "bsr"}, indices will be sorted.
    """
    if not isinstance(src, str):
        raise ValueError("src for load_matrix must be a str")

    mat = np.load(src)
    # decide whether it's dense or sparse
    if isinstance(mat, np.ndarray):
        pass
    elif isinstance(mat, np.lib.npyio.NpzFile):
        # Ref code: https://github.co[m/scipy/scipy/blob/v1.4.1/scipy/sparse/_matrix_io.py#L19-L80
        matrix_format = mat["format"].item()
        if not isinstance(matrix_format, str):
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            matrix_format = matrix_format.decode("ascii")
        try:
            cls = getattr(smat, "{}_matrix".format(matrix_format))
        except AttributeError:
            raise ValueError("Unknown matrix format {}".format(matrix_format))

        if matrix_format in ("csc", "csr", "bsr"):
            mat = cls((mat["data"], mat["indices"], mat["indptr"]), shape=mat["shape"])
            # This is in-place operation
            mat.sort_indices()
        elif matrix_format == "dia":
            mat = cls((mat["data"], mat["offsets"]), shape=mat["shape"])
        elif matrix_format == "coo":
            mat = cls((mat["data"], (mat["row"], mat["col"])), shape=mat["shape"])
        else:
            raise NotImplementedError(
                "Load is not implemented for sparse matrix of format {}.".format(matrix_format)
            )
    else:
        raise TypeError("load_feature_matrix encountered unknown input format {}".format(type(mat)))

    if dtype is None:
        return mat
    else:
        return mat.astype(dtype)
class Metrics(collections.namedtuple("Metrics", ["prec", "recall"])):
    """The metrics (precision, recall) for multi-label classification problems."""

    __slots__ = ()

    def __str__(self):
        """Format printing"""

        def fmt(key):
            return " ".join("{:4.2f}".format(100 * v) for v in getattr(self, key)[:])

        return "\n".join("{:7}= {}".format(key, fmt(key)) for key in self._fields)

    @classmethod
    def default(cls):
        """Default dummy metric"""
        return cls(prec=[], recall=[])

    @classmethod
    def generate(cls, tY, pY, topk=10):
        """Compute the metrics with given prediction and ground truth.

        Args:
            tY (csr_matrix): ground truth label matrix
            pY (csr_matrix): predicted logits
            topk (int, optional): only generate topk prediction. Default 10

        Returns:
            Metrics
        """
        assert isinstance(tY, smat.csr_matrix), type(tY)
        assert isinstance(pY, smat.csr_matrix), type(pY)
        assert tY.shape == pY.shape, "tY.shape = {}, pY.shape = {}".format(tY.shape, pY.shape)
        pY = sorted_csr(pY)
        total_matched = np.zeros(topk, dtype=np.uint64)
        recall = np.zeros(topk, dtype=np.float64)
        for i in range(tY.shape[0]):
            truth = tY.indices[tY.indptr[i] : tY.indptr[i + 1]]
            matched = np.isin(pY.indices[pY.indptr[i] : pY.indptr[i + 1]][:topk], truth)
            cum_matched = np.cumsum(matched, dtype=np.uint64)
            total_matched[: len(cum_matched)] += cum_matched
            recall[: len(cum_matched)] += cum_matched / max(len(truth), 1)
            if len(cum_matched) != 0:
                total_matched[len(cum_matched) :] += cum_matched[-1]
                recall[len(cum_matched) :] += cum_matched[-1] / max(len(truth), 1)
        prec = total_matched / tY.shape[0] / np.arange(1, topk + 1)
        recall = recall / tY.shape[0]
        return cls(prec=prec, recall=recall)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, no_bns=False):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.no_bns=no_bns

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.no_bns is False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, x, y_true, train_idx, optimizer, bsz_size, device=torch.device("cuda")):
    model.train()

    optimizer.zero_grad()

    if bsz_size == -1:
        out = model(x[train_idx])
        loss = F.nll_loss(out, y_true.squeeze()[train_idx])
        loss.backward()
        optimizer.step()
    else:
        train_idx_list = list(range(len(train_idx)))
        np.random.shuffle(train_idx_list)
        for i in tqdm(range(0, len(train_idx), bsz_size), total=int(len(train_idx)/bsz_size)):
            train_batch = train_idx[train_idx_list[i:i+bsz_size]].squeeze()
            out = model(x[train_batch].to(device))
            loss = F.nll_loss(out, y_true.squeeze()[train_batch])
            loss.backward()
            optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, device=torch.device("cuda")):
    model.eval()

    val_out = model(x[split_idx["valid"]].to(device)).cpu()
    test_out = model(x[split_idx['test']].to(device)).cpu()
    y_pred_val = val_out.argmax(dim=-1, keepdim=True)
    y_pred_val = y_pred_val.detach().cpu().numpy()

    y_pred_test = test_out.argmax(dim=-1, keepdim=True)
    y_pred_test = y_pred_test.detach().cpu().numpy()


    y_true = y_true.cpu().numpy()
    valid_acc = accuracy_score(
        y_true=y_true[split_idx['valid']],
        y_pred= y_pred_val)
    test_acc = accuracy_score(
        y_true=y_true[split_idx['test']],
        y_pred=y_pred_test)

    return -1, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--data_root_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--embed_path', type=str, default=None)
    parser.add_argument('--np_embed_path', type=str, default=None)
    parser.add_argument('--mmap_embed_path', type=str, default=None)
    parser.add_argument('--is_fi', action="store_true")
    parser.add_argument('--project', default="ogbn-arxiv", type=str)
    parser.add_argument('--no_bns', action='store_true')
    parser.add_argument('--is_use_ogb', action='store_true')
    parser.add_argument('--bsz_size', type=int, default=-1)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--K', type=int, default=-1)
    parser.add_argument('--is_float32', action='store_true')
    args = parser.parse_args()
    wandb.config.update(vars(args))
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    graph_dataset = torch.load(args.data_root_dir + f"/{args.project}-ogbn.torch")

    # graph, labels = graph_dataset[0]

    split_idx = graph_dataset['split_idx']
    labels = torch.tensor(graph_dataset['label'])
    num_classes = len([i for i in set(labels.reshape(-1).tolist()) if i >= 0])
    # print("*"*100)
    # print(num_classes, max(labels.reshape(-1).tolist()), list(sorted(set(labels.reshape(-1).tolist())))[:5])
    if args.is_use_ogb:
        from ogb.nodeproppred import PygNodePropPredDataset
        from ogb.linkproppred import PygLinkPropPredDataset
        if "citation2" in args.project:
            dataset = PygLinkPropPredDataset(name='ogbl-citation2', root=args.data_root_dir)
            data = dataset[0]
            if type(data.x) is torch.Tensor:
                x = data.x
            else:
                x = torch.from_numpy(data.x)

        else:
            dataset = PygNodePropPredDataset(name=args.project,
                                             root=args.data_root_dir)
            data = dataset[0]
            if type(data.x) is not torch.Tensor:
                x = torch.from_numpy(data.x)
            else:
                x = data.x
    else:
        if args.model_path is not None:
            x = torch.load(args.model_path+"/pytorch_model.bin", map_location='cpu')['node_embedding.weight']
            x = x[1:]
        elif args.mmap_embed_path is not None:
            x = torch.from_numpy(np.array(np.memmap(args.mmap_embed_path, mode='r', dtype=np.float32 if args.is_float32 else np.float16).astype(np.float32)).reshape((-1, 768)))
        elif args.np_embed_path is not None:
            x = torch.from_numpy(load_matrix(args.np_embed_path).astype(np.float32))
        else:

            x = torch.load(args.embed_path, map_location='cpu')
            if x.shape[0] == 169343 + 1:
                x = x[1:]
            elif x.shape[0] == 2449029 + 1:
                x = x[1:]
            elif x.shape[0] == 2927963 + 1:
                x = x[1:]
    if args.is_fi and x.shape[1] > 768:
        with torch.no_grad():
            x_gnn = x[:, 768:]
            x_lm = x[:, :768]
            feature_add = x_gnn + x_lm
            feature_minus = x_gnn - x_lm
            x = torch.cat([x_lm, x_gnn, feature_add, feature_minus], dim=1)
    assert x.shape[0] == 169343 or x.shape[0] == 2449029 or x.shape[0] == 2927963, x.shape
    if args.bsz_size == -1:
        x = x.to(device)
    y_true = labels.to(device)
    train_idx = torch.tensor(split_idx['train']).to(device)
    # get label




    model = MLP(x.size(-1), args.hidden_channels, num_classes,
                args.num_layers, args.dropout, args.no_bns).to(device)




    test_acc_list = []
    for run in tqdm(range(args.runs)):
        if args.K > 0:
            here_idx_list = []
            for i in range(num_classes):
                here_idx = (y_true[train_idx] == i).nonzero()
                here_idx = train_idx[torch.randperm(here_idx.shape[0])[:args.K]]
                here_idx_list.append(here_idx)
            train_idx_here = torch.concat(here_idx_list).to(device)
        else:
            train_idx_here = train_idx
        model.reset_parameters()
        best_valid_acc = -1
        best_test_acc = -1
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        reset_button = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_idx_here, optimizer, args.bsz_size)
            result = test(model, x, y_true, split_idx)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_test_acc = test_acc
                    reset_button = 0
                else:
                    reset_button += 1
                if reset_button > args.patience:
                    break

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')
                # wandb.log({f'run_{run}_test_acc': test_acc, 'global_step': epoch})
        test_acc_list.append(best_test_acc)
        print(f'Run: {run + 1:02d}, '
              f"Best Test Acc {100 * best_test_acc:.2f}%, "
              f"Best Valid Acc {100 * best_valid_acc:.2f}%"
              )
    wandb.log({f'avg_test_acc': np.mean(test_acc_list), "std_test_acc": np.std(test_acc_list)})
    print(np.mean(test_acc_list), np.std(test_acc_list))



if __name__ == "__main__":
    main()
    wandb.finish()