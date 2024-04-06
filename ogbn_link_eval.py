import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from ogb.linkproppred import Evaluator
from ogb.nodeproppred import NodePropPredDataset
import wandb
from pecos.utils import smat_util
print("FINISH INITLIZATION!!!")
wandb.init(project="ogbn-link-prediction", sync_tensorboard=True)

import torch.nn as nn


class CosSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y)

class DotSimilarity(nn.Module):
    def __init__(self):
        super(DotSimilarity, self).__init__()

    def forward(self, x, y):
        return torch.sigmoid(torch.bmm(x.view(x.shape[0], 1, x.shape[1]), y.view(*y.shape, 1)).squeeze())


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

# cosine similarity; dot product => 0-1 score
@torch.no_grad()
def test(sim, h, split_edge, evaluator, batch_size, device):
    sim.eval()
    sim = sim.to(device)
    def test_split():
        source = torch.tensor(split_edge['query'])
        target = torch.tensor(split_edge['pos_neighbors'])
        target_neg = torch.tensor(split_edge['neg_neighbors'])

        pos_preds = []

        for perm in tqdm(DataLoader(range(source.size(0)), batch_size), "Pos"):
            src, dst = source[perm], target[perm]
            pos_preds += [sim(h[src].to(device), h[dst].to(device)).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)
        index = torch.nonzero(pos_pred).squeeze()
        pos_pred_count = len(pos_pred) - len(torch.nonzero(pos_pred))
        total_count = len(pos_pred)
        print("POS PRED COUNT {}/{}={} ".format(pos_pred_count,total_count,pos_pred_count/total_count))
        neg_preds = []

        target_neg = target_neg.view(-1)
        number_neg = int(len(target_neg)/ len(source))
        source = source.view(-1, 1).repeat(1, number_neg).view(-1)
        for perm in tqdm(DataLoader(range(source.size(0)), batch_size), "neg"):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [sim(h[src].to(device), h[dst_neg].to(device)).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, number_neg)
        # print(zero_count_pos/len(target), zero_count_neg/len(source))
        # print some statistcal information
        neg_pred_max = neg_pred.max(dim=-1).values
        q = torch.tensor([0.25, 0.5, 0.75])
        print("Neg Percentage")
        print(torch.quantile(neg_pred_max, q, keepdim=True))
        print("Neg Average")
        print(neg_pred.mean(dim=-1))

        print("Pos Percentage")
        print(torch.quantile(pos_pred, q, keepdim=True))
        print("Pos Average")
        print(pos_pred.mean(dim=-1))
        print("\n\n")
        #
        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()
    #

    test_mrr = test_split()

    # return valid_mrr, test_mrr
    return test_mrr

def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (NS)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=512* 8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--project', type=str, default="ogbn-arxiv")
    parser.add_argument('--embed_path', type=str, default=None)
    parser.add_argument('--np_embed_path', type=str, default=None)
    parser.add_argument('--mmap_embed_path', type=str, default=None)
    parser.add_argument('--root_data_dir', type=str, default=None)
    parser.add_argument('--is_use_ogb', action="store_true")
    parser.add_argument('--sim', type=str, choices=['cos', 'dot'], default="cos")
    args = parser.parse_args()
    print(args)
    wandb.config.update(vars(args))
    if args.sim == "cos":
        sim = CosSimilarity()
    else:
        sim = DotSimilarity()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    mrr_data = torch.load("{}/{}/mrr_edges.torch".format(args.root_data_dir, args.project))
    # assume the missing nodes are feed with zeros
    # ATTENTION: Another preprocessing file align the matrix.
    dataset = NodePropPredDataset(name=args.project, root='{}/temp'.format(args.root_data_dir))
    query = dataset.get_idx_split()['test']
    if "query" not in mrr_data:
        mrr_data['query'] = query
    if args.is_use_ogb:
        embeddings = dataset[0][0]['node_feat']
        embeddings = torch.from_numpy(embeddings.astype(np.float32))
    else:
        if args.np_embed_path is not None:
            embeddings = torch.from_numpy(smat_util.load_matrix(args.np_embed_path).astype(np.float32))
        elif args.mmap_embed_path is not None:
            embeddings = torch.from_numpy(
                np.array(np.memmap(args.mmap_embed_path, mode='r', dtype=np.float16).astype(np.float32)).reshape(
                    (-1, 768)))
        else:
            embeddings = torch.load(args.embed_path)
        if embeddings.shape[0] == 169343 + 1 or embeddings.shape[0] == 2449029 + 1:
            embeddings = embeddings[1:]
    #     0.09593

    evaluator = Evaluator(name='ogbl-citation2')
    all_test_mrr = []
    for run in range(args.runs):

        # sim, h, split_edge, evaluator, batch_size, device
        result = test(sim,
                      embeddings,
                      mrr_data,
                      evaluator,
                      batch_size=512 * 1024,
                      device=device)

        test_mrr = result
        print(f'Run: {run + 1:02d}, '
              f'Test: {test_mrr:.4f}')
        wandb.log({f'run_{run}_test_mrr': test_mrr})

        all_test_mrr.append(test_mrr)

    mean_test_mrr = np.mean(all_test_mrr)
    std_test_mrr = np.std(all_test_mrr)



    print(f"Test Avg: {mean_test_mrr:.4f}, Valid Avg: {std_test_mrr:.4f}")
    wandb.log({
        "mean_test_mrr": mean_test_mrr,
        "std_test_mrr": std_test_mrr,})



if __name__ == "__main__":
    main()

# ./GLEM/arxiv_TA/l00/seed789seed789CRMaxIter2_inf_tr200000_temp0.2_LM-first__SAGE__GNNSAGE_l3_lr0.003_e300_do0.5_d256_es15_wd0.0_normBN_inT_liF_alpha0.05_1.0_redmean_fan5,10,15_bs1000/LMBert_lr2e-05_bsz30_wd0.01_do0.1_atdo0.1_cla_do0.4_cla_biasT_e3_we0.2_ef30460_loadT_ckptNone_lsf0.0_alpha0.8_1.0_redmean__em_info.pickle