import torch
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from argparse import ArgumentParser
import wandb
from pecos.utils import smat_util
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

parser = ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--project', type=str, default="ogbn-arxiv")
parser.add_argument('--embed_path', type=str, default=None)
parser.add_argument('--np_embed_path', type=str, default=None)
parser.add_argument('--mmap_embed_path', type=str, default=None)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument('--root_data_dir', type=str, default="/home/yli29/pecos/examples/giant-xrt/proc_data_xrt")
parser.add_argument("--is_use_ogb", action="store_true")
parser.add_argument("--is_agg_cluster", action="store_true")
parser.add_argument("--is_float32", action="store_true")
wandb.init(project="kmeans_clf", sync_tensorboard=True)
args = parser.parse_args()
wandb.config.update(vars(args))

def retrieve_info(cluster_labels,y_train, num_classes):
    # Initializing

    reference_labels = {}

    # For loop to run through each label of cluster label

    for i in range(num_classes):
        index = np.where(cluster_labels == i, 1, 0)
        try:
            num = np.bincount(y_train[index == 1]).argmax()
        except:
            num = np.random.randint(0, num_classes)
        reference_labels[i] = num

    return reference_labels


if args.is_use_ogb:
    if "citation2" not in args.project:
        embeddings = NodePropPredDataset(name=args.project, root=args.root_data_dir + "/temp")[0][0]['node_feat']
    else:
        dataset = LinkPropPredDataset(name='ogbl-citation2', root=args.root_data_dir)
        data = dataset[0]
        if type(data.x) is torch.Tensor:
            embeddings = data.x
        else:
            embeddings = torch.from_numpy(data.x)
        del data
else:
    if args.np_embed_path is not None:
        embeddings = torch.from_numpy(smat_util.load_matrix(args.np_embed_path).astype(np.float32))
    elif args.mmap_embed_path is not None:
        embeddings = torch.from_numpy(
            np.array(np.memmap(args.mmap_embed_path, mode='r', dtype=np.float32 if args.is_float32 else np.float16).astype(np.float32)).reshape(
                (-1, 768)))
    else:
        embeddings = torch.load(args.embed_path)
    if embeddings.shape[0] == 169343 + 1 or embeddings.shape[0] == 2449029 + 1:
        embeddings = embeddings[1:]

# load label information
data_dict = torch.load(f"{args.root_data_dir}/{args.project}-ogbn.torch")
test_index = data_dict['split_idx']['test']
label = data_dict['label']
if type(label) is list:
    label = np.array(label)
test_label = label[test_index].reshape(-1)
num_classes = len([i for i in set(test_label.reshape(-1).tolist()) if i >=0 ])
test_label_unique = {i: index for index, i in enumerate(np.unique(test_label).tolist())}
test_label = np.array([test_label_unique[i] for i in test_label.tolist()])
test_embeddings = embeddings[test_index]
if type(test_embeddings) is torch.Tensor:
    test_embeddings = test_embeddings.numpy()
del embeddings
# create the kmeans model
acc_list = []
nmi_list = []
arc_list = []
print(f"There are {num_classes} classes")
for i in range(args.runs):
    if args.is_agg_cluster:
        cluster_model = AgglomerativeClustering(n_clusters=num_classes).fit(X=test_embeddings, y=test_label)
    else:
        cluster_model = MiniBatchKMeans(n_clusters=num_classes).fit(X=test_embeddings, y=test_label)
    predict = cluster_model.labels_.reshape(-1)
    # eval the acc
    reference_labels = retrieve_info(predict, test_label, num_classes)
    predict = [reference_labels[i] for i in predict.tolist()]
    acc = accuracy_score(y_true=test_label, y_pred=predict)
    nmi = normalized_mutual_info_score(labels_true=test_label, labels_pred=predict)
    arc = adjusted_rand_score(labels_true=test_label, labels_pred=predict)
    wandb.log({"global_step":i, 'acc': acc, 'arc':arc, 'nmi':nmi})
    print(f"Runs: {i}, acc: {acc}")
    acc_list.append(acc)
    nmi_list.append(nmi)
    arc_list.append(arc)

avg_acc = np.mean(acc_list)
std_acc = np.std(acc_list)
avg_arc = np.mean(arc_list)
std_arc = np.std(arc_list)
avg_nmi = np.mean(nmi_list)
std_nmi = np.std(nmi_list)
print(f"STD Acc: {std_acc}, AVG acc: {avg_acc}")
wandb.log({"std_acc": std_acc,"avg_acc": avg_acc,
           "std_nmi": std_nmi, 'avg_nmi':avg_nmi,
           "std_arc": std_arc, 'avg_arc': avg_arc
           })

# python -m sklearnex Kmeans_eval.py --project ogbn-products --embed_path /home/yli29/OGBScripts/output/bert-ogbn-products-MM_1-Neighbor_10_5-gnnlr_5e-5-lm2manyContrastNoSym-sample_1-LSP_0.05-LinkPre/X_embed.torch
#python -m sklearnex Kmeans_eval.py --project ogbn-products --np_embed_path /home/yli29/pecos/examples/giant-xrt/proc_data_xrt/ogbn-products/X.all.xrt-emb.npy

