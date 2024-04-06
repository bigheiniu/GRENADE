from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertOnlyMLMHead
# BertForPreTraining
from transformers.modeling_outputs import ModelOutput
import torch
from typing import Optional, Union, List, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
from Metrics import Accuracy, ConfusionMatrix, Average
import math
try:
    from torch_geometric.nn import SAGEConv
except:
    print("ERROR in LOADING Geometric")


class GenClfOutput(ModelOutput):

    clf_loss: Optional[torch.FloatTensor] = None
    clf_logits: torch.FloatTensor = None
    loss=None
    gen_loss=None
    logits=None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        sim = self.cos(x, y)
        return sim / self.temp


class GraphSAGE(nn.Module):
    def __init__(self, hidden_size, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(hidden_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, hidden_size))

    def forward(self, x, adjs):
        for i, (edge_index, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x



class ProjectionMLP(nn.Module):
    def __init__(self, config, hidden_size=None):
        super().__init__()
        in_dim = config.hidden_size if hidden_size is None else hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class CMLM(BertPreTrainedModel):
    def __init__(self, config):
        super(CMLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        # graph neural network here
        self.lm_mlp = ProjectionMLP(config)
        self.gnn_mlp = ProjectionMLP(config)

        self.gnn_lp_mlp = ProjectionMLP(config)
        self.lm_lp_mlp = ProjectionMLP(config)

        # buffer for storing the sequence embedding
        self.momentum = config.momentum
        self.n_count = config.sample_neighbor_count
        self.is_2hop = config.is_2hop
        self.is_link_pre = config.is_link_pre
        self.contrast_lambda = getattr(config, "contrast_lambda", 1.0)
        self.is_pyg_gnn = getattr(config, "is_pyg_gnn", False)
        # two hop neighbors
        self.neighbor_sizes = 0 if getattr(config, "neighbor_sizes", None) is None else [int(i) for i in config.neighbor_sizes.split("_")]
        if self.is_pyg_gnn:
            # ATTENTION: We only exploit graphsage model
            self.graph_agg = GraphSAGE(
                hidden_size=config.hidden_size,
                num_layers=len(self.neighbor_sizes),
            )
        self.link_lambda = config.link_lambda
        self.is_pyg_gnn = config.is_pyg_gnn
        self.is_symmetric = getattr(config, "is_symmetric", False)


        self.is_self_kd = getattr(config, "is_self_kd", False)
        self.is_lsp_kd = getattr(config, "is_lsp_kd", False)
        self.lsp_temp = getattr(config, "lsp_temp", 0.05)
        self.is_lm2many = getattr(config, "is_lm2many", False)
        self.sample_neighbor_count = getattr(config, "sample_neighbor_count", 1)

        self.node_embedding = nn.Embedding(
            config.all_nodes_count+1,
            config.hidden_size,
            padding_idx=0
        )
        self.temp = config.temp
        self.sim = Similarity(self.temp)
        self.metrics = {
            'contrast_loss': Average(),
            "mlm_loss": Average(),
            "link_lm_loss": Average(),
            "link_gnn_loss": Average(),
            "link_con_loss": Average(),
            "gnn_masked_lm_contrast_loss": Average(),
            "lm_masked_lm_contrast_loss": Average(),
            "kd_loss": Average(),
            "lsp_kd_loss": Average(),
            "lm2many_loss": Average(),
        }

    def get_metrics(self, reset: bool = False):
        return {k: m.get_metric(reset=reset) for k, m in self.metrics.items()}

    def load_gnn(self, state_dict):
        if self.is_pyg_gnn:
            try:
                self.graph_agg.load_state_dict(state_dict)
            except:
                print(self.graph_agg.state_dict().keys())
                print(state_dict.keys())
                exit(-1)
        else:
            self.graph_agg_list.load_state_dict(state_dict)


    def message_agg_gnn(self, q, pooled_output, adjs, n_ids, is_update_node_embed=True, momentum=None, is_recover_node_embed=False):
        if is_recover_node_embed:
            original_query = self.node_embedding.weight[q].clone()

        if is_update_node_embed and pooled_output is not None:
            with torch.no_grad():
                momentum = momentum if momentum is not None else self.momentum
                self.node_embedding.weight[q] = self.node_embedding(q) * momentum + (1 - momentum) * pooled_output.detach()
        output = self.graph_agg(self.node_embedding.weight[n_ids], adjs)
        assert len(output.shape) == 2, output.shape
        if is_recover_node_embed:
            with torch.no_grad():
                self.node_embedding.weight[q] = original_query
        return output


    def init_node_embedding(self, node_embedding_weight: torch.Tensor):
        # Fix this stupid error here.
        self.node_embedding = nn.Embedding.from_pretrained(node_embedding_weight, freeze=True)




    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        q: Optional[torch.Tensor]=None,
        n: Optional[torch.Tensor]=None,
        n1: Optional[torch.Tensor]=None,
        original_input_ids: Optional[torch.Tensor]=None,
        is_only_update_gnn=False,
        is_only_update_lm=False,
        adjs=None,
        n_ids=None,
        **kwargs,
    ):
        if kwargs.get('is_only_node_embedding', False):
            return self.get_sentence_embedding(
                original_input_ids=original_input_ids,
                attention_mask=attention_mask,
                q=q,
                n=n,
                adjs=adjs,
                n_ids=n_ids
            )


        q = q.reshape(-1)
        if n is not None:
            n = n.reshape(-1, n.shape[-1])
        outputs = self.bert(
                original_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cls_input=None
            )
        # sentence embedding from BERT
        q_pooled_output = outputs[1]
        q_agg_output = self.message_agg_gnn(
                q,
                q_pooled_output if self.is_lm2many is False else None,
                adjs,
                n_ids
            )

        loss = 0
        if self.is_lm2many:
            lm_output = self.lm_mlp(q_pooled_output)
            gnn_output = self.gnn_mlp(q_agg_output)
            # lm-based contrastive learning
            lm2many_loss = self.row_col_contrast(
                lm_output,
                gnn_output,
                is_symmetric=self.is_symmetric
            )

            loss += lm2many_loss
            self.metrics['lm2many_loss'](lm2many_loss)
        if self.is_lsp_kd:
            lm_output = self.lm_mlp(q_pooled_output)
            gnn_output = self.gnn_mlp(q_agg_output)
            is_symmetric = self.is_symmetric
            # since this is symmetric, the roles did not matter.
            teacher = gnn_output
            student = lm_output

            lsp_kd_loss = self.lsp_kd(
                teacher=teacher,
                student=student,
                is_symmetric=is_symmetric
            )
            loss += lsp_kd_loss
            self.metrics['lsp_kd_loss'](lsp_kd_loss)
        if self.is_link_pre and self.link_lambda > 0:
            link_lm_loss = self.row_col_contrast(
                self.lm_lp_mlp(q_pooled_output.reshape(-1, q_pooled_output.shape[-1])),
                self.lm_lp_mlp(q_pooled_output.reshape(-1, q_pooled_output.shape[-1])),
                is_remove_self=True,
            )
            link_gnn_loss = self.row_col_contrast(
                self.gnn_lp_mlp(q_agg_output.reshape(-1, q_agg_output.shape[-1])),
                self.gnn_lp_mlp(q_agg_output.reshape(-1, q_agg_output.shape[-1])),
                is_remove_self=True,
            )
            loss += (link_lm_loss + link_gnn_loss) * self.link_lambda
            self.metrics['link_lm_loss'](link_lm_loss)
            self.metrics['link_gnn_loss'](link_gnn_loss)



        if torch.isnan(loss):
            raise ValueError("NAN for training LOSS!!!")

        return BertForPreTrainingOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_sentence_embedding(self,
                               original_input_ids=None,
                              attention_mask=None,
                              q=None,
                              n=None,
                              adjs=None,
                                n_ids=None
                              ):


        outputs = self.bert(
                original_input_ids,
                attention_mask=attention_mask,
                cls_input=None
            )
        q_pooled_output = outputs[1]
        q_agg_input = self.message_agg_gnn(
            q,
            None,
            adjs,
            n_ids)
        assert len(q_agg_input.shape) == 2, q_agg_input.shape

        hidden_states = torch.cat([q_pooled_output, q_agg_input],dim=-1)
        # hidden_states = q_agg_input
        return hidden_states, q


    def row_col_contrast(self,
                         lm_hidden,
                         gnn_hidden,
                         temperature=0.05,
                         is_symmetric=False,
                         is_remove_self=False,
                         ):

        def sim_matrix(a, b, eps=1e-8):
            """
            added eps for numerical stability
            """
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.clamp(a_n, min=eps)
            b_norm = b / torch.clamp(b_n, min=eps)
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
            return sim_mt

        device = lm_hidden.device
        lm_bsz = lm_hidden.shape[0]
        gnn_bsz = gnn_hidden.shape[0]
        # 32 * 64
        # if self.is_symmetric:
        #     # ATTENTION: here we only sample 1 neighbors
        single_block = torch.ones(self.sample_neighbor_count + 1, self.sample_neighbor_count + 1)
        label_matrix = torch.block_diag(*[single_block]*int(gnn_bsz/(self.sample_neighbor_count + 1)))
        # else:
        #     labels = torch.arange(lm_bsz, dtype=torch.long).unsqueeze(1).repeat(1, int(gnn_bsz/lm_bsz)).reshape(-1)
        #     label_matrix = F.one_hot(labels.cpu(), num_classes=lm_bsz).T
        label_matrix = label_matrix.to(device)
        if is_remove_self:
            label_matrix.fill_diagonal_(0)

        cos_sim_contrast = torch.div(sim_matrix(lm_hidden, gnn_hidden), temperature)
        # consider the augmentation as the query
        logits_max, _ = torch.max(cos_sim_contrast, dim=1, keepdim=True)
        logits = cos_sim_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (label_matrix * log_prob).sum(1) / label_matrix.sum(1)
        loss = -mean_log_prob_pos
        if is_symmetric:
            log_prob_sym = logits - torch.log(exp_logits.sum(0, keepdim=True))
            mean_log_prob_pos_sym = (label_matrix.T * log_prob_sym).sum(0) / label_matrix.T.sum(0)
            loss += -mean_log_prob_pos_sym
            loss = loss / 2.


        loss = loss.mean()
        return loss


    def lsp_kd(self,
               student,
               teacher,
               is_symmetric=False,
               ):
        z1_cos_sim = self.sim(student.unsqueeze(1), student.unsqueeze(0))
        z2_cos_sim = self.sim(teacher.unsqueeze(1), teacher.unsqueeze(0))
        device = z1_cos_sim.device
        # mask the center
        self_mask = torch.eye(z1_cos_sim.shape[0], device=device)
        z1_cos_sim = z1_cos_sim + -10000. * self_mask
        z2_cos_sim = z2_cos_sim + -10000. * self_mask
        z1_log_prob = torch.log_softmax(
            z1_cos_sim/self.lsp_temp, dim=-1
        )
        z2_prob = torch.softmax(
            z2_cos_sim/self.lsp_temp, dim=-1
        )
        loss_fn = nn.KLDivLoss()
        loss = loss_fn(z1_log_prob, z2_prob)
        if is_symmetric:
            z1_prob = torch.softmax(
                z1_cos_sim / self.lsp_temp, dim=-1
            )
            z2_log_prob = torch.log_softmax(
                z2_cos_sim / self.lsp_temp, dim=-1
            )
            loss += loss_fn(z2_log_prob, z1_prob)
            loss = loss / 2
        print("LSP Loss {}".format(loss))
        return loss



