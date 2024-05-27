
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss, InfoNCE



class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            # x = self.dropout(x)
        return x


class CRGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CRGCN, self).__init__()

        self.dislike = args.dislike
        self.b = args.b
        self.dislike_behaviors = args.dis_behaviors
        self.CL = args.CL
        self.CL_view = args.CL_view

        self.device = args.device
        self.data_name = args.data_name
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index  # 
        self.behaviors = args.behaviors  # 
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        # ,
        if self.dislike:
            self.dis_user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
            self.dis_item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

            self.Graph_encoder = nn.ModuleDict({
                behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior
                in enumerate(self.behaviors_add())
            })
        else:
            self.Graph_encoder = nn.ModuleDict({
                behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior
                in enumerate(self.behaviors)
            })

        self.reg_weight = args.reg_weight
        self.beta = args.cbeta
        self.alph = args.calph
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cl_loss = InfoNCE()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None
        self.dis_storage_all_embeddings = None
        self.apply(self._init_weights)

        self._load_model()

    # 
    def behaviors_add(self):
        return self.behaviors + self.dislike_behaviors

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self):
        """
        gcn propagate in each behavior
        """
        all_embeddings = {}
        dis_all_embeddings = {}

        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            indices = self.edge_index[behavior].to(self.device)
            layer_embeddings = self.Graph_encoder[behavior](layer_embeddings, indices)
            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            total_embeddings = layer_embeddings + total_embeddings
            all_embeddings[behavior] = total_embeddings

        if self.dislike:
            dis_total_embeddings = torch.cat([self.dis_user_embedding.weight, self.dis_item_embedding.weight], dim=0)
            for behavior in self.dislike_behaviors:
            # for behavior in ['dislike_sec','dislike_one']:
                layer_embeddings = dis_total_embeddings
                indices = self.edge_index[behavior].to(self.device)
                layer_embeddings = self.Graph_encoder[behavior](layer_embeddings, indices)
                layer_embeddings = F.normalize(layer_embeddings, dim=-1)
                dis_total_embeddings = layer_embeddings + dis_total_embeddings
                dis_all_embeddings[behavior] = dis_total_embeddings
        return all_embeddings, dis_all_embeddings

    def forward(self, batch_data):
        self.storage_all_embeddings = None
        all_embeddings, dis_all_embeddings = self.gcn_propagate()
        # all_embeddings = self.gcn_propagate()
        total_loss = 0
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior],
                                                                 [self.n_users + 1, self.n_items + 1])


            if (self.data_name == 'Tmall' and behavior == self.CL_view) or (
                    self.data_name == 'beibei' and behavior == self.CL_view):
                user_all_embedding_1, item_all_embedding_1 = torch.split(all_embeddings[self.CL_view],
                                                                     [self.n_users + 1, self.n_items + 1])
                total_loss += self.cl_loss(user_all_embedding_1, user_all_embedding) + self.cl_loss(item_all_embedding_1,
                                                                                                  item_all_embedding)
            user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
            item_feature = item_all_embedding[items]

            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight,
                                                                  self.item_embedding.weight)
        if self.dislike:
            for index, behavior in enumerate(self.dislike_behaviors):
            # for index, behavior in enumerate([ 'dislike_sec','dislike_one']):
                data = batch_data[:, index]
                users = data[:, 0].long()
                items = data[:, 1:].long()
                dis_user_all_embedding, dis_item_all_embedding = torch.split(dis_all_embeddings[behavior],
                                                                             [self.n_users + 1, self.n_items + 1])

                dis_user_feature = dis_user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
                dis_item_feature = dis_item_all_embedding[items]
                # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)

                scores = torch.sum(dis_user_feature * dis_item_feature, dim=2)
                total_loss += self.bpr_loss(self.b * scores[:, 0], scores[:, 1])*self.beta  

                # total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)  
            total_loss = total_loss + self.reg_weight * self.emb_loss(self.dis_user_embedding.weight,
                                                                      self.dis_item_embedding.weight)  # 


        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings, self.dis_storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]],
                                                     [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))

        if self.dislike:
            dis_user_embedding, dis_item_embedding = torch.split(self.dis_storage_all_embeddings[self.dislike_behaviors[-1]],
                                                                 [self.n_users + 1, self.n_items + 1])
            dis_user_emb = dis_user_embedding[users.long()]
            dis_scores = torch.matmul(dis_user_emb, dis_item_embedding.transpose(0, 1))
            dis_scores = F.normalize(dis_scores, dim=-1)
            mask = (dis_scores > 0.5)
            scores[mask] = -np.inf
        # return scores-dis_scores
        return scores
