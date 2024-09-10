import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph


class MHCR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MHCR, self).__init__(config, dataset)

        # Initialize configuration parameters
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.hc_loss_weight = config['hc_loss_weight']
        self.embloss_weight = config['embloss_weight']
        self.ghc_weight = config['ghc_weight']
        self.tau = config['tau']
        self.n_nodes = self.n_users + self.n_items
        self.hgnnLayer = HyperGNN(self.n_hyper_layer)

        # Prepare interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tensor(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj2 = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Load or create adjacency matrices for image, text, and video
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        video_adj_file = os.path.join(dataset_path, 'video_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # Initialize additional modules
        self.graph_contrast = GraphRankContrast()
        self.emb_reg = EmbRegularization()
        self.gate_module = Gate(self.embedding_dim)
        self.drop = nn.Dropout(p=1 - self.keep_rate)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize embeddings for image, text, and video features if available
        if self.i_feat is not None:
            self.image_embedding, self.image_trs, self.image_hyper, self.image_original_adj = self.initialize_embedding(
                self.i_feat, self.embedding_dim, self.hyper_num, image_adj_file)

        if self.t_feat is not None:
            self.text_embedding, self.text_trs, self.text_hyper, self.text_original_adj = self.initialize_embedding(
                self.t_feat, self.embedding_dim, self.hyper_num, text_adj_file)

        if self.v_feat is not None:
            self.video_embedding, self.video_trs, self.video_hyper, self.video_original_adj = self.initialize_embedding(
                self.v_feat, self.embedding_dim, self.hyper_num, video_adj_file)

    # Matrix normalizations and adjacency matrix functions
    def normalized_adj_single(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        norm_adj = norm_adj.dot(d_mat_inv)
        return norm_adj.tocoo()

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        norm_adj_mat = self.normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def load_or_build_adj(self, file_path, embedding, knn_k, sparse):
        if os.path.exists(file_path):
            adj = torch.load(file_path)
        else:
            adj = build_sim(embedding.weight.detach())
            adj = build_knn_normalized_graph(adj, topk=knn_k, is_sparse=sparse, norm_type='sym')
            torch.save(adj, file_path)
        return adj.cuda()

    def scipy_matrix_to_sparse_tensor(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tensor(L, torch.Size((self.n_nodes, self.n_nodes)))

    # Feature and embedding processing
    def initialize_embedding(self, feat, embedding_dim, hyper_num, adj_file):
        embedding_layer = nn.Embedding.from_pretrained(feat, freeze=False)
        transform_layer = nn.Linear(feat.shape[1], embedding_dim)
        hyper_matrix = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(feat.shape[1], hyper_num)))
        original_adj = self.load_or_build_adj(adj_file, embedding_layer, self.knn_k, self.sparse)
        return embedding_layer, transform_layer, hyper_matrix, original_adj

    def process_features(self, embedding_weight, transform_layer, hyper_matrix):
        features = transform_layer(embedding_weight)
        item_hyper = torch.mm(embedding_weight, hyper_matrix)
        user_hyper = torch.mm(self.adj, item_hyper)
        item_hyper = F.gumbel_softmax(item_hyper, self.tau, dim=1, hard=False)
        user_hyper = F.gumbel_softmax(user_hyper, self.tau, dim=1, hard=False)
        return features, item_hyper, user_hyper

    def process_embeddings(self, item_embeds, original_adj):
        for _ in range(self.n_layers):
            if self.sparse:
                item_embeds = torch.sparse.mm(original_adj, item_embeds)
            else:
                item_embeds = torch.mm(original_adj, item_embeds)
        user_embeds = torch.sparse.mm(self.R, item_embeds)
        return torch.cat([user_embeds, item_embeds], dim=0)

    def aggregate_user_item(self):
        combined_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        embeddings_history = [combined_embeddings]
        for _ in range(self.n_ui_layers):
            combined_embeddings = torch.sparse.mm(self.norm_adj2, combined_embeddings)
            embeddings_history.append(combined_embeddings)
        stacked_embeddings = torch.stack(embeddings_history, dim=1)
        averaged_embeddings = torch.mean(stacked_embeddings, dim=1)
        return averaged_embeddings

    # Forward propagation and loss calculation
    def forward(self, adj, train=False):
        # Process features for image, text, and video
        if self.i_feat is not None:
            image_feats, i_image_hyper, u_image_hyper = self.process_features(self.image_embedding.weight, self.image_trs, self.image_hyper)

        if self.t_feat is not None:
            text_feats, i_text_hyper, u_text_hyper = self.process_features(self.text_embedding.weight, self.text_trs, self.text_hyper)

        if self.v_feat is not None:
            video_feats, i_video_hyper, u_video_hyper = self.process_features(self.video_embedding.weight, self.video_trs, self.video_hyper)

        # Multiply item embeddings by gate outputs
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_module.forward_gate(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_module.forward_gate(text_feats))
        video_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_module.forward_gate(video_feats))

        # Aggregate user and item embeddings
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Process embeddings for image, text, and video
        image_embeds = self.process_embeddings(image_item_embeds, self.image_original_adj)
        text_embeds = self.process_embeddings(text_item_embeds, self.text_original_adj)
        video_embeds = self.process_embeddings(video_item_embeds, self.video_original_adj)

        # HGNN Layers
        u_image_hyper_embs, i_image_hyper_embs = self.hgnnLayer(self.drop(i_image_hyper), self.drop(u_image_hyper), self.aggregate_user_item()[self.n_users:])
        u_text_hyper_embs, i_text_hyper_embs = self.hgnnLayer(self.drop(i_text_hyper), self.drop(u_text_hyper), self.aggregate_user_item()[self.n_users:])
        u_video_hyper_embs, i_video_hyper_embs = self.hgnnLayer(self.drop(i_video_hyper), self.drop(u_video_hyper), self.aggregate_user_item()[self.n_users:])
        hyper_embeddings = [u_image_hyper_embs, i_image_hyper_embs, u_text_hyper_embs, i_text_hyper_embs, u_video_hyper_embs, i_video_hyper_embs]

        # Calculate attention for common embedding
        att_common = torch.cat([self.gate_module.forward_common(image_embeds), self.gate_module.forward_common(text_embeds), self.gate_module.forward_common(video_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(dim=1) * text_embeds + weight_common[:, 2].unsqueeze(dim=1) * video_embeds

        # Calculate separate embeddings
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds
        sep_video_embeds = video_embeds - common_embeds
        sep_image_embeds = torch.multiply(self.gate_module.forward_gate(content_embeds), sep_image_embeds)
        sep_text_embeds = torch.multiply(self.gate_module.forward_gate(content_embeds), sep_text_embeds)
        sep_video_embeds = torch.multiply(self.gate_module.forward_gate(content_embeds), sep_video_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + sep_video_embeds + common_embeds) / 4
        graph_embeds = [image_embeds, video_embeds, text_embeds, side_embeds, content_embeds]
        ai_hyper_embs = torch.concat([u_image_hyper_embs, i_image_hyper_embs], dim=0)
        at_hyper_embs = torch.concat([u_text_hyper_embs, i_text_hyper_embs], dim=0)
        av_hyper_embs = torch.concat([u_video_hyper_embs, i_video_hyper_embs], dim=0)
        hyper_embeds = [u_image_hyper_embs, i_image_hyper_embs, u_text_hyper_embs, i_text_hyper_embs, u_video_hyper_embs, i_video_hyper_embs, ai_hyper_embs, at_hyper_embs, av_hyper_embs]
        ghe_embs = ai_hyper_embs + at_hyper_embs + av_hyper_embs
        all_embeds = content_embeds + side_embeddings + self.alpha * F.normalize(ghe_embs)
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return (
                all_embeddings_users,
                all_embeddings_items,
                graph_embeds,
                hyper_embeds
            )

        return all_embeddings_users, all_embeddings_items

    # Loss calculation
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        emb_loss = self.reg_weight * regularizer
        bpr_loss = mf_loss + emb_loss
        return bpr_loss

    def contrastive_loss(self, emb1, emb2, emb3=None, all_emb=None, temperature=0.2):
        norm_emb1 = F.normalize(emb1, dim=1)
        norm_emb2 = F.normalize(emb2, dim=1)
        if emb3 is not None and all_emb is not None:
            norm_emb3 = F.normalize(emb3, dim=1)
            norm_all_emb = F.normalize(all_emb, dim=1)
            pos_score_1 = torch.exp((norm_emb1 * norm_emb2).sum(dim=-1) / temperature)
            pos_score_2 = torch.exp((norm_emb1 * norm_emb3).sum(dim=-1) / temperature)
            pos_score_3 = torch.exp((norm_emb2 * norm_emb3).sum(dim=-1) / temperature)
            pos_score = pos_score_1 + pos_score_2 + pos_score_3
            ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / temperature).sum(dim=1)
            ttl_score += torch.exp(torch.matmul(norm_emb2, norm_all_emb.T) / temperature).sum(dim=1)
            ttl_score += torch.exp(torch.matmul(norm_emb3, norm_all_emb.T) / temperature).sum(dim=1)
            cl_loss = -torch.log(pos_score / ttl_score).sum()
        else:
            pos_score = torch.exp((norm_emb1 * norm_emb2).sum(dim=-1) / temperature)
            ttl_score = torch.exp(torch.matmul(norm_emb1, norm_emb2.T) / temperature).sum(dim=1)
            cl_loss = -torch.log(pos_score / ttl_score).mean()
        return cl_loss

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        ua_embeddings, ia_embeddings, graph_embeds, hyper_embeds = self.forward(self.norm_adj, train=True)
        
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        
        [image_embeds, video_embeds, text_embeds, side_embeds, content_embeds] = graph_embeds
        [u_image_embs, i_image_embs, u_text_embs, i_text_embs, u_video_embs, i_video_embs, ai_hyper_embs, at_hyper_embs, av_hyper_embs] = hyper_embeds
        
        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)+ (self.contrastive_loss(side_embeds_items[pos_items], content_embeds_items[pos_items]) + \
                  self.contrastive_loss(side_embeds_users[users], content_embeds_user[users]))* self.cl_loss + self.embloss_weight * self.emb_reg(ua_embeddings, ia_embeddings)
        batch_hc_loss = self.contrastive_loss(u_image_embs[users], u_text_embs[users], u_video_embs[users], u_text_embs) + \
                         self.contrastive_loss(i_image_embs[pos_items], i_text_embs[pos_items], i_video_embs[pos_items], i_text_embs)
        graph_hypergraph_contrast_loss = self.graph_contrast(ai_hyper_embs, image_embeds) + self.graph_contrast(av_hyper_embs, video_embeds) + self.graph_contrast(at_hyper_embs, text_embeds)
        
        return bpr_loss + self.hc_loss_weight * batch_hc_loss + self.ghc_weight * graph_hypergraph_contrast_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

class EmbRegularization(nn.Module):
    def __init__(self, p_norm=2):
        super().__init__()
        self.p_norm = p_norm

    def forward(self, *inputs):
        total_loss = torch.tensor(0.0, device=inputs[-1].device)
        for tensor in inputs:
            total_loss += torch.norm(tensor, p=self.p_norm)
        regularized_loss = total_loss / inputs[-1].size(0)
        return regularized_loss

class Gate(nn.Module):
    def __init__(self, embedding_dim):
        super(Gate, self).__init__()
        self.embedding_dim = embedding_dim
        self.common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Sigmoid()
        )

    def forward_common(self, x):
        return self.common(x)

    def forward_gate(self, x):
        return self.gate(x)


class HyperGNN(nn.Module):
    def __init__(self, n_hyper_layer):
        super().__init__()
        self.num_layers = n_hyper_layer

    def forward(self, item_hyper, user_hyper, embeddings):
        item_output = embeddings
        for _ in range(self.num_layers):
            latent = torch.matmul(item_hyper.T, item_output)
            item_output = torch.matmul(item_hyper, latent)
            user_output = torch.matmul(user_hyper, latent)
        return user_output, item_output


class GraphRankContrast(nn.Module):
    def __init__(self, temperature=0.1):
        super(GraphRankContrast, self).__init__()
        self.temperature = temperature

    def forward(self, g1_emb, g2_emb):
        g1_emb_sorted, _ = torch.sort(g1_emb, dim=1)
        g2_emb_sorted, _ = torch.sort(g2_emb, dim=1)
        pos_score = F.cosine_similarity(g1_emb_sorted, g2_emb_sorted, dim=1)
        pos_score = torch.exp(pos_score / self.temperature)
        neg_emb = g2_emb_sorted[torch.randperm(g2_emb_sorted.size(0))]
        neg_score = F.cosine_similarity(g1_emb_sorted, neg_emb, dim=1)
        neg_score = torch.exp(neg_score / self.temperature)
        cl_loss = -torch.log(pos_score / (pos_score + neg_score))
        return torch.mean(cl_loss)
