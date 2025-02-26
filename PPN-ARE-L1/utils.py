
import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss
# def InfoNCE(self, view1, view2, temperature, b_cos=True):
#     if b_cos:
#         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)  # 对应元素相乘，按最后一维相加，如第一行累加
#     pos_score = torch.exp(pos_score / temperature)  # 对张量的每个元素执行指数运算
#     ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # 将视图2进行转置，再进行矩阵运算
#     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)  # 对矩阵逐元素进行指数运算，再按照行相加，即第一行累加得到1919列，反映某一物品对其他所有物品的相似度
#     cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
#     return torch.mean(cl_loss) * 0.5
#    内存优化
class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()

    # def forward(self, view1, view2, temperature=0.2, b_cos=True):
    #     if b_cos:
    #         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    #     pos_score = (view1 * view2).sum(dim=-1)  # 对应元素相乘，按最后一维相加，如第一行累加
    #     pos_score = torch.exp_(pos_score / temperature)  # 对张量的每个元素执行指数运算
    #     ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # 将视图2进行转置，再进行矩阵运算
    #     ttl_score = torch.exp_(ttl_score / temperature).sum(
    #         dim=1)  # 对矩阵逐元素进行指数运算，再按照行相加，即第一行累加得到1919列，反映某一物品对其他所有物品的相似度
    #     cl_loss = -torch.log(pos_score / ttl_score + 10e-6)

    #     return torch.mean(cl_loss)

    # # 只计算正样本 + 一个随机负样本，避免sum(dim=1) 操作。
    # def forward(self, view1, view2, temperature=0.2, b_cos=True):
    #     if b_cos:
    #         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    #     # 计算正样本得分
    #     pos_score = (view1 * view2).sum(dim=-1)
    #     pos_score = torch.exp(pos_score / temperature)
    #     # 生成随机负样本索引（避免计算整个batch）
    #     batch_size = view1.shape[0]
    #     neg_idx = torch.randint(0, batch_size, (batch_size,), device=view1.device)  # 随机选取一个负样本
    #     neg_view2 = view2[neg_idx]  # 采样随机负样本
    #     # 计算负样本得分
    #     neg_score = (view1 * neg_view2).sum(dim=-1)
    #     neg_score = torch.exp(neg_score / temperature)
    #     # 计算对比损失
    #     cl_loss = -torch.log(pos_score / (pos_score + neg_score + 1e-6))
    #     return torch.mean(cl_loss)



    def forward(self, view1, view2, temperature=0.2, num_negatives=5, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    
        batch_size = view1.shape[0]
    
        # 计算正样本得分
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
    
        # 生成多个随机负样本索引
        neg_idx = torch.randint(0, batch_size, (batch_size, num_negatives), device=view1.device)  # 采样多个负样本
        neg_view2 = view2[neg_idx]  # 采样多个负样本, shape: (batch_size, num_negatives, feature_dim)
    
        # 计算负样本得分
        neg_score = (view1.unsqueeze(1) * neg_view2).sum(dim=-1)  # shape: (batch_size, num_negatives)
        neg_score = torch.exp(neg_score / temperature)
    
        # 计算均值后再进行损失计算，避免sum操作导致过大梯度
        neg_score_mean = neg_score.mean(dim=1)
    
        # 计算对比损失
        cl_loss = -torch.log(pos_score / (pos_score + neg_score_mean + 1e-6))
        return torch.mean(cl_loss)



