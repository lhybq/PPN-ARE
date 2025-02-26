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

    def forward(self, view1, view2, temperature=0.2, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)  # 对应元素相乘，按最后一维相加，如第一行累加
        pos_score = torch.exp_(pos_score / temperature)  # 对张量的每个元素执行指数运算
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # 将视图2进行转置，再进行矩阵运算
        ttl_score = torch.exp_(ttl_score / temperature).sum(
            dim=1)  # 对矩阵逐元素进行指数运算，再按照行相加，即第一行累加得到1919列，反映某一物品对其他所有物品的相似度
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)

        return torch.mean(cl_loss)

