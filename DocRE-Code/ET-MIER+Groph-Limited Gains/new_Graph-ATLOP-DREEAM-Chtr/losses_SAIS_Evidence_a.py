import torch
import torch.nn as nn
from constant_SAIS_Evidence_a import *
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)  # theshold is norelation
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]  # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):
        """获取Top-k"""
        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)  # 前 num_labels 个最大值
        else:
            return logits[:, 1] - logits[:, 0], 0


class ATLoss_ET(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss


class SAISLoss:

    def __init__(self):
        pass

    def cal_ET_loss(self, batch_ET_reps, batch_epair_types):
        # batch_ET_reps [2620, Specific_num, Specific_num] 头尾实体 类型预测的概率
        # batch_epair_types [该bs下实体对个数(1310)，2(表示头尾实体类型)] [[5, 5], ...*2034]

        batch_epair_types = batch_epair_types.T.flatten()  # [1310, 2] -> [2, 1310] -> [2620]:1-6 -> 0-5
        if not IS_high_batch_ET_loss:
            if docred_ent2id_NA:
                # TODO: 之前no_evi 61.729889是batch_epair_types-1， 但逻辑上是batch_epair_types
                batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types)
            else:
                # batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types - 1)
                batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types - 1)
        else:
            batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types - 1)

        return batch_ET_loss

    def get_ET_reps(self, batch_epair_reps):
        # batch_epair_reps [h - [实体对数, Specific_num, 768], t - [实体对数, Specific_num, 768]]   对头尾实体进行 预测 实体类型
        batch_ET_reps = torch.tanh(torch.cat(batch_epair_reps, dim=0)).to(batch_epair_reps[0])

        return batch_ET_reps


class MLLTRSCLloss(nn.Module):
    def __init__(self, tau=2.0, tau_base=1.0):
        super().__init__()
        self.tau = tau  # 对应公式9中的超参数
        self.tau_base = tau_base  # (self.tau / self.tau_base) 为公式11中的超参

    def forward(self, features, labels, weights=None):
        """
        :param features: 经过L2正则化的 实体对嵌入 $x_{h,t} (所有文档实体对数, hidden_size*block_size)
        :param labels: 真实标签(实体对数, num_labels - 关系的个数97)
        :param weights:
        :return:
        """
        labels = labels.long()
        # 对有关系的实体对进行标明
        label_mask = (labels[:, 0] != 1.)
        """mask_s - (实体对数, 实体对数) 每对样本之间是否存在共同的关系， 且同一个样本与自己之间的交集被排除在外"""
        # labels.unsqueeze(1)  (实体对数, 1, labels)  labels (实体对数, labels)
        # & (实体对数, 实体对数, labels) - (i, j, c) 表示第 i 个样本的类别标签与第 j 个样本的类别标签在类别 c 上是否同时为1。
        # torch.any (实体对数, 实体对数) 每个元素表示对应样本之间是否存在交集。
        # fill_diagonal_(0) 将对角线变成0 - 将自己与自己之前的交集排除
        mask_s = torch.any((labels.unsqueeze(1) & labels).bool(), dim=-1).float().fill_diagonal_(0)

        """sims (实体对数, 实体对数) 中的每一行都表示 计算$x_{h,t}$与其他实体对嵌入$x_d$的相似度"""
        # .mm  让features乘自己的转置 (实体对数, 实体对数)
        # .div 让每个元素都除以tau
        sims = torch.div(features.mm(features.T), self.tau)

        """The Log-Sum-Exp Trick for numerical stability"""
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        # https://zhuanlan.zhihu.com/p/153535799
        # 为了数值稳定性，记录每行最大值，且最大值一定出现在对角线上（因为自己与自己的相似对最大）
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        # 让sims中最大的相似对为0，即让对角线上元素为0 - !!! 最大的为0，让之后的exp操作可以进行，否则可能会出现上溢的情况
        # 此时减了max，之后会经过exp后，为了保证计算结果准确性，应该在经过exp结果上 *epx(max)，但是在公式中由于该logits被用到了分子和分母上，因此抵消了，结果也自然相同
        logits = sims - logits_max.detach()  # logits - (实体对数, 实体对数)

        """计算 supervised contrastive learning 公式8  (-log_prob1) - $L_{scl}^{h,t}$"""
        # (实体对数, 实体对数)  除了对角线，其他位置是全1
        logits_mask = torch.ones_like(mask_s).fill_diagonal_(0)
        # 计算公式8中 $exp(x_{h,t} * x_d)$  - (实体对数, 实体对数)
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob是公式8的计算结果
        # exp_logits.sum 计算分母，而公式中logits 的 exp 和 log 抵消了，所以分子是logits
        # log_prob - (实体对数, 实体对数)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # mask_s (实体对数, 实体对数) - 两个不同的实体对之间是否有共同的关系;
        # 公式8中的$|S_{h,t}|$ - denom (实体对数) - 与该实体对 有共同关系的 其他实体对的个数
        denom = mask_s.sum(1)
        denom[denom == 0] = 1  # avoid div by zero

        # 公式8拉近的有相同关系的实体对嵌入，而相同关系实体对集合保存在子集 $S_{h,t}$ 中
        log_prob1 = (mask_s * log_prob).sum(1) / denom

        """解决long-tail问题：如果该bs中找不到某实体对的正例，则让其远离其他所有实体对嵌入 - 公式9 (-log_prob2) - $L_{lt}^{h,t}$"""
        # mask_s - (实体对数, 实体对数) 实体对 与 其他实体对之间 是否有共同的关系
        # mask_s.sum ==0 说明该实体对 与 其他实体对之间 没有共同关系，也将该bs中找不到其对应的关系相同的正例，要用到$L_{lt}$
        log_prob2 = - torch.log(exp_logits.sum(-1, keepdim=True)) * (mask_s.sum(-1, keepdim=True) == 0)

        """计算公式10 L2"""
        # label_mask 对有关系的实体对进行标明 即公式10中的 $(h,t) 属于 B_p$
        # mean_log_prob_pos - (实体对数, 实体对数)
        mean_log_prob_pos = (log_prob1 + log_prob2) * label_mask

        loss = - (self.tau / self.tau_base) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class Relation_Specificloss(nn.Module):
    def __init__(self, tau=2.0, tau_base=1.0):
        super().__init__()
        self.tau = tau  # 对应公式9中的超参数
        self.tau_base = tau_base  # (self.tau / self.tau_base) 为公式11中的超参

    def logsumexploss(self, emb, mask):
        # (n*Relation_Specific_num, n*Relation_Specific_num)
        sims = torch.div(emb.mm(emb.T), self.tau)

        """The Log-Sum-Exp Trick for numerical stability"""
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()  # (n*Relation_Specific_num, n*Relation_Specific_num)

        """计算 supervised contrastive learning 公式8  (-log_prob1) - $L_{scl}^{h,t}$"""
        # (实体对数, 实体对数)  除了对角线，其他位置是全1
        logits_mask = torch.ones_like(mask).fill_diagonal_(0).to(emb)
        # 计算公式8中 $exp(x_{h,t} * x_d)$  - (实体对数, 实体对数)
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob是公式8的计算结果
        # exp_logits.sum 计算分母，而公式中logits 的 exp 和 log 抵消了，所以分子是logits
        # log_prob - (实体对数, 实体对数)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # mask_s (实体对数, 实体对数) - 两个不同的实体对之间是否有共同的关系;
        # 公式8中的$|S_{h,t}|$ - denom (实体对数) - 与该实体对 有共同关系的 其他实体对的个数
        denom = mask.sum(1)
        denom[denom == 0] = 1  # avoid div by zero

        # 公式8拉近的有相同关系的实体对嵌入，而相同关系实体对集合保存在子集 $S_{h,t}$ 中
        # (bs*Relation_Specific_num, bs*Relation_Specific_num)
        log_prob1 = (mask * log_prob).sum(1) / denom

        loss = - (self.tau / self.tau_base) * log_prob1
        loss = loss.mean()

        return loss

    def MY_logsumexploss(self, emb, mask):
        # (n*Relation_Specific_num, n*Relation_Specific_num)
        sims = torch.div(emb.mm(emb.T), self.tau)

        """The Log-Sum-Exp Trick for numerical stability"""
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()  # (n*Relation_Specific_num, n*Relation_Specific_num)

        """计算 supervised contrastive learning 公式8  (-log_prob1) - $L_{scl}^{h,t}$"""
        # (实体对数, 实体对数)  除了对角线，其他位置是全1
        logits_mask = torch.ones_like(mask).fill_diagonal_(0).to(emb)
        # 分母 计算公式8中 $exp(x_{h,t} * x_d)$  - (实体对数, 1)
        exp_logits_htd = torch.exp(logits) * logits_mask

        # 分子
        exp_logits_htp = torch.exp(logits) * mask

        # 公式8中的$|S_{h,t}|$ - denom (实体对数) - 与该实体对 有共同关系的 其他实体对的个数
        denom = mask.sum(1, keepdim=True)
        denom[denom == 0] = 1  # avoid div by zero

        # log_prob是公式8的计算结果
        # exp_logits.sum 计算分母，而公式中logits 的 exp 和 log 抵消了，所以分子是logits
        # log_prob - (实体对数, 1)
        log_prob = torch.log(exp_logits_htp.sum(1, keepdim=True)) - torch.log(
            exp_logits_htd.sum(1, keepdim=True)) - torch.log(denom)

        loss = - (self.tau / self.tau_base) * log_prob
        loss = loss.mean()

        return loss

    def forward(self, Crs, emb, bs_ht_num, relation_vector):
        """ PEMSCL
        :param Crs: relation上下文嵌入 [bs * (Relation_Specific_num, reduced_dim)]
        :param emb: 实体对嵌入 [该bs下文档实体数， Relation_Specific_num, reduced_dim)]
        :param bs_ht_num: [0, 132, ...]
        :param relation_vector: Relation_Specific_num * relation_dim
        :return:
        """
        loss = 0

        """让特定于同一种关系的 上下文接近"""
        if Crht_loss:
            Relation_Specific_num, _ = Crs[0].shape
            bs = len(Crs)

            Crs = torch.cat(Crs, dim=0).to(emb)  # (bs * Relation_Specific_num, reduced_dim)

            """让同一关系的上下文接近"""
            # 创建一个 RelxRel 的单位矩阵, 之后进行拼接，得到同一关系下的mask
            unit_matrix = torch.eye(Relation_Specific_num)
            left_right_concat = torch.cat([unit_matrix] * bs, dim=1)
            mask_s = torch.cat([left_right_concat] * bs, dim=0).float().fill_diagonal_(0).to(emb)

            if Crht_MY_logsumexploss:
                loss_Crs = self.MY_logsumexploss(Crs, mask_s)
            else:
                loss_Crs = self.logsumexploss(Crs, mask_s)

            # print(loss_Crs, "   ", Crht_loss_lambda, "     ", loss_Crs * Crht_loss_lambda)
            del mask_s

            loss += loss_Crs * Crht_loss_lambda

        """让特定于同一种关系的 嵌入接近"""
        if Emb_loss:
            num_e, Relation_Specific_num, reduced_dim = emb.shape
            emb = emb.view(num_e * Relation_Specific_num, reduced_dim)  # (num_e * Relation_Specific_num, reduced_dim)

            """让同一关系的嵌入接近"""
            # 创建一个 RelxRel 的单位矩阵, 之后进行拼接，得到同一关系下的mask
            unit_matrix = torch.eye(Relation_Specific_num)
            left_right_concat = torch.cat([unit_matrix] * num_e, dim=1)
            mask_e = torch.cat([left_right_concat] * num_e, dim=0).float().fill_diagonal_(0).to(emb)

            if Emb_MY_logsumexploss:
                loss_emb = self.MY_logsumexploss(emb, mask_e)  # 计算过程中有inf
            else:
                loss_emb = self.logsumexploss(emb, mask_e)

            del mask_e
            loss += loss_emb * Emb_loss_lambda

        """让关系的嵌入 相互远离"""
        if Relation_loss:
            # Relation_Specific_num * relation_dim
            relation_vector_num, _ = relation_vector.shape
            sims = torch.div(relation_vector.mm(relation_vector.T), self.tau)

            """The Log-Sum-Exp Trick for numerical stability"""
            logits_max, _ = torch.max(sims, dim=1, keepdim=True)
            logits = sims - logits_max.detach()  # (n*Relation_Specific_num, n*Relation_Specific_num)

            # 去掉自己和自己
            logits_mask = torch.ones(relation_vector_num, relation_vector_num).fill_diagonal_(0).to(emb)
            exp_logits_htd = torch.exp(logits) * logits_mask

            # log_prob = torch.log(exp_logits_htd.sum(1, keepdim=True))
            log_prob = exp_logits_htd.sum(1, keepdim=True)

            loss += (self.tau / self.tau_base) * log_prob.mean()

        return loss
