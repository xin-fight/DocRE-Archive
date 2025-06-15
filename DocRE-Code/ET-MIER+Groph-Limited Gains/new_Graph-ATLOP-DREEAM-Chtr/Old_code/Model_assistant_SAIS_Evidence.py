import torch.nn as nn
from opt_einsum import contract
from constant_SAIS_Evidence import *


def Use_Cls_Info(cls_mask, sequence_output, Relation_Specific_num):
    """cls_info [batch_size, num_labels, reduced_dim] 对每个实体，每个label情况下，都加入一个整个文档的CLS嵌入"""
    # shape:[batch_size,max_sent_cnt,hidden_size] = [batch_size,max_sent_cnt] * [batch_size,max_sent_cnt,hidden_size]
    keep_mask = cls_mask.unsqueeze(-1)
    cls_rep = sequence_output * keep_mask
    # [batch_size, 1, reduced_dim]
    cls_rep = torch.mean(cls_rep, dim=1, keepdim=True)
    # TODO: 是否是否往特征里面加入CLS信息
    if docred_ent2id_NA:
        cls_info = cls_rep.repeat(1, Relation_Specific_num + 1, 1)  # cls_info [batch_size, Relation_Specific_num, reduced_dim]
    else:
        cls_info = cls_rep.repeat(1, Relation_Specific_num, 1)  # cls_info [batch_size, Relation_Specific_num, reduced_dim]
    return cls_info


def Rel_Mutil_men2ent(sequence_output, cls_mask, attention, entity_pos, hts, offset, relation_vector, reduced_dim,
                      Relation_Specific_num, dropout, tag):
    """
    relation_vector: Relation_Specific_num * relation_dim
    """
    n, h, _, c = attention.size()
    hss, tss, rss, Crs = [], [], [], []
    ht_atts, bs_ht_num = [], [0]

    if use_cls_info:
        """cls_info [batch_size, Relation_Specific_num, reduced_dim] 对每个实体，每个label情况下，都加入一个整个文档的CLS嵌入"""
        cls_info = Use_Cls_Info(cls_mask, sequence_output, Relation_Specific_num)

    # len(entity_pos) 为bs, i为其中的一篇文档
    for i in range(len(entity_pos)):
        entity_embs, entity_atts = [], []
        # obtain entity embedding from mention embeddings.
        """TODO:修复BUG 为了保持ATLOP_DREEAM_Chtr一致"""
        if not entity_pos_isAA:
            entity_pos_AA = entity_pos[i][:-1]  # 与AA不同，不包含anaphor的嵌入 —— 没有图 or 有图但想用高效果时的emb
        else:
            entity_pos_AA = entity_pos[i]  # 与AA同，包含anaphor效果低

        for eid, e in enumerate(entity_pos_AA):  # for each entity
            if len(e) > 1:  # 该实体有多个提及
                """e_emb (num_men, reduced_dim), e_att(num_men, num_heads, sequence_length)实体的嵌入和实体对于其他单词的注意力"""
                e_emb, e_att = [], []
                for start, end in e:
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        # 如果由于最大seq长度有限，实体名称被截断。
                        """使用 实体前的* 来代表实体"""
                        e_emb.append(sequence_output[i, start + offset])
                        e_att.append(attention[i, :, start + offset])

                if len(e_emb) > 0:  # 将原来logsumexp融合改成现在的基于关系融合
                    """原始 (1, 768)  (1, H, len)"""
                    e_emb_logsumexp = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0).unsqueeze(0)
                    e_att_mean = torch.stack(e_att, dim=0).mean(0).unsqueeze(0)

                    # 某实体下的所有提及 [men_cnt, reduced_dim]
                    men_rep = torch.stack(e_emb, dim=0)

                    """1. get relation-specific emb  将关系特征融入实体嵌入 RSMAN 公式3，4"""
                    # reduced_to_relation(men_rep) [men_cnt, relation_dim] * relation_vector.transpose [relation_dim, Relation_Specific_num]
                    # men_attention: [men_cnt, Relation_Specific_num]
                    men_attention = torch.matmul(nn.Tanh()(men_rep), relation_vector.transpose(0, 1).contiguous())
                    # shape:[men_cnt, Relation_Specific_num] - [Relation_Specific_num, men_cnt] 利用语义相关性得到注意力权重
                    men_attention = nn.Softmax(dim=0)(men_attention).permute(1, 0).contiguous()
                    """get relation-specific rep for each entity  实体$e_i^r$特定关系r的表示 RSMAN公式5"""
                    # [R, men_cnt] * [men_cnt, reduced_dim] = shape:[Relation_Specific_num, reduced_dim]
                    e_emb = torch.matmul(men_attention, men_rep)

                    if docred_ent2id_NA:
                        e_emb = torch.cat((e_emb_logsumexp, e_emb), dim=0)

                    """2. get relation-specific attention
                    但是由于attention长度适合每个batch有关系，与 关系向量 维度不匹配  所以 直接用 men嵌入与relation计算的权重
                    """
                    e_att = torch.stack(e_att, dim=0)
                    # [Relation_Specific_num, men_cnt] * [men_cnt, head, length] -> [Relation_Specific_num, head, length]
                    e_att = torch.matmul(men_attention, e_att.reshape(men_attention.shape[1], -1)).reshape(
                        Relation_Specific_num, h, -1)

                    if docred_ent2id_NA:
                        e_att = torch.cat((e_att_mean, e_att), dim=0)

                else:
                    if docred_ent2id_NA:
                        # TODO: e_emb [Relation_Specific_num, reduced_dim]
                        e_emb = torch.zeros(Relation_Specific_num + 1, reduced_dim).to(sequence_output)
                        e_att = torch.zeros(Relation_Specific_num + 1, h, c).to(attention)
                    else:
                        # TODO: e_emb [Relation_Specific_num, reduced_dim]
                        e_emb = torch.zeros(Relation_Specific_num, reduced_dim).to(sequence_output)
                        e_att = torch.zeros(Relation_Specific_num, h, c).to(attention)

            else:
                """实体下单提及"""
                start, end = e[0]
                if start + offset < c:
                    e_emb = sequence_output[i, start + offset]
                    e_att = attention[i, :, start + offset]

                    """[1, 768] [1, h, len]"""
                    e_emb_o = e_emb.unsqueeze(0)
                    e_att_o = e_att.unsqueeze(0)

                    """处理实体只有一个提及的情况，要变成[Relation_Specific_num, reduced_dim]"""
                    if one_mention_copy_or_addrel:
                        # TODO: 1. 直接用该提及嵌入扩充，但这样得到的是相同的嵌入
                        e_emb = e_emb.expand(Relation_Specific_num, -1)
                    else:
                        # TODO: 2. 用该提及嵌入直接和关系嵌入相加，得到不同的嵌入
                        men_rep = e_emb_o + relation_vector  # [Relation_Specific_num, reduced_dim]
                        # shape:[Relation_Specific_num, Relation_Specific_num]
                        men_attention = torch.matmul(nn.Tanh()(men_rep), relation_vector.transpose(0, 1).contiguous())
                        # shape:[Relation_Specific_num, Relation_Specific_num]   利用语义相关性得到注意力权重
                        men_attention = nn.Softmax(dim=0)(men_attention).permute(1, 0).contiguous()

                        # shape:[Relation_Specific_num, reduced_dim]
                        e_emb = torch.matmul(men_attention, men_rep)
                    if docred_ent2id_NA:
                        e_emb = torch.cat((e_emb_o, e_emb), dim=0)

                    if one_mention_copy_or_addrel:
                        # TODO: 1. 直接用该提及att扩充，但这样得到的是相同的att [Relation_Specific_num, h, len]
                        e_att = e_att.expand(Relation_Specific_num, h, -1)
                    else:
                        # TODO: 2. 使用该提及（emn+Rel）与Rel嵌入得到的权重
                        """利用伯努利分布进行噪声引入 —— ERA"""
                        # [Relation_Specific_num, h, len]
                        e_att = e_att.expand(Relation_Specific_num, h, -1)

                        # if tag in ["train"]:
                        if tag in ["train", "test"]:
                            p = torch.bernoulli(torch.rand_like(e_att))
                            e_att = e_att * p

                        # [Relation_Specific_num, Relation_Specific_num] * [Relation_Specific_num, head, length] ->
                        # [Relation_Specific_num, head, length]
                        e_att = torch.matmul(men_attention, e_att.reshape(men_attention.shape[1], -1)).reshape(
                            Relation_Specific_num, h, -1)
                    if docred_ent2id_NA:
                        e_att = torch.cat((e_att_o, e_att), dim=0)
                    else:
                        pass

                else:
                    # TODO: e_emb [Relation_Specific_num, reduced_dim]
                    if docred_ent2id_NA:
                        e_emb = torch.zeros(Relation_Specific_num + 1, reduced_dim).to(sequence_output)
                        e_att = torch.zeros(Relation_Specific_num + 1, h, c).to(attention)
                    else:
                        e_emb = torch.zeros(Relation_Specific_num, reduced_dim).to(sequence_output)
                        e_att = torch.zeros(Relation_Specific_num, h, c).to(attention)

            if use_cls_info:
                # TODO: 为了测试，加入CLS信息    [Relation_Specific_num, reduced_dim*2]
                e_emb = torch.cat([e_emb, cls_info[i]], dim=-1)
                e_emb = dropout(e_emb)
                e_att = dropout(e_att)

            entity_embs.append(e_emb)
            entity_atts.append(e_att)

        # 包含该文档中的所有实体
        entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, Relation_Specific_num, d]
        entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, Relation_Specific_num, h, seq_len]

        # hs,ts - (实体对数, Relation_Specific_num, d)  ht_i：该bs包含了实体和除了本身之前的 所有实体对
        ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
        hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
        ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

        # h_att, t_att - [实体对数, Relation_Specific_num, num_heads, sequence_length]
        h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
        t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

        ht_att = (h_att * t_att).mean(2)  # average over all heads
        # [实体对数, Relation_Specific_num, sequence_length]  归一化，使它们的总和等于 1
        ht_att = ht_att / (ht_att.sum(2, keepdim=True) + 1e-30)
        # obtain local context embeddings.  [实体对数, Relation_Specific_num, reduced_dim]
        rs = contract("ld,nal->nad", sequence_output[i], ht_att)

        """3. Relation-Specific Entity Pair Context Embedding
        以r1为例，1）现在要将所有Csor1融合成Cr1 或者  2）只将该bs下属于该关系r1的Csor1 融合
        法2）
        """
        if docred_ent2id_NA:
            rs_Csor = rs[:, 1:, :]  # [实体对数, Relation_Specific_num, reduced_dim]
        else:
            rs_Csor = rs  # [实体对数, Relation_Specific_num, reduced_dim]

        # TODO: TESTING 之后可以测试一下哪个效果好
        # # reduced_to_relation(rs) [实体对数, Relation_Specific_num, relation_dim] * relation_vector.transpose [relation_dim, Relation_Specific_num]
        # # Cso_attention: [实体对数, Relation_Specific_num, Relation_Specific_num]
        # Cso_attention = torch.matmul(nn.Tanh()(self.reduced_to_relation(rs_Csor)), self.relation_vector.transpose(0, 1).contiguous())
        # # Cso_attention: [实体对数, Relation_Specific_num]
        # Cso_attention = torch.diagonal(Cso_attention, dim1=-2, dim2=-1)

        # [实体对数, Relation_Specific_num, reduced_dim] * [Relation_Specific_num, relation_dim]
        # 计算Csor 与 r的相似度 Cso_attention: [实体对数, Relation_Specific_num]
        Cso_attention = (nn.Tanh()(rs_Csor) * relation_vector.unsqueeze(0)).sum(dim=2)

        # shape:[实体对数, Relation_Specific_num] - [Relation_Specific_num, 实体对数]   利用语义相关性得到注意力权重
        Cso_attention = nn.Softmax(dim=0)(Cso_attention).permute(1, 0).contiguous()
        # shape:[Relation_Specific_num, Relation_Specific_num, reduced_dim]
        Cr_rep_spec = torch.matmul(Cso_attention, rs_Csor.reshape(rs_Csor.shape[0], -1)).reshape(Relation_Specific_num,
                                                                                                 Relation_Specific_num,
                                                                                                 -1)

        # Cr_rep_spec: [Relation_Specific_num, reduced_dim]
        Cr_rep_spec = Cr_rep_spec.permute(2, 0, 1)
        Cr_rep_spec = torch.diagonal(Cr_rep_spec, dim1=-2, dim2=-1)
        Cr_rep_spec = Cr_rep_spec.permute(1, 0)

        hss.append(hs)  # hs,ts - (该bs中所有文档实体对数, Relation_Specific_num, d)
        tss.append(ts)
        rss.append(rs)
        Crs.append(Cr_rep_spec)
        ht_atts.append(ht_att)  # 头尾实体对文档关注[ [实体对数, Relation_Specific_num, sequence_length] * bs ]

        bs_ht_num.append(bs_ht_num[-1] + len(hs))

    return hss, tss, rss, Crs, ht_atts, bs_ht_num


def Mutil_men2ent_O(entity_pos, sequence_output, attention, offset, hts, config):
    """
    Get head, tail, context embeddings from token embeddings.
    Inputs:
        :sequence_output: (batch_size, doc_len, hidden_dim)
        :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
        :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
        :offset: 1 for bert and roberta. Offset caused by [CLS] token.
    Outputs:
        :hss: (num_ent_pairs_all_batches, emb_size)
        :tss: (num_ent_pairs_all_batches, emb_size)
        :rss: (num_ent_pairs_all_batches, emb_size)
        :ht_atts: (num_ent_pairs_all_batches, doc_len)
        :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
    """
    n, h, _, c = attention.size()
    hss, tss, rss = [], [], []
    ht_atts = []

    for i in range(len(entity_pos)):  # for each batch
        entity_embs, entity_atts = [], []

        # obtain entity embedding from mention embeddings.
        for eid, e in enumerate(entity_pos[i]):  # for each entity
            if len(e) > 1:
                e_emb, e_att = [], []
                for mid, (start, end) in enumerate(e):  # for every mention
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        e_emb.append(sequence_output[i, start + offset])
                        e_att.append(attention[i, :, start + offset])

                if len(e_emb) > 0:
                    e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_emb = torch.zeros(config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)
            else:
                start, end = e[0]
                if start + offset < c:
                    e_emb = sequence_output[i, start + offset]
                    e_att = attention[i, :, start + offset]
                else:
                    e_emb = torch.zeros(config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)

            entity_embs.append(e_emb)
            entity_atts.append(e_att)

        entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
        entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

        ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

        # obtain subject/object (head/tail) embeddings from entity embeddings.
        hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
        ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

        h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
        t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

        ht_att = (h_att * t_att).mean(1)  # average over all heads
        ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
        ht_atts.append(ht_att)

        # obtain local context embeddings.
        rs = contract("ld,rl->rd", sequence_output[i], ht_att)

        hss.append(hs)
        tss.append(ts)
        rss.append(rs)

    return hss, tss, rss, ht_atts
