import math
import torch
import pickle
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses_SAIS_Evidence import ATLoss, SAISLoss, MLLTRSCLloss, Relation_Specificloss
from constant_SAIS_Evidence import *
from Model_assistant_SAIS_Evidence import *
from prepro import docred_ent2id


class DocREModel(nn.Module):
    def __init__(self, config, model, tokenizer,
                 emb_size=768, block_size=64, num_labels=-1,
                 max_sent_num=25, evi_thresh=0.2):
        """
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        # TODO: 是否使用后三层注意力
        self.three_atten = three_atten

        # TODO: 是否使用cls_info
        self.use_cls_info = use_cls_info

        # TODO: 加入RSMAN中的关系嵌入
        self.dropout = nn.Dropout(0.1)
        self.reduced_dim = reduced_dim
        self.relation_dim = relation_dim

        # TODO: 是否使用use_bilinear
        self.use_bilinear = use_bilinear

        # TODO: 关系特定的特征 中 关系的数量
        self.Relation_Specific_num = Relation_Specific_num

        # TODO: 是否使用对比学习，将实体对嵌入与其Relation_Specific上下文拉近
        self.Rel_Spe_loss = Relation_Specificloss()

        # TODO: 是否使用嵌入对比学习，PEMSCL SCL:让同一类下的实体对嵌入接近
        self.SCL_loss = MLLTRSCLloss(tau=tau, tau_base=tau_base)
        self.lambda_3 = lambda_3

        self.extractor_dim = extractor_dim

        if not self.use_cls_info:
            # 由于嵌入维度变了，这里也要改 2self.reduced_dim * relation_dim
            self.head_extractor = nn.Linear(2 * self.reduced_dim, self.extractor_dim)
            self.tail_extractor = nn.Linear(2 * self.reduced_dim, self.extractor_dim)
        else:
            self.cls_info_head_extractor = nn.Linear(3 * self.reduced_dim, self.extractor_dim)
            self.cls_info_tail_extractor = nn.Linear(3 * self.reduced_dim, self.extractor_dim)

        """bilinear classifier  第一个维度要与head_extractor最后一个维度相关"""
        if self.use_bilinear:
            # extractor_dim*block_size * 97
            self.bilinear = nn.Linear(self.extractor_dim * block_size, config.num_labels)
        else:
            # TODO: 按照RSMAN实现bilinear classifier
            # Relation_Specific_num * 2reduced_dim * 2reduced_dim
            self.classifier = nn.Parameter(
                torch.randn(self.Relation_Specific_num, self.reduced_dim * 2, self.reduced_dim * 2))
            # Relation_Specific_num
            self.classifier_bais = nn.Parameter(torch.randn(self.Relation_Specific_num))

            nn.init.uniform_(self.classifier, a=-math.sqrt(1 / (2 * self.reduced_dim)),
                             b=math.sqrt(1 / (2 * self.reduced_dim)))
            nn.init.uniform_(self.classifier_bais, a=-math.sqrt(1 / (2 * self.reduced_dim)),
                             b=math.sqrt(1 / (2 * self.reduced_dim)))

        """Relation Prototypes"""
        # Relation_Specific_num * relation_dim
        self.relation_vector = nn.Parameter(torch.randn(self.Relation_Specific_num, self.relation_dim))
        nn.init.xavier_normal_(self.relation_vector)

        """SAIS - ET"""
        # if TASK_ET:
        self.SAISLoss = SAISLoss()
        # self.ET_predictor_module = nn.Linear(reduced_dim, len(docred_ent2id))
        self.ET_predictor_module = nn.Linear(reduced_dim, Relation_Specific_num+1)

        """清空中间结果，分析证据句"""
        with open('env_sent.pkl', 'wb') as f:
            pickle.dump([], f)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens,
                                                        self.three_atten)

        return sequence_output, attention

    def get_hrt(self, sequence_output, cls_mask, attention, entity_pos, hts, offset, tag):
        """正常使用：将实体下的提及特征融合，得到头尾实体特征以及上下文嵌入"""
        # hss, tss, rss, ht_atts = Mutil_men2ent_O(entity_pos, sequence_output, attention, offset,  hts, self.config)
        # rels_per_batch = [len(b) for b in hss]  # 该bs下每个文档中要预测实体对个数
        # hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, R+1, emb_size)
        # tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, R+1, emb_size)
        # rss = torch.cat(rss, dim=0)  # [num_ent_pairs_all_batches, R+1, R+1, reduced_dim] mean+R
        # rs_old = torch.cat(rss, dim=0)  # [num_ent_pairs_all_batches, R+1, reduced_dim] mean+R
        # ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, R+1, R+1, max_doc_len)
        # ht_atts_old = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, R+1, max_doc_len)
        # return hss, rss, tss, ht_atts, rels_per_batch

        # hss, tss, rss, Crs, ht_atts, bs_ht_num = Rel_Mutil_men2ent(sequence_output, cls_mask, attention, entity_pos,
        #                                                            hts, offset, self.relation_vector, self.reduced_dim,
        #                                                            self.Relation_Specific_num, self.dropout, tag)
        hss, tss, rss, rss_old, Crs, ht_atts, ht_atts_old, bs_ht_num = Rel_Mutil_men2ent(sequence_output, cls_mask, attention, entity_pos,
                                                                   hts, offset, self.relation_vector, self.reduced_dim,
                                                                   self.Relation_Specific_num, self.dropout, tag)
        rels_per_batch = [len(b) for b in hss]  # 该bs下每个文档中要预测实体对个数
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, R+1, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, R+1, emb_size)
        rss = torch.cat(rss, dim=0)  # localized context embedding   (num_ent_pairs_all_batches, R+1, R+1, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # 头尾实体对文档关注 (num_ent_pairs_all_batches, R+1, R+1, max_doc_len)

        if add_old:
            rss_old = torch.cat(rss_old, dim=0)  # localized context embedding   (num_ent_pairs_all_batches, R+1, R+1, emb_size)
            ht_atts_old = torch.cat(ht_atts_old, dim=0)  # 头尾实体对文档关注 (num_ent_pairs_all_batches, R+1, R+1, max_doc_len)
        else:
            rss_old = torch.tensor([])
            ht_atts_old = torch.tensor([])

        return hss, rss, rss_old, tss, Crs, bs_ht_num, ht_atts, ht_atts_old, rels_per_batch

    def forward_rel(self, hs, ts):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, 1 / Topk+1, emb_size)
            :ts: (num_ent_pairs_all_batches, 1 / Topk+1, emb_size)
            :rs: (num_ent_pairs_all_batches, R+1, R+1, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
            :SCL_bl: (所有文档实体对数, extractor_dim * block_size) 实体对的嵌入
        '''
        # split into several groups.
        b1 = hs.view(-1, hs.shape[1], self.extractor_dim // self.block_size, self.block_size)
        b2 = ts.view(-1, ts.shape[1], self.extractor_dim // self.block_size, self.block_size)
        # [num_ent_pairs_all_batches, 1 / Topk+1, self.extractor_dim * self.block_size]
        bl = (b1.unsqueeze(4) * b2.unsqueeze(3)).view(-1, hs.shape[1], self.extractor_dim * self.block_size)
        logits = self.bilinear(bl)  # logits (所有文档实体对数, 1 / Topk+1, labels - 关系的个数97)

        # (所有文档实体对数, extractor_dim * block_size)
        SCL_bl = bl.mean(1)

        """对预测结果中的Relation_Specific_num进行处理"""
        logits = logits.mean(1)  # (所有文档实体对数, label)
        # logits = logits[:, 0, :]  # (所有文档实体对数, label)

        return logits, SCL_bl

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. bs中每个文档下实体对数 Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''

        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):  # for each batch
            # 取出该bs下的实体对的atten  atten the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]
            # 该文档中句子的范围
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]  # + offset

            # 根据句子的范围取出该范围下的atten [所有实体对 对于 该句子的关注 * 句子数]
            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            # 进行扩充得到该bs下最大句子长度
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            # [num_ent_pairs_all_batches, max_sent_num] 将实体对对于某个句子下 所有的atten进行相加
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1)  # sum across those attentions
            rel_sent_attn.append(sum_attn)

        # (num_ent_pairs_all_batches, max_sent_all_batch) 每个实体对的句子级证据分布 实体对对于某句子下所有token关注之和
        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def get_hrt_logsumexp(self, sequence_output, attention, entity_pos, hts, offset):
        n, h, _, c = attention.size()
        ht_atts = []

        for i in range(len(entity_pos)):  # for each batch
            entity_atts = []
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1:
                    e_att = []
                    for mid, (start, end) in enumerate(e):  # for every mention
                        if start + offset < c:
                            e_att.append(attention[i, :, start + offset])

                    if len(e_att) > 0:
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_att = attention[i, :, start + offset]
                    else:
                        e_att = torch.zeros(h, c).to(attention)

                entity_atts.append(e_att)

            entity_atts = torch.stack(entity_atts, dim=0)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)  # average over all heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

        rels_per_batch = [len(b) for b in ht_atts]
        ht_atts = torch.cat(ht_atts, dim=0)

        return ht_atts, rels_per_batch


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,  # relation labels
                entity_pos=None,
                hts=None,  # entity pairs
                sent_pos=None,
                sent_labels=None,  # evidence labels (0/1)
                teacher_attns=None,  # evidence distribution from teacher model
                tag="train",
                epair_types=None,
                ):
        # batch_epair_types [该bs下实体对个数(1310)，2(表示头尾实体类型)]

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        if tag in enhance_evi:
            doc_attn_logsumexp, batch_rel_logsumexp = self.get_hrt_logsumexp(sequence_output, attention, entity_pos, hts, offset)

        # rs - localized context embedding (num_ent_pairs_all_batches, R+1, R+1, emb_size)，表明该实体对认为重要的上下文嵌入 mean+R
        # rs_old - localized context embedding (num_ent_pairs_all_batches, R+1, emb_size)，表明该实体对认为重要的上下文嵌入 mean+R
        # hs, ts - (该bs中所有文档实体对数, R+1, reduced_dim), 头实体和尾实体对应的向量集合 pooling+R
        # Crs - [ [实体对数, R, R, reduced_dim] * bs ] 关系 所关注的上下文 --- 用于Crht_loss
        # doc_attn 文档中实体对 对于整个文档的注意力      (num_ent_pairs_all_batches, R+1, R+1, max_doc_len) mean+R
        # doc_attn_old 文档中实体对 对于整个文档的注意力      (num_ent_pairs_all_batches, R+1, max_doc_len) mean+R
        # bs_ht_num,   batch_rel 该bs下每个文档中要预测实体对个数
        # hs, rs, ts, Crs, bs_ht_num, doc_attn, batch_rel = self.get_hrt(sequence_output, attention_mask, attention, entity_pos, hts, offset, tag)
        hs, rs, rs_old, ts, Crs, bs_ht_num, doc_attn, doc_attn_old, batch_rel = self.get_hrt(sequence_output, attention_mask, attention, entity_pos, hts, offset, tag)

        old_hs = hs.clone()  # (该bs中所有文档实体对数, R+1, reduced_dim)
        old_ts = ts.clone()
        # 原始
        doc_attn_old_mean = doc_attn_old.mean(1)  # [1310, 7, 360] -> [1310, 360]

        """最新 [1310, 7, 7, 360] -> [1310, 360]      epair_types [1310, 2]"""
        # doc_attn_sel = []
        # for sample in range(epair_types.shape[0]):
        #     doc_attn_sel.append(doc_attn[sample][epair_types[sample][0]][epair_types[sample][1]].unsqueeze(0))
        # doc_attn = torch.cat(doc_attn_sel, dim=0)  # [1310, 360]
        sample_indices = torch.arange(epair_types.shape[0])
        doc_attn = doc_attn[sample_indices, epair_types[:, 0], epair_types[:, 1], :]  # [1310, 360]

        """是否在构造新向量时，使用之前的向量"""
        if add_old:
            doc_attn = (doc_attn_old_mean + doc_attn) / 2

        if TASK_ET:
            # hs  ts   (1310, R+1, reduced_dim)
            # rs  (1310, R+1, R+1, emb_size)
            # epair_types  [该bs下实体对个数(1310)，2(表示头尾实体类型)]
            # rsh, rst = [], []
            # for sample in range(epair_types.shape[0]):
            #     rsh.append(rs[sample, :, epair_types[sample][1], :].unsqueeze(0))
            #     rst.append(rs[sample, epair_types[sample][0], :, :].unsqueeze(0))
            # rsh = torch.cat(rsh, dim=0)  # (1310, R+1, reduced_dim)
            # rst = torch.cat(rst, dim=0)
            """头或者尾 正确的类型 重要的上下文嵌入  (1310, R+1, emb_size)"""
            sample_indices = torch.arange(epair_types.shape[0])
            rsh = rs[sample_indices, :, epair_types[:, 1], :]  # t的类型选择正确的
            rst = rs[sample_indices, epair_types[:, 0], :, :]  # h的类型选择正确的

            """是否在构造新向量时，使用之前的向量"""
            if add_old:
                rsh = (rsh + rs_old) / 2
                rst = (rst + rs_old) / 2

            """hs ts 融合上下文构建最终的实体特征  (1310, R+1, emb_size) cat (1310, R+1, emb_size)"""
            if not change_rsh:
                hs = torch.tanh(self.head_extractor(torch.cat([hs, rsh], dim=-1)))  # dim维度拼接
                ts = torch.tanh(self.tail_extractor(torch.cat([ts, rst], dim=-1)))
            else:
                hs = torch.tanh(self.head_extractor(torch.cat([rsh, hs], dim=-1)))  # dim维度拼接
                ts = torch.tanh(self.tail_extractor(torch.cat([rst, ts], dim=-1)))

            """对实体对中所有实体进行 实体类型分类"""
            # hs, ts - (该bs中所有文档实体对数 (1310), R (7), reduced_dim), 头实体和尾实体对应的向量集合
            # batch_ET_reps [2620,7,768]
            if not ET_hs_old:  # hs ts 融合上下文构建最终的实体特征
                batch_ET_reps = self.SAISLoss.get_ET_reps([hs, ts])  # [2620, 7, 768]
            else:  # old不好含上下文
                batch_ET_reps = self.SAISLoss.get_ET_reps([old_hs, old_ts])  # [2620, 7, 768]

            batch_ET_reps = self.ET_predictor_module(batch_ET_reps)  # [2620, 7, 7]
            batch_ET_reps = torch.diagonal(batch_ET_reps, dim1=-2, dim2=-1)  # 该bs中所有文档实体对数*2, R+1

            """batch_ET_reps 由 pooling+R 组成，去除pooling影响
            batch_ET_reps 由 pooling+R 组成，但类型分类时之关心最后R产生的类别，和pooling无关
            batch_ET_reps_del  [2620, 6]
            """
            batch_ET_reps_del = batch_ET_reps[:, 1:]  # [2620, 6]

            """[1310, 7, 768] -> [1310, 1, 768]"""
            if tag in TSER_topk_tag:
                """选取Topk"""
                # batch_ET_reps_del (2620, 6)       取出最有可能的前k种类型    该bs中所有文档实体对数*2, R
                # top_indices in (0, 5)
                top_values, top_indices = torch.topk(batch_ET_reps_del, k=Top_k, dim=1)
                """类型真实的位置 = top_indices + 1"""
                top_indices += 1  # top_indices in (1, 6), hs中0号位置为pooling

                # mask (2620, 7)
                mask = torch.zeros_like(batch_ET_reps).to(batch_ET_reps)
                mask.scatter_(1, top_indices, 1)  # 将每行最大值对应的索引处设为 1
                # h_mask, t_mask: [1310, 7, 768]  最大索引处为True
                h_mask, t_mask = torch.chunk(mask.bool().unsqueeze(2).repeat(1, 1, hs.shape[2]), chunks=2, dim=0)

                # [1310, 7, 768] -> [1310, 1, 768]
                hs_new = torch.masked_select(hs, h_mask).reshape(hs.shape[0], Top_k, hs.shape[2])
                ts_new = torch.masked_select(ts, t_mask).reshape(ts.shape[0], Top_k, ts.shape[2])
            else:
                """直接使用数据集标记"""
                epair_types_h = epair_types[:, 0].view(epair_types.shape[0], 1, 1).expand(-1, -1, hs.shape[2])
                epair_types_t = epair_types[:, 1].view(epair_types.shape[0], 1, 1).expand(-1, -1, ts.shape[2])

                # [1310, 7, 768] -> [1310, 1, 768]
                hs_new = torch.gather(hs, 1, epair_types_h)
                ts_new = torch.gather(ts, 1, epair_types_t)

            if Hs_Ts_list == "TSER_Mean_Ground":  # R_mean+top/正确type
                hs = torch.cat((torch.mean(hs, dim=1, keepdim=True), hs_new), dim=1)  # [1310, Topk+1, 768]
                ts = torch.cat((torch.mean(ts, dim=1, keepdim=True), ts_new), dim=1)
            elif Hs_Ts_list == "Type_Ground_True":  # 使用top/正确的type
                hs = hs_new
                ts = ts_new
            elif Hs_Ts_list == "Mean":  # R_mean
                hs = torch.mean(hs, dim=1, keepdim=True)  # (该bs中所有文档实体对数, 1, reduced_dim)
                ts = torch.mean(ts, dim=1, keepdim=True)

            # """利用之前计算的rs和现在的rs一起计算"""
            # if add_old:
            #     batch_ET_reps_old = self.SAISLoss.get_ET_reps([old_hs, old_ts])
            #     batch_ET_reps_old = self.ET_predictor_module(batch_ET_reps_old)
            #     batch_ET_reps_old = torch.diagonal(batch_ET_reps_old, dim1=-2, dim2=-1)  # 该bs中所有文档实体对数*2, Relation_Specific_num
            #
            #     # 取出最有可能的前两种类型    该bs中所有文档实体对数*2, Relation_Specific_num
            #     top_values_old, top_indices_old = torch.topk(batch_ET_reps_old, k=Top_k, dim=1)
            #     mask_old = torch.zeros_like(batch_ET_reps_old).to(batch_ET_reps_old)
            #     mask_old.scatter_(1, top_indices_old, 1)  # 将每行最大值对应的索引处设为 1
            #     h_mask_old, t_mask_old = torch.chunk(mask_old.bool().unsqueeze(2).repeat(1, 1, old_hs.shape[2]), chunks=2, dim=0)
            #
            #     # # [1310, 7, 768] -> [1310, 1, 768]
            #     # hs_new = torch.masked_select(hs, h_mask).reshape(hs.shape[0], Top_k, hs.shape[2])
            #     # ts_new = torch.masked_select(ts, t_mask).reshape(ts.shape[0], Top_k, ts.shape[2])
            #
            #     ht_mask_old = (h_mask_old | t_mask_old)[:, :, 0].sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, rs_old.shape[2])  # [1310, 1, 768]
            #     # rs_new   (num_ent_pairs_all_batches, 1, emb_size)，表明该实体对认为重要的上下文嵌入
            #     rs_new = rs_old.sum(dim=1, keepdim=True) / ht_mask_old
            #     rs_old = torch.cat((torch.mean(rs_old, dim=1, keepdim=True), rs_new), dim=1)  # [1310, 2, 768]


        # logits (所有文档实体对数, label)            SCL_bl (所有文档实体对数, extractor_dim * block_size)
        # hs ts   (num_ent_pairs_all_batches, 1/Topk+1, emb_size)
        # rs  (num_ent_pairs_all_batches, R+1, R+1, emb_size)，表明该实体对认为重要的上下文嵌入
        logits, SCL_bl = self.forward_rel(hs, ts)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if sent_labels != None:  # human-annotated evidence available
            """doc_attn (num_ent_pairs, max_sent) 对于每个实体对的 句子级证据分布"""
            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
            if tag in enhance_evi:
                s_attn_logsumexp = self.forward_evi(doc_attn_logsumexp, sent_pos, batch_rel_logsumexp, offset)
                s_attn = s_attn_lambda * s_attn + s_attn_logsumexp_lambda * s_attn_logsumexp

            """模型预测的证据句, 预测每个实体对    evi_thresh=0.2"""
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

            # """保存中间结果，分析证据句"""
            # with open('env_sent.pkl', 'ab') as f:  # 使用追加模式打开文件
            #     pickle.dump([{'sent_labels': sent_labels, 's_attn': s_attn}], f)

        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        if tag in ["test", "dev", 'train']:  # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)  # Topk
            output["scores"] = scores_topk[0]  # Top-k 对应的分数
            output["topks"] = scores_topk[1]  # Top-k 对应的索引

        if tag == "infer":  # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else:  # training
            """论文中的adaptive-thresholding loss"""
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
            # print('atl: ', output["loss"], end="   ")

            if sent_labels != None:  # supervised training with human evidence
                """evidence retrieval loss (kldiv loss) 用人类证据$v^{(s,o)}$ 来指导 句子级别的重要性$p^{(s,o)}$"""
                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)  # 有label的那些实体对
                # s_attn (有关系的实体对个数, max_sent_all_batch) 句子级别证据句 - 每个实体对的 句子级证据分布
                s_attn = s_attn[idx_used]
                s_attn[s_attn == 0] = 1e-30

                # (有关系的实体对个数, max_sent_all_batch) 实体对所有关系的 evidence labels (0/1)
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30

                # 0.9 2.5016      1 2.5006     0.1  2.5164
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)  # KLDivLoss
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

            elif teacher_attns != None:  # self training with teacher attention
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

            if PEMSCLloss:
                """PEMSCL Supervised Contrastive Learning for Multi-Labels and Long-Tailed Relations"""
                # bl 公式14得到$x_{h,t}$ !!!
                # F.normalize - bl中的每个向量进行了单位L2范数规范化，使它们每行的L2范数（欧几里德范数）都变为1，计算相似性度量或正则化神经网络的权重时，非常有用
                scl_loss = self.SCL_loss(F.normalize(SCL_bl, dim=-1), labels)
                output["loss"]["pemscl_loss"] = scl_loss * self.lambda_3

            if Relation_Seploss:
                """Relation_Specific_loss  [实体对数, Relation_Specific_num, reduced_dim]"""
                # TODO: 1. 实体对的嵌入信息，直接用 hs*ts
                self.relation_seploss = self.Rel_Spe_loss(Crs, hs * ts, bs_ht_num, self.relation_vector)
                # TODO: 2. 实体对的嵌入用 bl 后进行维度变换

                output["loss"]["relation_seploss"] = self.relation_seploss * self.lambda_3

            if TASK_ET:
                """batch_ET_reps 由 pooling+R 组成，去除pooling影响
                batch_ET_reps 由 pooling+R 组成，但类型分类时之关心最后R产生的类别，和pooling无关
                batch_ET_reps_del  (2620, 6) in [0, 5]
                epair_types (2620, 2) in [1, 6]
                所以为了让cross_entropy计算，需要 epair_types - 1
                """
                etoss = self.SAISLoss.cal_ET_loss(batch_ET_reps_del, epair_types - 1)
                output["loss"]["etoss"] = etoss * loss_weight_ET

        return output
