from tqdm import tqdm
import ujson as json
import numpy as np
import pickle
import os

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}


def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''
    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        # add * marks to the beginning and end of entities
        new_map = {}

        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end

        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos


def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int,
                        tokenizer=None):
    ''' Construct pseudo documents from predictions.
    raw_feature {input_ids, entity_pos, relations, hts, sent_pos, sent_labels, sample['title']}
    pred_rels 为了fusion时 获取topk下所有结果 [{title, h_idx, t_idx, r, evidence, score}, ...]
    '''

    pos_samples = 0
    neg_samples = 0

    sent_grps = []
    pseudo_features = []

    # 对于该文档的Topk下某个预测，实际上只用到了evidence，对于某个实体对，会有topk个预测，但是每个预测的evidence是相同的
    for pred_rel in pred_rels:
        # pred_rel {title, h_idx, t_idx, r, evidence, score}
        curr_sents = pred_rel["evidence"]  # evidence sentence
        if len(curr_sents) == 0:
            continue

        # check if head/tail entity presents in evidence. if not, append sentence containing the first mention of head/tail into curr_sents
        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]])
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]])

        if len(set(head_sents) & set(curr_sents)) == 0:
            curr_sents.append(head_sents[0])
        if len(set(tail_sents) & set(curr_sents)) == 0:
            curr_sents.append(tail_sents[0])

        """对于某个实体对下Topk的evidence都是相同的"""
        curr_sents = sorted(set(curr_sents))  # 保存证据句
        if curr_sents in sent_grps:  # skip if such sentence group has already been created
            continue
        sent_grps.append(curr_sents)

        # 使用证据句构建伪文档 new sentence masks and input ids
        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents]  # 证据句的范围[(0, 62), (82, 119)]
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos]
        # 将所有子列表按顺序连接成一个单独的列表
        new_input_ids = sum(new_input_ids_each, [])
        new_input_ids = tokenizer.build_inputs_with_special_tokens(new_input_ids)  # 添加特殊标记

        # 将上述证据句拼接后获得最新的pos [(0, 62), (62, 99)]
        new_sent_pos = []
        prev_len = 0
        for sent in old_sent_pos:
            curr_sent_pos = (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        # 遍历所有实体，仅保留在curr_sents中提及的实体。 iterate through all entities, keep only entities with mention in curr_sents.
        # obtain entity positions w.r.t whole document
        curr_entities = []
        ent_new2old = {}  # 保存伪文档中的实体 对应 原文档哪个实体 head/tail of a relation should be selected
        new_entity_pos = []
        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                if mention["sent_id"] in curr_sents:  # curr_sents 保存证据句
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0]  # 找到提及所在句子的最新起始位置
                    # 提及新位置 = 新句子的位置 + 该提及的起始/终止位置距离所在原句子头的距离 sent_map update the start/end position of each token.
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in
                           mention['pos']]
                    curr_pos.append(pos)

            if curr != []:  # 说明当前实体有提及在伪文档中
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                # 保存伪文档中的实体 对应 原文档哪个实体
                ent_new2old[len(ent_new2old)] = i  # update dictionary

        # iterate through all entities to obtain all entity pairs and labels
        new_hts = []
        new_labels = []
        epair_types = []
        for h in range(len(curr_entities)):
            for t in range(len(curr_entities)):
                if h != t:
                    new_hts.append([h, t])
                    epair_types.append([docred_ent2id[entities[h][0]['type']], docred_ent2id[entities[t][0]['type']]])
                    old_h, old_t = ent_new2old[h], ent_new2old[t]
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])]
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0]
                    pos_samples += 1 - curr_label[0]

        pseudo_feature = {'input_ids': new_input_ids,  # 伪文档
                          'entity_pos': new_entity_pos,  # 记录实体提及在伪文档中位置
                          'labels': new_labels,  # 伪文档中实体对对应labels
                          'hts': new_hts,  # 在伪文档中的实体对
                          'sent_pos': new_sent_pos,  # 证据句拼接后获得最新的pos
                          'sent_labels': None,
                          'title': raw_feature['title'],
                          'entity_map': ent_new2old,  # 伪文档中的实体 对应 原文档哪个实体
                          'epair_types': epair_types,
                          }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples


def read_docred(file_in,
                tokenizer,
                transformer_type="bert",
                max_seq_length=1024,
                teacher_sig_path="",
                single_results=None):
    """
    teacher_sig_path 为了加载教师模型的atten
    single_results 为了fusion时 获取topk下所有结果 [{title:, h_idx, t_idx, r, evidence, score}, ...]
    """
    feature_file = './feature_file/' + file_in.split('.')[-2].split('/')[-1]
    feature_file += '_feature.pkl'
    if os.path.exists(feature_file) and teacher_sig_path == "" and single_results == None:
        print('Feature file:', feature_file)
        features = pickle.load(open(feature_file, "rb"))
        print('Feature loaded from', feature_file)
        return features

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    if teacher_sig_path != "":  # load logits
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        attns = pickle.load(open(attns_file, 'rb'))

    if single_results != None:
        # reorder predictions as relations by title 将相同title的预测放在一起
        pred_pos_samples = 0
        pred_neg_samples = 0
        pred_rels = single_results
        title2preds = {}
        for pred_rel in pred_rels:
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel)
            else:
                title2preds[pred_rel["title"]] = [pred_rel]

    # for doc_id in tqdm(range(len(data[:100])), desc="Loading examples"):
    for doc_id in tqdm(range(len(data)), desc="Loading examples"):
        sample = data[doc_id]
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        # add entity markers
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}  # 保存某篇文档中的实体对的关系和证据，一个如果实体对有多个关系 - {(h,t):[{r1,e1}, {r2,e2},...}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        """记录每个句子中 各个token进过分词和引入*后的 实体最新的位置 [[(0, 8) - 第一个实体更新后位置区间[0,8), ()...], []]"""
        # entity start, end position
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            assert len(e) != 0
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end,))

        """sent_labels 记录实体对下所有关系的证据句"""
        relations, hts, sent_labels, epair_types = [], [], [], []
        for h, t in train_triple.keys():  # for every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)
            for mention in train_triple[h, t]:  # for each relation mention with head h and tail t，遍历实体对下所有关系
                relation[mention["relation"]] = 1
                for i in mention["evidence"]:
                    sent_evi[i] += 1
            relations.append(relation)

            hts.append([h, t])
            epair_types.append([docred_ent2id[entities[h][0]['type']], docred_ent2id[entities[t][0]['type']]])
            sent_labels.append(sent_evi)
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                # all entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts:  # and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)

                    hts.append([h, t])
                    epair_types.append([docred_ent2id[entities[h][0]['type']], docred_ent2id[entities[t][0]['type']]])
                    sent_labels.append(sent_evi)
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)
        assert len(sents) < max_seq_length
        sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = [{'input_ids': input_ids,
                    'entity_pos': entity_pos,
                    'labels': relations,
                    'hts': hts,
                    'sent_pos': sent_pos,
                    'sent_labels': sent_labels,  # 证据句
                    'title': sample['title'],
                    'epair_types': epair_types,
                    }]

        if teacher_sig_path != '':  # add evidence distributions from the teacher model
            feature[0]['attns'] = attns[doc_id][:, :len(input_ids)]

        if single_results != None:  # get pseudo documents from predictions of the single run
            offset = 1 if transformer_type in ["bert", "roberta"] else 0
            if sample["title"] in title2preds:
                # 仅仅 利用某篇文档进行 预测的证据句 构建的 伪文档
                """{'input_ids': new_input_ids,  # 伪文档
                    'entity_pos': new_entity_pos,  # 记录实体提及在伪文档中位置
                    'labels': new_labels,  # 伪文档中实体对对应labels
                    'hts': new_hts,  # 在伪文档中的实体对
                    'sent_pos': new_sent_pos,  # 证据句拼接后获得最新的pos
                    'sent_labels': None,
                    'title': raw_feature['title'],
                    'entity_map': ent_new2old,  # 伪文档中的实体 对应 原文档哪个实体
                    'epair_types': epair_types,
                }"""
                feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]],
                                                                     entities, sent_map, offset, tokenizer)
                pred_pos_samples += pos_sample
                pred_neg_samples += neg_sample

        i_line += len(feature)
        features.extend(feature)

    print("# of documents {}.".format(i_line))
    if single_results != None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))
    else:
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))

    if not os.path.exists(feature_file) and teacher_sig_path == "" and single_results == None:
        print('Saving to', feature_file)
        pickle.dump(features, open(feature_file, "wb"))
        print('Save ', feature_file, " !!!")

    return features

# def read_docred(file_in,
#                 tokenizer,
#                 transformer_type="bert",
#                 max_seq_length=1024,
#                 teacher_sig_path="",
#                 single_results=None):
#     """
#     teacher_sig_path 为了加载教师模型的atten
#     single_results 为了fusion时 获取topk下所有结果 [{title:, h_idx, t_idx, r, evidence, score}, ...]
#     """
#     feature_file = './feature_file/' + file_in.split('.')[-2].split('/')[-1]
#     if teacher_sig_path != "":
#         feature_file += '_teacher_sig_path'
#     if single_results != None:
#         feature_file += '_single_results'
#     feature_file += '_feature.pkl'
#     if os.path.exists(feature_file):
#         print('Feature file:', feature_file)
#         features = pickle.load(open(feature_file, "rb"))
#         print('Feature loaded from', feature_file)
#         return features
#
#     i_line = 0
#     pos_samples = 0
#     neg_samples = 0
#     features = []
#     if file_in == "":
#         return None
#     with open(file_in, "r") as fh:
#         data = json.load(fh)
#
#     if teacher_sig_path != "":  # load logits
#         basename = os.path.splitext(os.path.basename(file_in))[0]
#         attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
#         attns = pickle.load(open(attns_file, 'rb'))
#
#     if single_results != None:
#         # reorder predictions as relations by title 将相同title的预测放在一起
#         pred_pos_samples = 0
#         pred_neg_samples = 0
#         # 为了fusion时 获取topk下所有结果 [title:[{title:, h_idx, t_idx, r, evidence, score}, ...], ...]
#         pred_rels = single_results
#         title2preds = {}
#         for pred_rel in pred_rels:
#             if pred_rel["title"] in title2preds:
#                 title2preds[pred_rel["title"]].append(pred_rel)
#             else:
#                 title2preds[pred_rel["title"]] = [pred_rel]
#
#     for doc_id in tqdm(range(len(data[:100])), desc="Loading examples"):
#         sample = data[doc_id]
#         entities = sample['vertexSet']
#         entity_start, entity_end = [], []
#         # record entities
#         for entity in entities:
#             for mention in entity:
#                 sent_id = mention["sent_id"]
#                 pos = mention["pos"]
#                 entity_start.append((sent_id, pos[0],))
#                 entity_end.append((sent_id, pos[1] - 1,))
#
#         # add entity markers
#         sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)
#
#         # training triples with positive examples (entity pairs with labels)
#         train_triple = {}
#         if "labels" in sample:
#             for label in sample['labels']:
#                 evidence = label['evidence']
#                 r = int(docred_rel2id[label['r']])
#                 # update training triples
#                 if (label['h'], label['t']) not in train_triple:
#                     train_triple[(label['h'], label['t'])] = [
#                         {'relation': r, 'evidence': evidence}]
#                 else:
#                     train_triple[(label['h'], label['t'])].append(
#                         {'relation': r, 'evidence': evidence})
#
#         """记录每个句子中 各个token进过分词和引入*后的 实体最新的位置 [[(0, 8) - 第一个实体更新后位置区间[0,8), ()...], []]"""
#         # entity start, end position
#         entity_pos = []
#         for e in entities:
#             entity_pos.append([])
#             assert len(e) != 0
#             for m in e:
#                 start = sent_map[m["sent_id"]][m["pos"][0]]
#                 end = sent_map[m["sent_id"]][m["pos"][1]]
#                 label = m["type"]
#                 entity_pos[-1].append((start, end,))
#
#         relations, hts, sent_labels = [], [], []
#         for h, t in train_triple.keys():  # for every entity pair with gold relation
#             relation = [0] * len(docred_rel2id)
#             sent_evi = [0] * len(sent_pos)
#             for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
#                 relation[mention["relation"]] = 1
#                 for i in mention["evidence"]:
#                     sent_evi[i] += 1
#             relations.append(relation)
#             hts.append([h, t])
#             sent_labels.append(sent_evi)
#             pos_samples += 1
#
#         for h in range(len(entities)):
#             for t in range(len(entities)):
#                 # all entity pairs that do not have relation are treated as negative samples
#                 if h != t and [h, t] not in hts:  # and [t, h] not in hts:
#                     relation = [1] + [0] * (len(docred_rel2id) - 1)
#                     sent_evi = [0] * len(sent_pos)
#                     relations.append(relation)
#
#                     hts.append([h, t])
#                     sent_labels.append(sent_evi)
#                     neg_samples += 1
#
#         assert len(relations) == len(entities) * (len(entities) - 1)
#         assert len(sents) < max_seq_length
#         sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
#         input_ids = tokenizer.convert_tokens_to_ids(sents)
#         input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
#
#         feature = [{'input_ids': input_ids,
#                     'entity_pos': entity_pos,
#                     'labels': relations,
#                     'hts': hts,
#                     'sent_pos': sent_pos,
#                     'sent_labels': sent_labels,
#                     'title': sample['title'],
#                     }]
#
#         if teacher_sig_path != '':  # add evidence distributions from the teacher model
#             feature[0]['attns'] = attns[doc_id][:, :len(input_ids)]
#
#         if single_results != None:  # get pseudo documents from predictions of the single run
#             offset = 1 if transformer_type in ["bert", "roberta"] else 0
#             if sample["title"] in title2preds:
#                 feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]],
#                                                                        entities, sent_map, offset, tokenizer)
#                 pred_pos_samples += pos_sample
#                 pred_neg_samples += neg_sample
#
#         i_line += len(feature)
#         features.extend(feature)
#
#     print("# of documents {}.".format(i_line))
#     if single_results != None:
#         print("# of positive examples {}.".format(pred_pos_samples))
#         print("# of negative examples {}.".format(pred_neg_samples))
#     else:
#         print("# of positive examples {}.".format(pos_samples))
#         print("# of negative examples {}.".format(neg_samples))
#
#     if not os.path.exists(feature_file):
#         print('Saving to', feature_file)
#         pickle.dump(features, open(feature_file, "wb"))
#     print('Save ', feature_file, " !!!")
#
#     return features
