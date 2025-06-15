import bisect
import numpy as np
from constant_SAIS_Evidence_a import *

if docred_ent2id_NA:
    docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}
else:
    docred_ent2id = {'ORG': 0, 'LOC': 1, 'NUM': 2, 'TIME': 3, 'MISC': 4, 'PER': 5}


def create_graph_add_typefa(entity_pos, entity_type):
    """创建图 - 使用numpy创建邻接矩阵"""
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1]
    mention_num = len([mention for entity in entity_pos for mention in entity])
    anaphor_num = len(anaphor_pos)

    N_nodes = mention_num + anaphor_num
    addType_N_nodes = N_nodes + len(docred_ent2id) - 1  # 去除NA
    All_N_nodes = addType_N_nodes + 1  # 增加父节点

    nodes_adj = np.zeros((All_N_nodes, All_N_nodes), dtype=np.int32)

    edges_cnt = 1  # 自连接
    # add self-loop
    for i in range(All_N_nodes):
        nodes_adj[i, i] = edges_cnt

    edges_cnt = edges_cnt + 1  # 连接mention和anaphor
    # add mention-anaphor edges          link each mention node with all anaphor nodes.
    for i in range(mention_num):
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    entities = []  # [e1[m, ...], []]
    i = 0
    for e in entity_pos:  # [e1[m1(st, end), m2(st, end), ...], ...]
        ms = []
        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)

    edges_cnt = edges_cnt + 1  # 连接同实体下的mention
    # add co-reference edges   Mentions that refer to the same entity are connected with each other.
    for e in entities:
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    # TODO: 增加type边  add mention-type edges
    edges_cnt = edges_cnt + 1  # 连接mention和type
    for e_i, e in enumerate(entities):
        type = entity_type[e_i]
        for m_i, m in enumerate(e):
            nodes_adj[m, N_nodes + type[m_i]] = edges_cnt
            nodes_adj[N_nodes + type[m_i], m] = edges_cnt

    # TODO: 增加type边  add anaphor-type edges
    edges_cnt = edges_cnt + 1  # 连接anaphor和type
    for a_i, a in enumerate(range(mention_num, N_nodes)):
        type = entity_type[-1][a_i]
        nodes_adj[a, N_nodes + type] = edges_cnt
        nodes_adj[N_nodes + type, a] = edges_cnt

    # TODO: 增加父节点边  add all-fa edges
    edges_cnt = edges_cnt + 1  # 连接anaphor和type
    for i in range(addType_N_nodes):
        for j in range(addType_N_nodes, All_N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    edges_cnt = edges_cnt + 1  # 如果两个提及指向不同的实体，则在图中相应提及节点之间存在一条边。
    # add inter-entity edges
    # if two mentions refer to different entities, an edge exists between the corresponding mention nodes in the graph.
    nodes_adj[nodes_adj == 0] = edges_cnt

    return nodes_adj


def create_graph_add_type(entity_pos, entity_type):
    """创建图 - 使用numpy创建邻接矩阵"""
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1]
    mention_num = len([mention for entity in entity_pos for mention in entity])
    anaphor_num = len(anaphor_pos)

    N_nodes = mention_num + anaphor_num
    addType_N_nodes = N_nodes + len(docred_ent2id) - 1  # 去除NA
    All_N_nodes = addType_N_nodes

    # nodes_adj = np.zeros((All_N_nodes, All_N_nodes), dtype=np.int32)
    nodes_adj = np.zeros((All_N_nodes, All_N_nodes), dtype=np.float32)

    edges_cnt = 1  # 自连接
    # add self-loop
    for i in range(All_N_nodes):
        nodes_adj[i, i] = edges_cnt

    edges_cnt = edges_cnt + 1  # 连接mention和anaphor
    # add mention-anaphor edges          link each mention node with all anaphor nodes.
    for i in range(mention_num):
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    entities = []  # [e1[m, ...], []]
    i = 0
    for e in entity_pos:  # [e1[m1(st, end), m2(st, end), ...], ...]
        ms = []
        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)

    edges_cnt = edges_cnt + 1  # 连接同实体下的mention
    # add co-reference edges   Mentions that refer to the same entity are connected with each other.
    for e in entities:
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    # TODO: 增加type边  add mention-type edges
    edges_cnt = edges_cnt + 1  # 连接mention和type
    for e_i, e in enumerate(entities):
        type = entity_type[e_i]
        for m_i, m in enumerate(e):
            nodes_adj[m, N_nodes + type[m_i]] = edges_cnt
            nodes_adj[N_nodes + type[m_i], m] = edges_cnt

    # TODO: 增加type边  add anaphor-type edges
    edges_cnt = edges_cnt + 1  # 连接anaphor和type
    for a_i, a in enumerate(range(mention_num, N_nodes)):
        type = entity_type[-1][a_i]
        nodes_adj[a, N_nodes + type] = edges_cnt
        nodes_adj[N_nodes + type, a] = edges_cnt

    edges_cnt = edges_cnt + 1  # 如果两个提及指向不同的实体，则在图中相应提及节点之间存在一条边。
    # add inter-entity edges
    # if two mentions refer to different entities, an edge exists between the corresponding mention nodes in the graph.
    nodes_adj[nodes_adj == 0] = edges_cnt

    """
    type-type 利用数据集中的先验分布
    """
    type_type = {(1, 1): 0.022341540073336826, (1, 2): 0.08216343635411211, (1, 3): 0.0, (1, 4): 0.009350445259298061, (1, 5): 0.006757464641173389, (1, 6): 0.012572027239392353, (2, 1): 0.011891042430591933, (2, 2): 0.3544525929806181, (2, 3): 2.6191723415400734e-05, (2, 4): 0.00392875851231011, (2, 5): 0.005526453640649555, (2, 6): 0.010240963855421687, (3, 1): 0.0001571503404924044, (3, 2): 0.00026191723415400735, (3, 3): 0.0, (3, 4): 5.238344683080147e-05, (3, 5): 5.238344683080147e-05, (3, 6): 5.238344683080147e-05, (4, 1): 2.6191723415400734e-05, (4, 2): 7.85751702462022e-05, (4, 3): 0.0, (4, 4): 0.00036668412781561024, (4, 5): 5.238344683080147e-05, (4, 6): 7.85751702462022e-05, (5, 1): 0.04444735463593504, (5, 2): 0.03664222105814562, (5, 3): 5.238344683080147e-05, (5, 4): 0.03787323205866946, (5, 5): 0.035804085908852805, (5, 6): 0.05940282870612886, (6, 1): 0.051021477213200626, (6, 2): 0.0960974332111053, (6, 3): 2.6191723415400734e-05, (6, 4): 0.048402304871660556, (6, 5): 0.029229963331587217, (6, 6): 0.04057097957045574}
    for pair in type_type:
        nodes_adj[N_nodes + pair[0] - 1, N_nodes + pair[1] - 1] = type_type[pair]

    return nodes_adj


def create_sentence_graph(entity_pos, entity_type, sent_pos):
    """创建图 - 使用numpy创建邻接矩阵"""
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1]
    mention_num = len([mention for entity in entity_pos for mention in entity])
    anaphor_num = len(anaphor_pos)
    sentence_num = len(sent_pos)

    starts = [s[0] for s in sent_pos]
    anaphor_sentence = [bisect.bisect_right(starts, a_start) - 1 for a_start, _ in anaphor_pos]

    N_nodes = mention_num + anaphor_num
    # addType_N_nodes = N_nodes + len(docred_ent2id) - 1  # 去除NA
    addSen_N_nodes = N_nodes + sentence_num
    All_N_nodes = addSen_N_nodes

    # nodes_adj = np.zeros((All_N_nodes, All_N_nodes), dtype=np.int32)
    nodes_adj = np.zeros((All_N_nodes, All_N_nodes), dtype=np.float32)

    """自连接"""
    edges_cnt = 1  # 自连接
    # add self-loop
    for i in range(All_N_nodes):
        nodes_adj[i, i] = edges_cnt

    """mention-anaphor，全连接"""
    edges_cnt = edges_cnt + 1  # 连接mention和anaphor
    # add mention-anaphor edges          link each mention node with all anaphor nodes.
    for i in range(mention_num):
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt



    entities = []  # [e1[m, ...], []]
    entities_sentence = []
    i = 0
    for e in entity_pos:  # [e1[m1(st, end), m2(st, end), ...], ...]
        ms = []
        men_sen = [bisect.bisect_right(starts, a_start) - 1 for a_start, _ in e]

        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)
        entities_sentence.append(men_sen)


    """Mentions that refer to the same entity"""
    edges_cnt = edges_cnt + 1  # 连接同实体下的mention
    # add co-reference edges   Mentions that refer to the same entity are connected with each other.
    for e in entities:
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    # # TODO: 增加type边  add mention-type edges
    # edges_cnt = edges_cnt + 1  # 连接mention和type
    # for e_i, e in enumerate(entities):
    #     type = entity_type[e_i]
    #     for m_i, m in enumerate(e):
    #         nodes_adj[m, N_nodes + type[m_i]] = edges_cnt
    #         nodes_adj[N_nodes + type[m_i], m] = edges_cnt

    # # TODO: 增加type边  add anaphor-type edges
    # edges_cnt = edges_cnt + 1  # 连接anaphor和type
    # for a_i, a in enumerate(range(mention_num, N_nodes)):
    #     type = entity_type[-1][a_i]
    #     nodes_adj[a, N_nodes + type] = edges_cnt
    #     nodes_adj[N_nodes + type, a] = edges_cnt

    edges_cnt = edges_cnt + 1  # 如果两个提及指向不同的实体，则在图中相应提及节点之间存在一条边。
    # add inter-entity edges
    # if two mentions refer to different entities, an edge exists between the corresponding mention nodes in the graph.
    nodes_adj[nodes_adj == 0] = edges_cnt

    # """
    # type-type 利用数据集中的先验分布
    # """
    # type_type = {(1, 1): 0.022341540073336826, (1, 2): 0.08216343635411211, (1, 3): 0.0, (1, 4): 0.009350445259298061, (1, 5): 0.006757464641173389, (1, 6): 0.012572027239392353, (2, 1): 0.011891042430591933, (2, 2): 0.3544525929806181, (2, 3): 2.6191723415400734e-05, (2, 4): 0.00392875851231011, (2, 5): 0.005526453640649555, (2, 6): 0.010240963855421687, (3, 1): 0.0001571503404924044, (3, 2): 0.00026191723415400735, (3, 3): 0.0, (3, 4): 5.238344683080147e-05, (3, 5): 5.238344683080147e-05, (3, 6): 5.238344683080147e-05, (4, 1): 2.6191723415400734e-05, (4, 2): 7.85751702462022e-05, (4, 3): 0.0, (4, 4): 0.00036668412781561024, (4, 5): 5.238344683080147e-05, (4, 6): 7.85751702462022e-05, (5, 1): 0.04444735463593504, (5, 2): 0.03664222105814562, (5, 3): 5.238344683080147e-05, (5, 4): 0.03787323205866946, (5, 5): 0.035804085908852805, (5, 6): 0.05940282870612886, (6, 1): 0.051021477213200626, (6, 2): 0.0960974332111053, (6, 3): 2.6191723415400734e-05, (6, 4): 0.048402304871660556, (6, 5): 0.029229963331587217, (6, 6): 0.04057097957045574}
    # for pair in type_type:
    #     nodes_adj[N_nodes + pair[0] - 1, N_nodes + pair[1] - 1] = type_type[pair]

    """
    mention-sentence / anaphor-sentence  共用edges_cnt
    """
    # 连接mention和sent
    edges_cnt = edges_cnt + 1
    for e_i, e in enumerate(entities):
        men_sent = entities_sentence[e_i]
        for m_i, m in enumerate(e):
            nodes_adj[m, N_nodes + men_sent[m_i]] = edges_cnt
            nodes_adj[N_nodes + men_sent[m_i], m] = edges_cnt

    # 连接anaphor和sent
    for a_i in range(anaphor_num):
        a_sent = anaphor_sentence[a_i]
        nodes_adj[mention_num + a_i, N_nodes + a_sent] = edges_cnt
        nodes_adj[N_nodes + a_sent, mention_num + a_i] = edges_cnt

    return nodes_adj


def create_graph(entity_pos):
    """创建图 - 使用numpy创建邻接矩阵"""
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1]
    mention_num = len([mention for entity in entity_pos for mention in entity])
    anaphor_num = len(anaphor_pos)

    N_nodes = mention_num + anaphor_num
    nodes_adj = np.zeros((N_nodes, N_nodes), dtype=np.int32)

    edges_cnt = 1
    # add self-loop
    for i in range(N_nodes):
        nodes_adj[i, i] = edges_cnt

    edges_cnt = 2
    # add mention-anaphor edges          link each mention node with all anaphor nodes.
    for i in range(mention_num):
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    entities = []
    i = 0
    for e in entity_pos:
        ms = []
        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)

    edges_cnt = 3
    # add co-reference edges   Mentions that refer to the same entity are connected with each other.
    for e in entities:
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    edges_cnt = 4
    # add inter-entity edges
    # if two mentions refer to different entities, an edge exists between the corresponding mention nodes in the graph.
    nodes_adj[nodes_adj == 0] = edges_cnt

    return nodes_adj