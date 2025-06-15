import torch
import argparse
from args_iscf import add_args

parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Load_feature_file = True  # 是否加载已有特征
Load_feature_file = False  # 是否加载已有特征

docred_ent2id_NA = True  # True {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}  Relation_Specific_num + 1
# 使用使用最高效果的ET_loss公式
IS_high_batch_ET_loss = True  # True batch_epair_types - 1 效果最高

"""add_graph"""
create_graph_ISori = True  # 是否是使用原始图，否则为add type fa
# create_graph_ISori = False  # 是否是使用原始图，否则为add type fa

"""graph"""
use_graph = args.use_graph
cat_graph = False  # 默True cat([hs, rs, h_graph]); 使用其他方法 - cat(hs, rs)+h_graph
cat_graph = True  # 默True cat([hs, rs, h_graph]); 使用其他方法 - cat(hs, rs)+h_graph

cat_htrs_htrsnew = False  # 默True 先cat后mean [1310, 1, 768]; 先mean后cat [1310, 2, 768]

entity_pos_isAA = True  # 影响entity_emb   True是entity_pos，包含anaphor效果低，    False是entity_pos[:-1]不包含anaphor —— 没有图 or 有图但想用高效果时的emb
# entity_pos_isAA = False  # 影响entity_emb   True是entity_pos，包含anaphor效果低，    False是entity_pos[:-1]不包含anaphor —— 没有图 or 有图但想用高效果时的emb

"""SAIS下：TSER & Logsumexp"""
TSER_Logsumexp = True  # False 表示 TSER_Logsumexp_mean
TSER_Logsumexp = args.TSER_Logsumexp

TSER_Logsumexp_new_mean = True
TSER_Logsumexp_new_mean = False  # 默认

tau = 2.0
tau_base = 1.0
lambda_3 = 1.0
"""loss"""
Crht_loss = False  # base 61.413232       61.627956

# MY_logsumexploss
Crht_MY_logsumexploss = True
Crht_loss_lambda = 0.01

Emb_loss = False  # Emb loss

Emb_MY_logsumexploss = True
Emb_loss_lambda = 0.01

Relation_loss = True
Relation_Seploss = Crht_loss or Emb_loss or Relation_loss

PEMSCLloss = False

evi_loss = True
attn_loss = True

"""SAIS"""
TASK_ET = True
Top_k = 1
loss_weight_ET = args.loss_weight_ET

"""frozen grid"""
frozen_relation_vector = False  # 正常为False
""""""


"""enhance evi"""
enhance_evi = ["test", "dev", "train"]
enhance_evi = []
s_attn_lambda = 0.08
s_attn_lambda = 0.4
s_attn_lambda = 0
s_attn_lambda = args.s_attn_lambda
s_attn_logsumexp_lambda = 1 - s_attn_lambda

"""对实体下单提及处理"""
one_mention_copy_or_addrel = False  # True 为copy，False为提及嵌入直接和关系嵌入相加

"""Tips"""
three_atten = True
three_atten = args.three_atten
use_cls_info = False
use_bilinear = True  # False为按照RSMAN实现bilinear classifier
full_fine_tuning = True

Relation_Specific_num = 6

reduced_dim = 768  # 将修改emb的维度
relation_dim = 768  # 维度和RSMAN不同，原256
extractor_dim = 768  # head_extractor  -  开始和relation_dim一样   最后计算bilinear classifier时需要该维度，需要整除64

# seed = 66
seed = args.seed  # 66
seed = 66
is_wandb = False
is_wandb = args.wandb

evi_lambda = args.evi_lambda
# save_path = "./Save/three_cls_info_reduced_dim_512_relation_dim_128_save"
path = ("use_graph{}_docred_ent2id_NA{}_REDocred_TASK_ET{}{}_{}_s_attn_lambda{}_three{}_cls{}_relation{}_extractor{}_seed{}_Relation_Seploss{}_Crht_loss{}_Emb_loss{}"
        "_Relation_loss{}").format(use_graph, docred_ent2id_NA, TASK_ET, loss_weight_ET, evi_lambda, s_attn_lambda, three_atten,
                                   use_cls_info, relation_dim, extractor_dim, seed, Relation_Seploss, Crht_loss, Emb_loss,
                                   Relation_loss
                                   )
save_path = "./Save/" + path

run_save_file = "./Logs/" + path + ".txt"
file = open(run_save_file, 'w')
file.write(str(save_path) + "\n")


def fprint(name, value):
    result = "{}: {}".format(name, value)
    print(result)
    file.write(result + "\n")


print("##" * 10)
# fprint("device", device)
print("*" * 10)

fprint("save_path", save_path)
fprint("全微调", full_fine_tuning)
fprint("seed", seed)
print("&" * 10)
print("&" * 10)

fprint("use_graph", use_graph)
fprint("cat_graph", cat_graph)
fprint("cat_htrs_htrsnew", cat_htrs_htrsnew)
fprint("entity_pos_isAA", entity_pos_isAA)
print("*" * 10)
fprint("create_graph_ISori", create_graph_ISori)
print("*" * 10)
print("*" * 10)

fprint("docred_ent2id_NA", docred_ent2id_NA)
fprint("IS_high_batch_ET_loss", IS_high_batch_ET_loss)
print("*" * 10)

fprint("three_atten", three_atten)
fprint("use_cls_info", use_cls_info)
fprint("use_bilinear", use_bilinear)
print("&" * 10)

fprint("reduced_dim", reduced_dim)
fprint("relation_dim", relation_dim)
fprint("extractor_dim", extractor_dim)
print("*&" * 10)

fprint("Relation_Specific_num", Relation_Specific_num)
fprint("Crht_loss", Crht_loss)
fprint("__Crht_loss", Crht_loss_lambda)
fprint("**Crht_MY_logsumexploss", Crht_MY_logsumexploss)
fprint("Emb_loss", Emb_loss)
fprint("__Emb_loss", Emb_loss_lambda)
fprint("**Emb_MY_logsumexploss", Emb_MY_logsumexploss)
fprint("Relation_loss", Relation_loss)
fprint("__Relation_Seploss", Relation_Seploss)
print('*$' * 10)

fprint("PEMSCLloss", PEMSCLloss)
fprint("evi_loss", evi_loss)
fprint("attn_loss", attn_loss)
print("&" * 10)

fprint("TASK_ET", TASK_ET)
fprint("**loss_weight_ET", loss_weight_ET)
print("&" * 10)

fprint("frozen_relation_vector", frozen_relation_vector)
print("&" * 10)

fprint("enhance_evi", enhance_evi)
fprint("s_attn_lambda", s_attn_lambda)
print("&" * 10)

fprint("TSER_Logsumexp", TSER_Logsumexp)
fprint("TSER_Logsumexp_new_mean", TSER_Logsumexp_new_mean)
print("_"*20)

fprint('one_mention_copy_or_addrel', one_mention_copy_or_addrel)

print("##" * 10)

file.close()
