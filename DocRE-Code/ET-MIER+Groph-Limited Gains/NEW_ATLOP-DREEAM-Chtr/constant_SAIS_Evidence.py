import torch
import argparse
from args_iscf import add_args

parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tau = 2.0
tau_base = 1.0
lambda_3 = 1.0

"""loss"""
Crht_loss = False  # base 61.413232       61.627956

# MY_logsumexploss
Crht_MY_logsumexploss = True  # ------------------ mylogsumexp
Crht_loss_lambda = 0.01  # 61.356251      61.798604
# Crht_loss_lambda = 0.02  # 61.240805    61.359481
# Crht_loss_lambda = 0.05  # 61.156535
# Crht_loss_lambda = 0.07  # 60.837996
# Crht_loss_lambda = 0.001  # 61.226245
# Crht_loss_lambda = 0.003  # 61.184549
# Crht_loss_lambda = 0.004  # 61.261954   61.209421
# Crht_loss_lambda = 0.005  # 61.622035 * 61.785534
# Crht_loss_lambda = 0.006  # 61.421623   61.455319
# Crht_loss_lambda = 0.008  # 61.209025
# Crht_MY_logsumexploss = False  # ------------------ no_mylogsumexp
# Crht_loss_lambda = 0.005  # 61.508951

Emb_loss = False  # Emb loss

Emb_MY_logsumexploss = True  # ------------------ mylogsumexp
Emb_loss_lambda = 0  # inf
Emb_loss_lambda = 0.01  # inf
Emb_MY_logsumexploss = False  # ------------------ no_mylogsumexp
Emb_loss_lambda = 0.01  # 59.589363       59.610655
# Emb_loss_lambda = 0.05  # 58.989778
Emb_loss_lambda = 0.001  # 60.432893      60.938483
Emb_loss_lambda = 0.002  # 60.544767
# Emb_loss_lambda = 0.005  # 58.989778
# Emb_loss_lambda = 0.008  # 59.958071
"""
Crht_loss_True_0.005   Emb_loss_False_0.001  61.087654
Crht_loss_True_0.005   Emb_loss_False_0.01   59.855478
Crht_loss_True_0.01    Emb_loss_False_0.001  60.330439    60.534374
"""

Relation_loss = True
Relation_Seploss = Crht_loss or Emb_loss or Relation_loss

PEMSCLloss = False

evi_loss = True
attn_loss = True

"""SAIS"""
TASK_ET = True
Top_k = 1
loss_weight_ET = 0.25
# loss_weight_ET = 0.1
# loss_weight_ET = 0.5
loss_weight_ET = args.loss_weight_ET

"""SAIS下：TSER & Logsumexp"""
TSER_Logsumexp = True  # False 表示 TSER_Logsumexp_mean
TSER_Logsumexp = args.TSER_Logsumexp

"""frozen grid"""
frozen_relation_vector = False  # 正常为False
""""""


"""enhance evi"""
"""enhance evi"""
if args.ishas_enhance_evi:
    enhance_evi = ["test", "dev", "train"]
else:
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

# Relation_Specific_num = 7  # 要与docred_ent2id（7）一致
Relation_Specific_num = 6  # 要与docred_ent2id（7）一致
# Relation_Specific_num = 5
# Relation_Specific_num = 3

reduced_dim = 768  # 将修改emb的维度
relation_dim = 768  # 维度和RSMAN不同，原256
extractor_dim = 768  # head_extractor  -  开始和relation_dim一样   最后计算bilinear classifier时需要该维度，需要整除64

# seed = 66
seed = args.seed  # 66
seed = 66
is_wandb = False

evi_lambda = args.evi_lambda
# save_path = "./Save/three_cls_info_reduced_dim_512_relation_dim_128_save"
path = ("REDocred_TASK_ET{}{}_{}_s_attn_lambda{}_three{}_cls{}_relation{}_extractor{}_seed{}_Relation_Seploss{}_Crht_loss{}_Emb_loss{}"
        "_Relation_loss{}").format(TASK_ET, loss_weight_ET, evi_lambda, s_attn_lambda, three_atten, use_cls_info, relation_dim,
                                                extractor_dim, seed, Relation_Seploss, Crht_loss, Emb_loss, Relation_loss
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
print("_"*20)

fprint('one_mention_copy_or_addrel', one_mention_copy_or_addrel)

print("##" * 10)

file.close()
