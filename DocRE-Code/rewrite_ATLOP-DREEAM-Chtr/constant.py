import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tau = 2.0
tau_base = 1.0
lambda_3 = 1.0

"""loss"""
Crht_loss = False
Emb_loss = False
Relation_loss = True
Relation_Seploss = Crht_loss or Emb_loss or Relation_loss

evi_loss = True
attn_loss = False

PEMSCLloss = False

"""对实体下单提及处理"""
one_mention_copy_or_addrel = False  # True 为copy，False为提及嵌入直接和关系嵌入相加

"""Tips"""
three_atten = False
use_cls_info = False
use_bilinear = True  # False为按照RSMAN实现bilinear classifier
full_fine_tuning = True

Relation_Specific_num = 5
# Relation_Specific_num = 3

reduced_dim = 768  # 将修改emb的维度
relation_dim = 768  # 维度和RSMAN不同，原256
extractor_dim = 768  # head_extractor  -  开始和relation_dim一样   最后计算bilinear classifier时需要该维度，需要整除64

seed = 66
is_wandb = False

# save_path = "./Save/three_cls_info_reduced_dim_512_relation_dim_128_save"
path = ("three{}_cls{}_bilinear{}_reduced{}_relation{}_extractor{}_seed{}_Relation_Seploss{}_Crht_loss{}_Emb_loss{}"
        "_PEMSCLloss{}_Relation_loss{}").format(three_atten, use_cls_info, use_bilinear, reduced_dim, relation_dim,
                                                extractor_dim, seed, Relation_Seploss, Crht_loss, Emb_loss, PEMSCLloss,
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
fprint("Emb_loss", Emb_loss)
fprint("Relation_loss", Relation_loss)
fprint("__Relation_Seploss", Relation_Seploss)
print('*$' * 10)
fprint("PEMSCLloss", PEMSCLloss)
fprint("evi_loss", evi_loss)
fprint("attn_loss", attn_loss)
print("&" * 10)
print("_"*20)

fprint('one_mention_copy_or_addrel', one_mention_copy_or_addrel)

print("##" * 10)

file.close()
