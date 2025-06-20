import torch
import os
import random
import numpy as np
from constant_SAIS_Evidence import *


def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d


def set_seed(args):
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.n_gpu > 0 and torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    # !!!!!
    # os.environ['PYTHONHASHSEED'] = str(args.seed)

    # 完全固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    # 训练时 - 使用默认卷积算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 测试时 - 为模型的每个卷积层搜索最适合它的卷积实现算法
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    max_sent = max([len(f["sent_pos"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    sent_pos = [f["sent_pos"] for f in batch]
    sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
    attns = [f["attns"] for f in batch if "attns" in f]
    epair_types = [type for f in batch if "epair_types" in f for type in f["epair_types"]]
    epair_types = torch.tensor(epair_types, dtype=torch.long)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    labels = [torch.tensor(label) for label in labels]
    labels = torch.cat(labels, dim=0)

    if sent_labels != [] and None not in sent_labels:
        sent_labels_tensor = []
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None

    if attns != []:

        attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
        attns = torch.from_numpy(np.concatenate(attns, axis=0))
    else:
        attns = None

    if use_graph:
        graph = [f["graph"] for f in batch]
    else:
        graph = None

    output = (input_ids, input_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns, epair_types, graph)

    return output
