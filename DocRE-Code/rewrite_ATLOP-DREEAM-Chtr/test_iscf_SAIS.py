import argparse
import os
import datetime

import numpy as np
import torch
import ujson as json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from args_iscf import add_args
from constant_SAIS import *
from model_SAIS_Primitive import DocREModel
from utils import set_seed, collate_fn, create_directory
from prepro import read_docred
from evaluation import to_official, official_evaluate, merge_results
if is_wandb:
    import wandb
from tqdm import tqdm

import pandas as pd
import pickle

"""
python test_iscf_SAIS.py --data_dir dataset/docred --transformer_type roberta --model_name_or_path roberta-large --load_path /root/autodl-tmp/Code/REdocred-SAIS-DREEAM/roberta_docred_lambda0.1_seed66/test --eval_mode single --train_file train_annotated.json --dev_file dev.json --train_batch_size 4 --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97 --save_path roberta_docred_lambda0.1_seed66/
"""

def load_input(batch, device, tag="dev"):
    input = {'input_ids': batch[0].to(device),
             'attention_mask': batch[1].to(device),
             'labels': batch[2].to(device),
             'entity_pos': batch[3],
             'hts': batch[4],
             'sent_pos': batch[5],
             'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
             'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
             'tag': tag,
             'epair_types': batch[8].to(device),
             }

    return input


def train(args, model, train_features, dev_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        """
        dev_scores: {dev_F1, dev_evi_F1, dev_F1_ign}
        dev_output: {dev_rel:[re_p, re_r, re_f1], dev_rel_ign:[evi_p, evi_r, evi_f1], dev_evi:[re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated]}
        official_results: [{title:, h_idx, t_idx, r, evidence, score}, ...] official results used for evaluation.
        results: topk下所有结果 results to be dumped into file, which can be further used during fushion.
        """
        dev_scores, dev_output, official_results, results = evaluate(args, model, dev_features, tag="dev")

        best_score = dev_scores["dev_F1_ign"]
        best_offi_results = official_results
        best_results = results
        best_output = dev_output

        pred_file = os.path.join(args.save_path, args.pred_file)
        score_file = os.path.join(args.save_path, "scores.csv")
        results_file = os.path.join(args.save_path, f"topk_{args.pred_file}")

        dump_to_file(best_offi_results, pred_file, best_output, score_file, best_results, results_file)

        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds, evi_preds = [], []
    scores, topks = [], []
    attns = []

    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()

        inputs = load_input(batch, args.device, tag)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())
                topks.append(outputs["topks"].cpu().numpy())

            if "evi_pred" in outputs:  # relation extraction and evidence extraction
                # (实体对个数， 最大句子个数)
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)

            if "attns" in outputs:  # attention recorded
                attn = outputs["attns"]
                attns.extend([a.cpu().numpy() for a in attn])

    preds = np.concatenate(preds, axis=0)

    if scores != []:
        scores = np.concatenate(scores, axis=0)
        topks = np.concatenate(topks, axis=0)

    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)  # (396790, 25) 数据集实体对数，25

    """
    official_results: official results used for evaluation. 保存Topk中那些关系不为Na
    results: topk results to be dumped into file, which can be further used during fushion. 保存top-k全部
    """
    official_results, results = to_official(preds, features, evi_preds=evi_preds, scores=scores, topks=topks)

    if len(official_results) > 0:
        if tag == "dev":
            # best_re [re_p, re_r, re_f1], best_evi [evi_p, evi_r, evi_f1],
            # best_re_ign [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated],
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.dev_file)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.test_file)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign],
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {"dev_F1": best_re[-1] * 100, "dev_evi_F1": best_evi[-1] * 100, "dev_F1_ign": best_re_ign[-1] * 100}

    if args.save_attn:
        attns_path = os.path.join(args.load_path, f"{os.path.splitext(args.test_file)[0]}.attns")
        print(f"saving attentions into {attns_path} ...")
        with open(attns_path, "wb") as f:
            pickle.dump(attns, f)

    return scores, output, official_results, results


def dump_to_file(offi: list, offi_path: str, scores: list, score_path: str, results: list = [], res_path: str = "",
                 thresh: float = None):
    '''
    offi: [{title:, h_idx, t_idx, r, evidence, score}, ...] official results used for evaluation.
    scores: {dev_rel:[re_p, re_r, re_f1], dev_rel_ign:[evi_p, evi_r, evi_f1], dev_evi:[re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated]}
    results: topk下所有结果 [{title:, h_idx, t_idx, r, evidence, score}, ...]
             results to be dumped into file, which can be further used during fushion.

    dump scores and (top-k) predictions to file.
    '''
    print(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))

    print(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns=headers)
    print(scores_pd)
    scores_pd.to_csv(score_path, sep=',')

    if len(results) != 0:
        assert res_path != ""
        print(f"saving topk results into {res_path} ...")
        json.dump(results, open(res_path, "w"))

    if thresh != None:
        thresh_path = os.path.join(os.path.dirname(offi_path), "thresh")
        if not os.path.exists(thresh_path):
            print(f"saving threshold into {thresh_path} ...")
            json.dump(thresh, open(thresh_path, "w"))

    return


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    # create directory to save checkpoints and predicted files
    time = str(datetime.datetime.now()).replace(' ', '_')
    save_path_ = os.path.join(args.save_path, f"{time}")

    args.n_gpu = torch.cuda.device_count()
    args.seed = seed
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.transformer_type = args.transformer_type

    set_seed(args)

    read = read_docred
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = DocREModel(config, model, tokenizer,
                       num_labels=args.num_labels,
                       max_sent_num=args.max_sent_num,
                       evi_thresh=args.evi_thresh)
    model.to(args.device)

    total_params = sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad])
    memory_size_bytes = total_params * 4  # 每个参数占据4个字节
    memory_size_gb = memory_size_bytes / (1024 ** 3)  # 将字节数转换为GB
    print('total parameters:', memory_size_gb)

    if args.load_path != "":  # load model from existing checkpoint
        model_path = os.path.join(args.load_path, "best.ckpt")
        model.load_state_dict(torch.load(model_path))

    create_directory(save_path_)
    args.save_path = save_path_

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    train_features = read(train_file, tokenizer, transformer_type=args.transformer_type,
                          max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
    dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type,
                        max_seq_length=args.max_seq_length)

    train(args, model, train_features, dev_features)



if __name__ == "__main__":
    import datetime
    start_time = datetime.datetime.now()

    main()

    end_time = datetime.datetime.now()
    # 计算经过的时间
    elapsed_time = end_time - start_time
    # 将时间差转换为时、分、秒
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    # 打印输出
    print(f"经过时间：{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
