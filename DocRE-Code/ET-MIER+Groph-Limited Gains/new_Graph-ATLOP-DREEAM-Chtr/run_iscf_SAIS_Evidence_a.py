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
from constant_SAIS_Evidence_a import *
from model_SAIS_Evidence_a import DocREModel

from utils_a import set_seed, collate_fn, create_directory
from prepro_a import read_docred
from evaluation import to_official, official_evaluate, merge_results
if is_wandb:
    import wandb
from tqdm import tqdm

import pandas as pd
import pickle


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
             'graph': batch[9],
             'sen_graph': batch[10],
             }

    return input


def train(args, model, train_features, dev_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        scaler = GradScaler()
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator, desc='Train epoch'):
            # torch.cuda.empty_cache()
            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                optimizer.zero_grad()
                model.train()

                inputs = load_input(batch, args.device, tag='train')
                outputs = model(**inputs)
                loss = [outputs["loss"]["rel_loss"]]

                if inputs["sent_labels"] != None and evi_loss:
                    loss.append(outputs["loss"]["evi_loss"] * args.evi_lambda)
                if inputs["teacher_attns"] != None and attn_loss:
                    loss.append(outputs["loss"]["attn_loss"] * args.attn_lambda)
                if PEMSCLloss:
                    loss.append(outputs["loss"]["pemscl_loss"])
                if Relation_Seploss:
                    loss.append(outputs["loss"]["relation_seploss"])
                if TASK_ET:
                    loss.append(outputs["loss"]["etoss"])

                loss = sum(loss) / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if is_wandb:
                    wandb.log(outputs["loss"], step=num_steps)

                if (step + 1) == len(train_dataloader) or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):

                    """
                    dev_scores: {dev_F1, dev_evi_F1, dev_F1_ign}
                    dev_output: {dev_rel:[re_p, re_r, re_f1], dev_rel_ign:[evi_p, evi_r, evi_f1], dev_evi:[re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated]}
                    official_results: [{title:, h_idx, t_idx, r, evidence, score}, ...] official results used for evaluation.
                    results: topk下所有结果 results to be dumped into file, which can be further used during fushion.
                    """
                    dev_scores, dev_output, official_results, results = evaluate(args, model, dev_features, tag="dev")
                    if is_wandb:
                        wandb.log(dev_scores, step=num_steps)

                    print(epoch, dev_output)
                    with open(run_save_file, 'a') as file:
                        # 写入内容到文件
                        file.write(str(epoch) + "  " + str(dev_output) + "\n")

                    """根据dev_F1_ign保存"""
                    if dev_scores["dev_F1_ign"] > best_score:
                        best_score = dev_scores["dev_F1_ign"]
                        best_offi_results = official_results
                        best_results = results
                        best_output = dev_output

                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        print(f"dev_F1_ign: {dev_scores['dev_F1_ign']} saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                    if epoch == train_iterator[-1]:  # last epoch
                        ckpt_file = os.path.join(args.save_path, "last.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        # torch.save(model.state_dict(), ckpt_file)

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

        if args.save_attn:
            tag = "infer"

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
    results: topk results to be dumped into file, which can be further used during fushion. 保存top-k全部
    official_results: official results used for evaluation. 保存Topk中那些关系不为Na
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

    if is_wandb:
        wandb.init(project="DocRED", name=args.display_name)

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

    if frozen_relation_vector:  # 是否 实体类型不动态调整
        for name, parameter in model.named_parameters():
            if 'relation_vector' in name:
                parameter.requires_grad = False

    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num, trainable_num

    total_num, trainable_num = get_parameter_number(model)
    print("trainable_num/total_num: %.2fM / %.2fM" % (trainable_num / 1e6, total_num / 1e6))


    if args.load_path != "":  # load model from existing checkpoint
        model_path = os.path.join(args.load_path, "best.ckpt")
        model.load_state_dict(torch.load(model_path))

    if args.do_train:  # Training
        create_directory(save_path_)
        args.save_path = save_path_

        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        train_features = read(train_file, tokenizer, transformer_type=args.transformer_type,
                              max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
        dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type,
                            max_seq_length=args.max_seq_length)

        train(args, model, train_features, dev_features)

    else:  # Testing
        basename = os.path.splitext(args.test_file)[0]
        test_file = os.path.join(args.data_dir, args.test_file)

        test_features = read(test_file, tokenizer, transformer_type=args.transformer_type,
                             max_seq_length=args.max_seq_length)

        if args.eval_mode != "fushion":
            """
            test_scores = {"dev_F1", "dev_evi_F1", "dev_F1_ign"}
            test_output = {tag + "_rel": [], tag + "_rel_ign": [], tag + "_evi": [], }
            official_results: official results used for evaluation. 保存Topk中那些关系不为Na
            results: topk results to be dumped into file, which can be further used during fushion. 保存top-k全部
            """
            test_scores, test_output, official_results, results = evaluate(args, model, test_features, tag="test")
            if is_wandb:
                wandb.log(test_scores)

            offi_path = os.path.join(args.load_path, args.pred_file)
            score_path = os.path.join(args.load_path, f"{basename}_scores.csv")
            res_path = os.path.join(args.load_path, f"topk_{args.pred_file}")

            dump_to_file(official_results, offi_path, test_output, score_path, results, res_path)

        else:  # inference stage fusion
            results = json.load(open(os.path.join(args.results_path, f"topk_{args.pred_file}")))

            # 仅仅 利用预测的top-k构建伪文档 formulate pseudo documents from top-k (k=num_labels in arguments) predictions
            pseudo_test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length,
                                        single_results=results)
            """
            pseudo_test_scores = {"dev_F1", "dev_evi_F1", "dev_F1_ign"}
            pseudo_output = {tag + "_rel": [], tag + "_rel_ign": [], tag + "_evi": [], }
            pseudo_official_results: official results used for evaluation. 保存Topk中那些关系不为Na
            pseudo_results: topk results to be dumped into file, which can be further used during fushion. 保存top-k全部
            """
            pseudo_test_scores, pseudo_output, pseudo_official_results, pseudo_results = evaluate(args, model,
                                                                                                  pseudo_test_features,
                                                                                                  tag="test")

            """thresh用于Merge relation predictions，阈值可以最大化test上F1"""
            if 'thresh' in os.listdir(args.results_path):
                with open(os.path.join(args.results_path, "thresh")) as f:
                    thresh = json.load(f)
                print(f"Threshold loaded from file: {thresh}")
            else:
                thresh = None

            # Merge relation predictions from the original document and psuedo documents.
            merged_offi, thresh = merge_results(results, pseudo_results, test_features, thresh)
            merged_re, merged_evi, merged_re_ign, _ = official_evaluate(merged_offi, args.data_dir, args.train_file,
                                                                        args.test_file)

            tag = args.test_file.split('.')[0]
            merged_output = {
                tag + "_rel": [i * 100 for i in merged_re],
                tag + "_rel_ign": [i * 100 for i in merged_re_ign],
                tag + "_evi": [i * 100 for i in merged_evi],
            }

            if is_wandb:
                wandb.log({"dev_F1": merged_re[-1] * 100, "dev_evi_F1": merged_evi[-1] * 100,
                           "dev_F1_ign": merged_re_ign[-1] * 100})

            offi_path = os.path.join(args.results_path, f"fused_{args.pred_file}")
            score_path = os.path.join(args.results_path, f"{basename}_fused_scores.csv")
            dump_to_file(merged_offi, offi_path, merged_output, score_path, thresh=thresh)


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
