import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import HeteroData
from model import FPN_GCN
from data_load import load_and_preprocess_data
from config import load_config


def print_10_fold_res(base_dir="./ablation/test/CV_results", flage=False): 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    prs = []
    auprs = []
    mean_recall = np.linspace(0, 1, 100)

    fold_accs = []
    fold_precs = []
    fold_recs = []
    fold_f1s = []
    all_thresholds = []

    print(f"data loading from {base_dir}")

    for i in range(1, 11):
        fold_path = os.path.join(base_dir, f"fold_{i}", "best_fold_tensors.pt")
        if not os.path.exists(fold_path):
            print(f"Can not find Fold {i} data: {fold_path}")
            continue

        data = torch.load(fold_path, map_location='cpu', weights_only=False)
        
        y_true = np.array(data['test_labels'])
        y_probs = np.array(data['test_probs'])

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0 
        tprs.append(interp_tpr)

        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        pr_auc = data['test_aupr'] 
        auprs.append(pr_auc)
        
        interp_pr = np.interp(mean_recall, recall[::-1], precision[::-1])
        prs.append(interp_pr)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        all_thresholds.append(best_threshold)
        y_preds = (y_probs >= best_threshold).astype(int)
        
        fold_accs.append(accuracy_score(y_true, y_preds))
        fold_precs.append(precision_score(y_true, y_preds, zero_division=0))
        fold_recs.append(recall_score(y_true, y_preds, zero_division=0))
        fold_f1s.append(f1_score(y_true, y_preds, zero_division=0))

    print(f"Loading successful, fold = {len(aucs)}")
    avg_auprs = np.mean(auprs)
    aupr_closest = min(range(len(auprs)), key=lambda i: abs(auprs[i] - avg_auprs))
    avg_aucs = np.mean(aucs)
    aucs_closest = min(range(len(aucs)), key=lambda i: abs(aucs[i] - avg_aucs))

    print("="*40)
    for i in range(len(aucs)):
        print(f"Fold {i+1} -> AUC: {aucs[i]:.4f}, AUPR: {auprs[i]:.4f}, Acc: {fold_accs[i]:.4f}, Precision: {fold_precs[i]:.4f}, Recall: {fold_recs[i]:.4f}, F1: {fold_f1s[i]:.4f}")
    print("="*40)
    print(f"Accuracy  : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"Precision : {np.mean(fold_precs):.4f} ± {np.std(fold_precs):.4f}")
    print(f"Recall    : {np.mean(fold_recs):.4f} ± {np.std(fold_recs):.4f}")
    print(f"F1-Score  : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"AUC       : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AUPR      : {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")
    print(f"Based on AUPR, the fold closest to the mean value is Fold {aupr_closest}.")
    print(f"Based on AUC, the fold closest to the mean value is Fold {aucs_closest}.")
    print("="*40 + "\n")

    if flage:
        model_filename = "best_model_weights.pth" 
        best_fold_num = aucs_closest
        parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
        src_model_path = os.path.join(base_dir, f"fold_{best_fold_num}", model_filename)
        dst_model_dir = os.path.join(parent_dir, "best_model")
        dst_model_path = os.path.join(dst_model_dir, "final_model.pth") 

        if os.path.exists(src_model_path):
            os.makedirs(dst_model_dir, exist_ok=True)
            
            shutil.copy2(src_model_path, dst_model_path)
            print(f"successfully, Fold {best_fold_num} model copy to: {dst_model_path}")
        else:
            print(f"Error: Can find original model {src_model_path}")
    
    return (np.mean(aucs), np.mean(auprs), np.mean(fold_f1s), np.mean(fold_recs), np.mean(fold_precs), np.mean(fold_accs))
    

def generate_negative_samples(pos_sample, N_m, N_d, mark_sample=None):
    num_neg = len(pos_sample)
    num_u = N_m
    num_v = N_d

    pos_set = set()
    for row in pos_sample:
        pos_set.add((int(row[0]), int(row[1])))

    ban_set = set()
    if mark_sample is not None:
        for row in mark_sample:
            ban_set.add((int(row[0]), int(row[1])))

    neg_edges = []
    count = 0

    while count < num_neg:
        u = np.random.randint(0, num_u)
        v = np.random.randint(0, num_v)
        
        
        if (u, v) not in pos_set:
            if (u, v) not in ban_set:
                neg_edges.append([u, v, 0])
                pos_set.add((u, v))
                count += 1

    neg_sample = np.array(neg_edges)

    final_sample = np.vstack([pos_sample, neg_sample])
            
    return final_sample, neg_sample

def load_parameters(path="./config.yaml"):
    cfg = load_config(path)
    return cfg

