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
    

def draw_beta_embedding_new(beta=[1.0, 5.0, 10.0, -1], embedding=[32, 64, 128], base_dir = "./ablation/GPU/final/ablation/beta/"):
    labels = ['AUC', 'AUPR', 'F1-Score', 'Recall', 'Precision', 'Accuracy']
    x = np.arange(len(labels))
    beta_colors = {
        1.0: '#2b5c8f',  
        5.0: "#66a61e",   
        10.0: "#403F3F", 
        -1: '#d95f02'     
    }
    embedding_linestyles = {32: '-', 64: '--', 128: '-.'}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=500) 

    all_values = {}


    for b in beta:
        for e in embedding:   
            if b != -1:
                base_dir += str(e) + "/beta_" + str(b)
            else:
                base_dir += str(e) + "/simple/adaptive1_10"
            values = print_10_fold_res(base_dir)
            all_values[(b, e)] = values
            label = f'β={b}, emb={e}' if b != -1 else f'β=dynamic, emb={e}'
            ax.plot(x, values,
                    color=beta_colors[b],
                    linestyle=embedding_linestyles[e],
                    marker='o', linewidth=2,
                    label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylim(0.75, 0.96)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95'], color="black", size=10)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)


    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)

    fig.canvas.draw() 
    bbox = ax.get_position() 

    sub_width = 0.15
    sub_height = bbox.height * 0.35
    sub_left = bbox.x0 + bbox.width + 0.035   
    sub_bottom = bbox.y0

    axins = fig.add_axes([sub_left, sub_bottom, sub_width, sub_height])

    zoom_combinations = [
        (1.0, 32), (1.0, 64), (1.0, 128),
        (5.0, 32),
        (-1, 32), (-1, 64)
    ]

    for b, e in zoom_combinations:
        if (b, e) in all_values:
            lbl = f'β={b}, e={e}' if b != -1 else f'β=dynamic, e={e}'
            axins.plot(x[:2], all_values[(b, e)][:2],
                       color=beta_colors[b],
                       linestyle=embedding_linestyles[e],
                       marker='o', linewidth=2,
                       label=lbl)

    axins.set_xticks(x[:2])
    axins.set_xticklabels(labels[:2], fontsize=10) 
    # axins.set_title('Selected Configurations', fontsize=10)
    axins.set_ylim(0.94, 0.95)
    axins.set_yticks([0.94, 0.95, 0.96])
    axins.tick_params(axis='y', labelsize=10)
    axins.grid(True, linestyle='--', alpha=0.6)
    # axins.legend(fontsize=5, loc='best')

    axins.patch.set_facecolor('white')
    axins.patch.set_alpha(0.9)
    plt.show()


def load_model_read_matrix(model_path, fold, embedding_size):
    
    N_M, N_G, N_D, x_m, x_g, x_d, m2g_mask, g2d_thresholds, g2d_mask, all_pos_samples = load_and_preprocess_data()

    data = HeteroData()
    data['miRNA'].x = x_m
    data['Target Gene'].x = x_g
    data['DOID'].x = x_d
    data['miRNA', 'targets', 'Target Gene'].edge_index = m2g_mask.nonzero().t()
    data['Target Gene', 'associates', 'DOID'].edge_index = g2d_mask.nonzero().t()
    data = T.ToUndirected(merge=False)(data)
    best_model_path = model_path

    print(f"The selected model in fold {fold}")
    # test_samples = np.array(splits[fold_name]["test"])
    model_path = best_model_path + f"fold_{fold}/best_model_weights.pth"

    model = FPN_GCN(
        hidden_dim=embedding_size, in_m=x_m.size(1), in_g=x_g.size(1), in_d=x_d.size(1),
        N_M=N_M, N_G=N_G, N_D=N_D,
        m2g_mask=m2g_mask, g2d_thresholds=g2d_thresholds, g2d_mask=g2d_mask)
    
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=torch.device('cpu')))
    
    model.eval()
    with torch.no_grad():
        test_preds, test_m2g, test_g2d, _, _, = model(data.x_dict, data.edge_index_dict)
        m2g_d_matrix = model.fpn_m2g.D_matrix
        g2d_d_matrix = model.fpn_g2d.D_matrix
        m2g_D_matrix = m2g_d_matrix.detach().cpu()
        g2d_D_matrix = g2d_d_matrix.detach().cpu()
    
    return (m2g_D_matrix, g2d_D_matrix)


def test_model_parameters(position=0):
        all_d_matrix = {}
        dfs_to_concat = []
        zoom_combinations = [
        (1.0, 32, 2), (1.0, 64, 9), (1.0, 128, 5),
        (5.0, 32, 7),
        (-1, 32, 5), (-1, 64, 6)]
        MAX_SAMPLE_SIZE = 50000

        for beta, emb, fold in zoom_combinations:
            base_dir = "./ablation/GPU/final/ablation/beta/"
            if beta != -1:
                base_dir += str(emb) + "/beta_" + str(beta) + "/"
            else:
                base_dir += str(emb) + "/simple/adaptive1_10/"
            
            D_matrix = load_model_read_matrix(base_dir, fold, emb)
            all_d_matrix[(beta, emb)] = D_matrix
            flat_vals = D_matrix[position].ravel()
            # linestyle = embedding_linestyles.get(emb, '-')
            # color = beta_colors.get(beta, 'gray')
            label = f'β={beta}, emb={emb}' if beta != -1 else f'β=dynamic, emb={emb}'
            if len(flat_vals) > MAX_SAMPLE_SIZE:
                sampled_vals = np.random.choice(flat_vals, size=MAX_SAMPLE_SIZE, replace=False)
            else:
                sampled_vals = flat_vals
            # sns.kdeplot(flat, color=color, linestyle=linestyle, label=label, fill=False, linewidth=2)
            df_temp = pd.DataFrame({
                    'Model': label,
                    'Value': sampled_vals
                })
            dfs_to_concat.append(df_temp)
            
        df = pd.concat(dfs_to_concat, ignore_index=True)
        

        plt.figure(figsize=(10, 6), dpi=500)
        sns.violinplot(x='Model', y='Value', data=df,
                    order=df['Model'].unique(),
                    inner='quartile',
                    cut=0,
                    color='steelblue')

        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        plt.xlabel('Configurations of β and Embedding Size')
        plt.ylabel('Value')
        # plt.ylabel(r'$(\mathbf{T}_{src}-\mathbf{\Lambda})$')
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show() 


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

