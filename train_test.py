import os
import json
import torch
import time
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import random
from sklearn.metrics import confusion_matrix
from data_load import load_and_preprocess_data
from model import FPN_GCN
from utils import load_parameters
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score


def run_10_fold_cv(beta=None):
    cfg = load_parameters()
    epoch_set = cfg["train"]["epochs"]
    seed = cfg["train"]["seed"]
    learning_rate = cfg["optimizer"]["lr"]
    fold_max = cfg["train"]["fold"]
    l1_weight = cfg["train"]["l1_weight"]
    hidden_dim_size = cfg["model"]["hidden_dim"]
    cv_results_dir = cfg["train"]["cv_results_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print("data loading\n")

    N_M, N_G, N_D, x_m, x_g, x_d, m2g_mask, g2d_thresholds, g2d_mask, all_pos_samples = load_and_preprocess_data()

    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    if beta is not None:
        print(f"\n Gating Function: Beta = {beta}")
    
        fold_dir_beta = os.path.join(cv_results_dir, f"beta_{beta}")
        os.makedirs(fold_dir_beta, exist_ok=True)
        cv_results_dir +=  f"/beta_{beta}"     


    with open("final_train_val_test_data.json", "r") as f:
        splits = json.load(f)

    print("=== Starting 10-Fold Cross-Validation ===")

    # for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_samples)):
    for i in range(fold_max):
        fold_num = i
        print(f"\n--- Fold {fold_num + 1} ---")
        fold_name = "fold_"+str(fold_num)
        # train_idx = splits[fold_name]["train"]
        # val_idx = splits[fold_name]["val"]
        # test_idx = splits[fold_name]["test"]


        # train_samples = np.array(splits[fold_name]["train"])
        test_samples = np.array(splits[fold_name]["test"])
        val_samples = np.array(splits[fold_name]["val"])


        # train_idx, val_idx = train_test_split(train_val_idx, test_size=1./9., random_state=42)

        # train_pos_samples = all_pos_samples[train_idx]
        # val_pos_samples = all_pos_samples[val_idx]
        # test_pos_samples = all_pos_samples[test_idx]

    
        # test_samples, test_neg_samples = generate_negative_samples(test_pos_samples, N_M, N_D)
        # val_samples, val_neg_samples = generate_negative_samples(val_pos_samples, N_M, N_D, test_neg_samples)

        # test_val_neg_samples = np.vstack([test_neg_samples, val_neg_samples])
        
        data = HeteroData()
        data['miRNA'].x = x_m
        data['Target Gene'].x = x_g
        data['DOID'].x = x_d
        data['miRNA', 'targets', 'Target Gene'].edge_index = m2g_mask.nonzero().t()
        data['Target Gene', 'associates', 'DOID'].edge_index = g2d_mask.nonzero().t()
        data = T.ToUndirected(merge=False)(data)
        data = data.to(device)

        val_target_labels = torch.zeros((N_M, N_D), device=device)
        val_calc_mask = torch.zeros((N_M, N_D), device=device)
        # target_labels = torch.zeros((N_M, N_D))
        # train_calc_mask = torch.zeros((N_M, N_D))

        # for u, v, rel_type in train_samples:
        #     u, v, rel_type = int(u), int(v), int(rel_type)
        #     train_calc_mask[u, v] = 1.0
        #     if rel_type > 0:
        #         target_labels[u, v] = 1.0
        
        
        for u, v, rel_type in val_samples:
            u, v, rel_type = int(u), int(v), int(rel_type)
            val_calc_mask[u, v] = 1.0
            if rel_type > 0:
                val_target_labels[u, v] = 1.0
                
        model = FPN_GCN(
            hidden_dim=hidden_dim_size, in_m=x_m.size(1), in_g=x_g.size(1), in_d=x_d.size(1),
            N_M=N_M, N_G=N_G, N_D=N_D,
            m2g_mask=m2g_mask, g2d_thresholds=g2d_thresholds, g2d_mask=g2d_mask)
        model = model.to(device)
        # model.fpn_beta = beta

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        fold_dir = os.path.join(cv_results_dir, f"fold_{fold_num+1}")
        os.makedirs(fold_dir, exist_ok=True)

        best_model_path = os.path.join(fold_dir, "best_model_weights.pth")
        best_tensor_path = os.path.join(fold_dir, "best_fold_tensors.pt")
        best_val_loss = float('inf')


        # all_time = []
        model.train()
        for epoch in range(1, epoch_set+1):
            # start_time = time.time()
            optimizer.zero_grad()
            
            train_samples = np.array(splits[fold_name]["train"][f"epoch_{epoch}"]["train_epoch"])

            target_labels = torch.zeros((N_M, N_D), device=device)
            train_calc_mask = torch.zeros((N_M, N_D), device=device)

            for u, v, rel_type in train_samples:
                u, v, rel_type = int(u), int(v), int(rel_type)
                train_calc_mask[u, v] = 1.0
                if rel_type > 0:
                    target_labels[u, v] = 1.0

            model.fpn_beta = 1.0 + 9.0 * ((epoch-1) / epoch_set) 
        
            train_preds, train_m2g, train_g2d, train_explore_m2g, train_explore_g2d = model(data.x_dict, data.edge_index_dict)


            loss_bce = F.binary_cross_entropy_with_logits(train_preds, target_labels, weight=train_calc_mask, reduction='sum') / len(train_samples)
        
            # l1_weight = 1e-3
            density_m2g = train_explore_m2g.mean()
            density_g2d = train_explore_g2d.mean()
            
            loss_l1 = l1_weight * (density_m2g + density_g2d)
            loss = loss_bce + loss_l1

            loss.backward()
            optimizer.step()
            # epoch_time = time.time() - start_time
            # print(f"epoch time: {epoch_time}")
            # all_time.append(epoch_time)

            model.eval()
            with torch.no_grad():
                val_preds, val_m2g, val_g2d, val_explore_m2g, val_explore_g2d = model(data.x_dict, data.edge_index_dict)

                val_density_m2g = val_explore_m2g.mean()
                val_density_g2d = val_explore_g2d.mean()
                
                val_loss_l1 = l1_weight * (val_density_m2g + val_density_g2d)
                val_loss = F.binary_cross_entropy_with_logits(val_preds, val_target_labels, weight=val_calc_mask, reduction='sum') / len(val_samples)
    
                val_loss += val_loss_l1

                val_u = val_samples[:, 0]
                val_v = val_samples[:, 1]

                val_labels_tensor = torch.tensor((val_samples[:, 2] > 0), dtype=torch.float32).to(val_preds.device)
                val_logits = val_preds[val_u, val_v]

                y_scores_val = val_logits.detach().cpu().numpy()
                y_true_val = val_labels_tensor.detach().cpu().numpy()
                val_auc = roc_auc_score(y_true_val, y_scores_val)
                val_aupr = average_precision_score(y_true_val, y_scores_val)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
                    torch.save(model.state_dict(), best_model_path)

                    torch.save({
                        'best_epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'val_auc': val_auc,
                        'val_aupr': val_aupr,
                        'opt_m2g': val_m2g.cpu(),
                        'opt_g2d': val_g2d.cpu(),
                        # 'train_idx': train_idx,
                        # 'val_idx': val_idx,
                        # 'test_idx': test_idx
                    }, best_tensor_path)
                
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
        model.eval()
        with torch.no_grad():
            saved_info = torch.load(best_tensor_path, weights_only=False)
            
            print(f"\n Loading the best model from epoch {saved_info['best_epoch']} for test set evaluation")
            test_preds, test_m2g, test_g2d, _, _ = model(data.x_dict, data.edge_index_dict)

            test_u = test_samples[:, 0]
            test_v = test_samples[:, 1]
            test_labels_tensor = torch.tensor((test_samples[:, 2] > 0), dtype=torch.float32).to(test_preds.device)
            test_logits = test_preds[test_u, test_v]

            y_scores_test = test_logits.cpu().numpy()
            y_true_test = test_labels_tensor.cpu().numpy()

            auc = roc_auc_score(y_true_test, y_scores_test)
            aupr = average_precision_score(y_true_test, y_scores_test)
            
            precisions, recalls, thresholds = precision_recall_curve(y_true_test, y_scores_test)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

            best_idx = np.argmax(f1_scores[:-1])
            best_threshold = thresholds[best_idx]

            y_preds = (y_scores_test >= best_threshold).astype(int)
            acc = accuracy_score(y_true_test, y_preds)
            precision = precision_score(y_true_test, y_preds)
            recall = recall_score(y_true_test, y_preds)
            f1 = f1_score(y_true_test, y_preds)

            saved_info['test_labels'] = y_true_test
            saved_info['test_probs'] = y_scores_test
            saved_info['test_auc'] = auc
            saved_info['test_aupr'] = aupr
            torch.save(saved_info, best_tensor_path)

            cm = confusion_matrix(y_true_test, y_preds, labels=[0, 1])

            print("---Confusion Matrix---\n")
            print(cm)

            print(f"Fold {fold_num+1} Test AUC: {auc:.4f}, AUPR: {aupr:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            # print(f"time: avg = {np.mean(all_time)}")


def train_model():
    run_10_fold_cv()


if __name__ == "__main__":
    train_model()