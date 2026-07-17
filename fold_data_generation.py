import json
import numpy as np
from data_load import load_and_preprocess_data
from sklearn.model_selection import KFold, train_test_split
from utils import generate_negative_samples
from config import load_config

def load_parameters(path="./config.yaml"):
    cfg = load_config(path)
    return cfg

def fold_generation():
    cfg = load_parameters()
    epoch_set = cfg["train"]["epochs"]
    seed = cfg["train"]["seed"]
    fold_max = cfg["train"]["fold"]

    N_M, N_G, N_D, x_m, x_g, x_d, m2g_mask, g2d_thresholds, g2d_mask, all_pos_samples = load_and_preprocess_data()

    kf = KFold(n_splits=fold_max, shuffle=True, random_state=seed)
    fold_indices = {}

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_pos_samples)):
        train_idx, val_idx = train_test_split(train_val_idx, test_size=1./9., random_state=seed)

        train_pos_samples = all_pos_samples[train_idx]
        val_pos_samples = all_pos_samples[val_idx]
        test_pos_samples = all_pos_samples[test_idx]


        # 在测试集中隔离训练集、验证集的正样本
        ban_for_test_samples = np.vstack([train_pos_samples, val_pos_samples])
        test_samples, test_neg_samples = generate_negative_samples(test_pos_samples, N_M, N_D, ban_for_test_samples)
        
        # 在验证集中隔离训练集的正样本，测试集的所有样本
        ban_for_val_samples = np.vstack([test_samples, train_pos_samples])
        val_samples, val_neg_samples = generate_negative_samples(val_pos_samples, N_M, N_D, ban_for_val_samples)

        ban_for_train_samples = np.vstack([test_samples, val_samples])

        # train_samples, _ = generate_negative_samples(train_pos_samples, N_M, N_D, ban_for_train_samples)

        print(f"============== fold: {fold+1} ==============")
        epoch_indices = {}
        for epoch in range(1, epoch_set+1):
            print(f"Generating training data at epoch {epoch}")
            
            train_samples, _ = generate_negative_samples(train_pos_samples, N_M, N_D, ban_for_train_samples)
            epoch_indices[f"epoch_{epoch}"] = {
                "train_epoch": train_samples.tolist()
            }
        
        
        fold_indices[f"fold_{fold}"] = {
            "train": epoch_indices,
            "val": val_samples.tolist(),
            "test": test_samples.tolist()
        }

    with open("final_train_val_test_data.json", "w") as f:
        json.dump(fold_indices, f)
    print("10-fold cross-validation indices have been successfully saved.")


def main():
    fold_generation()

if __name__ == "__main__":
    main()