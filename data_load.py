import pandas as pd
import numpy as np
import torch.nn as nn
import torch

def load_and_preprocess_data(case_study_flage=False):
    disease_s = pd.read_csv('./data/final/disease_s.csv', index_col=0) # [N_D, N_D]
    mirna_s = pd.read_csv('./data/final/miRNA_s.csv', index_col=0)     # [N_M, N_M]
    
    unique_mirnas = mirna_s.index.tolist()
    unique_diseases = disease_s.index.tolist()
    
    m2g_df = pd.read_csv('./data/final/miR2tar_final.csv') # columns: miRNA, Gene, score
    g2d_df = pd.read_csv('./data/final/tar2dis_final.csv') # columns: Gene, Disease
    hmdd_m2d = pd.read_csv('./data/final/hmdd_combined.csv', index_col=0) # [N_M, N_D] 矩阵
    
    unique_genes = pd.concat([m2g_df['Target Gene'], g2d_df['Target Gene']]).unique().tolist()
    
    # ID mapping dict
    m2id = {name: i for i, name in enumerate(unique_mirnas)}
    d2id = {name: i for i, name in enumerate(unique_diseases)}
    g2id = {name: i for i, name in enumerate(unique_genes)}

    
    N_M, N_D, N_G = len(m2id), len(d2id), len(g2id)
    
    x_m = torch.tensor(mirna_s.values, dtype=torch.float32)
    x_d = torch.tensor(disease_s.values, dtype=torch.float32)
    x_g = nn.Embedding(N_G, x_m.size(1)).weight.data.clone() 
    
    m2g_mask = torch.zeros((N_M, N_G), dtype=torch.float32)

    for _, row in m2g_df.iterrows():
        if row['miRNA'] in m2id and row['Target Gene'] in g2id:
            m = m2id[row['miRNA']]
            g = g2id[row['Target Gene']]
            m2g_mask[m, g] = 1.0

            
    # Gene -> Disease
    g2d_mask = torch.zeros((N_G, N_D), dtype=torch.float32)
    g2d_thresholds = torch.zeros((N_G, N_D), dtype=torch.float32)
    for _, row in g2d_df.iterrows():
        if row['Target Gene'] in g2id and row['DOID'] in d2id:
            g = g2id[row['Target Gene']]
            d = d2id[row['DOID']]
            g2d_mask[g, d] = 1.0
            g2d_thresholds[g, d] = 1.0 - row['score']
            
    hmdd_matrix = hmdd_m2d.values
    all_pos_hmdd_edges = []
    for i in range(N_M):
        for j in range(N_D):
            val = hmdd_matrix[i, j]
            if val == 1:
                all_pos_hmdd_edges.append((i, j, val))
                

    all_pos_samples = np.array(all_pos_hmdd_edges, dtype=int)

    
    if case_study_flage:
        id2m = {i: name for name, i in m2id.items()}
        id2d = {i: name for name, i in d2id.items()}
        id2g = {i: name for name, i in g2id.items()}
        return N_M, N_G, N_D, x_m, x_g, x_d, m2g_mask, g2d_thresholds, g2d_mask, id2m, id2g, id2d
    else:
        return N_M, N_G, N_D, x_m, x_g, x_d, m2g_mask, g2d_thresholds, g2d_mask, all_pos_samples

