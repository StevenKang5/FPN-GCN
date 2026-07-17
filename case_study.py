import os
import json
import torch
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from model import FPN_GCN
from data_load import load_and_preprocess_data


def fpn_max_product_chunked(m2g, g2d, chunk_size=256):
    M, G = m2g.shape
    G_b, D = g2d.shape
    assert G == G_b, "The intermediate transition dimension G must match"
    
    result = torch.empty((M, D), device=m2g.device, dtype=m2g.dtype)
    
    g2d_expanded = g2d.unsqueeze(0)
    

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        m2g_chunk = m2g[i:end].unsqueeze(2)
        result[i:end] = (m2g_chunk * g2d_expanded).max(dim=1)[0]
        
    return result

def case_study_extract(model, data, id2m, id2g, id2d, disease_id=10, top_k_mirnas=10, top_k_genes=1):
    disease_name = id2d[disease_id]
    edges_data = []
    print(f"\n" + "="*70)
    print(f"Case Study: Identifying Key miRNAs Associated with Disease [{disease_name}] (ID: {disease_id})")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        val_preds, opt_m2g, opt_g2d,_ ,_ = model(data.x_dict, data.edge_index_dict)
        
        val_preds = fpn_max_product_chunked(opt_m2g, opt_g2d).cpu()
        opt_m2g = opt_m2g.cpu()
        opt_g2d = opt_g2d.cpu()

        # 得到先验结构
        mask_m2g = model.fpn_m2g.prior_mask.cpu() 
        mask_g2d = model.fpn_g2d.prior_mask.cpu()


    disease_prebs = val_preds[:, disease_id] 
    print(f"mean: {disease_prebs.mean().item():.4f} | max: {disease_prebs.max().item():.4f} | min: {disease_prebs.min().item():.4f} | std: {disease_prebs.std().item():.4f}")
    top_mirnas = torch.topk(disease_prebs, k=top_k_mirnas)
    
    for i in range(top_k_mirnas):
        m_idx = top_mirnas.indices[i].item()
        final_prob = top_mirnas.values[i].item()
        mirna_name = id2m[m_idx]
        
        print(f"\n[Top {i+1} disease-related miRNA] {mirna_name} (Final Confidence: {final_prob:.4f})")
        pathway_scores = torch.minimum(opt_m2g[m_idx, :], opt_g2d[:, disease_id])
   
        top_genes = torch.topk(pathway_scores, k=top_k_genes)
        print(f"gene mean: {pathway_scores.mean()} | max: {pathway_scores.max()} | min: {pathway_scores.min()}")
        valid_bridge_found = False
        
        for j in range(top_k_genes):
            g_idx = top_genes.indices[j].item()
            p_score = top_genes.values[j].item()
            
            if p_score >= pathway_scores.mean():
                gene_name = id2g[g_idx]
                m2g_score = opt_m2g[m_idx, g_idx].item()
                g2d_score = opt_g2d[g_idx, disease_id].item()

                is_m2g_prior = int(mask_m2g[m_idx, g_idx].item())
                is_g2d_prior = int(mask_g2d[g_idx, disease_id].item())

                m2g_label = "Prior" if is_m2g_prior == 1 else "Implicit"
                g2d_label = "Prior" if is_g2d_prior == 1 else "Implicit"
                
                print(f"  ├─ [Intermediate Gene] {gene_name} (Score: {p_score:.4f})")
                print(f"  │    ├─ m->g score: {m2g_score:.4f} [{m2g_label}]")
                print(f"  │    └─ g->d score: {g2d_score:.4f} [{g2d_label}]")
                valid_bridge_found = True

                edges_data.append({
                    'mirna': mirna_name,
                    'gene': gene_name,
                    'disease': disease_name,
                    'm2g': m2g_score,
                    'g2d': g2d_score,
                    'm2g_is_prior': is_m2g_prior,
                    'g2d_is_prior': is_g2d_prior
                })
                
        
        if not valid_bridge_found:
            print("  └─ [!] Hidden Graph Association:")
            print("         No strong target-gene bridge was identified in the known prior knowledge graph (bridge missing or not recorded).")
            print("         This high-confidence prediction is directly inferred from the global topological patterns learned by the Graph Neural Network (GNN),")
            print("         indicating high potential for discovering novel regulatory pathways.")
        else:
            print("  └─ [Mechanistic Interpretation Completed]")
            
    print("\n" + "="*70 + "\n")
    return edges_data, disease_name



def plot_pathway_sankey(edges_data, disease_name):

    if not edges_data:
        print("No valid regulatory paths satisfying the threshold were identified. Visualization cannot be generated.")
        return

    dis_true_name = disease_name
    if dis_true_name == "DOID:9352":
        dis_true_name = "T2DM"
    elif dis_true_name == "DOID:12365":
        dis_true_name = "Malaria"
    elif dis_true_name == "DOID:0060081":
        dis_true_name = "TNBC"
    else:
        dis_true_name = disease_name

    df = pd.DataFrame(edges_data)


    all_mirnas = list(df['mirna'].unique())
    all_genes = df.groupby('gene')['g2d'].max().sort_values(ascending=False).index.tolist()
    all_diseases = list(df['disease'].unique())
    
    node_labels = all_mirnas + all_genes + all_diseases
    node_map = {name: i for i, name in enumerate(node_labels)}
    
    sources = []
    targets = []
    values = []
    colors = []
    raw_values = []
    hidden_edge_color = "rgba(220, 20, 60, 0.75)"


    # first layer：miRNA -> Gene
    m2g_color = "rgba(231,111,81,0.35)" 
    for _, row in df.iterrows():
        mirna = row['mirna']
        gene = row['gene']
        sources.append(node_map[mirna])
        targets.append(node_map[gene])
        values.append(row['m2g'] * 10) 
        raw_values.append(row['m2g'])

        is_m2g_prior = row.get('m2g_is_prior', True)
        if is_m2g_prior:
            colors.append(m2g_color)
        else:
            colors.append(hidden_edge_color)

    # The second layer：Gene -> Disease
    g2d_color = "rgba(42,157,143,0.35)" 
    unique_gene_disease_edges = df.drop_duplicates(subset=['gene', 'disease'])
    
    for _, row in unique_gene_disease_edges.iterrows():
        gene = row['gene']
        disease = row['disease']
        sources.append(node_map[gene])
        targets.append(node_map[disease])
        values.append(row['g2d'] * 10) 
        raw_values.append(row['g2d'])

        is_g2d_prior = row.get('g2d_is_prior', True)
        if is_g2d_prior:
            colors.append(g2d_color)
        else:   
            colors.append(hidden_edge_color)

    node_colors_data = []
    for node_name in node_labels:
        if node_name in all_mirnas:
            node_colors_data.append("#E76F51")  
        elif node_name in all_genes:
            node_colors_data.append("#2A9D8F")  
        else:
            node_colors_data.append("#5E81AC")

    display_labels = []
    for name in node_labels:
        if name in all_diseases:
            custom_disease_name = f"{dis_true_name}" 
            display_labels.append(custom_disease_name)
        else:
            display_labels.append(name)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,          
            thickness=20,    
            line=dict(color="gray", width=0.5), 
            # label=node_labels, 
            label=display_labels,
            color=node_colors_data
        ),
        link=dict(
            source=sources, 
            target=targets, 
            value=values,   
            customdata=raw_values,
            color=colors,   
            # hovertemplate="<b>%{source.label}</b> &rarr; <b>%{target.label}</b><br>Strength: %{value:.2f}<extra></extra>"
            hovertemplate="<b>%{source.label}</b> &rarr; <b>%{target.label}</b><br>Real Score: %{customdata:.4f}<extra></extra>"
        )
    )])

    if dis_true_name == "T2DM":
        fig.update_layout(
            # title={
            #     'text': f"Predicted Pathogenic Pathways for <b>{dis_true_name}</b>",
            #     'y': 0.95,
            #     'x': 0.5,
            #     'xanchor': 'center',
            #     'yanchor': 'top',
            #     'font': dict(size=16, family="Arial", color="black")
            # },
            # 整个图表的节点字体大小和颜色都在这里统一定义
            font=dict(color="black", size=18, family="Arial"),
            width=1000, 
            height=700,
            # margin=dict(l=10, r=10, t=60, b=30), 
            margin=dict(l=10, r=10, t=60, b=120),
            paper_bgcolor="rgba(255, 255, 255, 0)", 
            plot_bgcolor="rgba(255, 255, 255, 1)",
            
            annotations=[
                dict(
                    x=0.5,       
                    y=-0.15,     
                    xref="paper", 
                    yref="paper",
                    text=(
                        "<span style='color:rgba(231,111,81,0.35);font-size:30px;'>■</span> <b>Prior Associations (miRNA to Gene)</b>"
                        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        "<span style='color:rgba(42,157,143,0.35);font-size:30px;'>■</span> <b>Prior Associations (Gene to Disease)</b>"
                    ),
                    showarrow=False, 
                    font=dict(size=18, family="Arial", color="black"),
                    align="center"
                )
            ]
        )
    else:
         fig.update_layout(
            # title={
            #     'text': f"Predicted Pathogenic Pathways for <b>{dis_true_name}</b>",
            #     'y': 0.95,
            #     'x': 0.5,
            #     'xanchor': 'center',
            #     'yanchor': 'top',
            #     'font': dict(size=16, family="Arial", color="black")
            # },
            font=dict(color="black", size=18, family="Arial"),
            width=1000, 
            height=700,
            # margin=dict(l=10, r=10, t=60, b=30), 
            margin=dict(l=10, r=10, t=60, b=120),
            paper_bgcolor="rgba(255, 255, 255, 0)", 
            plot_bgcolor="rgba(255, 255, 255, 1)",
            
            annotations=[
                dict(
                    x=0.5,     
                    y=-0.15,   
                    xref="paper", 
                    yref="paper",
                    text=(
                        "<span style='color:rgba(220, 20, 60, 0.75);font-size:30px;'>■</span> <b>Implicit Associations</b>"
                        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        "<span style='color:rgba(231,111,81,0.35);font-size:30px;'>■</span> <b>Prior Associations (miRNA to Gene)</b>"
                        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        "<span style='color:rgba(42,157,143,0.35);font-size:30px;'>■</span> <b>Prior Associations (Gene to Disease)</b>"
                    ),
                    showarrow=False, 
                    font=dict(size=18, family="Arial", color="black"),
                    align="center"
                )
            ]
        )

    save_path = f"{dis_true_name}_pathway_sankey.png"
    
    try:
        fig.write_image(save_path, scale=3, width=1000, height=700)
        print(f"The Sankey diagram generated and saved to PNG: {save_path}")
    except Exception as e:
        print(f"Unable to export the static image automatically (Kaleido may not be installed):{e}")
        
    fig.show()


def test():

    N_M, N_G, N_D, x_m, x_g, x_d, m2g_mask, g2d_thresholds, g2d_mask, id2m, id2g, id2d = load_and_preprocess_data(case_study_flage=True)

    data = HeteroData()
    data['miRNA'].x = x_m
    data['Target Gene'].x = x_g
    data['DOID'].x = x_d
    data['miRNA', 'targets', 'Target Gene'].edge_index = m2g_mask.nonzero().t()
    data['Target Gene', 'associates', 'DOID'].edge_index = g2d_mask.nonzero().t()

    data = T.ToUndirected(merge=False)(data)

    model = FPN_GCN(
    hidden_dim=64, in_m=x_m.size(1), in_g=x_g.size(1), in_d=x_d.size(1),
    N_M=N_M, N_G=N_G, N_D=N_D,
    m2g_mask=m2g_mask, g2d_thresholds=g2d_thresholds, g2d_mask=g2d_mask)

    best_model_path = './res/best_model/save_model/best_model_weights.pth'

    model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=torch.device('cpu')))

    diabetes_data, diabetes_name = case_study_extract(model, data, id2m, id2g, id2d, disease_id=192)

    plot_pathway_sankey(diabetes_data, diabetes_name)


if __name__ == "__main__":
    test()