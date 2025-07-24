import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys

# from sklearn.metrics.cluster import adjusted_rand_score
import sklearn.metrics as metrics
import STGCM_pyG
from utils import fix_seed
section_id = '151674'

input_dir = os.path.join('D:/ST/STGCM/ST-GCM/Data/DLPFC', section_id)
output_dir = os.path.join('D:/ST/STGCM/ST-GCM/results') #保存路径
# adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')

adata.var_names_make_unique()

print(adata)
fix_seed(2025)
#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)

# read the annotation
Ann_df = pd.read_csv(os.path.join('D:/ST/STGCM/ST-GCM/Data/DLPFC/DLPFC_annotations', section_id+'_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']

adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"], show=False)
plt.savefig(os.path.join(output_dir, f'Ground Truth.svg'), bbox_inches='tight', dpi=300)

STGCM_pyG.Cal_Spatial_Net(adata, rad_cutoff=150)
# STGCM_pyG.Stats_Spatial_Net(adata)

adata = STGCM_pyG.train_STGCM(adata, Conv_type='GCNConv')

#保存训练后的adata
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{section_id}_processed.h5ad')
adata.write(output_path)
print(f"Saved processed adata to: {output_path}")

# print(adata.X)
print("STGCM的维度：", adata.obsm['STGCM'].shape)

sc.pp.neighbors(adata, use_rep='STGCM')
sc.tl.umap(adata)
adata = STGCM_pyG.mclust_R(adata, used_obsm='STGCM', num_cluster=7)

obs_df = adata.obs.dropna()
ARI = metrics.adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
AMI = metrics.adjusted_mutual_info_score(obs_df['mclust'], obs_df['Ground Truth']) #添加了三个新的指标：AMI、NMI、HC
NMI = metrics.normalized_mutual_info_score(obs_df['mclust'], obs_df['Ground Truth'])
HC =  metrics.homogeneity_score(obs_df['mclust'], obs_df['Ground Truth'])
print('ARI = %.4f' %ARI)
print('AMI = %.4f' %AMI)
print('NMI = %.4f' %NMI)
print('HC = %.4f' %HC)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=["mclust", "Ground Truth"], title=['STGCM (ARI=%.4f)'%ARI, "Ground Truth"], show=False)
plt.savefig(os.path.join(output_dir, f'ump_domain.svg'), bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['STGCM (ARI=%.4f)'%ARI, "Ground Truth"], show=False) #聚类
plt.savefig(os.path.join(output_dir, f'pred_domain_cluster.svg'), bbox_inches='tight', dpi=300)


used_adata = adata[~adata.obs['Ground Truth'].isin([None,np.nan])].copy()
print(used_adata)

sc.tl.paga(used_adata, groups='Ground Truth')
plt.rcParams["figure.figsize"] = (4,3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                   title=section_id+'_STGCM', legend_fontoutline=2, show=False) #轨迹分析
plt.savefig(os.path.join(output_dir, f'Trajectory_Inference.svg'), bbox_inches='tight', dpi=300)