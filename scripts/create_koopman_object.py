import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from pathlib import Path
import os

from kcm.koopman_category_model import KoopmanCategoryModel
from kcm.discovery import train_test_split_indices

plt.style.use('dark_background')




# Reproducibility Code
seed=42
rng = np.random.default_rng(seed)



############### Shared Inputs ###############
num_cats = 4 # 10, 4
num_samples = 100 # 500, 100
system_dimension = 3 # 2
test_size = 0.2
category_discovery=True
train_classes = range(2) # range(3) # range(7)
noisy_data=True
noise_std=0.01
samples_name = f'noisy_{noise_std}_samples' if noisy_data else 'samples'
data_path = (rf"C:\Users\peterdb1\Documents\Masters in ACM\(i-j) 625.801-802 - ACM Master's Research\Technical Work\koopman-category-discovery\data",
                f"{system_dimension}-dimensional-systems",
                f"dataset_{num_cats}_class_{num_samples}_{samples_name}.pkl"
            )
data_path = os.path.join(*data_path)
use_gpu = False
#############################################


############ kcm-specific inputs ############
delay_embeddings = 5
num_segments = 20 # 30, 8
svd_rank = None
dmd_rank = None
q = 1
num_clusters = 8 # 15, 5
codebook_training_size = 500 # 490 # divides <num training classes>
normalize_kcm_inputs=True
soft_clustering=True
tau = 0.1
#############################################


train_counts, test_counts = train_test_split_indices(num_cats,num_samples,test_size,train_classes,category_discovery,rng)


KCM = KoopmanCategoryModel(num_cats=num_cats,
                           num_samples=num_samples,
                           system_dimension=system_dimension,
                           delay_embeddings=delay_embeddings,
                           num_segments=num_segments,
                           svd_rank=svd_rank,
                           dmd_rank=dmd_rank,
                           q=q,
                           data_path=data_path,
                           cluster_method='kmeans',
                           num_clusters=num_clusters,
                           noisy_data=noisy_data,
                           noise_std=noise_std,
                           normalize_inputs=normalize_kcm_inputs,
                           train_classes=train_classes,
                           soft_clustering=soft_clustering,
                           tau=tau,
                           seed=seed,
                           use_gpu=use_gpu)

print(KCM.run_dir.stem)

KCM.train_counts = train_counts
KCM.test_counts = test_counts
KCM.generate_data()

kcm = KCM.df
kcm_train_data = kcm.loc[kcm['count'].isin(train_counts)].reset_index(drop=True)
kcm_test_data = kcm.loc[kcm['count'].isin(test_counts)].reset_index(drop=True)

KCM.df_train = kcm_train_data
KCM.df_test = kcm_test_data

assert int(kcm_train_data.shape[0]/num_segments) == len(train_counts), 'training samples not correct shape'
assert int(kcm_test_data.shape[0]/num_segments) == len(test_counts), 'testing samples not correct shape'





KCM.create_codebook(codebook_training_size=codebook_training_size,
                    category_discovery=category_discovery,
                    include_plots=True)

KCM.create_feature_outputs()

KCM.save()
KCM.shutdown_logger()
