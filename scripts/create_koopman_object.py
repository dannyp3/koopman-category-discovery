import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from pathlib import Path
import argparse
import os

from kcm.koopman_category_model import KoopmanCategoryModel
from kcm.discovery import train_test_split_indices

plt.style.use('dark_background')


REPO_ROOT = Path(__file__).resolve().parents[1]

# Optional command-line overrides (defaults match the values used for the thesis)
parser = argparse.ArgumentParser(description='Build and save a Koopman Category Model')
parser.add_argument('--num-cats', type=int, default=4)
parser.add_argument('--num-samples', type=int, default=100)
parser.add_argument('--num-train-classes', type=int, default=2)
parser.add_argument('--num-segments', type=int, default=20)
parser.add_argument('--num-clusters', type=int, default=8)
parser.add_argument('--codebook-training-size', type=int, default=500)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


# Reproducibility Code
seed=args.seed
rng = np.random.default_rng(seed)



############### Shared Inputs ###############
num_cats = args.num_cats # 10, 4
num_samples = args.num_samples # 500, 100
system_dimension = 3 # 2
test_size = 0.2
category_discovery=True
train_classes = range(args.num_train_classes) # range(3) # range(7)
noisy_data=True
noise_std=0.01
samples_name = f'noisy_{noise_std}_samples' if noisy_data else 'samples'
data_path = (str(REPO_ROOT / "data"),
                f"{system_dimension}-dimensional-systems",
                f"dataset_{num_cats}_class_{num_samples}_{samples_name}.pkl"
            )
data_path = os.path.join(*data_path)
use_gpu = False
#############################################


############ kcm-specific inputs ############
delay_embeddings = 5
num_segments = args.num_segments # 30, 8
svd_rank = None
dmd_rank = None
q = 1
num_clusters = args.num_clusters # 15, 5
codebook_training_size = args.codebook_training_size # 490 # divides <num training classes>
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
