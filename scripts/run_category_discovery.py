import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from pathlib import Path, PureWindowsPath
import pathlib
import argparse
import json
import os

from kcm.utils import load_koopman_model, create_discovery_run_dir, save_discovery_params, save_plot, save_artifact, copy_koopman_params_to_discovery

from kcm.basic_feature_extract import BasicFeatureExtractor
from kcm.discovery import (
    CategoryDiscoveryTrainer,
    prep_data_for_discovery,
    check_histograms,
)
plt.style.use('default')


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / 'experiments'


parser = argparse.ArgumentParser(description='Run category discovery on a saved Koopman Category Model')
parser.add_argument('--kcm-name', default=None,
                    help='KCM run folder inside <repo>/experiments (default: most recent KCM_* folder)')
parser.add_argument('--output-dims', type=int, nargs='+', default=[4, 6, 8])
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()


######## Koopman Category Model Version ########
KCM_name = args.kcm_name # e.g. "KCM_20250818_174401_7dc47c3f"
output_dims = args.output_dims
################################################

if KCM_name is None:
    KCM_name = sorted(p.name for p in EXPERIMENTS_DIR.glob('KCM_*') if (p / 'koopman_model.pkl').exists())[-1]

# Models pickled on Linux/Colab contain PosixPath objects; this lets Windows unpickle them
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath


print(f'Loading in Koopman Category Model at: {KCM_name}')

full_path = EXPERIMENTS_DIR
koopman_dir = full_path / KCM_name
KCM, KCM_params = load_koopman_model(koopman_dir)
for par in KCM_params:
    print(par)

for output_dim in output_dims:

    category_params = {'koopman_path' : str(koopman_dir),
                       'drop_na' : True,
                       'normalize_basic_inputs' : False,
                       'input_dim' : KCM.num_clusters,
                       'output_dim' : output_dim,
                       'hidden_dims' : [200,200], # [200, 200], [1024, 512, 256]
                       'dropout' : 0.3,
                       'classes' : KCM.num_cats,
                       'epochs' : args.epochs,
                       'model_type' : 'SMILE', # baseline, SMILE
                       'temperature' : 0.2}
    
    
    
    discovery_run_dir = create_discovery_run_dir()
    save_discovery_params(category_params, discovery_run_dir)
    copy_koopman_params_to_discovery(koopman_dir, discovery_run_dir)
    
    print('\n\n', discovery_run_dir.stem, output_dim)
    
    
    # If the model was built on another machine, look for the dataset under <repo>/data instead
    if not os.path.isfile(KCM.data_path):
        raw = str(KCM.data_path)
        path = PureWindowsPath(raw) if '\\' in raw else Path(raw)
        KCM.data_path = REPO_ROOT / 'data' / path.parent.name / (path.stem + '.pkl')
    
    
    
    # train test split
    kcm_X_train, kcm_X_test, kcm_y_train, kcm_y_test, kcm_stacked = prep_data_for_discovery(train=KCM.train_data,
                                                                                            test=KCM.test_data,
                                                                                            normalize_final_data=False, # already normalized within kcm.create_codebook()
                                                                                            pca_reduction=False,
                                                                                            n_components=None,
                                                                                            feat_extractor='kcm')
    
    Extractor = BasicFeatureExtractor(num_cats=KCM.num_cats,
                                      num_samples=KCM.num_samples,
                                      system_dimension=KCM.system_dimension,
                                      data_path=KCM.data_path,
                                      noisy_data=KCM.noisy_data,
                                      noise_std=KCM.noise_std,
                                      seed=KCM.seed)
    
    Extractor.batch_extract_features(normalize_inputs=category_params['normalize_basic_inputs'],
                                     drop_na=category_params['drop_na'])
    
    basic = Extractor.df
    basic_train = basic.loc[basic['count'].isin(KCM.train_counts)]
    basic_test = basic.loc[basic['count'].isin(KCM.test_counts)]
    
    # train test split
    basic_X_train, basic_X_test, basic_y_train, basic_y_test, basic_stacked = prep_data_for_discovery(train=basic_train,
                                                                                                      test=basic_test,
                                                                                                      normalize_final_data=True,
                                                                                                      pca_reduction=True,
                                                                                                      n_components=KCM.num_clusters, # make kcm and basic extract have same feature dimension
                                                                                                      feat_extractor='basic')
    
    
    
    check_histograms(basic_stacked), print('')
    check_histograms(kcm_stacked)
    
    
    # Comparing system-target mappings for basic extractor and kcm
    mapping = basic[['system_name','target']].drop_duplicates().values
    basic_system_dict = {row[1] : row[0] for row in mapping}
    kcm_system_dict = {tgt : cat for cat,tgt in zip(KCM.cats,KCM.df['target'].drop_duplicates())}
    assert basic_system_dict == kcm_system_dict, 'Dictionaries do not match between extractors'
    
    assert (basic.target.values == KCM.df[['target','sample']].drop_duplicates()['target'].values).all()
    assert (basic_train.target.values == KCM.df_train[['target','sample']].drop_duplicates()['target'].values).all()
    assert (basic_test.target.values == KCM.df_test[['target','sample']].drop_duplicates()['target'].values).all()
    
    
    
    kcm_trainer = CategoryDiscoveryTrainer(input_dim=category_params['input_dim'],
                                           output_dim=category_params['output_dim'],
                                           hidden_dims=category_params['hidden_dims'],
                                           dropout=category_params['dropout'],
                                           classes=category_params['classes'],
                                           epochs=category_params['epochs'],
                                           model_type=category_params['model_type'],
                                           temperature=category_params['temperature'])
    
    basic_trainer = CategoryDiscoveryTrainer(input_dim=category_params['input_dim'],
                                             output_dim=category_params['output_dim'],
                                             hidden_dims=category_params['hidden_dims'],
                                             dropout=category_params['dropout'],
                                             classes=category_params['classes'],
                                             epochs=category_params['epochs'],
                                             model_type=category_params['model_type'],
                                             temperature=category_params['temperature'])
    
    
    
    kcm_trainer.train_model(kcm_X_train, kcm_X_test, kcm_y_train, kcm_y_test)
    save_artifact(kcm_trainer, discovery_run_dir, 'kcm_trainer')
    
    basic_trainer.train_model(basic_X_train, basic_X_test, basic_y_train, basic_y_test)
    save_artifact(basic_trainer, discovery_run_dir, 'basic_trainer')
    
    
    
    # kcm_trainer.plot_loss(log=True, show=True)
    kcm_trainer.plot_loss(log=True, show=False)
    save_plot(discovery_run_dir, filename="koopman_loss.png")
    
    # basic_trainer.plot_loss(log=True, show=True)
    basic_trainer.plot_loss(log=True, show=False)
    save_plot(discovery_run_dir, filename="basic_loss.png")
    
    
    
    # kcm_trainer.plot_unique_hash_count(show=True)
    kcm_trainer.plot_unique_hash_count(show=False)
    save_plot(discovery_run_dir, filename="koopman_unique_hash_count.png")
    
    # basic_trainer.plot_unique_hash_count(show=True)
    basic_trainer.plot_unique_hash_count(show=False)
    save_plot(discovery_run_dir, filename="basic_unique_hash_count.png")
    
    
    
    # kcm_trainer.plot_scores(show=True)
    kcm_trainer.plot_scores(show=False)
    save_plot(discovery_run_dir, filename="koopman_scores.png")
    
    # basic_trainer.plot_scores(show=True)
    basic_trainer.plot_scores(show=False)
    save_plot(discovery_run_dir, filename="basic_scores.png")
    
    
    
    index = -1
    # kcm_trainer.plot_hashes(index=index,split_testing=False,show=True,save_dir=discovery_run_dir,base_filename="koopman")
    kcm_trainer.plot_hashes(index=index,split_testing=False,show=False,save_dir=discovery_run_dir,base_filename="koopman")
    
    # basic_trainer.plot_hashes(index=index,split_testing=False,show=True,save_dir=discovery_run_dir,base_filename="basic")
    basic_trainer.plot_hashes(index=index,split_testing=False,show=False,save_dir=discovery_run_dir,base_filename="basic")
    
    
    
    # Save final-epoch metrics for quick comparison across runs
    def _final_metrics(trainer):
        return {'test_acc_all' : float(trainer.global_scores[-1]),
                'test_acc_old' : float(trainer.old_global_scores[-1]),
                'test_acc_new' : float(trainer.new_global_scores[-1]),
                'test_ari' : float(trainer.test_aris[-1]),
                'test_nmi' : float(trainer.test_nmis[-1]),
                'train_ari' : float(trainer.train_aris[-1]),
                'unique_train_hashes' : int(trainer.unique_training_hashes[-1]),
                'unique_test_hashes' : int(trainer.unique_testing_hashes[-1]),
                'final_train_loss' : float(trainer.losses[-1])}
    
    results = {'output_dim' : output_dim, 'kcm' : _final_metrics(kcm_trainer), 'basic' : _final_metrics(basic_trainer)}
    with open(discovery_run_dir / 'final_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))