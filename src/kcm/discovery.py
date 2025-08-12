import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from tqdm import tqdm

from kcm.utils import save_plot

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')



    




def sup_con_loss(features, labels, temp=1):
    
    labels = labels.contiguous().view(-1, 1)
    
    dot_matrix = torch.matmul(features, features.T) / temp
    
    mask = ~torch.eye(len(features), dtype=bool)
    
    exp_matrix = torch.exp(dot_matrix) * mask
    
    positive_mask = (labels == labels.T) & mask
    
    log_denom = torch.log(exp_matrix.sum(dim=1, keepdim=True))
    
    log_proba = dot_matrix - log_denom
    
    losses = (positive_mask * log_proba).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    
    loss = - losses.mean()
    
    return loss





class BaselineModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512, 256], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim,hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.head = nn.Linear(in_dim,output_dim,bias=False)

        
    def forward(self, x):
        x = self.mlp(x)
        x = self.head(x)
        x = nn.functional.normalize(x, dim=1)
        return x, x, x






class HASHHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512, 256], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim,hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.hash = nn.Linear(in_dim,output_dim,bias=False)
        self.variance = nn.Linear(in_dim,output_dim,bias=False)
        self.bn_h = nn.BatchNorm1d(output_dim)
        self.bn_v = nn.BatchNorm1d(output_dim)

        
    def forward(self, x):
        x = self.mlp(x)
        h = self.hash(x)
        v = self.variance(x)

        h = self.bn_h(h)
        v = self.bn_v(v)

        # From OCD SMILE code
        v = v / (nn.Tanh()(v * 1))
        h = nn.Tanh()(h * 1) 

        x = h * v
        
        x = nn.functional.normalize(x, dim=1)
        return x, h, v


def train_test_split_indices(num_cats,num_samples,test_size,train_classes,category_discovery,rng):
    
    target = (np.ones((num_cats,num_samples )) * np.arange(num_cats)[:,np.newaxis]).ravel().astype(int)
    count = np.arange(num_samples * num_cats).astype(int)
    
    train_classes = np.array(train_classes)
    
    test_classes = np.array(list(set(range(num_cats)) - set(train_classes)))
    
    if category_discovery:
        print('Category Discovery Split...')
        
        known_class_train_counts = []
        
        for cl in train_classes:
            class_counts = count[target == cl]
            random_sample = rng.choice(class_counts, size=int((1-test_size)*num_samples), replace=False)
            known_class_train_counts.extend(list(random_sample))
        
        known_class_train_counts = np.array(known_class_train_counts)
        
        mask1 = np.isin(count, known_class_train_counts, invert=True)
        mask2 = np.isin(target, train_classes)
        known_class_test_counts = count[mask1 & mask2]
        
        unknown_class_counts = count[np.isin(target, test_classes)]
        
        train_counts = known_class_train_counts
        test_counts = np.array(list(known_class_test_counts) + list(unknown_class_counts))
        
    
    else:
        print('Classification Split...')

        train_counts = []
        test_counts = []
        
        for cl in range(num_cats):
            class_counts = count[target == cl]
            random_sample = rng.choice(class_counts, size=int((1-test_size)*num_samples), replace=False)
            train_counts.extend(list(random_sample))
        
        train_counts = np.array(train_counts)
        test_counts = count[np.isin(count, train_counts, invert=True)]

    
    if len(train_counts) + len(test_counts) != num_cats*num_samples:
        raise ValueError(f'num training set ({len(train_counts)}) + num testing set ({len(test_counts)}) != total samples ({num_cats*num_samples})')
    
    print(f'    Training Size: {len(train_counts)}')
    print(f'    Testing Size: {len(test_counts)}')

    return train_counts, test_counts


## Backup Code for Train Test Split
# target = (np.ones((num_cats,num_samples * num_segments)) * np.arange(num_cats)[:,np.newaxis]).ravel().astype(int)
# sample = np.array(list((np.ones((num_samples,num_segments)) * np.arange(num_samples)[:,np.newaxis]).ravel()) * num_cats).astype(int)
# segment = np.array(list(np.arange(num_segments)) * (num_cats * num_samples)).astype(int)

# df_kcm = pd.DataFrame(data=dict(target=target,sample=sample,segment=segment))
# df = df_kcm.drop(columns=['segment']).drop_duplicates().reset_index(drop=True)

# df_kcm['count'] = (np.ones((num_cats * num_samples,num_segments)) * np.arange(num_cats * num_samples)[:,np.newaxis]).ravel().astype(int)
# df['count'] = np.arange(num_samples * num_cats).astype(int)







def prep_data_for_discovery(train,test,normalize_final_data=False,pca_reduction=True,n_components=None,feat_extractor=None):

    if feat_extractor == 'basic':
        X_train = train.drop(columns=['system_name','count','target']).values
        y_train = train['target'].values
        
        X_test = test.drop(columns=['system_name','count','target']).values
        y_test = test['target'].values
        
    elif feat_extractor == 'kcm': 
        X_train = train[:,:-1]
        y_train = train[:,-1]

        X_test = test[:,:-1]
        y_test = test[:,-1]
    
    # Normalize based on training set
    if normalize_final_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Perform PCA
    if pca_reduction:
        assert X_train.shape[1] >= n_components
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    X_train = torch.Tensor(X_train)
    y_train = torch.tensor(y_train).long().squeeze()
    
    X_test = torch.Tensor(X_test)
    y_test = torch.tensor(y_test).long().squeeze()

    # Combine tensors into a single dataframe
    Xs = torch.vstack((X_train,X_test))
    ys = torch.vstack((torch.unsqueeze(y_train, 1),torch.unsqueeze(y_test, 1)))

    stacked = torch.hstack((Xs,ys))
    
    return X_train, X_test, y_train, y_test, stacked


def check_histograms(stacked):

    X = stacked[:,:-1]
    y = stacked[:,-1]

    num_cats = len(y.unique())
    num_samples = len(y[y == 0])

    print('Histograms with duplicate labels:')

    unique_histograms, indices = torch.unique(X, dim=0, return_inverse=True)

    num_issue_hists = 0
    num_issue_samples = 0
    for uni, ind in zip(unique_histograms,torch.unique(indices)):
    
        histogram_count = X[indices == ind].shape[0]
        unique_labels = torch.unique(y[indices == ind])
    
        if len(unique_labels) > 1:
            num_issue_hists += 1
            num_issue_samples += histogram_count
            print(f'{np.round(np.array(uni),3)} has {histogram_count} instances with {np.array(unique_labels)} unique_labels')
    
            
    percent_hist_issues = round(num_issue_hists*100/len(unique_histograms),2)
    print(f'\n{num_issue_hists}/{len(unique_histograms)} ({percent_hist_issues} %) of histograms had duplicate labels,')
    perc_sample_issues = round(num_issue_samples*100/(num_cats*num_samples),2)
    print(f'\taffecting {num_issue_samples}/{num_cats*num_samples} samples ({perc_sample_issues} %)')




def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc




def create_hash_ids(output, y):
    
    hashes = torch.where(output > 0, 1, 0)
    unique_hashes = torch.unique(hashes, dim=0)
    hashes = np.array(hashes)
    hash_ids = np.array([int("".join(map(str, row)), 2) for row in hashes])

    _, new_labels = np.unique(hash_ids, return_inverse=True)

    return hash_ids, hashes, new_labels





class CategoryDiscoveryTrainer():
    def __init__(self, input_dim=None, output_dim=None, hidden_dims=None, dropout=None, classes=None, epochs=None, model_type=None, temperature=None):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.classes = classes
        self.epochs = epochs
        self.model_type = model_type
        self.temperature = temperature


    def train_model(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


        # Create mask for training class locations in target
        self.train_classes = np.unique(np.array(self.y_train))
        self.mask = np.isin(np.array(self.y_test),self.train_classes)
        
        if self.model_type == 'baseline':
            self.model = BaselineModel(self.input_dim, self.output_dim, self.hidden_dims, self.dropout)
        elif self.model_type == 'SMILE':
            self.model = HASHHead(self.input_dim, self.output_dim, self.hidden_dims, self.dropout)

        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        
        self.train_aris = []
        self.train_nmis = []
        self.train_amis = []
        
        self.test_aris = []
        self.test_nmis = []
        self.test_amis = []
        
        self.weighted_scores = []
        self.old_scores = []
        self.new_scores = []
        
        self.global_scores = []
        self.old_global_scores = []
        self.new_global_scores = []
        
        self.unique_training_hashes = []
        self.unique_testing_hashes = []

        self.all_training_hash_ids = []
        self.all_testing_hash_ids = []

        self.all_training_hashes = []
        self.all_testing_hashes = []
        
        self.losses = []
        for i in tqdm(range(self.epochs), desc=f'Training {self.model_type} Model'):
        
            if self.model_type == 'baseline':
                training_features, _, _ = self.model(self.X_train)
                testing_features, _, _ = self.model(self.X_test)
                sc_loss = sup_con_loss(training_features, self.y_train, temp=self.temperature)
                reg_loss = 0
                
            elif self.model_type == 'SMILE':
                training_features, training_hash_features, _ = self.model(self.X_train)
                testing_features, testing_hash_features, _ = self.model(self.X_test)
                sc_loss = sup_con_loss(training_features, self.y_train, temp=self.temperature)
        
                reg_loss = (1 - torch.abs(training_hash_features)).mean()
                
        
            loss = sc_loss * 1 + reg_loss * 3
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            self.losses.append(loss.item())
        
            # Create Cluster ids for train, test, and full data sets
            training_hash_ids, training_hashes, new_training_labels = create_hash_ids(training_features, self.y_train)
            testing_hash_ids, testing_hashes, new_testing_labels = create_hash_ids(testing_features, self.y_test)

            self.all_training_hash_ids.append(training_hash_ids)
            self.all_testing_hash_ids.append(testing_hash_ids)
    
            self.all_training_hashes.append(training_hashes)
            self.all_testing_hashes.append(testing_hashes)
            
            # # Full dataset clusters
            # features, _, _ = model(X)
            # full_has_ids, full_hashes, new_full_labels = create_hash_ids(features, y)
            
            # Report ARI, NMI, and AMI for training and testing splits
            self.train_aris.append(adjusted_rand_score(self.y_train, training_hash_ids))
            self.train_nmis.append(normalized_mutual_info_score(self.y_train, training_hash_ids))
            self.train_amis.append(adjusted_mutual_info_score(self.y_train, training_hash_ids))
        
            self.test_aris.append(adjusted_rand_score(self.y_test, testing_hash_ids))
            self.test_nmis.append(normalized_mutual_info_score(self.y_test, testing_hash_ids))
            self.test_amis.append(adjusted_mutual_info_score(self.y_test, testing_hash_ids))
        
            # Report category discovery metrics
            total_acc, old_acc, new_acc = split_cluster_acc_v1(np.array(self.y_test), testing_hash_ids, self.mask)
            self.weighted_scores.append(total_acc)
            self.old_scores.append(old_acc)
            self.new_scores.append(new_acc)
            
            total_acc, old_acc, new_acc = split_cluster_acc_v2(np.array(self.y_test), testing_hash_ids, self.mask)
            self.global_scores.append(total_acc)
            self.old_global_scores.append(old_acc)
            self.new_global_scores.append(new_acc)
        
            self.unique_training_hashes.append(len(np.unique(training_hash_ids)))
            self.unique_testing_hashes.append(len(np.unique(testing_hash_ids)))


    def plot_loss(self, log=True, show=False):
    
        plt.figure(figsize=(5,2))
        if log:
            plt.plot(np.log(np.array(self.losses)))
        else:
            plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Supervised Contrastive Loss')
        if show:
            plt.show()


    def plot_unique_hash_count(self, show=False):
        
        plt.figure(figsize=(5,2))
        plt.plot(self.unique_training_hashes,label='Training')
        plt.plot(self.unique_testing_hashes,label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Unique Hashes')
        plt.title('Number of Unique Hashes')
        plt.legend()
        if show:
            plt.show()


    def plot_scores(self, show=False):
        
        figs, axes = plt.subplots(1,3,figsize=(12,3))
        
        minimum = -0.05
        maximum = 1.05
        
        # Plot independent & weighted scores
        axes[0].plot(self.old_scores,linewidth=0.8,color='blue',label='Old Classes')
        axes[0].plot(self.new_scores,linewidth=0.8,color='red',label='New Classes')
        axes[0].plot(self.weighted_scores,linewidth=2,color='purple',label='Total')
        axes[0].set_title('Independent Scores')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim([minimum, maximum])
        axes[0].legend()
        
        # Plot global scores
        axes[1].plot(self.old_global_scores,linewidth=0.8,color='blue',label='Old Classes')
        axes[1].plot(self.new_global_scores,linewidth=0.8,color='red',label='New Classes')
        axes[1].plot(self.global_scores,linewidth=2,color='purple',label='Total')
        axes[1].set_title('Global Scores')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylim([minimum, maximum])
        axes[1].legend()
        
        # Plot training and testing clustering scores
        axes[2].plot(self.train_aris,linewidth=2,color='lightblue',label='Train ARI')
        axes[2].plot(self.train_nmis,linewidth=1,color='mediumblue',label='Train NMI')
        axes[2].plot(self.train_amis,linewidth=0.5,color='darkblue',label='Train AMI')
        axes[2].plot(self.test_aris,linewidth=2,color='lightcoral',label='Test ARI')
        axes[2].plot(self.test_nmis,linewidth=1,color='red',label='Test NMI')
        axes[2].plot(self.test_amis,linewidth=0.5,color='darkred',label='Test AMI')
        axes[2].set_title('Clustering Scores')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylim([minimum, maximum])
        axes[2].legend()
        
        
        plt.suptitle('Category Discovery Metrics')
        plt.tight_layout()
        if show:
            plt.show()


    def plot_hashes(self, index=-1, split_testing=False, show=False, save_dir=None, base_filename="koopman_hashes"):

        data = {'cluster_id' : self.all_training_hash_ids[index], 'y' : self.y_train}
        train = pd.DataFrame(data)
        
        data = {'cluster_id' : self.all_testing_hash_ids[index], 'y' : self.y_test}
        test = pd.DataFrame(data)
        
        for i in range(self.output_dim):
            train[f'hash_{i}'] = self.all_training_hashes[index][:,i]
            test[f'hash_{i}'] = self.all_testing_hashes[index][:,i]
        
        all_y_vals = sorted(set(train['y']).union(set(test['y'])))
        
        # Create consistent color mapping
        cmap = plt.get_cmap('tab10')
        y_to_color = {y: cmap(i % 10) for i, y in enumerate(all_y_vals)}


        # === Training Plot ===
        self._plot_histograms(train, combine=True, y_to_color=y_to_color, show=show)
        if not show:
            save_plot(save_dir, f"{base_filename}_hashes_train_epoch_{index}.png", subfolder="plots")
    
        # === Testing Plot(s) ===
        if split_testing:
            test_known = test[self.mask]
            test_unknown = test[~self.mask]
    
            self._plot_histograms(test_known, combine=True, y_to_color=y_to_color, show=show)
            if not show:
                save_plot(save_dir, f"{base_filename}_hashes_test_known_epoch_{index}.png", subfolder="plots")
    
            self._plot_histograms(test_unknown, combine=True, y_to_color=y_to_color, show=show)
            if not show:
                save_plot(save_dir, f"{base_filename}_hashes_test_unknown_epoch_{index}.png", subfolder="plots")
        else:
            self._plot_histograms(test, combine=True, y_to_color=y_to_color, show=show)
            if not show:
                save_plot(save_dir, f"{base_filename}_hashes_test_epoch_{index}.png", subfolder="plots")
                

    def _plot_histograms(self,df,combine=True,y_to_color=None,all_y_vals=None, show=False):

        all_clusters = sorted(df['cluster_id'].unique())
        unique_y = sorted(df['y'].unique()) if all_y_vals is None else sorted(all_y_vals)
        
        x = np.arange(len(all_clusters))
        width = 0.8 / len(unique_y)
    
        if y_to_color is None:
            cmap = plt.get_cmap('tab10')
            y_to_color = {y: cmap(i % 10) for i, y in enumerate(unique_y)}
            
    
        if combine:
            plt.figure(figsize=(8,3))
            for i, y_val in enumerate(unique_y):
                counts = df.loc[df['y'].eq(y_val),'cluster_id'].value_counts().reindex(all_clusters, fill_value=0)
                pmf = counts / (counts.sum()+1)
                bar_positions = x + i * width
                plt.bar(bar_positions, pmf, width=width, label=f'y = {y_val}', color=y_to_color.get(y_val, 'gray'))

            # Add vertical lines between cluster groups
            for i in range(len(x) + 1):
                xpos = x[0] + width * (len(unique_y) - 1) / 2 - 1/2 + i
                plt.axvline(x=xpos, linestyle='--', color='gray', alpha=0.6, linewidth=0.8)


            plt.xlabel('cluster_id')
            plt.ylabel('Frequency')
            plt.title('Cluster ID Counts by y')

            # xtick_positions = x + (group_width - width) / 2
            # plt.xticks(xtick_positions, labels=all_clusters)

            plt.xticks(x + width * (len(unique_y) - 1) / 2, labels=all_clusters)
            plt.legend(title='System')
            if show:
                plt.show()
    
        
        else:
            fig, axs = plt.subplots(len(unique_y), 1, figsize=(6, 10), constrained_layout=True)
            for i, y_val in enumerate(unique_y):
                counts = train.loc[train['y'].eq(y_val),'cluster_id'].value_counts().reindex(all_clusters, fill_value=0)
                pmf = counts / (counts.sum()+1)
                axs[i].bar(all_clusters, pmf, color=y_to_color.get(y_val, 'gray'))
                axs[i].set_title(f'Histogram of cluster_ids for y = {y_val}')
                axs[i].set_xlabel('cluster_id')
                axs[i].set_ylabel('Frequency')
                axs[i].set_xticks(all_clusters)
            # for i in range(4):
            #     axs[i].vlines(x=i+2*width,ymin=0,ymax=1,linestyles='dashed',color='gray',alpha=0.5)
            if show:
                plt.show()











class Plotter():
    def __init__(self,models):
        self.models = models