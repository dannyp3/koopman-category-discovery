import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform, cdist

from sklearn.manifold import MDS
from sklearn.cluster import k_means
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import pickle
from datetime import datetime
from pathlib import Path
import uuid
import logging
import joblib



class KoopmanCategoryModel:
    def __init__(self, num_cats=None, num_samples=None, system_dimension=None, data_path=None, delay_embeddings=0, num_segments=5,
                 svd_rank=None, dmd_rank=None, q=1, cluster_method='kmeans', num_clusters=None, run_root: str | Path | None = None, seed=None):

        logger = logging.getLogger("KCM")
        
        # ---------- unique output directory ----------
        run_root = self._repo_root() / "runs" if run_root is None else Path(run_root)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid  = uuid.uuid4().hex[:8]                 # 8-char suffix
        self.run_dir = Path(run_root) / f"KCM_{ts}_{uid}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        
        # ----------------- robust logger ------------------
        self.logger = logging.getLogger(f"KCM.{uid}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # # console
        # ch = logging.StreamHandler()
        # ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        # self.logger.addHandler(ch)

        # file
        fh = logging.FileHandler(self.run_dir / "run.log")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

        self.logger.info("Results will be saved in %s", self.run_dir)

        self.seed = seed if seed is not None else 0
        self.rng  = np.random.default_rng(self.seed)


        
        # Dataset Parameters
        self.num_cats = num_cats
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.system_dimension = system_dimension
        self.data_path = data_path
        self.dataset = self._load_data(data_path)
        self.cats = list(self.dataset.keys())
        self.m, self.n = self.dataset[self.cats[0]][0]['y'].shape
        self.segment_length = int(self.n/self.num_segments)

        # DMD Parameters
        self.delay_embeddings = delay_embeddings
        self.total_observables = self.m * (self.delay_embeddings + 1)
        self.svd_rank = self.total_observables if svd_rank is None else svd_rank
        self.dmd_rank = self.svd_rank if dmd_rank is None else dmd_rank

        if (self.dmd_rank > self.svd_rank) or (self.svd_rank > self.total_observables) or (self.dmd_rank > self.total_observables):
            raise ValueError(f'Should have dmd_rank < svd_rank < total_observables, but have values of {self.dmd_rank}, {self.svd_rank}, and {self.total_observables}')

        if np.mod(self.n,self.num_segments) != 0:
            raise ValueError(f'Number of segments {self.num_segments} must divide given data size {self.n-self.delay_embeddings}')
        
        # Clustering Parameters
        self.q = q
        self.MDS_dimension = 10
        self.cluster_method = cluster_method
        self.num_clusters = num_clusters
        self.labels = None
        
        # Classification Parameters
        self.scaler = StandardScaler()

        # Category Discovery Parameters


        param_details = [f'num_cats: {self.num_cats}',
                         f'num_samples: {self.num_samples}',
                         f'num_segments: {self.num_segments}',
                         f'data_path: {self.data_path}',
                         f'cats (categories) : {self.cats}',
                         f'delay_embeddings: {self.delay_embeddings}',
                         f'total_observables: {self.total_observables}',
                         f'svd_rank: {self.svd_rank}',
                         f'dmd_rank: {self.dmd_rank}',
                         f'q: {self.q}',
                         f'MDS_dimension: {self.MDS_dimension}',
                         f'cluster_method: {self.cluster_method}',
                         f'num_cluseters (k) : {self.num_clusters}']

        joined = '\n'.join(param_details)
        self.logger.info(f"Parameters:\n{joined}")


    def _load_data(self, data_path):
        """
        Load in the data based on either the provided data path
        or the default path if none is provided.
        """

        if self.data_path is None:
            file_path = self._data_path(
                f"{self.system_dimension}-dimensional-systems",
                f"dataset_{self.num_cats}_class_{self.num_samples}_noisy_samples.pkl"
            )
            self.data_path = file_path
        else:
            file_path = Path(self.data_path)

        print(f'Loading data in at {self.data_path}...')
        
        with open(file_path, "rb") as f:
            return pickle.load(f)
        

    def generate_data(self):

        all_eigs = []
        all_modes = []
        all_amps = []
        all_data = []
        
        self.total_dmd_calculations = self.num_cats * self.num_samples * self.num_segments
        self.logger.info(f'Generating {self.total_dmd_calculations} DMD eigs/modes each with dimensionality {self.svd_rank}')
        
        for cat in self.cats:
        
            curr_data = self.dataset[cat]
        
            for index in range(self.num_samples):
        
                X = curr_data[index]['y'].T
                
                if self.delay_embeddings > 0:
                    X = np.hstack([X[i:self.n-self.delay_embeddings+i,:] for i in range(self.delay_embeddings+1)])
                
                # DMD
                for sp in range(self.num_segments):
                    start_ind = sp * self.segment_length
                    end_ind = (sp + 1) * self.segment_length
                    X_split = X[start_ind:end_ind,:]
                    
                    eigs, modes, b = self._compute_dmd(X_split.T)
                    all_eigs.append(eigs)
                    all_modes.append(modes)
                    all_amps.append(b)
                    all_data.append(X_split)

        self.all_eigs = all_eigs
        self.all_modes = all_modes
        self.all_amps = all_amps
        self.all_data = all_data
        
        if len(self.all_eigs) != self.total_dmd_calculations:
            raise ValueError(f'Incorrect Number of DMD Run: expected {self.total_dmd_calculations}, but ran {len(self.all_eigs)}')

        self.num_observables = min([self.svd_rank, self.total_observables])
        if self.all_eigs[0].shape[0] != self.num_observables:
            raise ValueError(f'Incorrect Observables Count: expected {self.num_observables}, but got {self.all_eigs[0].shape[0]}')



        # Combine DMD Data into Dataframe
        real_eigs = []
        imag_eigs = []
        full_modes = []
        
        for i in range(len(self.all_eigs)):
            eigs = self.all_eigs[i]
            modes = self.all_modes[i]
            
            real_eigs.append(eigs.real)
            imag_eigs.append(eigs.imag)
        
            norm_modes = [np.linalg.norm(modes[:,k]) / sum(np.linalg.norm(modes,axis=0)) for k in range(self.num_observables)]
            
            full_modes.append(norm_modes)
        
        real_eigs = np.array(real_eigs)
        imag_eigs = np.array(imag_eigs)
        full_modes = np.array(full_modes)
        target = np.array([i for i in range(self.num_cats) for _ in range(self.num_segments) for _ in range(self.num_samples)])[:,np.newaxis]
        sample = np.array([j for _ in range(self.num_cats) for j in range(self.num_samples) for _ in range(self.num_segments)])[:,np.newaxis]
        segment = np.array([k for _ in range(self.num_cats) for _ in range(self.num_samples) for k in range(self.num_segments)])[:,np.newaxis]
        
        columns = [f'eig_{i}' for i in range(self.num_observables)] + [f'real_{i}' for i in range(self.num_observables)] + [f'imag_{i}' for i in range(self.num_observables)] + [f'norm_mode_{i}' for i in range(self.num_observables)] + ['target','sample','segment']
        df = pd.DataFrame(data=np.hstack([np.array(self.all_eigs),real_eigs,imag_eigs,full_modes,target,sample,segment]),columns=columns)
        
        for col in df.columns:
            if 'eig' not in col:
                df[col] = df[col].astype(float)
        df['target'] = df['target'].astype(int)
        df['sample'] = df['sample'].astype(int)
        df['segment'] = df['segment'].astype(int)

        self.df = df

    def train_test_split(self, test_size, codebook_training_size, category_discovery=False, train_classes=range(3)):

        self.codebook_training_size = codebook_training_size

        # Define stratified splitter for even class representation
        self.test_size = test_size
        n_splits = int(1 / self.test_size)
        sgkf  = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        

        # Exclude certain classes from training set for category discovery
        if category_discovery:
            
            self.train_classes = np.array(train_classes)
            
            # Define classes used for training and testing
            # self.train_classes = self.rng.choice(range(len(self.cats)),size=num_train_classes,replace=False)
            self.test_classes = np.array(list(set(range(len(self.cats))) - set(self.train_classes)))

            self.logger.info(f"Train classes: {self.train_classes}, Test classes: {self.test_classes}")

            # create train and test split in the dataframe for training classes only
            # (preserving train/test proportion across target classes)
            df_train_subset = self.df.loc[self.df['target'].isin(self.train_classes)]
            groups = df_train_subset['sample']
            y      = df_train_subset['target']

            train_ilocs, test_ilocs = next(sgkf.split(df_train_subset, y=y.values, groups=groups))
            self.known_class_train_idx = df_train_subset.index[train_ilocs]
            known_class_test_idx = df_train_subset.index[test_ilocs]
            
            unknown_class_idx = self.df.loc[self.df['target'].isin(self.test_classes)].index

            self.test_idx = list(set(known_class_test_idx).union(set(unknown_class_idx)))

            
            if len(self.test_idx) != len(known_class_test_idx) + len(unknown_class_idx):
                raise ValueError(f'Train test split for category discovery mismatch: total test segments {len(self.test_idx)} != {len(known_class_test_idx)} + {len(unknown_class_idx)}')

            # Create train/test dataframes based on chosen indices
            self.df_train = df_train_subset.loc[self.known_class_train_idx].reset_index(drop=True)
            self.df_test  = self.df.loc[self.test_idx].reset_index(drop=True)

            self.logger.info(f"Train class counts:\n{self.df_train['target'].value_counts()}")
            self.logger.info(f"Test class counts:\n{self.df_test['target'].value_counts()}")

            class_size = self.codebook_training_size // len(self.train_classes) // self.num_segments


            
        
        # Train on all classes
        if not category_discovery:
    
            # create train and test split in the dataframe for all classes
            # (preserving train/test proportion across target classes)
            groups = self.df['sample']
            y      = self.df['target']
            train_idx, test_idx = next(sgkf.split(self.df, y=y, groups=groups))

            # Create train/test dataframes based on chosen indices
            self.df_train = self.df.iloc[train_idx].reset_index(drop=True)
            self.df_test  = self.df.iloc[test_idx].reset_index(drop=True)

            class_size = self.codebook_training_size // self.num_cats // self.num_segments



        self.df_train = self.df_train.sort_values(['target', 'sample', 'segment']).reset_index(drop=True)
        self.df_test  = self.df_test.sort_values(['target', 'sample', 'segment']).reset_index(drop=True)
        
        
        self.logger.info(f'Training set size: {self.df_train.shape[0]}')
        self.logger.info(f'Testing set size: {self.df_test.shape[0]}')
        
        
        samples_to_keep = self.df_train[['target','sample']].drop_duplicates().groupby('target').apply(lambda g: g.sample(class_size, random_state=self.seed)).reset_index(drop=True)
        self.df_sample = self.df.merge(samples_to_keep, on=["target", "sample"], how="inner")

        if len(self.df_sample) < self.codebook_training_size:
            print(f'Reducing codebook size from {self.codebook_training_size} to {len(self.df_sample)} based on codebook_training_size // <number of training classes> // num_segments')
        
        self.logger.info(f'Codebook training size: {self.codebook_training_size}')


    def create_codebook(self,include_plots=False):

        # Create Wasserstein Distance Matrix from downsampled training points
        metric_matrix = self._create_metric_matrix(self.df_sample)

        
        # # Optimize embedding dimension for metric matrix reconstruction accuracy
        # self.MDS_dimension = self._optimize_MDS_dim() # Need to write internal function
        

        # Create Euclidean Representation Based off Distance Matrix
        embedding = MDS(n_components=self.MDS_dimension, dissimilarity='precomputed', random_state=self.seed)
        self.X_transformed = embedding.fit_transform(metric_matrix)
        
        D_reconstructed = squareform(pdist(self.X_transformed))

        if include_plots:
            plt.imshow(D_reconstructed)
            plt.colorbar()
            plt.title('Reconstructed Wasserstein Distance Matrix') # Include reconstruction percent error
            self._save_current_fig("distance_matrix")


        if self.cluster_method == 'kmeans':
            self.centroid, label, inertia = k_means(self.X_transformed, self.num_clusters, random_state=self.seed)
        
        # Find the 5 closest points in the training data to the cluster centers (surrogate cluster centers)
        distances = cdist(self.centroid,self.X_transformed)
        cluster_indices = np.argmin(distances, axis=1)
        self.cluster_centers = self.X_transformed[cluster_indices,:]
        
        # Use those surrogate cluster centers to make the codebook dataframe
        self.codebook = self.df_sample.loc[cluster_indices].reset_index(drop=True)
        
        # Reassign the labels for the downsampled dataframe based on surrogate cluster centers
        distances = cdist(self.cluster_centers,self.X_transformed)
        self.sample_label = np.argmin(distances,axis=0)
        
        # Compare percent change in labels from using surrogate cluster centers
        tol = 1e-8
        num_changed = np.sum(abs(self.sample_label - label) >= tol)
        self.logger.info(f'{num_changed}/{self.df_sample.shape[0]} labels changed ({round(num_changed/self.df_sample.shape[0] * 100,2)}%) when choosing training points as cluster centers')

    
        # Represent training and testing datasets with codebook
        self.train_metric_matrix = self._create_metric_matrix(self.codebook,self.df_train)
        self.test_metric_matrix = self._create_metric_matrix(self.codebook,self.df_test)

        # Assign training samples to clusters
        train_label = np.argmin(self.train_metric_matrix,axis=0)
        self.df_train['cluster'] = train_label
        
        
        all_clusters = sorted(self.df_train['cluster'].unique())
        
        if include_plots:

            plt.imshow(self.train_metric_matrix[:,:100])
            plt.colorbar()
            plt.title('Training Metric Matrix (100 pts)')
            self._save_current_fig("train_metric_matrix")

            plt.imshow(self.test_metric_matrix[:,:100])
            plt.colorbar()
            plt.title('Testing Metric Matrix (100 pts)')
            self._save_current_fig("test_metric_matrix")
            
            for c in self.df_train.target.unique():
                counts = self.df_train.loc[self.df_train.target.eq(c),'cluster'].value_counts().reindex(all_clusters, fill_value=0).sort_index()
                counts.plot(kind='bar',figsize=(3,2), title=f'{self.cats[c]}')
                plt.xlabel('Cluster')
                plt.ylabel('Count')
                plt.tight_layout()
                self._save_current_fig(f"{c}_clusters")

            # Plot the data
            self.plot_data(3)
            self.plot_MDS(color_by_target=True)
            self.plot_MDS(color_by_target=False)




        # Create histogram data
        self.training_histograms, self.c_train_matrix, self.train_target = self._compute_cluster_histograms(self.df_train,self.train_metric_matrix)
        self.testing_histograms, self.c_test_matrix, self.test_target = self._compute_cluster_histograms(self.df_test,self.test_metric_matrix)
        
        self.N_ks = np.sum(self.training_histograms > 0,axis=0)
        self.N = len(self.train_target)
        
        self.inv_c_train_matrix = self.c_train_matrix * np.log(self.N / self.N_ks)
        self.inv_c_test_matrix = self.c_test_matrix * np.log(self.N / self.N_ks)

        


        # # Identify number of samples from each system category in both training and testing samples
        # self.num_train_samples = int(len(self.df_train.groupby(['target','sample']).groups) / self.num_cats)
        # self.num_test_samples = int(len(self.df_test.groupby(['target','sample']).groups) / self.num_cats)
        
        # # Create cluster assignments for each segment in training and testing set
        # self.train_cluster_assignments = np.argmin(self.train_metric_matrix,axis=0).reshape(self.num_cats,self.num_train_samples,self.num_segments)
        # self.test_cluster_assignments = np.argmin(self.test_metric_matrix,axis=0).reshape(self.num_cats,self.num_test_samples,self.num_segments)
        
        # # Count number of each cluster assignment in training set
        # self.N_ks = np.array([(np.sum(self.train_cluster_assignments == i,axis=2) > 0).ravel().sum() for i in range(self.num_clusters)])
        # self.N = self.df_train.shape[0] # Total number of time series samples
        
        # # Create histogram of cluster assignments for training data
        # self.n_train_matrix = np.concatenate([np.sum(self.train_cluster_assignments == i,axis=2).ravel() for i in range(self.num_clusters)]).reshape(self.num_clusters,self.num_cats*self.num_train_samples).T
        # self.c_train_matrix = self.n_train_matrix / self.num_segments
        # self.inv_c_train_matrix = self.c_train_matrix * np.log(self.N / self.N_ks)
        
        # # Create histogram of cluster assignments for testing data
        # self.n_test_matrix = np.concatenate([np.sum(self.test_cluster_assignments == i,axis=2).ravel() for i in range(self.num_clusters)]).reshape(self.num_clusters,self.num_cats*self.num_test_samples).T
        # self.c_test_matrix = self.n_test_matrix / self.num_segments
        # self.inv_c_test_matrix = self.c_test_matrix * np.log(self.N / self.N_ks)

        # # Extract target assignments (classification labels)
        # self.train_target = self.df_train['target'].values[-self.num_segments::-self.num_segments][-1::-1]
        # self.test_target = self.df_test['target'].values[-self.num_segments::-self.num_segments][-1::-1]
        

    def perform_classification(self, classifier=None, return_statement=False):
        """
        Perform classification with given classifer
        """


        class_name = classifier.__class__.__name__
        self.logger.info(f'Classifier Chosen: {class_name}')
        
        
        # Prepare classifier
        self.y_pred = {}
        scores, confs = {}, {}
        
        for c_train, c_test, name in [(self.c_train_matrix, self.c_test_matrix, 'reg_c'),(self.inv_c_train_matrix, self.inv_c_test_matrix, 'inverse_c')]:

            # Fit and score classifier
            classifier.fit(c_train, self.train_target)
            y_pred = classifier.predict(c_test)
            self.y_pred[name] = y_pred
            
            scores[name] = classifier.score(c_test,self.test_target)
            confs[name] = confusion_matrix(self.test_target, y_pred)

            self.logger.info(f'{name} score: {scores[name]}')
            self.logger.info("Confusion Matrix:\n%s", confs[name])

            
            # Save confusion matrix figure
            plt.figure(figsize=(8, 8))
            sns.heatmap(confs[name], annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=[f'Predicted {cat}' for cat in self.cats],
                        yticklabels=[f'True {cat}' for cat in self.cats])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix {scores[name]}:\n{class_name}, {name}\n')
            self._save_current_fig(f"confusion_matrix_{class_name}_{name}")
        
        if return_statement:
            return scores, confs


    def save(self, name: str = "model.pkl"):
        """Save *this* model object to <run_dir>/<name>."""
        path = self.run_dir / name
        joblib.dump(self, path)
        self.logger.info("Model saved to %s", path)

    @staticmethod
    def load(path):
        """Load a previously-saved model."""
        return joblib.load(path)

        
    def _repo_root(self) -> Path:
        """Return <repo root> regardless of where code is run."""
        # file …/src/kcm/koopman_category_model.py – two parents up is repo/
        return Path(__file__).resolve().parents[2]

    def _data_path(self, *parts) -> Path:
        """
        Convenience helper:
            data_path("dir", "file.ext") → <repo>/data/dir/file.ext
        Works from notebooks, scripts, tests, anywhere.
        """
        return self._repo_root() / "data" / Path(*parts)



    def _compute_dmd(self, X_full):
        """
        Compute DMD Eigenvalues, Modes, and Amplitudes using numpy matrix operations
        """

        m, n = X_full.shape
    
        if self.svd_rank < 0 or self.svd_rank > m:
            raise ValueError(f'Given svd rank is {self.svd_rank} yet data given only has dimension {m}')
        
        X = X_full[:,:-1]
        Y = X_full[:,1:]
        
        # SVD of X
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        
        # r rank of svd of X
        U_r = U[:,:self.svd_rank]
        s_r = s[:self.svd_rank]
        Vh_r = Vh[:self.svd_rank,:]
        
        # Calculate A_tilde
        U_r_star = U_r.conj().T
        V_r = Vh_r.conj().T
        S_r_inv = np.diag(1.0 / s_r)
        
        A_tilde = U_r_star @ Y @ V_r @ S_r_inv
        
        # Eigendecomposition of A_tilde
        eigs, W = np.linalg.eig(A_tilde)
        
        # Koopman Modes & amplitudes
        modes = Y @ V_r @ S_r_inv @ W
        b, _, _, _ = np.linalg.lstsq(modes,X[:,0])
    
        return eigs, modes, b


    def _create_metric_matrix(self, df1, df2=None, plot_matrix=False):

        square_matrix = False

        # df2 should only be 'None' when creating codebook from self.df_sample 
        if df2 is None:
            square_matrix = True
            df2 = df1.copy()
            dim1 = dim2 = df1.shape[0]
            total_metric_calculations = int(self.codebook_training_size**2/2)

            # Square Matrix
            metric_statement = f'(1/2) * {self.codebook_training_size}^2 = {total_metric_calculations} Wasserstein distance metrics'

        else:
            dim1, dim2 = df1.shape[0], df2.shape[0]
            total_metric_calculations = int(dim1 * dim2)
            metric_statement = f'{dim1} * {dim2} = {total_metric_calculations} Wasserstein distance metrics'
        
        metric_matrix = np.zeros((dim1,dim2))
        
        
        # Columns for extracting data
        eig_columns = [col for col in df1.columns if 'eig' in col]
        norm_mode_columns = [col for col in df1.columns if 'norm_mode' in col]

        
        df1.reset_index(drop=True,inplace=True)
        df2.reset_index(drop=True,inplace=True)

        for i in tqdm(df1.index, desc=metric_statement):
            
            # Only include half of computations if square matrix
            df2_indices = range(i) if square_matrix else df2.index
            
            for j in df2_indices:
        
                # Eigenvalues
                l1 = df1.loc[i,eig_columns].values[:,np.newaxis]
                l2 = df2.loc[j,eig_columns].values[:,np.newaxis]
                
                # Mass vectors (normed modes)
                m1 = df1.loc[i,norm_mode_columns].values.real
                m2 = df2.loc[j,norm_mode_columns].values.real
        
                # Compute Wasserstein Metric
                metric_matrix[i,j] = self._compute_wasserstein_metric(l1,l2,m1,m2)

        if square_matrix:
            i_triu = np.triu_indices_from(metric_matrix, k=1)
            metric_matrix[i_triu] = metric_matrix.T[i_triu]

        return metric_matrix


        
    def _compute_wasserstein_metric(self, l1, l2, m1, m2):
        
        # --- Step 1: Sample Inputs ---
        n = self.num_observables
        n_bar = self.num_observables
    
        m1 = m1 / np.sum(m1)
        m2 = m2 / np.sum(m2)
        
        assert np.isclose(np.sum(m1), 1.0)
        assert np.isclose(np.sum(m2), 1.0)
        
        # --- Step 2: Compute cost matrix C (shape n x n_bar) ---
        C = np.linalg.norm(l1[:, None, :] - l2[None, :, :], axis=2) ** 2
        # Flatten to 1D for linprog
        c = C.flatten()  # size (n * n_bar,)
        
        # --- Step 3: Equality Constraints ---
        
        # Total number of variables
        N = n * n_bar
        
        # 1. Row sum constraints: each row i must sum to m[i]
        A_eq_rows = np.zeros((n, N))
        for i in range(n):
            for j in range(n_bar):
                A_eq_rows[i, i * n_bar + j] = 1
        b_eq_rows = m1
        
        # 2. Column sum constraints: each column j must sum to m_bar[j]
        A_eq_cols = np.zeros((n_bar, N))
        for j in range(n_bar):
            for i in range(n):
                A_eq_cols[j, i * n_bar + j] = 1
        b_eq_cols = m2
        
        # 3. Total mass constraint
        A_eq_total = np.ones((1, N))
        b_eq_total = np.array([1.0])
        
        # Combine constraints
        A_eq = np.vstack([A_eq_rows, A_eq_cols, A_eq_total])
        b_eq = np.concatenate([b_eq_rows, b_eq_cols, b_eq_total])
        
        # --- Step 4: Bounds ---
        bounds = [(0, None)] * N  # rho_ij >= 0
        
        # --- Step 5: Solve ---
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        # --- Step 6: Extract solution ---
        if result.success:
            rho_star = result.x.reshape((n, n_bar))
            transport_cost = np.sum(rho_star * C)
            wasserstein_metric = transport_cost ** (1 / self.q)
        else:
            self.logger.info("Optimization failed:", result.message)
    
        return wasserstein_metric



    # Create histogram of cluster assignments for a given dataframe and metric matrix
    def _compute_cluster_histograms(self,df,metric_matrix):

        samples = (
            df
            .sort_values(['target', 'sample', 'segment'])
            .drop_duplicates(['target', 'sample'])
            [['target', 'sample']]
            .reset_index(drop=True)
        )
    
        cluster_assignments = np.argmin(metric_matrix, axis=0)
    
        histograms = np.zeros((samples.shape[0],self.num_clusters))

        # Loop through each individual sample
        for i, (target, sample) in samples.iterrows():
        
            seg_start = i * self.num_segments
            seg_end = seg_start + self.num_segments
            assign_clusters = cluster_assignments[seg_start:seg_end]
        
            histogram = np.bincount(assign_clusters, minlength=self.num_clusters)
            histograms[i,:] = histogram
    
        
        c_matrix = histograms / self.num_segments
    
        targets = samples.target
    
        return histograms, c_matrix, targets


    def shutdown_logger(self):
        """Immediately close all handlers to release the log file."""
        for h in self.logger.handlers[:]:
            h.close()
            self.logger.removeHandler(h)

    def reconstruction_error(self):
        pass

    def _optimize_MDS_dim(self):
        pass
        

    def get_params(self):
        return {
            'dmd_rank': self.dmd_rank,
            'cluster_method': self.cluster_method,
            'classifier': self.classifier.__class__.__name__
        }

    def _save_current_fig(self, name: str, ext="png", dpi=300):
        plt.gcf().savefig(self.run_dir / f"{name}.{ext}", dpi=dpi,
                          bbox_inches="tight")
        plt.close()


    def plot_data(self, samples_to_plot=3):
        
        fig = make_subplots(
            rows=1, cols=self.num_cats,
            # specs=[[{'type': 'scene'}]*num_cats],
            # subplot_titles=[f'System {i+1}' for i in range(num_cats)]
            subplot_titles=[val.replace('_','<br>') for val in self.cats]
        )

        samples_to_plot = self.num_samples if samples_to_plot is None else samples_to_plot
        
        for i, cat in enumerate(self.cats):
        
            for j in range(samples_to_plot):
                trace = go.Scatter(
                    x=self.dataset[cat][j]['t'], y=self.dataset[cat][j]['y'][1],
                    mode='lines',
                    line=dict(width=1),
                    name=f"{cat}"
                )
            
                # Add trace to the correct subplot
                fig.add_trace(trace, row=1, col=i+1)
        
        # Update layout
        fig.update_layout(
            # title=dict(x=0.5, text="3D Line Plot Subplots"),
            height=350,
            width=1300,
            showlegend=False
        )
        fig.update_yaxes(range=[-20, 20])

        fig.write_html(self.run_dir / "time_series_samples.html")



    def plot_MDS(self,color_by_target=True):


        if color_by_target:
            title = f'Data Clusters (3/{self.MDS_dimension} dimensions)<br>Colored by True Target'
            data_color = self.df_sample.target
            true_centroid_color = 'black'
            data_centroid_color = 'green'
        else:
            title = f'Data Clusters (3/{self.MDS_dimension} dimensions)<br>Colored by Cluster Label'
            data_color = self.sample_label
            true_centroid_color = np.arange(1,self.num_clusters+1)
            data_centroid_color = 'black'
    
            
        full_data = go.Scatter3d(x=self.X_transformed[:,0],
                                 y=self.X_transformed[:,1],
                                 z=self.X_transformed[:,2],
                                 mode='markers',
                                 marker=dict(color=data_color,size=3),
                                 name='Data')
    
        centers = go.Scatter3d(x=self.centroid[:,0],
                               y=self.centroid[:,1],
                               z=self.centroid[:,2],
                               mode='markers',
                               marker=dict(color=true_centroid_color,size=8),
                               name='Centroids')
        
        data_centers = go.Scatter3d(x=self.cluster_centers[:,0],
                                    y=self.cluster_centers[:,1],
                                    z=self.cluster_centers[:,2],
                                    mode='markers',
                                    marker=dict(color=data_centroid_color,size=8),
                                    name='Surrogate Centroids')
    

        data = [full_data,centers,data_centers]

        
        layout=go.Layout(width=900,height=600,
                         title=dict(x=0.5,text=title))
        
        fig = go.Figure(data=data,layout=layout)

        name = 'colored_by_target' if color_by_target else 'colored_by_cluster'
        fig.write_html(self.run_dir / f"MDS_{name}.html")
        

    def create_phase_diagrams(self, num_to_plot=20, inv_c_pred=True):

        # Choose y_predictions based on whether inv_c matrix was used or not
        c_name = 'inverse_c' if inv_c_pred else 'reg_c'
        y_pred = self.y_pred[c_name]
        
        # Extract Dynamic System Names
        train_sets = self.df_train[['target', 'sample']].drop_duplicates().reset_index(drop=True)
        test_sets = self.df_test[['target', 'sample']].drop_duplicates().reset_index(drop=True)
        systems = [cat.replace('_',' ').capitalize() for cat in self.cats]


        # Identify max and min position and velocity for plotting
        max_x = 0
        max_v = 0
        for i in range(len(self.all_data)):
            x = max(abs(self.all_data[i][:,0]))
            v = max(abs(self.all_data[i][:,1]))
            if x > max_x:
                max_x = x
            if v > max_v:
                max_v = v


        # Plot Phase Diagrams for Full Test Set
        test_index = 0
        
        for s, system in tqdm(enumerate(systems), desc=f'Creating Phase Diagrams'):
            
            figs, axes = plt.subplots(5,4,figsize=(14,14)) # Should be based on num_to_plot
            ax = axes.ravel()
            
            for ind in range(num_to_plot):
                
                # Identify target & sample number
                target = test_sets.loc[test_index,'target']
                sample = test_sets.loc[test_index,'sample']
                
                # Identify segment indices in self.all_data
                start = (self.num_samples * self.num_segments) * target + self.num_segments * sample
                segment_indices = np.arange(start,start+5)
                
                start = self.num_samples
                for i,d in enumerate(segment_indices):
                
                    x, dx = self.all_data[d][:,:2].T
                    
                    ax[ind].axhline(y=0, color='k', linewidth=2)
                    ax[ind].axvline(x=0, color='k', linewidth=2)
                    ax[ind].plot(x,dx,label=f'Segment {i+1}')
                    for j in range(0, len(x)-1, 50):
                        ax[ind].annotate("",
                                     xy=(x[j+1], dx[j+1]),
                                     xytext=(x[j], dx[j]),
                                     arrowprops=dict(arrowstyle="->", color="blue", lw=1))
                
                title = [f'True: {systems[target]}',
                         f'Pred: {systems[y_pred[test_index]]}']
            
                if self.test_target[test_index] == y_pred[test_index]:
                    ax[ind].set_facecolor('palegreen')
                else:
                    ax[ind].set_facecolor('lightcoral')
                
                ax[ind].set_title('\n'.join(title))
                ax[ind].set_xlim([-np.ceil(abs(max_x)),np.ceil(abs(max_x))])
                ax[ind].set_ylim([-np.ceil(max_v),np.ceil(max_v)])
                ax[ind].grid()
                ax[ind].legend(fontsize=6)
        
                test_index += 1
        
            nrows, ncols = 5, 4
        
            for i, a in enumerate(ax):
                row = i // ncols
                col = i % ncols
            
                if row == nrows - 1:  # bottom row
                    a.set_xlabel("x")
            
                if col == 0:  # leftmost column
                    a.set_ylabel("dx/dt")
        
            
            plt.suptitle(f'{system} Phase Diagrams\n({c_name} training matrix)',fontsize=20)
            plt.tight_layout()
            self._save_current_fig(f"results_{self.cats[s]}")