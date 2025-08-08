import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler



class BasicFeatureExtractor():
    def __init__(self, num_cats=None, num_samples=None, system_dimension=None, data_path=None, noisy_data=True, noise_std=None,
                 run_root: str | Path | None = None, seed=None):

       
        # Set seed for reproducibility
        self.seed = seed if seed is not None else 0
        self.rng  = np.random.default_rng(self.seed)
        
        # Dataset Parameters
        self.num_cats = num_cats
        self.num_samples = num_samples
        self.system_dimension = system_dimension
        self.data_path = data_path
        self.noisy_data = noisy_data
        self.noise_std = noise_std
        self.dataset = self._load_data(data_path)
        self.cats = list(self.dataset.keys())
        self.m, self.n = self.dataset[self.cats[0]][0]['y'].shape
        
        # Clustering Parameters
        self.labels = None
        
        # Classification Parameters
        self.scaler = StandardScaler()



    def _load_data(self, data_path):
        """
        Load in the data based on either the provided data path
        or the default path if none is provided.
        """

        if self.data_path is None:
            
            samples_name = f'noisy_{self.noise_std}_samples' if self.noisy_data else 'samples'
            
            file_path = self._data_path(
                f"{self.system_dimension}-dimensional-systems",
                f"dataset_{self.num_cats}_class_{self.num_samples}_{samples_name}.pkl"
            )
            self.data_path = file_path
            
        else:
            file_path = Path(self.data_path)

        print(f'Loading data in at {self.data_path}...')
        
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    
    def batch_extract_features(self, normalize_inputs=False, drop_na=True):


        def _flatten_features(features):
        
            flattened = {}
            for key, item in features.items():
                if isinstance(item, np.ndarray):
                    for i in range(len(item)):
                        flattened[f'{key}_{i}'] = item[i]
                else:
                    flattened[key] = item
            return flattened
        
    
        rows = []
        for system_name, records in tqdm(self.dataset.items(), desc='Calculating features for all systems'):
            for rec in records:
                t = rec['t']
                y = rec['y']
                features = self._extract_feature_vector(t, y, normalize=normalize_inputs)
    
    
                flat = _flatten_features(features)
                flat['system_name'] = system_name
                rows.append(flat)
    
        # Create DataFrame
        df = pd.DataFrame(rows)
    
        system_mapping = {system: i for i, system in enumerate(self.dataset.keys())}
        df['target'] = df['system_name'].map(system_mapping)
    
    
        # max_size = 1e4
        if drop_na:
            dropping_columns = [col for col in df if df[col].isna().sum() > 0]
            df.drop(columns=dropping_columns,inplace=True)
            print(f'Dropped columns: {dropping_columns}')
            print(f'Final shape: {df.shape}')

    
            # df_feats = df.drop(columns=['system_name','system'])
            # large_value_columns = df_feats.max() > max_size

        df['count'] = np.arange(self.num_samples * self.num_cats).astype(int)

        self.df = df
                
    def train_test_split(self, test_size, category_discovery=False, train_classes=range(3)):

        self.test_size = test_size
        self.df['count'] = np.arange(len(self.df))
        

        # Exclude certain classes from training set for category discovery
        if category_discovery:
            
            self.train_classes = np.array(train_classes)
            
            # Define classes used for training and testing
            # self.train_classes = self.rng.choice(range(len(self.cats)),size=num_train_classes,replace=False)
            self.test_classes = np.array(list(set(range(self.num_cats)) - set(self.train_classes)))
            

            # create train and test split in the dataframe for training classes only
            # (preserving train/test proportion across target classes)
            df_train_subset = self.df.loc[self.df['target'].isin(self.train_classes)]

            # self.df_train = df_train_subset.groupby('target').apply(lambda x : x.sample(n=int((1-self.test_size)*self.num_samples), random_state=self.seed)).reset_index(drop=True)
            self.df_train = df_train_subset.groupby('target').apply(lambda x : x.sample(frac=(1 - self.test_size), random_state=self.seed)).reset_index(drop=True)
            self.known_class_train_idx = np.array(list(set(self.df_train['count'])))
            known_class_test_idx = self.df.loc[self.df['target'].isin(self.train_classes) & ~self.df['count'].isin(self.known_class_train_idx),'count'].values
            
            unknown_class_idx = self.df.loc[self.df['target'].isin(self.test_classes)].index

            self.test_idx = list(set(known_class_test_idx).union(set(unknown_class_idx)))
            
            if len(self.test_idx) != len(known_class_test_idx) + len(unknown_class_idx):
                raise ValueError(f'Train test split for category discovery mismatch: total test segments {len(self.test_idx)} != {len(known_class_test_idx)} + {len(unknown_class_idx)}')

            # Create train/test dataframes based on chosen indices
            # self.df_train = df_train_subset.loc[df_train_subset['count'].isin(self.known_class_train_idx)].reset_index(drop=True)
            self.df_test  = self.df.loc[self.df['count'].isin(self.test_idx)].reset_index(drop=True)

            # print(f"Train class counts:\n{self.df_train['target'].value_counts()}")
            # print(f"Test class counts:\n{self.df_test['target'].value_counts()}")
            
            
        
        # Train on all classes
        if not category_discovery:

            self.df_train = self.df.groupby('target').apply(lambda x : x.sample(n=int((1-test_size)*self.num_samples), random_state=self.seed)).reset_index(drop=True)
            training_indices = list(set(self.df_train['count']))
            self.df_test = self.df.loc[~self.df['count'].isin(training_indices)].reset_index(drop=True)

            print(f"Train class counts:\n{self.df_train['target'].value_counts()}")
            print(f"Test class counts:\n{self.df_test['target'].value_counts()}")



        self.df_train = self.df_train.sort_values(['target']).reset_index(drop=True)
        self.df_test  = self.df_test.sort_values(['target']).reset_index(drop=True)
        
        
        print(f'Training set size: {self.df_train.shape[0]}')
        print(f'Testing set size: {self.df_test.shape[0]}')
        
        
    
    # def extract_important_features():
    #     pass


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
        

    def _extract_feature_vector(self, t, series, normalize):

        assert series.shape[0] == self.system_dimension, \
        f"Expected shape (dims, time), got {series.shape}"
            
        dt = np.mean(np.diff(t))
    
        if normalize:
            series = (series - series.mean(axis=0)) / series.std(axis=0)
    
        features = {}
        
        # skew
        features['skew'] = skew(series,axis=1)
        
        # kurtosis
        features['kurt'] = kurtosis(series,axis=1)
        
        _rows, _cols = np.where(np.diff(np.sign(series)))
        _spacings = [np.diff(_cols[_rows == i]) * dt for i in range(series.shape[0])]
        
        # zeros
        features['num_zero_crossings'] = np.array([(_rows == i).sum() for i in range(series.shape[0])]) / t.max()
        features['avg_zero_spacing'] = np.array([s.mean() if len(s) > 0 else np.nan for s in _spacings])
        features['var_zero_spacing'] = np.array([s.var() if len(s) > 0 else np.nan for s in _spacings])
        
        # mean sign
        features['mean_sign'] = np.sign(series).mean(axis=1)
        
        # energy
        features['energy'] = np.sum(series**2,axis=1)
        
        _dy = np.gradient(series, dt, axis=1)
        
        # mean absolute derivative
        features['mean_abs_deriv'] = abs(_dy).mean(axis=1)
        
        # max derivative
        features['max_deriv'] = abs(_dy).max(axis=1)
        
        # derivative:energy ratio
        features['deriv_energy_ratio'] = (_dy**2).sum(axis=1) / (features['energy'] + 1e-8)
        
        _peak_diffs = [np.diff(find_peaks(xi)[0]) for xi in series]
        
        # average peak period
        # features['avg_peak_period'] = np.array([diffs.mean() for diffs in _peak_diffs])
        features['avg_peak_period'] = np.array([
            diffs.mean() if len(diffs) > 0 else np.nan for diffs in _peak_diffs
        ])

        
        # variance peak period
        # features['var_peak_period'] = np.array([diffs.var() for diffs in _peak_diffs])
        features['var_peak_period'] = np.array([
            diffs.var() if len(diffs) > 0 else np.nan for diffs in _peak_diffs
        ])
        
        # correlation coefficients and covariances
        features['corrs'] = np.array([np.corrcoef(series[i], series[j])[0, 1] for i in range(self.system_dimension) for j in range(i + 1, self.system_dimension) ])
        features['covs'] = np.array([np.cov(series[i], series[j])[0, 1] for i in range(self.system_dimension) for j in range(i + 1, self.system_dimension) ])
        
        # trajectory length
        if self.system_dimension == 3:
            _trajectory_diff = np.diff(series, axis=1)
            _segment_lengths = np.linalg.norm(_trajectory_diff, axis=0)
            features['trajectory_length'] = np.sum(_segment_lengths)
        
        # pca variance ratios
        _pca = PCA(n_components=min(self.system_dimension, series.shape[1]))
        _pca.fit(series.T)
        features['pca_variance_ratios'] = _pca.explained_variance_ratio_
        
        # anisotropy
        max_var = features['pca_variance_ratios'][0]
        min_var = features['pca_variance_ratios'][-1]
        features['pca_anisotropy'] = max_var / (min_var + 1e-8)
    
        # Fourier Transform (FFT)   
    
        dominant_freq_fft = []
        centroid_fft = []
        bandwidth_fft = []
        energy_fft = []
    
        if self.system_dimension == 3:
            magnitude = np.linalg.norm(series, axis=0, keepdims=True)
            series = np.vstack([series, magnitude])
        
        for xi in series:
            
            N = len(xi)
            yf = fft(xi)
            xf = fftfreq(N, dt)[:N//2]
            
            amplitudes = 2.0/N * np.abs(yf[:N//2])
            dominant_freq = xf[np.argmax(amplitudes)]
            centroid = np.sum(xf * amplitudes) / np.sum(amplitudes)
            bandwidth = np.sqrt(np.sum(((xf - centroid)**2) * amplitudes) / np.sum(amplitudes))
            energy = np.sum(amplitudes**2)
        
            dominant_freq_fft.append(dominant_freq)
            centroid_fft.append(centroid)
            bandwidth_fft.append(bandwidth)
            energy_fft.append(energy)
    
        features['dominant_freq_fft'] = np.array(dominant_freq_fft)
        features['centroid_fft'] = np.array(centroid_fft)
        features['bandwidth_fft'] = np.array(bandwidth_fft)
        features['energy_fft'] = np.array(energy_fft)


        # Consider adding the following features:
            # Spectral entropy
            # Approximate entropy
            # Sample entropy
            # Average deviation from zero over time (across each dimension, or for absolute magnitude)
            # Signal smoothness
            # Number of peaks
            # Phase shift (fft)
    
        return features