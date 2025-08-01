import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.decomposition import PCA


def extract_feature_vector(t, series, normalize=False):

    dt = np.mean(np.diff(t))

    if normalize:
        series = (series - series.mean(axis=0)) / series.std(axis=0)
    
    # skew
    sk = skew(series,axis=1)
    
    # kurtosis
    kurt = kurtosis(series,axis=1)
    
    rows, cols = np.where(np.diff(np.sign(series)))
    spacings = [np.diff(cols[rows == i]) * dt for i in range(series.shape[0])]
    
    # zeros
    num_zero_crossings = np.array([(rows == i).sum() for i in range(series.shape[0])])
    avg_zero_spacing = np.array([s.mean() if len(s) > 0 else np.nan for s in spacings])
    var_zero_spacing = np.array([s.var() if len(s) > 0 else np.nan for s in spacings])
    
    # mean sign
    mean_sign = np.sign(series).mean(axis=1)
    
    # energy
    energy = np.sum(series**2,axis=1)
    
    _dy = np.gradient(series, dt, axis=1)
    
    # mean absolute derivative
    mean_abs_deriv = abs(_dy).mean(axis=1)
    
    # max derivative
    max_deriv = abs(_dy).max(axis=1)
    
    # derivative:energy ratio
    deriv_energy_ratio = (_dy**2).sum(axis=1) / (energy + 1e-8)
    
    _peak_diffs = [np.diff(find_peaks(xi)[0]) for xi in series]
    
    # average peak period
    avg_peak_period = np.array([diffs.mean() for diffs in _peak_diffs])
    
    # variance peak period
    var_peak_period = np.array([diffs.var() for diffs in _peak_diffs])
    
    # correlation coefficients and covariances
    corrs = np.array([np.corrcoef(series[i], series[j])[0, 1] for i in range(3) for j in range(i + 1, 3) ])
    covs = np.array([np.cov(series[i], series[j])[0, 1] for i in range(3) for j in range(i + 1, 3) ])
    
    # trajectory length
    _trajectory_diff = np.diff(series, axis=1)
    _segment_lengths = np.linalg.norm(_trajectory_diff, axis=0)
    trajectory_length = np.sum(_segment_lengths)
    
    # pca variance ratios
    pca = PCA(n_components=min(3, series.shape[1]))
    pca.fit(series.T)
    pca_variance_ratios = pca.explained_variance_ratio_
    
    # anisotropy
    max_var = pca_variance_ratios[0]
    min_var = pca_variance_ratios[-1]
    pca_anisotropy = max_var / (min_var + 1e-8)

        # Concatenate all features into one vector
    features = np.concatenate([
        sk, kurt, num_zero_crossings, avg_zero_spacing, var_zero_spacing,
        mean_sign, energy, mean_abs_deriv, max_deriv, deriv_energy_ratio,
        avg_peak_period, var_peak_period, corrs, covs,
        [trajectory_length], pca_variance_ratios, [pca_anisotropy]
    ])

    return features




def extract_important_features():
    pass