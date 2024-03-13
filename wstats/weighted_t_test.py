import numpy as np
from numba import njit, prange
from scipy.stats import t

def weighted_mean(x, w):
    """Calculate the weighted mean."""
    return np.average(x, weights=w)


def weighted_var(x, w):
    """Calculate the weighted variance."""
    mean = weighted_mean(x, w)
    return np.average((x-mean)**2, weights=w)


#def bootstrap_se(x, w, n_boot=1000):
#    """Estimate the standard error of the mean using bootstrapping."""
#    np.random.seed(42)  # For reproducible results
#    boot_means = np.empty(n_boot)
#    n = len(x)
#    for i in range(n_boot):
#        sample_indices = np.random.choice(n, size=n, replace=True, p=w/np.sum(w))
#        sample = x[sample_indices]
#        boot_means[i] = weighted_mean(sample, w[sample_indices])
#    return np.std(boot_means, ddof=1)


@njit
def weighted_sample(weights, n_samples):
    """Perform weighted sampling of indices based on weights."""
    cumulative_weights = np.cumsum(weights)
    total = cumulative_weights[-1]
    rnd = np.random.random(n_samples) * total
    return np.searchsorted(cumulative_weights, rnd)


@njit
def manual_std(values, ddof=1):
    """Manually compute standard deviation, compatible with Numba."""
    mean = np.sum(values) / len(values)
    variance = np.sum((values - mean) ** 2) / (len(values) - ddof)
    return np.sqrt(variance)


@njit
def bootstrap_se(x, w, n_boot=1000):
    """Estimate the bootstrap standard error using weighted sampling, with manual standard deviation calculation."""
    boot_means = np.empty(n_boot)
    for i in prange(n_boot):
        indices = weighted_sample(w, len(x))
        sampled_means = np.sum(x[indices] * w[indices]) / np.sum(w[indices])
        boot_means[i] = sampled_means
    return manual_std(boot_means, ddof=1)


def wtd_t_test(x, y=None, wx=None, wy=None, alternative="two-sided", bootse=False, bootn=1000):
    if wx is None:
        wx = np.ones_like(x)
    if isinstance(y, np.ndarray) and wy is None:
        wy = np.ones_like(y)
    if y is None:
        y = np.array([0])
        wy = np.array([1])
    #
    mx = weighted_mean(x, wx)
    vx = weighted_var(x, wx)
    #
    my = weighted_mean(y, wy)
    vy = weighted_var(y, wy)
    #
    n = np.sum(wx)
    n2 = np.sum(wy)
    #
    if bootse:
        sx = bootstrap_se(x, wx, n_boot=bootn)
        sy = bootstrap_se(y, wy, n_boot=bootn) if y.size > 1 else 0
        sxy = np.sqrt(sx**2 + sy**2)
    else:
        sxy = np.sqrt(vx/n + vy/n2) if y.size > 1 else np.sqrt(vx/n)
    #
    t_stat = (mx - my) / sxy
    df = n + n2 - 2
    #
    # Adjust degrees of freedom for Welch's t-test if applicable
    if y.size > 1 and not bootse:
        df = (((vx/n) + (vy/n2))**2) / (((vx/n)**2)/(n-1) + ((vy/n2)**2)/(n2-1))
    #
    # Calculate p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df))
    elif alternative == "less":
        p_value = t.cdf(t_stat, df)
    else:  # alternative == "greater"
        p_value = 1 - t.cdf(t_stat, df)
    #
    return {"t_stat": t_stat, "p_value": p_value, "df": df}



