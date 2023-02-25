#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm

plt.style.use(r'..\casper_style.mplstyle')
#%%
def generate_gaussian(mu, sigma, size=50):
    values = np.random.normal(mu, sigma, size)
    return values
def gaussian_function(x, mu, sigma):
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x-mu)**2/(2*sigma**2))

def likelihood(data, function, **kwargs):
    likelihood_value = np.prod(function(data, **kwargs))
    return likelihood_value

def log_likelihood(data, function, **kwargs):
    log_likelihood_value = -np.sum(np.log(function(data, **kwargs)))
    if log_likelihood_value > 10000:
        log_likelihood_value = 10000
    return log_likelihood_value

def max_likelihood():
    pass

def raster_scan(data):
    fig, ax = plt.subplots()
    sns.heatmap(data, ax=ax, norm=LogNorm())
    
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\sigma$')
    ax.set_title('Raster Plot')

#%%
np.seed = 42
mu_true = 0.2
sigma_true = 0.1
value_range = 4*sigma_true
data = generate_gaussian(mu_true, sigma_true)
mu_raster = np.linspace(mu_true-value_range, mu_true+value_range, 50)
sigma_raster = np.linspace(0.01, sigma_true+value_range/2, 50)
#%%
mesh = np.meshgrid(mu_raster, sigma_raster)
mu_list = mesh[0].flatten()
sigma_list = mesh[1].flatten()
raster_data = np.array([log_likelihood(data, gaussian_function, mu = mu, sigma = sigma) for mu, sigma in zip(mu_list, sigma_list)]).reshape(len(mu_raster),len(sigma_raster))
raster_data += np.abs(np.min(raster_data)) + 1
result_df = pd.DataFrame(raster_data, columns=np.around(mu_raster,2), index=np.around(sigma_raster,2))
# %%
raster_scan(result_df)

# %%
