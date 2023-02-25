#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import sympy
from sympy import symbols, lambdify
plt.style.use(r'..\casper_style.mplstyle')
#%%
n_small = 60
k_big = 100
k_small = 10
#n_big = k_big/k_small*n_small
# %%
def hypergeometric_pmf(n_big, n_small, k_big, k_small):
    probability = stats.hypergeom.pmf(k=k_small,M=n_big, n=k_big, N=n_small)
    return probability

def bayes(likelihood, prior, marginal_likelihood):
    return (likelihood*prior)/marginal_likelihood

def plot_posterior_pdf(x_list, likelihood_function, priors, k_small, n_small, k_big, marginal_likelihood=1, ax=None):
    likelihoods = np.array([likelihood_function(n_big, n_small, k_big, k_small) for n_big in x_list])
    posteriors = np.array([bayes(likelihood=likelihood, prior=prior, marginal_likelihood=marginal_likelihood) for likelihood, prior in zip(likelihoods, priors)])
    if not ax:
        fig, ax = plt.subplots()
    norm = np.max(likelihoods)/np.max(posteriors)
    sns.lineplot(x=x_list, y=posteriors*norm, label=f'Posterior PDF: k={k_small}',ax=ax)
    sns.lineplot(x=x_list, y=likelihoods, label=f'Likelihood: k={k_small}',ax=ax, linestyle='--')
    ax.set_xlabel('Fish population size')
    ax.set_ylabel('Probability')
    ax.set_title('Fish population with Bayesian Estimation')

    ax.legend()
    return ax

def generate_gaussian_prior(x_list, mean, sigma):
    return stats.norm.pdf(x_list, mean, sigma)

def z(vals, sigmas):
    xSym, ySym = symbols("x y")
    symlist = [xSym, ySym]
    
    z = xSym/ySym
    
    def variances(func, symbols, values, sigmas):
        variance = np.zeros(len(symbols))
        for idx, (symbol, sigma) in enumerate(zip(symbols,sigmas)):
            f = lambdify(symbols, func.diff(symbol)**2 * sigma **2)
            variance[idx] = f(*values)
        return variance
    Vz = variances(z, symlist, vals, sigmas)
    sigmaz = np.sqrt(np.sum(Vz))
    zvalue = lambdify(symlist, z)(*vals)
    
    return zvalue, sigmaz

vals = [5000, 10]
sigmas = [300, 1]

z(vals, sigmas)
#%%
# Prior = 1
x_list = np.linspace(k_big, 2000, 200)
priors = np.array([1]*len(x_list))
ax = plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors,n_small=n_small, k_big=k_big, k_small=10)
plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=15, ax=ax)

# %%
# Prior = 1/N
x_list = np.linspace(k_big, 2000, 200)
priors = [1/N for N in x_list]
ax = plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=10)
plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=15, ax=ax)

# %%
# Prior = Gaussian
values = [5000, 10]
sigmas = [300, 1]
mean, sigma = z(values, sigmas)
x_list = np.linspace(k_big, 2000, 200)
priors = generate_gaussian_prior(x_list, mean, sigma)
ax = plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=10)
plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=15, ax=ax)

# %%
n_small = 30
k_big = 50
# Prior = Gaussian
values = [5000, 10]
sigmas = [300, 1]
mean, sigma = z(values, sigmas)
x_list = np.linspace(k_big, 2000, 200)
priors = generate_gaussian_prior(x_list, mean, sigma)
ax = plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=4)
plot_posterior_pdf(x_list, hypergeometric_pmf, priors=priors, n_small=n_small, k_big=k_big, k_small=8, ax=ax)

# %%
top = np.random.normal(5000, 300, 10000)
bottom = np.random.normal(10,1,10000)
total = np.random.normal(500, 58.309518948453004, 10000)
fig, ax = plt.subplots()
sns.histplot(sorted(total), label='total', ax=ax, color='b', alpha=0.5)
sns.histplot((sorted(top/bottom)), label='comb', ax=ax, alpha=0.5)
ax.legend()
print(np.mean(top/bottom))
np.std(top/bottom)
# %%
