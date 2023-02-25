#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import scipy.optimize
from scipy import stats, integrate
import pandas as pd
import seaborn as sns
plt.style.use(r'..\casper_style.mplstyle')

#%%
def beta_distrib(x,alpha,beta):
    return (gamma(alpha + beta)/(gamma(alpha)*gamma(beta)))*(x**(alpha-1))*((1-x)**(beta-1))

def binomial(p, k, n):
    probability = stats.binom.pmf(k=k,n=n, p=p)
    return probability

def bayes(likelihood, prior, marginal_likelihood):
    return (likelihood*prior)/marginal_likelihood

def plot_posterior_pdf(p_list, likelihood_function, k, n, marginal_likelihood=1, ax=None):
    likelihoods = np.array([likelihood_function(p,k,n) for p in p_list])
    priors = [beta_distrib(p, 5, 17) for p in p_list]
    posteriors = np.array([bayes(likelihood=likelihood, prior=prior, marginal_likelihood=marginal_likelihood) for likelihood, prior in zip(likelihoods, priors)])
    if not ax:
        fig, ax = plt.subplots()
    norm_posterior = integrate.trapz(posteriors, p_list)
    norm_prior = integrate.trapz(priors, p_list)
    norm_likelihood = integrate.trapz(likelihoods, p_list)

    sns.lineplot(x=p_list, y=posteriors/norm_posterior, label=f'Posterior: k={k}',ax=ax)
    sns.lineplot(x=p_list, y=likelihoods/norm_likelihood, label=f'Likelihood: k={k}',ax=ax)
    sns.lineplot(x=p_list, y=priors/norm_prior, label=f'Priors: k={k}',ax=ax)

    ax.set_xlabel(f'Probability of {k} heads out of {n} coinflips')
    ax.set_ylabel(f'Bayesian density')
    ax.set_title('Bayesian probability as function of coinflip probability')

    ax.legend()
    return ax
#%%
n = 100
k = 66
p = np.linspace(0.01, 0.99, 100)

plot_posterior_pdf(p, binomial, k, n)
# %%
def mc_gaussian_function(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*sigma)) * np.exp(-(x-0.5*mu)**2/(2*sigma))

def accept_reject(mu, sigma):
    guess = []
    while len(guess) < 1:
        guess_x = np.random.uniform(0.5*mu-5*sigma, 0.5*mu+5*sigma, 1)
        guess_y = np.random.uniform(0, stats.norm.pdf(mu, mu, sigma), 1)
        y_func = mc_gaussian_function(guess_x, mu, sigma)
        guess = guess_x[guess_y < y_func]
    return guess

def markov_chain(mu, sigma, steps):
    mu_array = np.zeros(steps)
    mu_array[0] = mu

    for step in range(1,steps):
        guess = accept_reject(mu, sigma)
        mu = guess
        mu_array[step] = mu

    return mu_array

sigma = 1
mu = 100 
mc_100 = markov_chain(100, sigma, 100)
mc_neg_27 = markov_chain(-27, sigma, 100)

# %%
plt.plot(mc_100)
plt.plot(mc_neg_27)
#%%
def acceptreject(func, xmin, xmax, ymin, ymax, N_points, **kwargs):
    # Recursive function to do accept/reject monte carlo simulation
    xran = np.random.uniform(xmin, xmax, N_points)
    yran = np.random.uniform(ymin, ymax, N_points)
    yfunc = func(xran, **kwargs)
    xkeep = xran[yran < yfunc]
    ykeep = yran[yran < yfunc]
    missing = N_points - len(xkeep)
    if missing > 0:
        xrest, yrest, all_xrest, all_yrest, tries = acceptreject(func, xmin, xmax, ymin, ymax, missing, **kwargs)
    else:
        xrest = np.array([])
        yrest = np.array([])
        all_xrest = np.array([])
        all_yrest = np.array([])
        tries = 0
    finalx = np.append(xkeep, xrest)
    finaly = np.append(ykeep, yrest)
    allx = np.append(xran, all_xrest)
    ally = np.append(yran, all_yrest)
    finaltries = N_points + tries
    return finalx, finaly, allx, ally, finaltries

def acceptrejectdata(func, xmin, xmax, ymin, ymax, N_points, **kwargs):
    x, y, all_x, all_y, tries = acceptreject(func, xmin, xmax, ymin, ymax, N_points, **kwargs)
    eff = N_points/tries
    area = (xmax - xmin) * (ymax - ymin) * eff
    return x, y, area, all_x, all_y, eff, tries

def metropolis_hastings(theta_old, theta_cand, likelihood_func, prior_func, likelihood_args, prior_args):
    likelihood_old = likelihood_func(theta_old, *likelihood_args)
    likelihood_cand = likelihood_func(theta_cand, *likelihood_args)
    prior_old = prior_func(theta_old, *prior_args)
    prior_cand = prior_func(theta_old, *prior_args)
    posterior_old = bayes(likelihood_old, prior_old, 1)
    posterior_cand = bayes(likelihood_cand, prior_cand, 1)
    r = posterior_cand/posterior_old
    acceptance = [1, r]
    p = np.min(acceptance)
    theta_new = np.random.choice([theta_cand, theta_old], p=[p,1-p])
    return theta_new

def mcmc_mh(init_guess, steps,likelihood_func, prior_func, likelihood_args, prior_args, mu=0, sigma=0.3, burn_in_percent=0.2):
    theta_array = np.zeros(steps)
    theta_array[0] = init_guess
    theta_old = np.copy(init_guess)
    steps_burn = np.floor(steps*burn_in_percent)
    for step in range(1,steps):
        theta_cand = -1
        while not 0 < theta_cand < 1:
            theta_cand = theta_old + stats.norm.rvs(mu, sigma)
        theta_new = metropolis_hastings(theta_old, theta_cand, likelihood_func, prior_func, likelihood_args, prior_args)
        theta_array[step] = theta_new
        theta_old = np.copy(theta_new)
    #theta_burn = theta_array[:steps_burn]
    #theta_sample = theta_array[steps_burn:]
    return theta_array
#%%
n = 100
k = 66
likelihood_args = (k,n)
alpha = 5
beta = 17
prior_args = (alpha, beta)
init_prob_guess = 0.5
theta_array = mcmc_mh(init_prob_guess, 10000, likelihood_func=binomial, prior_func=beta_distrib, likelihood_args=likelihood_args, prior_args=prior_args, sigma=0.05)
# %%
plt.plot(theta_array)
# %%
sns.histplot(theta_array)
# %%
