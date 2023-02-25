#%%
import numpy as np
import pandas as pd
from scipy import stats, optimize
import seaborn as sns
import matplotlib.pyplot as plt
from iminuit import Minuit
plt.style.use(r'..\casper_style.mplstyle')

#%%
mu = 10
sigma = np.sqrt(2.3)
x = np.linspace(5,15,10000)
y = stats.norm.pdf(x, loc = mu, scale = sigma)
fig, ax = plt.subplots()
ax.plot(x,y)
ax.vlines(x=mu, ymin=np.min(y), ymax=np.max(y),linestyles='--', label=f'$\mu$ = {mu}', color='#a11502')
ax.hlines(y=np.interp(mu+sigma,x,y),xmin=mu, xmax=mu+sigma, linestyles='--', label=f'$\sigma$ = {sigma:.2f}', color='#fc4931')
ax.hlines(y=np.interp(mu+2*sigma,x,y),xmin=mu, xmax=mu+2*sigma, linestyles='--', label=f'2$\sigma$ = {2*sigma:.2f}', color='#fc6c59')

ax.fill_between(x, y, 0, where = (x>=mu-sigma) & (x<=mu+sigma), color='#fd7f6f', alpha=0.9)
ax.fill_between(x, y, 0, where = (x>=mu-2*sigma) & (x<=mu-sigma), color='#fd9081', alpha=0.8)
ax.fill_between(x, y, 0, where =  (x<=mu+2*sigma) & (x>=mu+sigma), color='#fd9081', alpha=0.8)
ax.fill_between(x, y, 0, where = (x>=mu-3*sigma) & (x<=mu-2*sigma), color='#fda69b', alpha=0.6)
ax.fill_between(x, y, 0, where =  (x<=mu+3*sigma) & (x>=mu+2*sigma), color='#fda69b', alpha=0.6)
ax.fill_between(x, y, 0, where = (x>=mu-4*sigma) & (x<=mu-3*sigma), color='#febcb4', alpha=0.5)
ax.fill_between(x, y, 0, where =  (x<=mu+4*sigma) & (x>=mu+3*sigma), color='#febcb4', alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gaussian distribution')
ax.grid(False)
ax.legend()
# %%
def alpha_beta_func(x, alpha, beta):
    xmin, xmax = -1.02, 1.11
    return (1+alpha*x+beta*x**2)/((beta*(-xmin**3+xmax**3)/3)+(alpha*(-xmin**2+xmax**2)/2)+(xmax-xmin))


def accept_reject(func, xmin, xmax, ymin, ymax, N_points, **kwargs):
    # Recursive function to do accept/reject monte carlo simulation
    xran = np.random.uniform(xmin, xmax, N_points)
    yran = np.random.uniform(ymin, ymax, N_points)
    yfunc = func(xran, **kwargs)
    bool_mask = yran <= yfunc
    xkeep = xran[bool_mask]
    ykeep = yran[bool_mask] 
    missing = N_points - np.sum(bool_mask)
    if missing > 0:
        xrest, yrest, all_xrest, all_yrest, tries, bool_rest = accept_reject(func, xmin, xmax, ymin, ymax, missing, **kwargs)
    else:
        xrest = np.array([])
        yrest = np.array([])
        all_xrest = np.array([])
        all_yrest = np.array([])
        tries = 0
        bool_rest = np.array([], dtype=bool)
    finalx = np.append(xkeep, xrest)
    finaly = np.append(ykeep, yrest)
    allx = np.append(xran, all_xrest)
    ally = np.append(yran, all_yrest)
    final_bool = np.append(bool_mask, bool_rest)
    finaltries = N_points + tries
    return finalx, finaly, allx, ally, finaltries, final_bool

def accept_reject_df(func, xmin, xmax, ymin, ymax, N_points,x_name='x',y_name='y', **kwargs):
    x, y, x_all, y_all, tries, bool_mask = accept_reject(func, xmin, xmax, ymin, ymax, N_points, **kwargs)
    df = pd.DataFrame(data={f'{x_name}':x_all,f'{y_name}':y_all,'accept':bool_mask})
    eff = N_points/tries
    area = (xmax - xmin) * (ymax - ymin) * eff
    return df, {'eff':eff,'area':area,'tries':tries}, x, y

def ullhfit(x, fitfunc, **kwargs):

    def obt(*args):
        logf = np.zeros_like(x)
        
        # compute the function value
        f = fitfunc(x, *args)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = f > 0

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive])
        # set everywhere else to badvalue
        logf[~mask_f_positive] = -1000000
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        return llh

    ullh_Min = Minuit(obt, **kwargs, name = [*kwargs])
    ullh_Min.errordef = 0.5
    ullh_Min.migrad()
    valuesfit = np.array(ullh_Min.values, dtype = np.float64)
    errorsfit = np.array(ullh_Min.errors, dtype = np.float64)
    if not ullh_Min.valid:
        print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")
    # * Impliment p-value for ullh fit
    
    return valuesfit, errorsfit

# %%
alpha, beta = 0.9, 0.55
xmin, xmax = -1.02, 1.11 
ymin, ymax = 0, np.max(alpha_beta_func(np.linspace(xmin, xmax, 10000), alpha, beta))

df, extra_dict, x, y = accept_reject_df(alpha_beta_func, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, N_points=807, alpha=alpha, beta=beta)

values,errors = ullhfit(x, alpha_beta_func, alpha=0.5, beta=0.5)
alpha_est,beta_est = values[0], values[1]
#%%
fig,ax = plt.subplots()
sns.scatterplot(data=df, x='x', y='y', hue='accept', ax=ax)
ax.plot(np.linspace(xmin,xmax,10000), alpha_beta_func(np.linspace(xmin,xmax,10000),alpha=alpha, beta=beta), color='k')
ax.plot(np.linspace(xmin,xmax,10000), alpha_beta_func(np.linspace(xmin,xmax,10000),alpha=alpha_est, beta=beta_est), color='orange')

# %%
alpha_beta_func(np.linspace(xmin,xmax,1000), alpha, beta)
# %%
(1+alpha*-1.02+beta*(-1.02**2)) / (xmax - xmin + (alpha * ((xmax**2) - ((xmin)**2))/2) + (beta*((xmax**3)-((xmin)**3))/3))
# %%
# %%
values
# %%
