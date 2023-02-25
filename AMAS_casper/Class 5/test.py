#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
import seaborn as sns
#%%
def function(x, alpha, beta):
    return (1+alpha*x+beta*(x**2))/(0.95 - (-0.95) + (alpha*((0.95**2)-((-0.95)**2))/2) + (beta*((0.95**3)-((-0.95)
    **3))/3))


def likelihood(data, function, **kwargs):
    likelihood_value = np.prod(function(data, **kwargs))
    return likelihood_value
#%%
alpha = 0.5
beta = 0.5
n = 2000
#%%
def montecarlo(Nsize, func,  xrange, yrange, **kwargs):
	xmin, xmax = xrange
	ymin, ymax = yrange
	x_rnd = np.random.uniform(xmin, xmax, Nsize)
	y_rnd = np.random.uniform(ymin, ymax, Nsize)
	mask = func(x_rnd, **kwargs)>y_rnd
	N_accept = np.sum(mask)
	if N_accept < Nsize:
		x_add, y_add, _, Npoints = montecarlo(Nsize - N_accept, func,  xrange, yrange, **kwargs)
	else: 
		x_add, y_add, _, Npoints = np.array([]), np.array([]), 0, 0
	x = np.append(x_rnd[mask], x_add)
	y = np.append(y_rnd[mask], y_add)
	N_total = Npoints + Nsize
	integral = Nsize/N_total
	area = (xmax - xmin)*(ymax-ymin)*integral
	return x, y, area, N_total
# %%
xmin, xmax, ymin = -0.95, 0.95, 0 
ymax = np.max(function(np.linspace(xmin, xmax, 10000), alpha, beta))
x_monte, y_monte, area, N_total = montecarlo(Nsize=n, func=function, xrange=(xmin, xmax), yrange=(ymin, ymax), alpha=0.5, beta=0.5)
# %%
#Ludvig Marcussen
alpha_list = []
beta_list = []


for i in range(500):
    x, y, _, _ = montecarlo(Nsize=n, func=function, xrange=(xmin, xmax), yrange=(ymin, ymax), alpha=0.5, beta=0.5)

    def function_to_minimize(params):
        a = params[0] 
        b = params[1]
        xmin, xmax = -0.95, 0.95
        poly = (1+a*x+b*x**2)/((b*(-xmin**3+xmax**3)/3)+(a*(-xmin**2+xmax**2)/2)+(xmax-xmin))
        LLH = -np.sum(np.log(poly))
        return LLH

    init_params = [0.5, 0.5]
    result = scipy.optimize.minimize(function_to_minimize, init_params, method='Nelder-Mead')
    alpha, beta = result.x
    alpha_list.append(alpha)
    beta_list.append(beta)
# %%
plt.scatter(alpha_list,beta_list)
# %%
df = pd.read_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS2023\AMAS_casper\Class 5\MLE_Variance_data.txt', header=None, names=('x','y', 'delete'), sep=' ')
df = df.drop('delete', axis=1)

# %%
x = df['x']
y = df['y']
def function_to_minimize(params):
    a = params[0] 
    b = params[1]
    xmin, xmax = -0.95, 0.95
    poly = (1+a*x+b*x**2)/((b*(-xmin**3+xmax**3)/3)+(a*(-xmin**2+xmax**2)/2)+(xmax-xmin))
    LLH = -np.sum(np.log(poly))
    return LLH

init_params = [0.5, 0.5]
result = scipy.optimize.minimize(function_to_minimize, init_params, method='Nelder-Mead')
alpha_est, beta_est = result.x
print(alpha_est,beta_est)
# %%
#Ludvig Marcussen
alpha_list = []
beta_list = []


for i in range(500):
    x, y, _, _ = montecarlo(Nsize=n, func=function, xrange=(xmin, xmax), yrange=(ymin, ymax), alpha=alpha_est, beta=beta_est)

    def function_to_minimize(params):
        a = params[0] 
        b = params[1]
        xmin, xmax = -0.95, 0.95
        poly = (1+a*x+b*x**2)/((b*(-xmin**3+xmax**3)/3)+(a*(-xmin**2+xmax**2)/2)+(xmax-xmin))
        LLH = -np.sum(np.log(poly))
        return LLH

    init_params = [0, 1]
    result = scipy.optimize.minimize(function_to_minimize, init_params, method='Nelder-Mead')
    alpha, beta = result.x
    alpha_list.append(alpha)
    beta_list.append(beta)
# %%
plt.scatter(alpha_list,beta_list)

# %%
df_pseudo = pd.DataFrame(data={'alpha':alpha_list, 'beta': beta_list})
sns.histplot(data=df_pseudo, x='alpha', y='beta')
# %%
sns.histplot(beta_list)
# %%
