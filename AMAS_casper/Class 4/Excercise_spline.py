#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate, integrate, fft
import seaborn as sns
import pandas as pd
import sympy
from sympy import symbols, lambdify
plt.style.use(r'..\casper_style.mplstyle')
#%%
df = pd.read_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS2023\AMAS_casper\Class 4\DustLog_forClass.txt', sep=' ',names=['cm', 'dust'])
df['Type'] = np.array(['Normal']*len(df['cm']))
#%%
f_linear = interpolate.interp1d(x=df['cm'], y=df['dust'])
df_iterp_linear = pd.DataFrame()
df_iterp_linear['cm'] = np.linspace(np.min(df['cm']),np.max(df['cm']), 10000)
df_iterp_linear['dust'] = f_linear(df_iterp_linear['cm'])
df_iterp_linear['Type'] = np.array(['Linear']*len(df_iterp_linear['cm']))

f_cubic = interpolate.interp1d(x=df['cm'], y=df['dust'], kind='cubic')
df_iterp_cubic = pd.DataFrame()
df_iterp_cubic['cm'] = np.linspace(np.min(df['cm']),np.max(df['cm']), 10000)
df_iterp_cubic['dust'] = f_cubic(df_iterp_cubic['cm'])
df_iterp_cubic['Type'] = np.array(['Cubic']*len(df_iterp_cubic['cm']))

df_merge = pd.concat([df, df_iterp_linear,df_iterp_cubic], axis=0)
#%%
sns.scatterplot(data=df_merge.where(df_merge['Type']=='Normal').dropna()
, x='cm', y='dust', label='Original')
sns.lineplot(data=df_merge.where(df_merge['Type']=='Linear').dropna()
, x='cm', y='dust', color='#7eb0d5', label='Linear interpolation')
plt.xlim(1700,1750)
plt.legend()
# %%
sns.scatterplot(data=df_merge.where(df_merge['Type']=='Normal').dropna()
, x='cm', y='dust', label='Original')
sns.lineplot(data=df_merge.where(df_merge['Type']=='Cubic').dropna()
, x='cm', y='dust', color='#7eb0d5', label='Cubic interpolation')
plt.xlim(1700,1750)
plt.legend()
# %%
df = pd.read_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS2023\AMAS_casper\Class 4\SplineCubic.txt', sep=' ',names=['cm', 'dust'])
df['Type'] = np.array(['Normal']*len(df['cm']))

f_linear = interpolate.interp1d(x=df['cm'], y=df['dust'])
df_iterp_linear = pd.DataFrame()
df_iterp_linear['cm'] = np.linspace(np.min(df['cm']),np.max(df['cm']), 10000)
df_iterp_linear['dust'] = f_linear(df_iterp_linear['cm'])
df_iterp_linear['Type'] = np.array(['Linear']*len(df_iterp_linear['cm']))

f_quadratic = interpolate.interp1d(x=df['cm'], y=df['dust'], kind='quadratic')
df_iterp_quadratic = pd.DataFrame()
df_iterp_quadratic['cm'] = np.linspace(np.min(df['cm']),np.max(df['cm']), 10000)
df_iterp_quadratic['dust'] = f_quadratic(df_iterp_quadratic['cm'])
df_iterp_quadratic['Type'] = np.array(['Quadratic']*len(df_iterp_quadratic['cm']))

f_cubic = interpolate.interp1d(x=df['cm'], y=df['dust'], kind='cubic')
df_iterp_cubic = pd.DataFrame()
df_iterp_cubic['cm'] = np.linspace(np.min(df['cm']),np.max(df['cm']), 10000)
df_iterp_cubic['dust'] = f_cubic(df_iterp_cubic['cm'])
df_iterp_cubic['Type'] = np.array(['Cubic']*len(df_iterp_cubic['cm']))

f_smooth = interpolate.UnivariateSpline(x=df['cm'], y=df['dust'], s=1)
df_iterp_smooth = pd.DataFrame()
df_iterp_smooth['cm'] = np.linspace(np.min(df['cm']),np.max(df['cm']), 10000)
df_iterp_smooth['dust'] = f_smooth(df_iterp_smooth['cm'])
df_iterp_smooth['Type'] = np.array(['Smooth']*len(df_iterp_smooth['cm']))

df_merge = pd.concat([df, df_iterp_linear,df_iterp_quadratic ,df_iterp_cubic, df_iterp_smooth], axis=0)
# %%
cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

sns.scatterplot(data=df_merge.where(df_merge['Type']=='Normal').dropna()
, x='cm', y='dust', label='Original', color=cycler[0], s=200)

sns.lineplot(data=df_merge.where(df_merge['Type']=='Linear').dropna()
, x='cm', y='dust', label='Linear interpolation', color=cycler[1])


sns.lineplot(data=df_merge.where(df_merge['Type']=='Quadratic').dropna()
, x='cm', y='dust', label='Quadratic interpolation', color=cycler[2])


sns.lineplot(data=df_merge.where(df_merge['Type']=='Cubic').dropna()
, x='cm', y='dust', label='Cubic interpolation', color=cycler[3])

sns.lineplot(data=df_merge.where(df_merge['Type']=='Smooth').dropna()
, x='cm', y='dust', label='Smooth interpolation', color=cycler[4])
plt.yscale('log')
plt.legend()
# %%
df_to_integrate = df_merge.where(df_merge['Type']=='Cubic').dropna()
integrate.trapz(df_to_integrate['dust'],df_to_integrate['cm'] )
# %%
# %%
df = pd.read_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS2023\AMAS_casper\Class 4\SplineCubic.txt', sep=' ',names=['cm', 'dust'])
df['Type'] = np.array(['Normal']*len(df['cm']))

df['cm_log'] = np.log(df['cm'])
f_cubic = interpolate.interp1d(x=df['cm_log'], y=df['dust'], kind='cubic')
df_iterp_cubic = pd.DataFrame()
df_iterp_cubic['cm_log'] = np.linspace(np.min(df['cm_log']),np.max(df['cm_log']), 10000)
df_iterp_cubic['dust'] = f_cubic(df_iterp_cubic['cm_log'])
df_iterp_cubic['Type'] = np.array(['Cubic']*len(df_iterp_cubic['cm_log']))
df_iterp_cubic['cm'] = np.exp(df_iterp_cubic['cm_log'])

df_merge = pd.concat([df, df_iterp_cubic], axis=0)

# %%
sns.scatterplot(data=df_merge.where(df_merge['Type']=='Normal').dropna()
, x='cm', y='dust', label='Original', color=cycler[0], s=200)

sns.lineplot(data=df_merge.where(df_merge['Type']=='Cubic').dropna()
, x='cm', y='dust', label='Cubic interpolation', color=cycler[1])
plt.yscale('log')
# %%
