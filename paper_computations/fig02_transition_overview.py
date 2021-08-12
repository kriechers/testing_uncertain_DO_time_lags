import numpy as np
import pandas as pd
import xarray as xr
import joblib as jl
import matplotlib.pyplot as plt
from sample_mean import sample_mean_distr
from scipy.stats import gaussian_kde
from combined_kde import combined_kde
from functions import plt_individual_pdfs, quantiles
from pyplot_setting_paper import set_pyplot
from pyplot_setting_paper import make_patch_spines_invisible

set_pyplot(plt.rc, plt.rcParams)
inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}'))

#NGRIP = jl.load('../data/MCMC_data/full_data/%s.gz' % ('NGRIP')).sel(
#    model=['t0'])
#NEEM = jl.load('../data/MCMC_data/full_data/%s.gz' % ('NEEM')).sel(
#    model=['t0'])

proxies = {'NGRIP': ['Ca', 'Na', 'lt', 'd18O'],
           'NEEM': ['Ca', 'Na']}

colors = {'d18O': 'C0',
          'Ca': 'C1',
          'Na': 'C2',
          'lt': 'C4'}

data = {'NGRIP' : jl.load('../data/MCMC_data/%s.gz' % ('NGRIP')).sel(
    model=['t0']),
        'NEEM' : jl.load('../data/MCMC_data/%s.gz' % ('NEEM')).sel(
    model=['t0'])}

events = data['NGRIP'].event.values
kdes = dict()

t_ax = np.arange(-150,100,0.1)
x_ax = np.arange(0,2,0.1)
fig = plt.figure(figsize=(twocol, onecol *0.8))
height_ratios = [3,1,3]
gs = fig.add_gridspec(3, 1,
                      hspace=0.1,
                      wspace=0.1,
                      top=0.95,
                      left=0.15,
                      bottom=0.15,
                      right=0.85,
                      height_ratios = height_ratios)

ax3 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0:2, 0])
ax2 = fig.add_subplot(gs[1:3, 0])
axes = [ax1, ax2]

for core,ax in zip(['NGRIP', 'NEEM'], [ax1, ax2]):
    make_patch_spines_invisible(ax)
    for p in proxies[core]:
        offset = 0.15
        for e in events:
            MCMC_sample = data[core].sel(
                param = p, event = e).values.squeeze()
            if all(np.isnan(MCMC_sample)):
                offset +=0.15
                continue
            else:
                kde = gaussian_kde(MCMC_sample)
                y_data = kde(t_ax) 
                mask = y_data > 0.005
                if p in ['Na', 'd18O']:
                    ax.fill_betweenx(t_ax[mask],
                                     offset-y_data[mask],
                                     offset,
                                     color = colors[p],
                                     alpha = 0.5,
                                     lw = 0.2)
                   
                else:
                    ax.fill_betweenx(t_ax[mask],
                                     offset,
                                     offset+y_data[mask], 
                                     color = colors[p],
                                     alpha = 0.5,
                                     lw = 0.2)

                offset +=0.15

                
ax1.spines['left'].set_visible(True)
ax1.xaxis.set_visible(False)
ax1.set_ylabel(r'$\Delta t\;[\text{y}]$ - NGRIP')
ax1.set_ylim(-120,30)
ax1.set_xlim((0, 24 * 0.15))

legend_ax = fig.add_subplot(gs[0, 0])
make_patch_spines_invisible(legend_ax)
legend_ax.plot((),(), color = 'C0', label = r'$\delta^{18}\text{O}$')
legend_ax.plot((),(), color = 'C1', label = r'$\text{Ca}^{2+}$')
legend_ax.plot((),(), color = 'C2', label = r'$\text{Na}^{+}$')
legend_ax.plot((), (), color = 'C4', label = r'$\lambda$')
legend_ax.yaxis.set_visible(False)
legend_ax.xaxis.set_visible(False)
legend_ax.legend(loc = 'upper left',
                 bbox_to_anchor = (1.01,0.9),
                 frameon = False, 
                 ncol = 1,
                 markerscale=1,
                 framealpha=1,
                 handlelength=1)

ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.spines['right'].set_visible(True)
ax2.spines['bottom'].set_visible(True)
ax2.xaxis.set_visible(True)
ax2.set_ylabel(r'$\Delta t\;[\text{y}]$ - NEEM')
ax2.set_xticks(np.arange(0.15,30,0.15))
ax2.set_xlim((0, 24 * 0.15))
ax2.set_xticklabels(events,
                    rotation=45,
                    ha = 'right')

make_patch_spines_invisible(ax3)
ax3.set_xlim((0, 24 * 0.15))
for x in np.arange(0.15,3.7,0.15)[:-1]:
    ax3.axvline(x, lw = 0.4,
                color='lightsteelblue')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
ax1.axhline(0, color = 'k')
ax2.axhline(0, color = 'k')

figname = '../../latex/revised_manuscript/figures/fig02.pdf'
plt.savefig(figname,
            transparent = True,
            dpi = 300)



test = data['NGRIP'].sel(event = 'GI-7c', param = 'lt')