import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from tests import wilcoxon_distribution
from scipy.stats.distributions import t
from scipy.stats import gaussian_kde
from itertools import product
from tests import hypothesis_testing
from pyplot_setting_paper import set_pyplot, make_patch_spines_invisible


set_pyplot(plt.rc, plt.rcParams)
inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}'))

DO_events = pd.read_table('../data/GIS_table.txt',
                          usecols=[0, 1],
                          header=2)

combinations = {'NGRIP': [('Ca', 'Na')]}  # ,
#                          ('lt', 'Na'),
#                          ('Ca', 'd18O'),
#                          ('lt', 'd18O')],
#                          ('lt', 'Ca')],
#                'NEEM': [('Ca', 'Na')]}
core = 'NGRIP'
par, ref_par = combinations[core][0]

data = jl.load('../data/MCMC_data/%s.gz' % (core)).sel(
    param=[par, ref_par], model='t0').dropna(dim='event')
events = data.event.values


label = {'Ca': r'$\text{Ca}^{2+}$',
         'Na': r'$\text{Na}^{+}$',
         'd18O': r'$\delta^{18}$O]',
         'lt': r'$\lambda$'}

uncertain, expected, bs_null = hypothesis_testing(
    *combinations['NGRIP'][0],
    'NGRIP')


height_ratios = [list(np.full((4), 0.8)) + list(np.full((4), 1))]
fig = plt.figure(figsize=(onecol, onecol))
gs = fig.add_gridspec(8, 10,
                      hspace=0.6,
                      wspace=0.4,
                      top=0.95,
                      left=0.15,
                      bottom=0.15,
                      right=0.85)


axes = []
t_ax = np.arange(-200, 200, 0.1)
for i, e in enumerate(data.event.values):
    idx = np.unravel_index(i, (4, 4))
    axes.append(fig.add_subplot(gs[idx]))
    lag_data = -data.sel(event=e).diff(dim='param').values.squeeze()
    kde = gaussian_kde(lag_data)
    axes[i].plot(t_ax, kde(t_ax), lw=0.5)
    #axes[i].hist(lag_data, bins=t_ax, density=True, lw=0.3)
    for sp in axes[i].spines.values():
        sp.set_visible(False)
        sp.set_linewidth(0.5)
    axes[i].spines['bottom'].set_visible(True)
    axes[i].set_ylim((0, 0.05))
    axes[i].set_yticks([0, 0.03])
    axes[i].set_yticklabels([0, 0.03])
    axes[i].set_xlim(-50, 50)
    axes[i].set_xticks(np.arange(-30, 32, 30))
    axes[i].vlines(0, 0, 0.05, color='k',
                   lw=0.3)
    axes[i].tick_params(length=1.5,
                        width=0.6,
                        labelsize=4.5)
    if idx[1] == 0:
        axes[i].spines['left'].set_visible(True)
        axes[i].tick_params(axis='y',
                            which='major',
                            labelsize=4.5)

    else:
        axes[i].yaxis.set_visible(False)
    if idx[0] == 3:
        axes[i].tick_params(axis='x',
                            which='major',
                            labelsize=4.5,
                            rotation=90)
    else:
        axes[i].xaxis.set_visible(False)

axes[4].set_ylabel(r'$\rho(\Delta\hat{t})$[1/yr]',
                   fontsize=6)
axes[4].yaxis.set_label_coords(-1.3, -1)
axes[14].set_xlabel(r'$\Delta \hat{t}$[yr]', fontsize=6)
axes[14].xaxis.set_label_coords(-1., -1.2)
axes[0].annotate('(a)', (-2, 0.85),
                 fontsize = 6,
                 xycoords='axes fraction')



hyp_axes = []


hyp_axes.append(fig.add_subplot(gs[2:5, 4:6]))
hyp_axes.append(fig.add_subplot(gs[2:5, 6:8]))
hyp_axes.append(fig.add_subplot(gs[2:5, 8:10]))

for ax in hyp_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.set_visible(False)
hyp_axes[2].spines['right'].set_visible(True)
hyp_axes[2].yaxis.set_ticks_position('right')
hyp_axes[2].yaxis.set_label_position('right')
hyp_axes[2].set_ylabel('rel. frequence / density [au]',
                       fontsize = 5)
hyp_axes[2].annotate('(b)', (1.2, 0.85),
                 fontsize = 6,
                 xycoords='axes fraction')


n = data.event.values.shape[0]
z_ax = np.arange(-20, 10, 0.1)
hyp_axes[0].hist(uncertain['ttest']['stat'], z_ax, density=True,
                 label=r'hist[$\rho^{\text{emp}}_{z}(z)]$',
                 color='C0')
hyp_axes[0].plot(z_ax, t.pdf(z_ax, df=n-1),
                 label=r'$\rho_{H_0}(z)$',
                 color='k')
hyp_axes[0].set_xlim((-5, 5))
hyp_axes[0].vlines(t.ppf(0.05, df=n-1), 0, 1,
                   color='firebrick',
                   linestyle='dotted',
                   lw=0.8)
hyp_axes[0].set_ylim((0, 0.6))
hyp_axes[0].set_xlabel('$\hat{z}$',
                       fontsize = 5)


# plot the distribution of w-values together with w-distribution
wilc_dist = wilcoxon_distribution(n)
w_ax = wilc_dist['w'].values
w_crit = w_ax[np.cumsum(wilc_dist['p(w)']) > 0.05][0]
hyp_axes[1].hist(uncertain['WSR']['stat'], w_ax, density=True,
                 label=r'hist[$\rho^{\text{emp}}_{w}(w)]$',
                 color='C1')
hyp_axes[1].plot(w_ax, wilc_dist['p(w)'],
                 label=r'$\rho_{H_0}(w)$',
                 color='k')
hyp_axes[1].vlines(w_crit, 0, 1,
                   color='firebrick',
                   linestyle='dotted',
                   lw=0.8)
hyp_axes[1].set_ylim((0, 0.035))
hyp_axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
hyp_axes[1].set_xlabel('$\hat{w}$',
                       fontsize = 5)


u_ax = np.arange(-100, 100, 0.1)
u_crit = np.sort(bs_null)[int(0.05 * bs_null.size)]
hyp_axes[2].hist(bs_null, u_ax, density=True,
                 histtype='step',
                 label=r'hist$[\hat{\rho}^{\text{bs}}_{H_0}(u)]$',
                 color='k',
                 zorder=2)
hyp_axes[2].hist(uncertain['bs']['stat'], u_ax, density=True,
                 label=r'hist$[\rho^{\text{bs}}(u)$',
                 color='C2', zorder=1)
#hyp_axes[2].vlines(u_crit, 0, 1,
#                   color='firebrick',
#                   linestyle='dotted',
#                   lw=0.8, zorder=3)
hyp_axes[2].set_ylim((0, 0.03))
hyp_axes[2].set_xlim((-100, 100))
hyp_axes[2].set_xlabel('$\hat{v}$',
                       fontsize = 5)


p_ax = np.arange(0, 1.1, 0.01)
p_axis = fig.add_subplot(gs[5:8, 1:10])
p_axis.hist(uncertain['ttest', 'pval'], p_ax,
            weights=np.ones(6000)/6000,
            histtype='step',
            label='t-test')

p_axis.hist(uncertain['WSR', 'pval'], p_ax,
            weights=np.ones(6000)/6000,
            histtype='step',
            label='Wilcoxon')

p_axis.hist(uncertain['bs', 'pval'], p_ax,
            weights=np.ones(6000)/6000,
            histtype='step',
            label='bootstrap')
p_axis.vlines(0.05, 0, 1,
              color='firebrick',
              linestyle='dotted',
              lw=0.8)
p_axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
p_axis.yaxis.set_ticks([])
p_axis.set_ylabel('rel. frequency [au]')
p_axis.set_xlabel(r'$\hat{p}$')
p_axis.set_xlim(0, 0.5)
p_axis.set_ylim(0, 0.125)
p_axis.spines['top'].set_visible(False)
p_axis.spines['right'].set_visible(False)
p_axis.annotate('(c)', (-0.2, 0.85),
                 fontsize = 6,
                 xycoords='axes fraction')


# axes[0].set_ylabel(r'$\rho^{\text{emp}}(\cdot)\;[\text{au}]$')
# axes[0].yaxis.set_label_coords(-0.1,1.02)
# axes[0].yaxis.label.set_rotation(0)
#axes[3].set_ylabel('rel. frequency')
# axes[3].yaxis.set_label_coords(-0.05,1.3)
# axes[3].yaxis.label.set_rotation(0)


legend_ax = fig.add_subplot(gs[2, 6])
make_patch_spines_invisible(legend_ax)
legend_ax.plot((), (), color='C0', label='t-test')
legend_ax.plot((), (), color='C1', label='WSR-test')
legend_ax.plot((), (), color='C2', label='BS-test')
legend_ax.plot((), (), color='k', label=r'$\rho_{0}(\cdot)$')
legend_ax.yaxis.set_visible(False)
legend_ax.xaxis.set_visible(False)
legend_ax.legend(loc='lower left',
                 bbox_to_anchor=(-0.5, 1.5),
                 ncol=2,
                 markerscale=1,
                 framealpha=0,
                 handlelength=1)
figname = '../../latex/revised_manuscript/figures/fig06.pdf'
plt.savefig(figname,
            dpi=300,
            transparent=True,
            bbox_to_inches='tight')
