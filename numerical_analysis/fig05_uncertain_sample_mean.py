import numpy as np
import pandas as pd
import xarray as xr
import joblib as jl
import matplotlib.pyplot as plt
from sample_mean import sample_mean_distr
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

combinations = {'NGRIP': [('Ca', 'Na'),
                          ('lt', 'Na'),
                          ('Ca', 'd18O'),
                          ('lt', 'd18O')],
                #                          ('lt','Ca')],
                'NEEM': [('Ca', 'Na')]}

core_list = ['NGRIP'] * 4
core_list.append('NEEM')
pair_list = ['Ca-Na', 'lt-Na', 'Ca-d18O', 'lt-d18O', 'Ca-Na']

col_idx = pd.MultiIndex.from_arrays([core_list, pair_list])
row_idx = pd.MultiIndex.from_product([['Erhardt', 'Riechers'],
                                      [0.05, 0.5, 0.95, 'P>0']])
statistics = pd.DataFrame(columns=col_idx, index=row_idx)

label = {'Ca': r'$\text{Ca}^{2+}$',
         'Na': r'$\text{Na}^{+}$',
         'd18O': r'$\delta^{18}$O',
         'lt': r'$\lambda$'}

time = 't0'
# ATTENTION: the sample mean function requires a symmetric time
# axis due to the convolution
t_ax = np.arange(-600, 600.1, 0.05)


fig = plt.figure(figsize=(onecol, onecol * 1.8))
gs = fig.add_gridspec(12, 5,
                      hspace=0,
                      wspace=0.1,
                      top=0.95,
                      left=0.15,
                      bottom=0.1,
                      right=0.85)
axes = []
axes.append(fig.add_subplot(gs[0:3, 0:3]))
axes.append(fig.add_subplot(gs[2:5, 2:5]))
axes.append(fig.add_subplot(gs[4:7, 0:3]))
axes.append(fig.add_subplot(gs[6:9, 2:5]))
axes.append(fig.add_subplot(gs[9:12, 0:3]))
axes.append(fig.add_subplot(gs[8, 0:2]))
#axes.append(fig.add_subplot(gs[10:13, 2:5]))

i = 0

for core in ['NGRIP', 'NEEM']:
    data = jl.load('../data/MCMC_data/%s.gz' % (core)).sel(
        model=['t0'])

    for (par, ref_par) in combinations[core]:
        print(par, ref_par)
        label1 = label[par]
        label2 = label[ref_par]
        lag_data = -data.diff(dim='param').values.squeeze()
        dt = -data.sel(param=[par, ref_par]).dropna(
            dim='event').diff(dim='param').values.squeeze()
        events = data.sel(param=[par, ref_par]).dropna(
            dim='event').event.values
        print(len(events))
        kde = combined_kde(dt.T)
        ce = kde.pdf(t_ax)
        print(quantiles(t_ax, ce, [0.05, 0.5, 0.95]))
        pdfs = [pdf(t_ax) for pdf in kde.kdes]
        u_ax, u_pdf = sample_mean_distr(t_ax, pdfs, dx=0.05)

        e, f, g = quantiles(t_ax, ce, [0.05, 0.5, 0.95])
        h = np.sum(ce[t_ax > 0]*0.1)
        statistics[core,
                   (par + '-' +
                    ref_par)]['Erhardt'] = [e, f, g, h]

        a, b, c = quantiles(u_ax, u_pdf, [0.05, 0.5, 0.95])
        d = np.sum(u_pdf[u_ax > 0]*(u_ax[1]-u_ax[0]))
        statistics[core,
                   (par + '-' +
                    ref_par)]['Riechers'] = [a, b, c, d]

        #plt_individual_pdfs(t_ax, pdfs, par, ref_par, names=events)

        make_patch_spines_invisible(axes[i])
        if (i+1) % 2 == 0:
            axes[i].spines['right'].set_visible(True)
            axes[i].annotate('(' + chr(97+i) + ')',
                             (0.87, 0.9),
                             xycoords='axes fraction')

            axes[i].annotate(r'$%0.1f_{%0.1f}^{+%0.1f}$' % (f, e-f, g-f),
                             (0.62, 0.75),
                             xycoords='axes fraction',
                             color='C0',
                             fontsize=5)

            axes[i].annotate(r'$%0.1f_{%0.1f}^{+%0.1f}$' % (b, a-b, c-b),
                             (0.62, 0.6),
                             xycoords='axes fraction',
                             color='C1',
                             fontsize=5)

            axes[i].yaxis.set_ticks_position('right')
            axes[i].yaxis.set_label_position('right')

        else:
            axes[i].spines['left'].set_visible(True)
            axes[i].annotate('(' + chr(97+i) + ')',
                             (0.1, 0.9),
                             xycoords='axes fraction')

            axes[i].annotate(r'$%0.1f_{%0.1f}^{+%0.1f}$' % (f, e-f, g-f),
                             (0.05, 0.75),
                             xycoords='axes fraction',
                             color='C0',
                             fontsize=5)

            axes[i].annotate(r'$%0.1f_{%0.1f}^{+%0.1f}$' % (b, a-b, c-b),
                             (0.05, 0.6),
                             xycoords='axes fraction',
                             color='C1',
                             fontsize=5)

        axes[i].spines['bottom'].set_visible(True)
        axes[i].plot(t_ax, ce, color='C0',
                     label='combined evidence (Erhardt et al., 2019)')
        axes[i].plot(u_ax, u_pdf, color='C1',
                     label='uncertain sample mean (this study)')
        axes[i].set_xlim(-35, 15)
        axes[i].set_ylabel(r'$\rho(\Delta\hat{t})\;$[1/y]')
        axes[i].set_xlabel((r'$\Delta\hat{t}$('
                            + label1 + ',' + label2
                            + r')$\;$[y]'))

        i += 1
axes[0].legend(loc='upper left', bbox_to_anchor=(0.9, 0.95))

axes[-1].plot([0, 0], [0, 1], color='lightsteelblue',
              lw=2.2)
axes[-1].set_xlim((-1, 0))
axes[-1].set_ylim((0, 1.8))
make_patch_spines_invisible(axes[-1])
axes[-1].spines['top'].set_visible(True)
axes[-1].spines['top'].set_color('lightsteelblue')
axes[-1].spines['right'].set_visible(False)
axes[-1].yaxis.set_visible(False)
axes[-1].xaxis.set_visible(False)
axes[-1].spines['top'].set_position(('axes', 0.6))
axes[-1].annotate('NGRIP',
                  (0.5, 0.66),
                  xycoords='axes fraction',
                  color='k')
axes[-1].annotate('NEEM',
                  (0.5, 0.4),
                  xycoords='axes fraction',
                  color='k')

figname = 'figures/fig05.pdf'
plt.savefig(figname,
            bbox_to_inches='tight')
statistics.to_csv('outputs/sample_mean_statistics.csv')
