import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import joblib as jl
from scipy.stats import gaussian_kde
from functions import normalize
from pyplot_setting_paper import set_pyplot
set_pyplot(plt.rc, plt.rcParams)

inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}'))


def linear_ramp(t, t0=0.0, dt=1.0, y0=0.0, dy=1.0):
    """Linear Ramp Function

    This function describes the linear transition between two constant values.

    Parameter
    ---------
    t : np.ndarray
        Time variable
    t0 : float
        Start time of the ramp
    dt : float
        Transition length
    y0 : float
        Function value before the transition
    dy : float
        Hight of the transion

    Return
    ------
    y : np.ndarray
        Function values of the linear transiton
    """
    lt_t0 = t < t0
    gt_t1 = t > t0 + dt
    condlist = [lt_t0,
                ~np.logical_or(lt_t0, gt_t1),
                gt_t1]
    funclist = [lambda t: y0,
                lambda t: y0 + dy * (t - t0) / dt,
                lambda t: y0 + dy]
    y = np.piecewise(t, condlist, funclist)
    return y


def calc_med_iqr(y, q=[5, 95], axis=None):
    qs = np.percentile(y, [q[0], 50, q[-1]], axis=axis)
    yerr = np.diff(qs, axis=0)
    y = qs[1]
    return y, yerr


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


core = 'NGRIP'
time = 't0'
par = 'Ca'
ref_par = 'Na'
colordict = {'Ca': 'C1', ref_par: 'C2'}

events = pd.read_table('../data/GIS_table.txt',
                       header=2,
                       usecols=[0])

for event in [events['Event'][14]]:

    data = jl.load('../data/MCMC_data/full_data/%s.gz' % (core)).sel(
        param=[par, ref_par], event=[event]).dropna(dim='event')
    lag_data = -data.sel(model=time).diff(dim='param').values.squeeze()
    if lag_data.size == 0:
        continue

    gi_table = pd.read_table('../data/GIS_table.txt', comment='#')
    ref_age = gi_table.loc[gi_table['Event'] == event, 'Age'].values
    lag_kde = gaussian_kde(lag_data)

    t_ax = np.arange(-100, 100, 1)
    lag_pdf = lag_kde(t_ax)

    fig = plt.figure(figsize=(onecol, 0.8 * onecol))
    gs = fig.add_gridspec(11, 10,
                          hspace=0,
                          wspace=0,
                          left=0.25,
                          right=0.88,
                          bottom=0.2)

    ax1 = fig.add_subplot(gs[2:6, :6])
    ax2 = fig.add_subplot(gs[4:8, :6])
    ax3 = fig.add_subplot(gs[0:2, :6])
    ax4 = fig.add_subplot(gs[6:, 5:])

    # axes[0].set_title('Onset of %s' % event)
    # axes[0].set_ylabel(r'arbitrary units')
    #    raxis = (axes[0].twinx())

    make_patch_spines_invisible(ax1)
    make_patch_spines_invisible(ax2)
    make_patch_spines_invisible(ax3)
    make_patch_spines_invisible(ax4)
    ax2.spines["left"].set_position(("axes", -0.32))
    ax1.spines['left'].set_position(('axes', -0.05))
    ax1.yaxis.set_label_coords(-0.18, 0.5)
    ax2.yaxis.set_label_coords(-0.45, 0.5)
    ax3.spines['right'].set_position(('axes', 1.05))
    ax2.spines['left'].set_visible(True)
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.set_ticks_position('right')

    ax2.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax3.xaxis.set_visible(False)
    ax3.spines['right'].set_visible(True)

    for p, ax in zip([par, ref_par], [(ax1, ax3), (ax2, ax3)]):

        print(p, ax)
        data_file = '../data/ramp_data/%s_%s_%s.csv' % (core, event, p)

        t, obs = pd.read_csv(data_file).values.T

        t_plot = (ref_age - t) / 1000

        temp = data.sel(param=p, event=event).values
        ramps = np.array([linear_ramp(t, *p)
                          for p in temp[:, :4]])

        r_med, r_err = calc_med_iqr(ramps, axis=0)

        ax[0].plot(t_plot, obs,
                   color=colordict[p],
                   lw=0.5, label=p)
        ax[0].plot(t_plot, r_med, color='k')
        ax[0].fill_between(t_plot, r_med - r_err[0], r_med + r_err[1],
                           alpha=.5, color='slategray')

        pdf = gaussian_kde(temp[:, 0])(t) * 1000
        ax[1].plot(t_plot, pdf, color=colordict[p])

    xlim = (46.95, 46.75)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylim((3, 6.5))
    ax1.set_ylabel(r'$ln$(Ca$^{2+})\;$[ng/g]',
                   color=colordict['Ca'])
    ax1.spines['left'].set_color(colordict['Ca'])
    ax1.tick_params(axis='y', colors=colordict['Ca'])
    ax1.set_xlim(xlim)
    ax3.annotate('(a)', (0.90, 0.85),
                 xycoords='axes fraction')

    ax2.set_xlabel('GICC05 Age [ky b2k]')
    ax2.xaxis.set_label_coords(0.3, -0.4)
    ax2.set_xlim(xlim)
    labels = [str(l) for l in ax2.get_xticks()]
    labels[0] = ''
    labels[-1] = ''
    ax2.set_xticklabels(labels)
    ax3.set_ylabel(r'$\rho_{T_{0}}(t_0)\;$[ky$^{-1}$]')
    ax3.yaxis.set_label_coords(1.25, 0.5)
    ax3.set_xlim(xlim)
    ax1.annotate('(b)', (0.90, 0.75),
                 xycoords='axes fraction')

    ax2.set_ylim((2.5, 6))
    ax2.set_ylabel(r'$ln$(Na$^{+})\;$[ng/g]',
                   color=colordict[ref_par])
    ax2.tick_params(axis='y', colors=colordict[ref_par])
    ax2.spines['left'].set_color(colordict[ref_par])

    ax4.hist(lag_data, t_ax,
             density=True,
             color='C4',
             label=r'MCMC samples')
    ax4.plot(t_ax, gaussian_kde(lag_data)(t_ax),
             label=r'Gaussian KDE')
#    ax4.set_ylabel(r'$\rho^{\text{emp}}(\Delta t$)')
    ax4.set_xlabel(r'$\Delta t$[y]')
    ax4.set_ylabel('rel.frequence / density [1/y]')
    ax4.annotate('(c)', (0.85, 0.9),
                 xycoords='axes fraction')
    ax4.legend(frameon=False,
               bbox_to_anchor=(0.88, 1.3, 0.2, 0.2))
    ax4.yaxis.set_ticks_position('right')
    ax4.yaxis.set_label_position('right')
    ax4.set_xlim(-75, 50)
    ax4.spines['bottom'].set_visible(True)
    ax4.spines['right'].set_visible(True)
    ax4.set_yticklabels(ax4.get_yticks(),
                        rotation=90,
                        va='center')
    ax4.yaxis.set_label_coords(1.2, 0.5)
    figname = 'figures/fig03.pdf'
    plt.savefig(figname,
                dpi=300,
                transparent=True)

# fit_example_event_%s_par_%s_ref_par_%s.png  % (event, par, ref_par))
