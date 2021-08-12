import numpy as np
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt


def normalize(time_series):
    '''
    normalizes a time series to the range [-1,1]
    '''

    normed = 2*(time_series - np.min(time_series)) / \
        (np.max(time_series)-np.min(time_series)) - 1
    return normed


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def quantiles(axis, pdf, percentiles):
    dx = axis[1] - axis[0]
    cdf = np.cumsum(pdf * dx)
    q = [axis[np.sum(cdf < p)] for p in percentiles]
    return q


def plt_individual_pdfs(t_ax, pdfs, p, ref_par, names=None):

    while len(pdfs) < 20:
        pdfs.append(np.nan)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 5,
                          hspace=0.25,
                          wspace=0.25,
                          top=0.9,
                          left=0.1,
                          bottom=0.1,
                          right=0.9)
    axes = []
    for i, pdf in enumerate(pdfs):
        idx = np.unravel_index(i, (4, 5))
        axes.append(fig.add_subplot(gs[idx]))
        if np.size(pdf) == np.size(t_ax):
            axes[i].plot(t_ax, pdf)
            if names is not None:
                axes[i].annotate(names[i], (0.9, 0.9),
                                 xycoords='axes fraction')

        for sp in axes[i].spines.values():
            sp.set_visible(False)
            axes[i].spines['bottom'].set_visible(True)
            axes[i].set_ylim((0, 0.05))
            axes[i].set_yticks([0, 0.02])
            axes[i].set_yticklabels([0, 0.02])
            axes[i].set_xlim(-50, 50)
            axes[i].set_xticks(np.arange(-50, 30, 25))
            axes[i].vlines(0, 0, 0.05, color='red',
                           linestyle=':',
                           lw=0.8)
        if idx[1] == 0:
            axes[i].spines['left'].set_visible(True)
            axes[i].tick_params(axis='y',
                                which='major')

        else:
            axes[i].yaxis.set_visible(False)
        if idx[0] == 3:
            axes[i].tick_params(axis='x',
                                which='major',
                                rotation=90)
        else:
            axes[i].xaxis.set_visible(False)

    axes[4].set_ylabel(r'$\rho(\Delta t)$[1/yr]')
    axes[4].yaxis.set_label_coords(-1, -0.5)
    axes[18].set_xlabel(r'$\Delta t$[yr]')
    axes[18].xaxis.set_label_coords(-0.22, -1.5)
    plt.savefig('figures/indv_events_p_%s_ref_par_%s.pdf' % (p, ref_par),
                bbox_to_inches='tight')
    plt.close()
