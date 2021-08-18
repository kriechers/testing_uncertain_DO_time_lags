import numpy as np
from scipy.stats import norm, gaussian_kde, t
from tests import ttest_1sided
import matplotlib.pyplot as plt
import matplotlib
from pyplot_setting_paper import set_pyplot
import matplotlib.patches as patches

set_pyplot(plt.rc, plt.rcParams)


inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def skewnorm(mu, sigma, skewness, xx):
    a = norm.cdf(skewness * xx, loc=mu, scale=sigma)
    b = norm.pdf(xx, loc=mu, scale=sigma)
    c = 2*b*a
    delta = np.diff(xx)[0]
    c /= np.sum(c) * delta
    return c


sigma = 10
mu = 0
xx = np.arange(-100, 100, 0.1)
pop = norm.pdf(xx, loc=mu, scale=sigma)
n = 6
m = 2000
certain_sample = np.sort([-12, 1, -5, 4, 7, 16])
skewness = np.array([2.75221492, -0.71965699,  0.78841478,
                     1.02189754,  1.44862946, -0.92476172])
uncertainty = np.array([1.72744132,  2.71276211,  3.37113784,  1.57697431,  2.01936496,
                        1.79375484])
uncertain_sample = []
sampled = np.zeros((n, m))

for i, (x, u, s) in enumerate(zip(certain_sample, uncertainty, skewness)):
    uncertain_sample.append(skewnorm(x, u, s, xx))
    sampled[i, :] = np.random.choice(xx, size=m, p=uncertain_sample[i]*0.1)

certain_sample = np.array(
    [np.sum(xx * pdf) * 0.1 for pdf in uncertain_sample])

z, pz = ttest_1sided(sampled.T, mu=2)
zz = np.arange(-10, 10, 0.01)
pp = np.arange(0, 1, 0.01)
smooth_z = gaussian_kde(z).pdf(zz)
smooth_p = gaussian_kde(pz).pdf(pp)


fig = plt.figure(figsize=(twocol,
                          0.4*twocol))
gs = fig.add_gridspec(1, 3,
                      hspace=0.2,
                      wspace=0.8,
                      left=0.1,
                      right=0.9,
                      bottom=0.15,
                      top=0.65)

ax = fig.add_subplot(gs[:, 0], label='ax')
ax.patch.set_visible(False)
ax.tick_params(axis='y',
               color='C0',
               labelcolor='C0')

rax = fig.add_subplot(gs[:, 0], label='rax')
rax.patch.set_visible(False)
rax.yaxis.set_label_position('right')
rax.yaxis.set_ticks_position('right')
rax.xaxis.set_label_position('bottom')
rax.xaxis.set_ticks_position('bottom')

ax.plot(xx, pop,
        label=r'$\mathcal{P}_{X}\propto\mathcal{N}(\mu=0)$')
ax.legend(loc='lower left',
          bbox_to_anchor=(-0.68, 1.5),
          ncol=1,
          markerscale=0,
          framealpha=1,
          handlelength=0,
          frameon=False)
ax.legend_.texts[0].set_color('C0')


ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\rho_{X}(x)$',
              color='C0')
ax.spines['top'].set_visible(True)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
ax.spines['left'].set_color('C0')
ax.annotate('',
            xy=(2.2, 1.3),
            xycoords='axes fraction',
            xytext=(0.5, 1.3),
            textcoords='axes fraction',
            annotation_clip=False,
            arrowprops=dict(arrowstyle="->", color='C5',
                            shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            connectionstyle='arc3,rad = -0.3',
                            ))
ax.annotate('(a)', xy=(0.1, 0.85), xycoords='axes fraction')
cc = 'C2'

rax.set_xticks(certain_sample)
rax.set_xticklabels([r'$x_{%s}$' % (i)
                     for i in range(1, len(certain_sample)+1)])
# rax.xaxis.set_visible(False)
rax.set_ylabel(r'$\rho_{\hat{Y}_{i}}(\hat{y}_{i})$',
               color='C4')
rax.tick_params(axis='x',
                color=cc,
                labelcolor=cc,
                direction='inout',
                length=10)
for i, tick in enumerate(rax.xaxis.majorTicks):
    offset = 0.18 * (i//2 - i/2)
    tick.label.set_y(offset)
    tick.label.set_fontsize(4.5)

rax.tick_params(axis='y',
                color='C4',
                labelcolor='C4')
rax.spines['right'].set_color('C4')
rax.spines['left'].set_visible(False)
# rax.vlines(certain_sample, 0, 1 +  np.random.randn(n)/30 ,
#           color=cc)
# for i, x in enumerate(certain_sample):
#    rax.annotate(r'$x_{%i}$' % (i), (x, 1.05 + np.random.randn(1)/30),
#                 color=cc,
#                 xycoords='data',
#                 annotation_clip=False)
rax.spines['top'].set_visible(True)
rax.set_xlim((-20, 20))
ax.set_xlim((-20, 20))
for pdf in uncertain_sample:
    mask = pdf > 0.01
    rax.plot(xx[mask], pdf[mask], color='C4')
ax.set_yticks([])
rax.set_yticks([])

eta = 0.05

ax2 = fig.add_subplot(gs[:, 1], label='ax2')
ax2.patch.set_visible(False)
ax2.spines['top'].set_visible(True)
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('olive')

rax2 = fig.add_subplot(gs[:, 1], label='rax2')
rax2.patch.set_visible(False)
rax2.yaxis.set_label_position('right')
rax2.yaxis.set_ticks_position('right')

rax2.set_ylabel(r'$\rho_{\hat{\Phi}}(\hat{\phi})$',
                color='C4')
rax2.tick_params(axis='x',
                 color=cc,
                 labelcolor=cc)
rax2.tick_params(axis='y',
                 color='C4',
                 labelcolor='C4')
rax2.spines['right'].set_color('C4')
rax2.spines['top'].set_visible(True)
rax2.spines['left'].set_visible(False)
# rax2.vlines(ttest_1sided(certain_sample[None,:], mu = 2),
#            0, 1 +  np.random.randn(n)/30 ,
#            color=cc)

idx = np.where(t.cdf(zz, df=5, loc=2) > eta)[0]
threshold = zz[idx[0]]

rax2.plot(zz, smooth_z,
          color='C4',
          label=r'uncertain $\phi$')

ax2.plot(zz, t.pdf(zz, df=5, loc=2),
         color='olive',
         label=r'$H_{0}:\mu \geq 2$')
ax2.legend(loc='lower left',
           bbox_to_anchor=(-0.85, 1.5),
           ncol=1,
           markerscale=0,
           framealpha=1,
           handlelength=0,
           frameon=False)
ax2.legend_.texts[0].set_color('olive')

ax2.set_ylabel(r'$\rho^{0}_{\Phi}(\phi)$',
               color='olive')
ax2.set_xlim((-5, 5))
ax2.set_xlabel(r'$\phi$')
rax2.set_xlim((-5, 5))
ax2.set_yticks([])
rax2.set_yticks([])
rax2.set_xticklabels([r'$\phi(x_{1},...,x_{6})$'], color=cc)
rax2.set_xticks(ttest_1sided(certain_sample[None, :], mu=2)[0])
rax2.tick_params(axis='x',
                 color=cc,
                 length=10,
                 direction='inout')
# rax2.xaxis.majorTicks[0].label.set_y()
ax2.annotate('',
             xy=(2.2, 1.3),
             xycoords='axes fraction',
             xytext=(0.6, 1.3),
             textcoords='axes fraction',
             annotation_clip=False,
             arrowprops=dict(arrowstyle="->", color='C5',
                             shrinkA=5, shrinkB=5,
                             patchA=None, patchB=None,
                             connectionstyle='arc3,rad = -0.3',
                             ))
ax2.annotate('(b)', xy=(0.1, 0.85), xycoords='axes fraction')
ylim = ax2.get_ylim()
ax2.vlines(threshold, *ylim,
           color='red',
           lw=0.8,
           linestyle=':')
ax2.set_ylim(*ylim)

ax3 = fig.add_subplot(gs[:, 2], label='ax2')
ax3.plot(pp, smooth_p,
         color='C4',
         label=r'$\alpha = 0.05$')
ax3.legend(loc='lower left',
           bbox_to_anchor=(-0.7, 1.5),
           ncol=1,
           markerscale=0,
           framealpha=1,
           handlelength=0,
           frameon=False)
ax3.legend_.texts[0].set_color('r')

ax3.spines['top'].set_visible(True)
ax3.xaxis.set_ticks_position('top')
ax3.xaxis.set_label_position('top')
ax3.spines['right'].set_color('C4')
ax3.yaxis.set_ticks_position('right')
ax3.yaxis.set_label_position('right')
ax3.set_yticks([])
ax3.set_ylabel(r'$\rho_{\hat{P}}(\hat{p})$', color='C4')
ax3.set_xlim((0, 0.6))
ylim = ax3.get_ylim()
ax3.vlines(0.05, *ylim,
           color='red',
           lw=0.8,
           linestyle=':')
ax3.annotate('(c)', xy=(0.8, 0.85), xycoords='axes fraction')

ax3.set_xlabel(r'$p$')
rax3 = fig.add_subplot(gs[:, 2], label='rax3')
rax3.set_xticks(ttest_1sided(certain_sample[None, :], mu=2)[1])
rax3.set_xticklabels([r'$p(x_{1},...,x_{6})$'], color=cc)
rax3.spines['right'].set_visible(False)
rax3.yaxis.set_visible(False)
rax3.tick_params(axis='x',
                 color=cc,
                 length=10,
                 direction='inout')
rax3.set_xlim((0, 0.6))

figname = 'figures/fig04.pdf'
plt.savefig(figname,
            dpi=300,
            transparent=True)
