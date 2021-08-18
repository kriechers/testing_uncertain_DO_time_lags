import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gs
from matplotlib.patches import ConnectionPatch
from pyplot_setting_paper import set_pyplot

#################################################################
# NOTE: I crossed checked, and the data provided by Erhardt is  #
# in fact given in BP while the DO events by Rasmussen are      #
# indicated in b2k - hence, age of the Erhardt data require a   #
# 50 year shift in order to use the common b2k time scale       #
#################################################################


set_pyplot(plt.rc, plt.rcParams,
           MEDIUM_SIZE=6,
           SMALL_SIZE=4)
inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def vmarker(x0, x1, ax0, ax1, **kwargs):
    xy0 = (x0, ax0.get_ylim()[0])
    xy1 = (x1, ax1.get_ylim()[1])
    ax0.axvline(x0, **kwargs)
    ax1.axvline(x1, **kwargs)
    con = ConnectionPatch(xy0, xy1, 'data', 'data',
                          ax0, ax1, **kwargs)
    ax0.add_artist(con)


filename = '../data/NGRIP_10yr.csv'
colnames = ['age', 'Na', 'Ca', 'lt', 'd18O']
figsize = (twocol, twocol*0.5)

NGRIP_data = pd.read_csv(filename,
                         header=17,
                         usecols=[0, 1, 2, 3, 4],
                         names=colnames)
NGRIP_data['Ca'] = np.log(NGRIP_data['Ca'])
NGRIP_data['Na'] = np.log(NGRIP_data['Na'])
NGRIP_data['age'] = (NGRIP_data['age'] + 50)/1000

filename = '../data/NEEM_10yr.csv'
NEEM_data = pd.read_csv(filename,
                        header=7,
                        usecols=[0, 1, 2],
                        names=colnames[:3])
NEEM_data['Ca'] = np.log(NEEM_data['Ca'])
NEEM_data['Na'] = np.log(NEEM_data['Na'])
NEEM_data['age'] = (NEEM_data['age'] + 50)/1000

DO_events = pd.read_table('../data/GIS_table.txt',
                          usecols=[0, 1],
                          header=2)

fig = plt.figure(figsize=figsize)
height_ratios = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
gs = fig.add_gridspec(14, 1,
                      hspace=0.5,
                      left=0.1,
                      bottom=0.15,
                      top=0.85,
                      right=0.9,
                      height_ratios=height_ratios)

ax1 = fig.add_subplot(gs[2:6])
ax2 = fig.add_subplot(gs[3:7])
ax3a = fig.add_subplot(gs[6:10])
ax3b = fig.add_subplot(gs[7:11])
ax4a = fig.add_subplot(gs[9:13])
ax4b = fig.add_subplot(gs[10:14])
axe = fig.add_subplot(gs[0])
axa = fig.add_subplot(gs[2:])

NGRIP_axdict = dict(zip(colnames[1:], [ax4a, ax3a, ax2, ax1]))
NEEM_axdict = dict(zip(colnames[1:3], [ax4b, ax3b]))
coldict = {'d18O': 'C0',
           'Ca': 'C1',
           'Na': 'C2',
           'lt': 'C9'}
#################################################################
# plot NGRIP data                                               #
#################################################################

for i, ax in NGRIP_axdict.items():
    ax.plot(NGRIP_data['age'], NGRIP_data[i],
            color=coldict[i],
            lw=0.7)
    make_patch_spines_invisible(NGRIP_axdict[i])
    ax.set_xlim(63, 7)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(coldict[i])
        ax.yaxis.label.set_color(coldict[i])
        ax.tick_params(axis='y', colors=coldict[i])

#################################################################
# plot NEEM data                                               #
#################################################################

for i, ax in NEEM_axdict.items():
    ax.plot(NEEM_data['age'], NEEM_data[i],
            color=coldict[i],
            lw=0.7)
    make_patch_spines_invisible(NEEM_axdict[i])
    ax.set_xlim(63, 7)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(coldict[i])
        ax.yaxis.label.set_color(coldict[i])
        ax.tick_params(axis='y', colors=coldict[i])


make_patch_spines_invisible(axe)
axe.yaxis.set_visible(False)
axe.xaxis.set_label_position('top')
axe.xaxis.set_ticks_position('top')
axe.spines['top'].set_visible(True)
axe.set_xticks(DO_events.index.values)
axe.set_xlim((0, 23))
axe.set_xticklabels(DO_events['Event'].values[::-1],
                    rotation=-45)
# axe.set_ylim((0,24))

make_patch_spines_invisible(axa)
axa.yaxis.set_visible(False)
# axa.vlines(DO_events['Age'].values / 1000, 0, 1,
#          lw=0.7,
#          linestyle='dotted')
axa.set_xlim(ax1.get_xlim())
axa.spines['bottom'].set_visible(True)
axa.set_xlabel('GICC05 Age [ky b2k]')
axa.set_ylim((0, 1.05))
####################################################
# all spines invisible, now make individual spines #
# visible and customize outer appearence           #
####################################################

ax1.set_ylabel('$\delta^{18}$O$\;$[\u2030]')
ax1.yaxis.set_label_coords(-0.04, 0.7)
ax1.spines['left'].set_position(('axes', 0.035))
ax1.spines['left'].set_visible(True)

ax2.set_ylabel(r'$\lambda\;$[cm/y]')
# ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.spines['right'].set_visible(True)
ax2.spines["right"].set_position(("axes", 0.95))
ax2.yaxis.set_label_coords(1.04, 0.5)

# ax3.set_ylim(ax3.get_ylim()[::-1])
ax3a.set_ylabel(r'$ln$(Ca$^{2+}$)[ng/g]')
ax3a.spines['left'].set_visible(True)
ax3a.set_ylim((7, 2))
ax3a.annotate('NGRIP',
              (0.97, 0.8),
              xycoords='axes fraction',
              color='C1')


# ax3b.set_ylabel(r'$ln$(Ca$^{2+}$)[ng/g]')
ax3b.spines['left'].set_visible(True)
ax3b.set_ylim((7, 2))
ax3b.lines[0].set_color('tomato')
ax3b.yaxis.set_tick_params(direction='in',
                           color='tomato',
                           labelcolor='tomato',
                           pad=-12)
ax3b.annotate('NEEM',
              (0.97, 0.8),
              xycoords='axes fraction',
              color='tomato')


ax4a.set_ylabel(r'$ln$(Na$^{+}$)[ng/g]')
ax4a.yaxis.set_ticks_position('right')
ax4a.yaxis.set_label_position('right')
ax4a.set_ylim((5.5, 2))
ax4a.spines['right'].set_visible(True)
ax4a.annotate('NGRIP',
              (-0.02, 0.3),
              xycoords='axes fraction',
              color='C2')

# ax4b.set_ylabel(r'$ln$(Na$^{+}$)[ng/g]')
ax4b.lines[0].set_color('limegreen')
ax4b.yaxis.set_ticks_position('right')
ax4b.yaxis.set_label_position('right')
ax4b.yaxis.set_tick_params(direction='in',
                           color='limegreen',
                           labelcolor='limegreen',
                           pad=-10)

ax4b.set_ylim((5.5, 2))
ax4b.spines['right'].set_visible(True)
ax4b.annotate('NEEM',
              (-0.02, 0.3),
              xycoords='axes fraction',
              color='limegreen')


for i, a in enumerate((DO_events['Age'][::-1]) / 1000):
    vmarker(i, a, axe, axa,
            lw=0.5,
            ls='solid',
            zorder=-100,
            color='lightsteelblue')

figname = 'figures/fig01.pdf'
plt.savefig(figname,
            dpi=300,
            transparent=True,
            bbox_to_inches=True)
